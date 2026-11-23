# SPDX-License-Identifier: Apache-2.0
"""Adaptive LayerNorm + AdaLN-zero for the diffusion (token) transformer.

These are functional helpers (not hk.Module classes) because the diffusion
track in AF3 is composed of Transformer blocks that pass single_cond through
to every sub-step. Mirroring AF3's name-prefix convention keeps the param
keys identical, so the same weight converter works for both.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from lantern_engine_jax.haiku._tokamax import dot_product_attention as _dpa
from lantern_engine_jax.haiku.primitives import LayerNorm, Linear


def adaptive_layernorm(
    x: jax.Array,
    single_cond: jax.Array | None,
    *,
    name: str = "",
) -> jax.Array:
    """LayerNorm modulated by `single_cond` (DiT-style AdaLN).

    When single_cond is None, behaves like a plain LayerNorm.
    """
    if single_cond is None:
        return LayerNorm(name=f"{name}layer_norm")(x)

    x = LayerNorm(
        name=f"{name}layer_norm",
        create_scale=False,
        create_offset=False,
    )(x)
    s = LayerNorm(
        name=f"{name}single_cond_layer_norm",
        create_offset=False,
    )(single_cond)
    scale = Linear(
        x.shape[-1], initializer="zeros", use_bias=True,
        name=f"{name}single_cond_scale",
    )(s)
    # OF3 PT linear_s has no bias.
    bias = Linear(
        x.shape[-1], use_bias=False, initializer="zeros",
        name=f"{name}single_cond_bias",
    )(s)
    return jax.nn.sigmoid(scale) * x + bias


def adaptive_zero_init(
    x: jax.Array,
    num_channels: int,
    single_cond: jax.Array | None,
    *,
    name: str = "",
) -> jax.Array:
    """Final-init projection, optionally gated by single_cond (AdaLN-zero).

    OF3's `linear_o` has no bias. The cond linear (only used when single_cond
    is given) keeps `bias_init=-2.0` to match AF3/Protenix v2 — OF3 doesn't
    use it on the pairformer single track (single_cond=None there).
    """
    if single_cond is None:
        return Linear(
            num_channels, use_bias=False, initializer="zeros",
            name=f"{name}transition2",
        )(x)

    out = Linear(
        num_channels, use_bias=False, name=f"{name}transition2",
    )(x)
    cond = Linear(
        out.shape[-1],
        initializer="zeros",
        use_bias=True,
        bias_init=-2.0,
        name=f"{name}adaptive_zero_cond",
    )(single_cond)
    return jax.nn.sigmoid(cond) * out


def transition_block(
    x: jax.Array,
    *,
    num_intermediate_factor: int,
    single_cond: jax.Array | None = None,
    name: str = "",
) -> jax.Array:
    """SwiGLU FFW with AdaLN, used inside diffusion transformer blocks."""
    num_channels = x.shape[-1]
    num_intermediates = num_intermediate_factor * num_channels

    x = adaptive_layernorm(x, single_cond, name=f"{name}ffw_")
    # OF3 swiglu has no biases.
    x = Linear(
        num_intermediates * 2, use_bias=False, initializer="relu",
        name=f"{name}ffw_transition1",
    )(x)
    a, b = jnp.split(x, 2, axis=-1)
    c = jax.nn.swish(a) * b
    return adaptive_zero_init(c, num_channels, single_cond, name=f"{name}ffw_")


def self_attention(
    x: jax.Array,
    mask: jax.Array,
    pair_logits: jax.Array | None,
    *,
    num_head: int,
    key_dim: int | None = None,
    value_dim: int | None = None,
    single_cond: jax.Array | None = None,
    pre_norm: bool = True,
    name: str = "",
) -> jax.Array:
    """Multihead self-attention with AdaLN, optional pair-bias logits.

    Numerically matches AF3's diffusion_transformer.self_attention: q/k/bias
    are upcast to float32 for the qk^T einsum, then softmax + cast back.

    Set ``pre_norm=False`` to skip the input AdaLN/LayerNorm — needed for
    Boltz-style blocks where the AdaLN is applied at the layer level
    *before* attention.
    """
    assert x.ndim == mask.ndim + 1
    num_channels = x.shape[-1]
    key_dim = (key_dim or num_channels) // num_head
    value_dim = (value_dim or num_channels) // num_head
    qk_shape = (num_head, key_dim)

    bias = (1e9 * (mask - 1.0))[..., None, None, :]

    if pre_norm:
        x = adaptive_layernorm(x, single_cond, name=name)

    # OF3 attn_pair_bias.mha: q has bias, k/v don't. tri_att.mha: none have bias.
    # We pass `q has bias` here since this fn is used for the single track.
    q = Linear(qk_shape, use_bias=True, name=f"{name}q_projection")(x)
    k = Linear(qk_shape, use_bias=False, name=f"{name}k_projection")(x)
    v = Linear(
        (num_head, value_dim), use_bias=False, name=f"{name}v_projection"
    )(x)

    # Try fused flash attention via _dpa (tokamax → cudnn → xla). This handles
    # mask + bias + scale in a single Pallas/cudnn kernel — much faster than
    # the einsum-softmax-einsum chain at BF16 on H100/B200.
    # We only need tokamax DPA when input is BF16/FP16 + we have a Pallas kernel;
    # for FP32 we keep the einsum path to preserve AF3-style precision.
    if x.dtype != jnp.float32:
        # Tokamax / jax.nn DPA require ``bias.ndim == q.ndim``. q is
        # ``[*B, T, num_head, head_dim]`` so bias must be
        # ``[*B (1-broadcast), num_head, T, T]`` with the same ndim.
        # Mask bias: mask shape ``[*B, T]`` -> ``[*B, 1, 1, T]`` (ndim = q.ndim).
        b = (1e9 * (mask.astype(q.dtype) - 1.0))[..., None, None, :]
        if pair_logits is not None:
            # pair_logits is ``[num_head, T, T]`` (3 dims). Pad with leading 1s
            # to match q.ndim before adding (broadcasts over batch dims).
            pl = pair_logits.astype(q.dtype)
            while pl.ndim < q.ndim:
                pl = pl[None]
            b = b + pl
        scale = float(key_dim ** -0.5)
        avg = _dpa(q, k, v, bias=b.astype(q.dtype), scale=scale)
        avg = jnp.reshape(avg, avg.shape[:-2] + (-1,))
    else:
        # Manual einsum path — kept for FP32 numerical reference.
        scale = key_dim ** -0.5
        logits = jnp.einsum("...qhc,...khc->...hqk", q * scale, k) + bias
        if pair_logits is not None:
            logits = logits + pair_logits.astype(logits.dtype)
        weights = jax.nn.softmax(logits, axis=-1).astype(x.dtype)
        avg = jnp.einsum("...hqk,...khc->...qhc", weights, v)
        avg = jnp.reshape(avg, avg.shape[:-2] + (-1,))

    # OF3 mha.linear_g has no bias.
    gate = Linear(
        num_head * value_dim,
        use_bias=False,
        initializer="zeros",
        name=f"{name}gating_query",
    )(x)
    avg = avg * jax.nn.sigmoid(gate)

    return adaptive_zero_init(avg, num_channels, single_cond, name=name)
