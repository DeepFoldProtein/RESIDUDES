# SPDX-License-Identifier: Apache-2.0
"""Triangle multiplication + triangle attention.

Two implementations live here:

 * Manual JAX path — `Linear` + `LayerNorm` + `jax.nn.dot_product_attention`
   (cuDNN flash on GPU). Works everywhere JAX runs, including environments
   without the ``cuequivariance_jax`` plugin.

 * cuequivariance_jax path — `cuex.triangle_multiplicative_update` and
   `cuex.triangle_attention`. These are the JAX bindings for NVIDIA's
   cuEquivariance fused triangle kernels (the same kernels the TRT engines
   use). On RTX 4090 a single `tri_mul` call on N=76, c_z=256 takes
   ~0.09 ms vs >5 ms for the manual path. Activated by
   ``BIO_TRT_JAX_CUEQ=1`` (default off until measured end-to-end).
"""

from __future__ import annotations

import os
from typing import Literal

import haiku as hk
import jax
import jax.numpy as jnp

from lantern_engine_jax.haiku._tokamax import (
    dot_product_attention as _dpa,
    has_tokamax as _has_tokamax,
)
from lantern_engine_jax.haiku.primitives import LayerNorm, Linear


TriangleEquation = Literal["ikc,jkc->ijc", "kic,kjc->ijc"]


# Lazily import cuequivariance_jax once and cache the result.
_CUEX = None
_CUEX_TRIED = False


def _get_cuex():
    global _CUEX, _CUEX_TRIED
    if _CUEX_TRIED:
        return _CUEX
    _CUEX_TRIED = True
    if os.environ.get("BIO_TRT_JAX_CUEQ", "0").lower() not in (
        "1", "true", "yes", "on"
    ):
        return None
    try:
        import cuequivariance_jax as _cx  # type: ignore
        _CUEX = _cx
    except Exception:
        _CUEX = None
    return _CUEX


def _pick_attention_impl(requested: str | None, dtype=None) -> str | None:
    """Resolve attention impl: ``'cudnn'`` only when GPU + BF16/FP16."""
    if requested == "xla":
        return None
    if requested == "auto":
        if dtype is not None and dtype not in (jnp.bfloat16, jnp.float16):
            return None
        try:
            if any(d.platform == "gpu" for d in jax.devices()):
                return "cudnn"
        except Exception:
            pass
        return None
    return requested


class TriangleMultiplication(hk.Module):
    """Triangle multiplicative update (outgoing or incoming).

    Args:
      equation: 'ikc,jkc->ijc' (outgoing) or 'kic,kjc->ijc' (incoming).
    """

    def __init__(
        self,
        *,
        equation: TriangleEquation,
        name: str | None = None,
    ):
        super().__init__(name=name)
        self.equation = equation

    def __call__(self, act: jax.Array, mask: jax.Array) -> jax.Array:
        # act: [N, N, C], mask: [N, N]
        num_channels = act.shape[-1]

        cuex = _get_cuex()
        if cuex is not None:
            return self._cuex_call(act, mask, cuex)

        # ── Manual JAX path (default) ──────────────────────────────────────
        # Masks arrive as fp32; multiplying a bf16 projection by an fp32 mask
        # promotes the whole pair stack to fp32 for the rest of the block (and,
        # through the residual, for every later block) — cuDNN attention then
        # refuses the fp32 q/k/v. Keep the mask in the activation dtype.
        mask_b = mask[None, ...].astype(act.dtype)  # [1, N, N]
        equation = {
            "ikc,jkc->ijc": "cik,cjk->cij",
            "kic,kjc->ijc": "cki,ckj->cij",
        }[self.equation]

        act = LayerNorm(name="left_norm_input")(act)
        input_act = act

        projection = Linear(
            num_channels * 2, use_bias=False, name="projection",
        )(act)
        gate = Linear(
            num_channels * 2, use_bias=False, initializer="zeros", name="gate",
        )(act)
        projection = jnp.transpose(projection, (2, 0, 1))
        gate = jnp.transpose(gate, (2, 0, 1))
        projection = projection * mask_b
        projection = projection * jax.nn.sigmoid(gate)

        a, b = jnp.split(projection, 2, axis=0)
        act = jnp.einsum(equation, a, b)
        act = LayerNorm(axis=0, param_axis=0, name="center_norm")(act)

        act = jnp.transpose(act, (1, 2, 0))
        act = Linear(
            num_channels, use_bias=False, initializer="zeros",
            name="output_projection",
        )(act)
        gate_out = Linear(
            num_channels, use_bias=False, initializer="zeros",
            name="gating_linear",
        )(input_act)
        return act * jax.nn.sigmoid(gate_out)

    @hk.transparent
    def _cuex_call(self, act, mask, cuex):
        """Fused tri-mul via ``cuex.triangle_multiplicative_update``.

        Allocates the same parameter names the manual path does so the
        same checkpoint converter works in both modes. The Linear weights
        are transposed at call time to match cuex's ``[out, in]`` layout.
        """
        from lantern_engine_jax.haiku.primitives import linear_get_params  # local import

        num_channels = act.shape[-1]

        # Same name-scopes as the manual path so converter rules are unchanged.
        proj_w, _ = linear_get_params(
            num_channels, num_channels * 2,
            dtype=act.dtype, use_bias=False,
            initializer="linear", name="projection",
        )
        gate_w, _ = linear_get_params(
            num_channels, num_channels * 2,
            dtype=act.dtype, use_bias=False,
            initializer="zeros", name="gate",
        )
        out_proj_w, _ = linear_get_params(
            num_channels, num_channels,
            dtype=act.dtype, use_bias=False,
            initializer="zeros", name="output_projection",
        )
        out_gate_w, _ = linear_get_params(
            num_channels, num_channels,
            dtype=act.dtype, use_bias=False,
            initializer="zeros", name="gating_linear",
        )
        with hk.name_scope("left_norm_input"):
            ln_in_scale = hk.get_parameter(
                "scale", (num_channels,), act.dtype, init=jnp.ones)
            ln_in_offset = hk.get_parameter(
                "offset", (num_channels,), act.dtype, init=jnp.zeros)
        with hk.name_scope("center_norm"):
            ln_out_scale = hk.get_parameter(
                "scale", (num_channels,), act.dtype, init=jnp.ones)
            ln_out_offset = hk.get_parameter(
                "offset", (num_channels,), act.dtype, init=jnp.zeros)

        direction = "outgoing" if self.equation == "ikc,jkc->ijc" else "incoming"
        return cuex.triangle_multiplicative_update(
            act,
            direction=direction,
            mask=mask,
            norm_in_weight=ln_in_scale,
            norm_in_bias=ln_in_offset,
            p_in_weight=proj_w.T,    # cuex wants [out, in]
            g_in_weight=gate_w.T,
            norm_out_weight=ln_out_scale,
            norm_out_bias=ln_out_offset,
            p_out_weight=out_proj_w.T,
            g_out_weight=out_gate_w.T,
        )


class TriangleAttention(hk.Module):
    """Pair self-attention with learned nonbatched bias (= AF3 GridSelfAttention).

    Args:
      num_head: heads; channels must divide unless ``head_dim`` is set.
      head_dim: explicit per-head dim. When None, derived from input channels.
        Required for boltz template/MSA pairformers where head_dim != C/H
        (e.g. token_z=64, num_head=4, head_dim=32 — qkv proj is [128, 64]).
      transpose: if True, attend over the second pair axis (j) instead of i.
    """

    def __init__(
        self,
        *,
        num_head: int = 4,
        head_dim: int | None = None,
        transpose: bool = False,
        impl: str | None = "auto",
        name: str | None = None,
    ):
        super().__init__(name=name)
        self.num_head = num_head
        self.head_dim = head_dim
        self.transpose = transpose
        self.impl = impl

    @hk.transparent
    def _attention(
        self, act: jax.Array, mask: jax.Array, bias: jax.Array
    ) -> jax.Array:
        # act: [num_seq, num_res, C], mask: [num_seq, 1, 1, num_res] (bool),
        # bias: [num_head, num_seq?, num_res] (we expand to [1, H, N, N])
        num_channels = act.shape[-1]
        if self.head_dim is not None:
            head_dim = self.head_dim
        else:
            assert num_channels % self.num_head == 0
            head_dim = max(num_channels // self.num_head, 16)
        qkv_shape = (self.num_head, head_dim)

        q = Linear(qkv_shape, use_bias=False, name="q_projection")(act)
        k = Linear(qkv_shape, use_bias=False, name="k_projection")(act)
        v = Linear(qkv_shape, use_bias=False, name="v_projection")(act)

        # q,k,v: [num_seq, num_res, num_head, head_dim] — treat num_seq as batch.
        bias_b = jnp.expand_dims(bias, 0)  # [1, num_head, q, k]

        # cuex.triangle_attention shapes (q/k/v: [B, N, H, S_qo, D],
        # bias: [B, 1, H, S_qo, S_kv]). Both B and N act as parallel batch
        # axes; bias broadcasts over N. Two valid mappings:
        #   * B=num_seq, N=1  — bias materializes to [B, 1, H, N, N] (~N³·H).
        #     Faster at small N (more independent batches → more SM occupancy)
        #     but OOMs at N≳300 due to bias size.
        #   * B=1, N=num_seq — bias only [1, 1, H, N, N] (~N²·H). Always fits
        #     but slower at small N (less parallelism).
        # Cross-over on RTX 4090 is around N≈300; default switch at 256.
        cuex = _get_cuex()
        if cuex is not None and self.impl != "xla":
            num_seq = q.shape[0]; num_res = q.shape[1]
            try:
                bn_swap_at = int(os.environ.get(
                    "BIO_TRT_JAX_CUEQ_TRI_ATTN_BN_SWAP_AT", "256"))
            except ValueError:
                bn_swap_at = 256

            if num_res <= bn_swap_at:
                # B=num_seq, N=1 — high-parallel path, broadcast small bias.
                q_c = jnp.transpose(q, (0, 2, 1, 3))[:, None]  # [B, 1, H, N, D]
                k_c = jnp.transpose(k, (0, 2, 1, 3))[:, None]
                v_c = jnp.transpose(v, (0, 2, 1, 3))[:, None]
                bias_c = bias_b[None].astype(jnp.float32)      # [1, 1, H, N, N]
                bias_c = jnp.broadcast_to(
                    bias_c, (num_seq, 1, self.num_head, num_res, num_res))
                mask_c = mask[:, None]                          # [B, 1, 1, 1, N]
                scale = float(head_dim ** -0.5)
                out_5d, _, _ = cuex.triangle_attention(
                    q_c, k_c, v_c, bias_c, mask_c, scale=scale,
                )
                out = jnp.squeeze(out_5d, axis=1)              # [B, H, N, D]
                out = jnp.transpose(out, (0, 2, 1, 3))         # [B, N, H, D]
            else:
                # B=1, N=num_seq — bias-light path, always fits.
                q_c = jnp.transpose(q, (0, 2, 1, 3))[None]      # [1, num_seq, H, N, D]
                k_c = jnp.transpose(k, (0, 2, 1, 3))[None]
                v_c = jnp.transpose(v, (0, 2, 1, 3))[None]
                bias_c = bias_b[None].astype(jnp.float32)      # [1, 1, H, N, N]
                mask_c = mask[None]                             # [1, num_seq, 1, 1, N]
                scale = float(head_dim ** -0.5)
                out_5d, _, _ = cuex.triangle_attention(
                    q_c, k_c, v_c, bias_c, mask_c, scale=scale,
                )
                out = jnp.squeeze(out_5d, axis=0)              # [num_seq, H, N, D]
                out = jnp.transpose(out, (0, 2, 1, 3))         # [num_seq, N, H, D]
        elif self.impl == "xla":
            out = jax.nn.dot_product_attention(q, k, v, bias=bias_b, mask=mask)
        else:
            out = _dpa(q, k, v, bias=bias_b, mask=mask)

        # Collapse heads.
        out = jnp.reshape(out, out.shape[:-2] + (self.num_head * head_dim,))

        gate_values = Linear(
            self.num_head * head_dim,
            use_bias=False,
            initializer="zeros",
            name="gating_query",
        )(act)
        out = out * jax.nn.sigmoid(gate_values)

        return Linear(
            num_channels, use_bias=False, initializer="zeros",
            name="output_projection",
        )(out)

    def __call__(self, act: jax.Array, pair_mask: jax.Array) -> jax.Array:
        # act: [N, N, C]; pair_mask: [N, N]
        assert act.ndim == 3 and pair_mask.ndim == 2

        if self.transpose:
            act = jnp.swapaxes(act, -2, -3)
            pair_mask = jnp.swapaxes(pair_mask, -1, -2)

        act = LayerNorm(name="act_norm")(act)

        nonbatched_bias = Linear(
            self.num_head, use_bias=False, name="pair_bias_projection",
        )(act)
        nonbatched_bias = jnp.transpose(nonbatched_bias, [2, 0, 1])  # [H, I, J]

        attn_mask = pair_mask[:, None, None, :].astype(jnp.bool_)
        # Fold the key mask into the shared bias. With a per-row mask [N,1,1,N]
        # next to a bias [1,H,N,N], both XLA and the cuDNN path first materialise
        # the broadcast [N,H,N,N] logits bias in fp32 (4·H·N³ bytes: 16 GB at
        # N=1002, 36 GB at N=1315 — the single allocation behind the ≥1000-token
        # JAX OOMs) and the attention runs ~2.5× slower than bias-only. The pair
        # mask is an outer product of a token mask, so every valid row shares the
        # same key mask: adding -1e9 on masked keys to the [H,N,N] bias is exact
        # for valid rows and only changes fully-masked rows (garbage either way).
        # cuEquivariance's kernel takes the mask natively, so it keeps the mask.
        # BIO_TRT_JAX_FOLD_MASK=0 restores the old call.
        use_cuex = _get_cuex() is not None and self.impl != "xla"
        if not use_cuex and os.environ.get("BIO_TRT_JAX_FOLD_MASK", "1") == "1":
            key_valid = jnp.any(pair_mask.astype(jnp.bool_), axis=0)          # [N]
            nonbatched_bias = nonbatched_bias + jnp.where(
                key_valid, 0.0, -1e9).astype(nonbatched_bias.dtype)[None, None, :]
            attn_mask = None
        out = self._attention(act, attn_mask, nonbatched_bias)

        if self.transpose:
            out = jnp.swapaxes(out, -2, -3)
        return out
