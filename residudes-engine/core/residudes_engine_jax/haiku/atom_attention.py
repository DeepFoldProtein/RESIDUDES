# SPDX-License-Identifier: Apache-2.0
"""AtomTransformer + AtomAttentionEncoder + AtomAttentionDecoder (JAX).

Reference:
  models/protenix/protenix/model/modules/transformer.py
    ::AtomTransformer (= DiffusionTransformer with cross_attention_mode=True,
       n_queries=32, n_keys=128, c_a=128, c_s=128, c_z=16, n_heads=4)
    ::AtomAttentionEncoder (Algorithm 5)
    ::AtomAttentionDecoder (Algorithm 6)

Cross-attention windowed attention layout:
  q [..., N_atom, c_a]  -> windowed [..., n_blocks, n_queries, c_a]
  kv [..., N_atom, c_a] -> windowed [..., n_blocks, n_keys, c_a]
  p [..., n_blocks, n_queries, n_keys, c_z]  -> per-window pair bias

Each window does standard self-attention against the wider key window,
masked outside the valid range. Adjacent blocks overlap on the k side.
"""

from __future__ import annotations

from dataclasses import dataclass

import haiku as hk
import jax
import jax.numpy as jnp

from residudes_engine_jax.haiku._tokamax import dot_product_attention as _dpa
from residudes_engine_jax.haiku.adaln import adaptive_layernorm
from residudes_engine_jax.haiku.atom_helpers import (
    aggregate_atom_to_token,
    broadcast_token_to_atom,
    broadcast_token_to_local_atom_pair,
    rearrange_qk_to_dense_trunk,
)
from residudes_engine_jax.haiku.primitives import LayerNorm, Linear


# ── AtomTransformer building block ───────────────────────────────────────

@dataclass(frozen=True)
class AtomTransformerVariant:
    n_blocks: int = 3
    n_heads: int = 4
    c_atom: int = 128
    c_atompair: int = 16
    n_queries: int = 32
    n_keys: int = 128


def _windowed_cross_attention(
    q: jax.Array,                       # [N_atom, c_atom]
    kv: jax.Array,                      # [N_atom, c_atom]
    p_windowed: jax.Array,              # [n_blocks, n_queries, n_keys, c_atompair]
    *,
    n_heads: int,
    n_queries: int,
    n_keys: int,
    name: str,
) -> jax.Array:
    """One layer of cross-attention with sequence-local windowing + pair bias.

    Returns updated q, shape [N_atom, c_atom].
    """
    c_a = q.shape[-1]
    head_dim = c_a // n_heads
    qkv_shape = (n_heads, head_dim)
    n_atom = q.shape[-2]

    # Pair bias: [n_blocks, n_queries, n_keys, c_z] -> [n_heads, n_blocks, n_queries, n_keys]
    p_norm = LayerNorm(create_offset=False, name=f"{name}layernorm_z")(p_windowed)
    bias = Linear(n_heads, use_bias=False, name=f"{name}linear_nobias_z")(p_norm)
    bias = jnp.transpose(bias, (3, 0, 1, 2))  # [H, n_blocks, n_queries, n_keys]

    # Q/K/V projections (q has bias, others no bias)
    q_p = Linear(qkv_shape, use_bias=True, name=f"{name}attention_linear_q")(q)
    k_p = Linear(qkv_shape, use_bias=False, name=f"{name}attention_linear_k")(kv)
    v_p = Linear(qkv_shape, use_bias=False, name=f"{name}attention_linear_v")(kv)
    # shapes: [N_atom, H, D]

    # Window q (size n_queries) and k/v (size n_keys)
    # Use the helper: pad/reshape q, slide-window k/v
    q_w, _, info_q = rearrange_qk_to_dense_trunk(
        q_p, q_p, n_queries=n_queries, n_keys=n_queries, compute_mask=False,
    )
    # q_w: [n_blocks, n_queries, H, D]
    _, kv_w, info_kv = rearrange_qk_to_dense_trunk(
        k_p, k_p, n_queries=n_queries, n_keys=n_keys, compute_mask=True,
    )
    # _ here is the q-window of k_p; we want k_w windowed by n_keys (from kv_w second arg)
    # Re-do for v:
    _, v_w, _ = rearrange_qk_to_dense_trunk(
        v_p, v_p, n_queries=n_queries, n_keys=n_keys, compute_mask=False,
    )
    # k_w / v_w shapes: [n_blocks, n_keys, H, D]
    k_w = kv_w
    mask_w = info_kv["mask_trunked"]  # [n_blocks, n_queries, n_keys] (bool)

    # Per-window attention
    # logits[b, h, i, j] = sum_d q_w[b, i, h, d] * k_w[b, j, h, d] / sqrt(D)
    scale = head_dim ** -0.5
    logits = jnp.einsum("bihd,bjhd->bhij", q_w * scale, k_w)
    # Add pair bias
    logits = logits + bias  # broadcasts: [H, n_blocks, n_queries, n_keys]
    # NOTE: bias shape is [H, n_blocks, n_queries, n_keys], logits is
    # [n_blocks, H, n_queries, n_keys]. We need to swap axes of bias.
    # Above einsum gives [b, H, i, j] = [n_blocks, H, n_queries, n_keys]
    # bias is [H, n_blocks, ...]. Permute to [n_blocks, H, n_queries, n_keys].
    # Actually let me redo: easier to compute bias in correct order.
    # (Let's correct below.)
    return None  # placeholder, will rewrite


# Rewrite cleanly
def _windowed_cross_attn_block(
    q: jax.Array,              # [N_atom, c_a]
    kv: jax.Array,             # [N_atom, c_a]
    p_windowed: jax.Array,     # [n_blocks, n_queries, n_keys, c_z]
    *,
    n_heads: int,
    n_queries: int,
    n_keys: int,
    name: str,
) -> jax.Array:
    c_a = q.shape[-1]
    head_dim = c_a // n_heads

    # Pair bias projection (per-block, per-head)
    p_norm = LayerNorm(create_offset=False, name=f"{name}layernorm_z")(p_windowed)
    # [n_blocks, n_queries, n_keys, n_heads]
    bias = Linear(n_heads, use_bias=False, name=f"{name}linear_nobias_z")(p_norm)
    # -> [n_blocks, n_heads, n_queries, n_keys]
    bias = jnp.transpose(bias, (0, 3, 1, 2))

    # Project as flat [N_atom, c_a] outputs so we can window along axis -2.
    q_p = Linear(c_a, use_bias=True, name=f"{name}attention_linear_q")(q)
    k_p = Linear(c_a, use_bias=False, name=f"{name}attention_linear_k")(kv)
    v_p = Linear(c_a, use_bias=False, name=f"{name}attention_linear_v")(kv)

    # Window q (n_queries) and k/v (n_keys), with shapes:
    #   q_w: [n_blocks, n_queries, c_a], k_w / v_w: [n_blocks, n_keys, c_a]
    q_w, _, _ = rearrange_qk_to_dense_trunk(
        q_p, q_p, n_queries=n_queries, n_keys=n_queries, compute_mask=False,
    )
    _, k_w, info = rearrange_qk_to_dense_trunk(
        k_p, k_p, n_queries=n_queries, n_keys=n_keys, compute_mask=True,
    )
    _, v_w, _ = rearrange_qk_to_dense_trunk(
        v_p, v_p, n_queries=n_queries, n_keys=n_keys, compute_mask=False,
    )
    # Split heads: [n_blocks, n_q/k, c_a] -> [n_blocks, n_q/k, H, D]
    q_w = q_w.reshape(*q_w.shape[:-1], n_heads, head_dim)
    k_w = k_w.reshape(*k_w.shape[:-1], n_heads, head_dim)
    v_w = v_w.reshape(*v_w.shape[:-1], n_heads, head_dim)
    mask_w = info["mask_trunked"]  # [n_blocks, n_queries, n_keys] bool

    # Try tokamax/cudnn fused flash attn first. q_w / k_w / v_w are
    # [n_blocks, T, num_head, head_dim]; bias is [n_blocks, num_head, q, k];
    # mask_w[:, None, :, :] is [n_blocks, 1, q, k] — exactly the layout
    # tokamax.dot_product_attention expects.
    if q_w.dtype != jnp.float32:
        scale = float(head_dim ** -0.5)
        out_w = _dpa(
            q_w, k_w, v_w,
            bias=bias.astype(q_w.dtype),
            mask=mask_w[:, None, :, :].astype(jnp.bool_),
            scale=scale,
        )
    else:
        # FP32 fallback — keep manual einsum path for numerical reference.
        scale = jnp.asarray(head_dim ** -0.5, dtype=q_w.dtype)
        logits = jnp.einsum("bqhd,bkhd->bhqk", q_w * scale, k_w) + bias.astype(q_w.dtype)
        logits = jnp.where(mask_w[:, None, :, :], logits, jnp.asarray(-1e4, dtype=logits.dtype))
        weights = jax.nn.softmax(logits, axis=-1)
        out_w = jnp.einsum("bhqk,bkhd->bqhd", weights, v_w)

    # Unwindow q axis: [n_blocks, n_queries, H, D] -> [N_atom + pad, H, D]
    n_blocks_dim, _, H, D = out_w.shape
    n_atom = q.shape[-2]
    out_unw = out_w.reshape(n_blocks_dim * n_queries, H, D)
    out_unw = out_unw[:n_atom]                # trim padding
    out_flat = out_unw.reshape(n_atom, H * D)  # [N_atom, c_a]

    # Output projection (no bias; zero-init)
    # Note: we apply gate (from q) via element-wise sigmoid.
    gate = Linear(c_a, use_bias=False, initializer="zeros",
                  name=f"{name}attention_linear_g")(q)
    out_gated = jax.nn.sigmoid(gate) * out_flat
    return Linear(c_a, use_bias=False, name=f"{name}attention_linear_o")(out_gated)


def _conditioned_transition_protenix(
    a: jax.Array,
    s: jax.Array,
    *,
    expansion: int,
    name: str,
) -> jax.Array:
    """Protenix ConditionedTransitionBlock: AdaLN + 3-linear FFW + AdaLN-zero gate."""
    c_a = a.shape[-1]
    inner = expansion * c_a
    a_n = adaptive_layernorm(a, s, name=f"{name}adaln_")
    a1 = Linear(inner, use_bias=False, initializer="relu",
                name=f"{name}linear_nobias_a1")(a_n)
    a2 = Linear(inner, use_bias=False, initializer="relu",
                name=f"{name}linear_nobias_a2")(a_n)
    b = jax.nn.silu(a1) * a2
    out = Linear(c_a, use_bias=False, name=f"{name}linear_nobias_b")(b)
    gate = jax.nn.sigmoid(Linear(
        c_a, use_bias=True, bias_init=-2.0, initializer="zeros",
        name=f"{name}linear_s",
    )(s))
    return gate * out


class AtomTransformerBlock(hk.Module):
    """One layer: cross-attention + AdaLN-zero gate + ConditionedTransition."""

    def __init__(self, *, variant: AtomTransformerVariant, name: str | None = None):
        super().__init__(name=name)
        self.variant = variant

    def __call__(self, q, c, p_windowed):
        v = self.variant
        # AdaLN for q (a=q, s=c)
        q_n = adaptive_layernorm(q, c, name="attention_pair_bias_layernorm_a_")
        # AdaLN for kv: PT applies the second AdaLN on the post-AdaLN_a value
        # of q (NOT on c). c is only the conditioning input for both AdaLNs.
        kv_n = adaptive_layernorm(q_n, c, name="attention_pair_bias_layernorm_kv_")
        attn = _windowed_cross_attn_block(
            q_n, kv_n, p_windowed,
            n_heads=v.n_heads, n_queries=v.n_queries, n_keys=v.n_keys,
            name="attention_pair_bias_",
        )
        # AdaLN-zero gate: sigmoid(linear_a_last(c)) * attn
        gate = jax.nn.sigmoid(Linear(
            v.c_atom, use_bias=True, bias_init=-2.0, initializer="zeros",
            name="attention_pair_bias_linear_a_last",
        )(c))
        q = q + gate * attn
        # ConditionedTransition (3-linear gated FFW)
        q = q + _conditioned_transition_protenix(
            q, c, expansion=2, name="conditioned_transition_block_",
        )
        return q


class AtomTransformer(hk.Module):
    """Stack of n_blocks AtomTransformer blocks."""

    def __init__(self, *, variant: AtomTransformerVariant, name: str = "atom_transformer"):
        super().__init__(name=name)
        self.variant = variant

    def __call__(self, q, c, p_windowed):
        for i in range(self.variant.n_blocks):
            q = AtomTransformerBlock(
                variant=self.variant, name=f"diffusion_transformer_blocks_{i}",
            )(q, c, p_windowed)
        return q


# ── AtomAttentionEncoder (Algorithm 5) ────────────────────────────────────

@dataclass(frozen=True)
class AtomAttentionEncoderConfig:
    has_coords: bool = True
    c_token: int = 768
    c_atom: int = 128
    c_atompair: int = 16
    c_s: int = 384
    c_z: int = 256                # Protenix v2
    n_blocks: int = 3
    n_heads: int = 4
    n_queries: int = 32
    n_keys: int = 128
    # Sum of input feature dims for linear_no_bias_f
    f_in_features: int = 1 + 128 + 4 * 64    # ref_mask + ref_element + ref_atom_name_chars

    @property
    def transformer_variant(self) -> AtomTransformerVariant:
        return AtomTransformerVariant(
            n_blocks=self.n_blocks, n_heads=self.n_heads,
            c_atom=self.c_atom, c_atompair=self.c_atompair,
            n_queries=self.n_queries, n_keys=self.n_keys,
        )


class AtomAttentionEncoder(hk.Module):
    """Algorithm 5: atom-level features -> token-level activations.

    Inputs (per atom unless noted):
      ref_pos:               [N_atom, 3]
      ref_charge:            [N_atom]    (will be reshaped to [N_atom, 1])
      ref_mask:              [N_atom]    (0/1)
      ref_element:           [N_atom, 128]   (one-hot)
      ref_atom_name_chars:   [N_atom, 4*64]  (4 chars × 64 categories)
      atom_to_token_idx:     [N_atom]    (token index for each atom)
      d_lm, v_lm:            atom-pair features in WINDOWED layout
                             [n_blocks_atom, n_queries, n_keys, 3 / 1]
      pad_info_mask_trunked: [n_blocks_atom, n_queries, n_keys] (bool/float)
      r_l (optional):        [N_sample, N_atom, 3]   noisy positions
      s (optional):          [N_sample, N_token, c_s] single trunk
      z (optional):          [N_sample, N_token, N_token, c_z] pair trunk

    Outputs:
      a:    [N_sample, N_token, c_token]  token-level activations
      q_l:  [N_sample, N_atom, c_atom]    final atom representation
      c_l:  [N_sample, N_atom, c_atom]    atom conditioning
      p_lm: [n_blocks_atom, n_queries, n_keys, c_atompair]   atom pair (windowed)
    """

    def __init__(
        self,
        *,
        config: AtomAttentionEncoderConfig,
        name: str = "atom_attention_encoder",
    ):
        super().__init__(name=name)
        self.config = config

    @hk.transparent
    def _prepare_cache_atom_pair(self, d_lm, v_lm, mask_trunked):
        """p_lm computation from d_lm, v_lm, pad_info."""
        c_p = self.config.c_atompair
        # d_lm: [..., n_blocks, n_q, n_k, 3]; v_lm: [..., n_blocks, n_q, n_k, 1]
        p_lm = Linear(c_p, use_bias=False, name="linear_no_bias_d")(d_lm) * v_lm
        p_lm = p_lm * mask_trunked[..., None]
        # invd
        invd = 1.0 / (1.0 + (d_lm ** 2).sum(axis=-1, keepdims=True))  # [..., n_b, n_q, n_k, 1]
        p_lm = p_lm + Linear(c_p, use_bias=False, name="linear_no_bias_invd")(invd) * v_lm
        # v
        p_lm = p_lm + Linear(c_p, use_bias=False, name="linear_no_bias_v")(v_lm)
        return p_lm

    @hk.transparent
    def _prepare_cache_atom_single(self, ref_pos, ref_charge, ref_mask, ref_element, ref_atom_name_chars):
        """c_l computation from ref atom features."""
        c_a = self.config.c_atom
        # ref_pos -> linear_no_bias_ref_pos
        c_l = Linear(c_a, use_bias=False, name="linear_no_bias_ref_pos")(ref_pos)
        # ref_charge -> arcsinh + linear_no_bias_ref_charge
        rc = jnp.arcsinh(ref_charge)
        if rc.ndim == ref_pos.ndim - 1:
            rc = rc[..., None]
        c_l = c_l + Linear(c_a, use_bias=False, name="linear_no_bias_ref_charge")(rc)
        # f: concat(ref_mask, ref_element, ref_atom_name_chars)
        rm = ref_mask if ref_mask.ndim == ref_pos.ndim else ref_mask[..., None]
        f_concat = jnp.concatenate([rm, ref_element, ref_atom_name_chars], axis=-1).astype(c_l.dtype)
        c_l = c_l + Linear(c_a, use_bias=False, name="linear_no_bias_f")(f_concat)
        # mask
        c_l = c_l * rm
        return c_l

    def __call__(
        self,
        atom_to_token_idx: jax.Array,
        ref_pos: jax.Array,
        ref_charge: jax.Array,
        ref_mask: jax.Array,
        ref_element: jax.Array,
        ref_atom_name_chars: jax.Array,
        d_lm: jax.Array,
        v_lm: jax.Array,
        mask_trunked: jax.Array,
        *,
        r_l: jax.Array | None = None,
        s: jax.Array | None = None,
        z: jax.Array | None = None,
        n_token: int | None = None,
        p_lm_cache: jax.Array | None = None,
        c_l_cache: jax.Array | None = None,
    ):
        cfg = self.config
        if p_lm_cache is not None:
            p_lm = p_lm_cache
        else:
            # Atom-pair cache (without trunk z)
            p_lm = self._prepare_cache_atom_pair(d_lm, v_lm, mask_trunked)
        if c_l_cache is not None:
            c_l = c_l_cache
        else:
            c_l = self._prepare_cache_atom_single(
                ref_pos, ref_charge, ref_mask, ref_element, ref_atom_name_chars,
            )
        # Add trunk pair contribution (only if no p_lm_cache; cache already
        # includes the trunk contribution in PT)
        add_trunk_z_to_plm = (p_lm_cache is None)
        if r_l is not None and add_trunk_z_to_plm:
            assert z is not None
            z_norm = LayerNorm(create_offset=False, name="layernorm_z")(z)
            z_proj = Linear(cfg.c_atompair, use_bias=False, initializer="zeros",
                            name="linear_no_bias_z")(z_norm)
            z_atom_w, _ = broadcast_token_to_local_atom_pair(
                z_proj, atom_to_token_idx,
                n_queries=cfg.n_queries, n_keys=cfg.n_keys,
                compute_mask=False,
            )
            p_lm = p_lm[..., None, :, :, :, :] if z_atom_w.ndim == p_lm.ndim + 1 else p_lm
            p_lm = p_lm + z_atom_w
        elif r_l is not None and not add_trunk_z_to_plm:
            # When p_lm_cache is provided, layernorm_z + linear_no_bias_z params
            # still need to exist (so converter rules find their targets), but
            # they're not applied here. Touch them as no-ops to register params.
            _ = LayerNorm(create_offset=False, name="layernorm_z")(jnp.zeros((1, 1, cfg.c_z), dtype=p_lm.dtype))
            _ = Linear(cfg.c_atompair, use_bias=False, initializer="zeros",
                       name="linear_no_bias_z")(jnp.zeros((1, 1, cfg.c_z), dtype=p_lm.dtype))

        # Add trunk single + noisy positions
        if r_l is not None:
            assert s is not None
            n_token_inferred = s.shape[-2]
            n_token = n_token if n_token is not None else n_token_inferred
            # Broadcast s to atoms: linear_no_bias_s(layernorm_s(s)) -> [N_s, N_token, c_a]
            s_norm = LayerNorm(create_offset=False, name="layernorm_s")(s)
            s_proj = Linear(cfg.c_atom, use_bias=False, initializer="zeros",
                            name="linear_no_bias_s")(s_norm)
            # Broadcast token->atom
            s_atom = broadcast_token_to_atom(s_proj, atom_to_token_idx)  # [N_s, N_atom, c_a]
            # c_l broadcasts across N_sample
            c_l_b = c_l[..., None, :, :] if c_l.ndim < s_atom.ndim else c_l
            c_l_b = c_l_b + s_atom
            # noisy position
            r_proj = Linear(cfg.c_atom, use_bias=False, name="linear_no_bias_r")(r_l)
            q_l = c_l_b + r_proj
            c_l_used = c_l_b
        else:
            c_l_used = c_l
            q_l = c_l

        # Update p_lm with c_l_q / c_l_k contributions
        c_l_q, c_l_k, _ = rearrange_qk_to_dense_trunk(
            c_l_used, c_l_used,
            n_queries=cfg.n_queries, n_keys=cfg.n_keys, compute_mask=False,
        )
        # c_l_q: [..., n_blocks, n_queries, c_a]
        # c_l_k: [..., n_blocks, n_keys, c_a]
        # p_lm: [..., n_blocks, n_queries, n_keys, c_atompair]
        cl_q_lin = Linear(cfg.c_atompair, use_bias=False,
                          name="linear_no_bias_cl")(jax.nn.relu(c_l_q[..., :, :, None, :]))
        cl_k_lin = Linear(cfg.c_atompair, use_bias=False,
                          name="linear_no_bias_cm")(jax.nn.relu(c_l_k[..., :, None, :, :]))
        p_lm = p_lm + cl_q_lin + cl_k_lin
        # small_mlp on p_lm: ReLU -> Linear -> ReLU -> Linear -> ReLU -> Linear(zeros)
        x = jax.nn.relu(p_lm)
        x = Linear(cfg.c_atompair, use_bias=False, initializer="relu",
                   name="small_mlp_1")(x)
        x = jax.nn.relu(x)
        x = Linear(cfg.c_atompair, use_bias=False, initializer="relu",
                   name="small_mlp_3")(x)
        x = jax.nn.relu(x)
        x = Linear(cfg.c_atompair, use_bias=False, initializer="zeros",
                   name="small_mlp_5")(x)
        p_lm = p_lm + x

        # AtomTransformer (cross-attention with windowing).
        # AtomTransformer takes 2-D inputs per sample [N_atom, c_a].
        # vmap only over axes that actually have N_sample as leading dim.
        tt = AtomTransformer(variant=cfg.transformer_variant)
        if q_l.ndim == 3:  # [N_sample, N_atom, c_atom]
            in_axes = (
                0,                                # q_l: sample axis
                0 if c_l_used.ndim == 3 else None,  # c_l_used may share or per-sample
                0 if p_lm.ndim == 5 else None,     # p_lm may share or per-sample
            )
            q_l = jax.vmap(tt, in_axes=in_axes)(q_l, c_l_used, p_lm)
        else:
            q_l = tt(q_l, c_l_used, p_lm)

        # Aggregate to token: PT does ReLU(Linear(q_l)) — linear THEN relu.
        q_proj = jax.nn.relu(Linear(cfg.c_token, use_bias=False, name="linear_no_bias_q")(q_l))
        a = aggregate_atom_to_token(
            q_proj, atom_to_token_idx, n_token=n_token, reduce="mean",
        )
        return a, q_l, c_l_used, p_lm


# ── AtomAttentionDecoder (Algorithm 6) ────────────────────────────────────

@dataclass(frozen=True)
class AtomAttentionDecoderConfig:
    c_token: int = 768
    c_atom: int = 128
    c_atompair: int = 16
    n_blocks: int = 3
    n_heads: int = 4
    n_queries: int = 32
    n_keys: int = 128

    @property
    def transformer_variant(self) -> AtomTransformerVariant:
        return AtomTransformerVariant(
            n_blocks=self.n_blocks, n_heads=self.n_heads,
            c_atom=self.c_atom, c_atompair=self.c_atompair,
            n_queries=self.n_queries, n_keys=self.n_keys,
        )


class AtomAttentionDecoder(hk.Module):
    """Algorithm 6: token-level activations -> atom position updates.

    Inputs:
      a:                  [N_sample, N_token, c_token]
      atom_to_token_idx:  [N_atom]
      q_skip, c_skip:     [N_sample, N_atom, c_atom]   (encoder cache)
      p_skip:             [n_blocks_atom, n_queries, n_keys, c_atompair]

    Output: r_update [N_sample, N_atom, 3]
    """

    def __init__(
        self, *,
        config: AtomAttentionDecoderConfig,
        name: str = "atom_attention_decoder",
    ):
        super().__init__(name=name)
        self.config = config

    def __call__(self, atom_to_token_idx, a, q_skip, c_skip, p_skip):
        cfg = self.config
        # token -> atom broadcast
        a_atom_proj = Linear(cfg.c_atom, use_bias=False, name="linear_no_bias_a")(a)
        a_atom = broadcast_token_to_atom(a_atom_proj, atom_to_token_idx)
        q = q_skip + a_atom

        # AtomTransformer (vmap only axes that have sample dim).
        tt = AtomTransformer(variant=cfg.transformer_variant)
        if q.ndim == 3:
            in_axes = (
                0,
                0 if c_skip.ndim == 3 else None,
                0 if p_skip.ndim == 5 else None,
            )
            q = jax.vmap(tt, in_axes=in_axes)(q, c_skip, p_skip)
        else:
            q = tt(q, c_skip, p_skip)

        # LayerNorm + linear_no_bias_out -> 3D positions
        q = LayerNorm(create_offset=False, name="layernorm_q")(q)
        r = Linear(3, use_bias=False, name="linear_no_bias_out")(q)
        return r
