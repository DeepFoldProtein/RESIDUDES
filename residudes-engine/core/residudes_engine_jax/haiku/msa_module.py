# SPDX-License-Identifier: Apache-2.0
"""MSAModule (AF3 Algorithm 8) — JAX/Haiku port for Bio-TRT.

Mirrors Protenix's ``MSAModule`` block stack. Inputs already-embedded MSA
(``m: [N_msa, N_token, c_m]``) + pair (``z: [N_token, N_token, c_z]``) +
pair_mask, returns updated ``z``. The PT-side preprocessing (one-hot
encoding, sampling, ``linear_no_bias_m`` / ``linear_no_bias_s`` input
projections) stays in PyTorch — those are ~50 ms on 9fm7. The 4 blocks
are the bottleneck (~750 ms/cycle on N=644).

Each block:
  z += OuterProductMean(m)              # MSA→pair communication
  m += MSAStack(m, z)                   # only on non-last blocks
  z = PairformerBlock(no single)(z, pair_mask)

Reuses the existing ``PairformerBlock`` (with_single=False) for the pair
stack so the cuex tri_mul / tri_attn / scan optimizations all apply.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import haiku as hk
import jax
import jax.numpy as jnp

from residudes_engine_jax.haiku.pairformer import PairformerBlock, PairformerVariant
from residudes_engine_jax.haiku.primitives import LayerNorm, Linear, linear_get_params


def _chunked_ln(x: jax.Array, *, name: str, leading_axis: int = 0,
                chunk_size: int | None = None) -> jax.Array:
    """LayerNorm with the leading axis chunked via ``hk.scan``.

    JAX's ``jnp.mean`` upcasts bf16 inputs to fp32 during reduction, which
    materializes a [leading, ..., c] fp32 tensor (e.g. 1 GB at MSA depth
    3194 × N=644 × c_m=128). Splitting the leading axis into ``chunk_size``
    slices bounds the peak fp32 footprint to one chunk's worth.

    Params (scale/offset) are allocated OUTSIDE the scan body so Haiku
    auto-naming doesn't create per-iteration duplicates.
    """
    if chunk_size is None:
        try:
            chunk_size = int(os.environ.get(
                "BIO_TRT_JAX_MSA_LN_CHUNK", "256"))
        except ValueError:
            chunk_size = 256

    c = x.shape[-1]
    with hk.name_scope(name):
        scale = hk.get_parameter("scale", (c,), x.dtype, init=jnp.ones)
        offset = hk.get_parameter("offset", (c,), x.dtype, init=jnp.zeros)

    def manual_ln(slc):
        # bf16-friendly LN with single fp32 conversion of the slice only.
        s32 = slc.astype(jnp.float32)
        mean = jnp.mean(s32, axis=-1, keepdims=True)
        cent = s32 - mean
        var = jnp.mean(cent * cent, axis=-1, keepdims=True)
        inv = jax.lax.rsqrt(var + 1e-5)
        return ((cent * inv) * scale.astype(jnp.float32)
                + offset.astype(jnp.float32)).astype(x.dtype)

    leading = x.shape[leading_axis]
    if leading <= chunk_size:
        return manual_ln(x)

    pad = (-leading) % chunk_size
    if pad:
        pad_shape = list(x.shape); pad_shape[leading_axis] = pad
        x_p = jnp.concatenate(
            [x, jnp.zeros(pad_shape, dtype=x.dtype)], axis=leading_axis)
    else:
        x_p = x
    out_buf = jnp.zeros(x_p.shape, dtype=x.dtype)

    def step(buf, idx):
        slc = jax.lax.dynamic_slice_in_dim(x_p, idx, chunk_size, axis=leading_axis)
        slc_out = manual_ln(slc)
        buf = jax.lax.dynamic_update_slice_in_dim(buf, slc_out, idx, axis=leading_axis)
        return buf, ()

    slice_starts = jnp.arange(0, x_p.shape[leading_axis], chunk_size)
    out_buf, _ = hk.scan(step, out_buf, slice_starts)
    if pad:
        out_buf = jax.lax.dynamic_slice_in_dim(
            out_buf, 0, leading, axis=leading_axis)
    return out_buf


@dataclass(frozen=True)
class MsaModuleVariant:
    """Hyperparameters that vary by model. Protenix v2 defaults."""
    num_blocks: int = 4
    c_m: int = 128
    c_z: int = 256
    opm_c_hidden: int = 32          # OuterProductMean hidden
    pwa_c: int = 8                  # MSAPairWeightedAveraging head dim
    pwa_n_heads: int = 8
    msa_transition_factor: int = 4
    pair_num_head: int = 8          # tri_attn heads (matches main pairformer v2)
    pair_transition_factor: int = 4

    @classmethod
    def protenix_v2(cls, num_blocks: int = 4) -> "MsaModuleVariant":
        return cls(num_blocks=num_blocks)


class OuterProductMean(hk.Module):
    """AF3 Algorithm 9. ``m -> z`` communication via outer product.

    Uses AF3's structure:
      output_w: [c_outer, c_outer, c_z]   (3-D, NOT a flat Linear)
      out[i,j,f] = sum_{c,e,s} a[s,i,c] * b[s,j,e] * output_w[c,e,f] + b_out[f]

    Done as two einsums per residue chunk. Chunking is over the OUTPUT
    residue axis (i), which keeps the peak intermediate
    ``[chunk_i, c_outer, c_outer, N]`` ~84 MB at chunk=64, N=644 — vs the
    naive [N, N, c_outer, c_outer] = 1.6 GB full-shape allocation. Implemented
    with ``hk.scan`` + ``dynamic_update_slice`` (AF3's sharded_apply pattern)
    so XLA cannot unroll the loop.

    PT checkpoint compatibility: Protenix stores ``linear_out`` as a flat
    ``[c_z, c_outer²]`` Linear. Our converter reshapes that to
    ``[c_outer, c_outer, c_z]`` for ``output_w``; bias unchanged.
    """

    def __init__(self, *, c_z: int, c_hidden: int = 32, eps: float = 1e-3,
                 name: str = "outer_product_mean"):
        super().__init__(name=name)
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.eps = eps

    def __call__(self, m: jax.Array, mask: jax.Array | None = None) -> jax.Array:
        # m: [N_msa, N_token, c_m]; mask: [N_msa, N_token] (or None=all-ones).
        m = _chunked_ln(m, name="layer_norm", leading_axis=0)
        if mask is None:
            mask = jnp.ones(m.shape[:-1], dtype=m.dtype)
        mask3 = mask[..., None].astype(m.dtype)             # [S, N, 1]; keep bf16 activations bf16
        a = Linear(self.c_hidden, use_bias=False, name="linear_1")(m) * mask3
        b = Linear(self.c_hidden, use_bias=False, name="linear_2")(m) * mask3

        S, N = a.shape[-3], a.shape[-2]
        c = self.c_hidden

        # 3-D output_w + bias. Init=zero to match Protenix's "final" init
        # in the original linear_out (which is also zero-init at training start
        # per Protenix's checkpointing).
        output_w = hk.get_parameter(
            "output_w", (c, c, self.c_z), a.dtype, init=jnp.zeros)
        output_b = hk.get_parameter(
            "output_b", (self.c_z,), a.dtype, init=jnp.zeros)

        try:
            chunk = int(os.environ.get("BIO_TRT_JAX_MSA_OPM_CHUNK", "64"))
        except ValueError:
            chunk = 64
        chunk = min(chunk, N)

        # Chunk over the OUTPUT first-residue axis (i). Each chunk slices
        # ``a`` along its N-axis (axis -2), keeps ``b`` full, and produces
        # ``out[chunk_i, N, c_z]``. AF3's einsum chain:
        #   a_slice [S, chunk_i, c] → transpose → [S, c, chunk_i]
        #   mid = einsum('Scb,Sde->dceb', a_slice_t, b)  # [N, c, c, chunk_i]
        #   out_chunk = einsum('dceb,cef->dbf', mid, output_w) + output_b
        #   → [N, chunk_i, c_z]; transpose → [chunk_i, N, c_z]
        def compute_chunk(a_slice):
            # a_slice: [S, chunk_i, c]
            # b:       [S, N, c]
            # mid[d=N, c, e, b=chunk_i] = sum_S a_slice[S, b, c] * b[S, d, e]
            # Reformulate via explicit matmul to give XLA a clear contraction
            # order (avoids a [S, c, chunk_i, N, c] intermediate that some
            # paths materialize at large S):
            #   a_flat = reshape(a_slice, [S, chunk_i*c])
            #   b_flat = reshape(b, [S, N*c])
            #   m_flat[chunk_i*c, N*c] = a_flat.T @ b_flat
            #   reshape to [chunk_i, c_left, N, c_right] → transpose
            S_, chunk_i, c1 = a_slice.shape
            N_full, c2 = b.shape[1], b.shape[2]
            a_flat = a_slice.reshape(S_, chunk_i * c1)
            b_flat = b.reshape(S_, N_full * c2)
            m_flat = jnp.matmul(a_flat.T, b_flat)            # [chunk_i*c1, N*c2]
            mid = m_flat.reshape(chunk_i, c1, N_full, c2)    # [chunk_i, c, N, c]
            mid = jnp.transpose(mid, (2, 1, 3, 0))           # [N, c, c, chunk_i]
            out = jnp.einsum("dceb,cef->dbf", mid, output_w) + output_b
            return jnp.transpose(out, (1, 0, 2))             # [chunk_i, N, c_z]

        # Pad ``a`` on its N-axis (axis=-2) to a multiple of chunk.
        pad = (-N) % chunk
        if pad:
            pad_shape = list(a.shape); pad_shape[-2] = pad
            a_p = jnp.concatenate([a, jnp.zeros(pad_shape, dtype=a.dtype)], axis=-2)
        else:
            a_p = a
        N_p = a_p.shape[-2]
        out_buf = jnp.zeros((N_p, N, self.c_z), dtype=a.dtype)

        def step(buf, idx):
            slc = jax.lax.dynamic_slice_in_dim(a_p, idx, chunk, axis=-2)
            slc_out = compute_chunk(slc)
            buf = jax.lax.dynamic_update_slice_in_dim(buf, slc_out, idx, axis=0)
            return buf, ()

        slice_starts = jnp.arange(0, N_p, chunk)
        out_buf, _ = hk.scan(step, out_buf, slice_starts)
        outer = out_buf[:N]                                  # drop pad

        # Normalize by valid (i, j) count: sum_S mask[s,i] * mask[s,j].
        norm = jnp.einsum("si,sj->ij", mask, mask)[..., None]
        # divide in the mask's (fp32) precision, hand back the activation dtype
        return (outer / (norm + self.eps)).astype(outer.dtype)


class MSAPairWeightedAveraging(hk.Module):
    """AF3 Algorithm 10. ``z -> m`` communication: pair-bias weighted average."""

    def __init__(self, *, c: int = 8, n_heads: int = 8,
                 name: str = "msa_pair_weighted_averaging"):
        super().__init__(name=name)
        self.c = c
        self.n_heads = n_heads

    def __call__(self, m: jax.Array, z: jax.Array) -> jax.Array:
        H, C = self.n_heads, self.c
        m = _chunked_ln(m, name="layernorm_m", leading_axis=0)
        v = Linear(H * C, use_bias=False, name="linear_no_bias_mv")(m)
        v = v.reshape(v.shape[:-1] + (H, C))                # [S, N, H, C]
        b = Linear(H, use_bias=False, name="linear_no_bias_z")(
            LayerNorm(name="layernorm_z")(z))               # [N, N, H]
        g = Linear(H * C, use_bias=False, initializer="zeros",
                   name="linear_no_bias_mg")(m)
        g = jax.nn.sigmoid(g.reshape(g.shape[:-1] + (H, C)))  # [S, N, H, C]
        w = jax.nn.softmax(b, axis=-2)                       # softmax over j (kv axis)
        # wv[s,i,h,c] = sum_j w[i,j,h] * v[s,j,h,c]
        wv = jnp.einsum("ijh,sjhc->sihc", w, v)
        o = (g * wv).reshape(g.shape[:-2] + (H * C,))        # [S, N, H*C]
        return Linear(m.shape[-1], use_bias=False, initializer="zeros",
                      name="linear_no_bias_out")(o)


class MSATransition(hk.Module):
    """SwiGLU FFW for MSA track.

    Two design notes:

    1. Naming: Protenix's PT checkpoint uses ``transition_m.linear_no_bias_a/b``
       + ``linear_no_bias`` (3-linear form). The converter concatenates a/b
       into the 2-linear ``transition1/transition2`` form, so the JAX side
       matches the existing ``TransitionBlock`` layout.

    2. S-chunking: at inference the MSA depth can reach several thousand
       sequences (e.g. 9fm7 has S≈3194). The intermediate tensor
       ``[S, N, 2*intermediate]`` would be 16+ GB in bf16. We split the S
       axis into chunks (default 256) and run the FFN per chunk. Output is
       gathered. Chunk size controlled by ``BIO_TRT_JAX_MSA_TRANSITION_CHUNK``
       (default 256, set ≥S to disable chunking).
    """

    def __init__(self, *, num_intermediate_factor: int = 4,
                 name: str = "transition_m"):
        super().__init__(name=name)
        self.num_intermediate_factor = num_intermediate_factor

    def __call__(self, act: jax.Array) -> jax.Array:
        num_channels = act.shape[-1]
        num_intermediate = num_channels * self.num_intermediate_factor
        # Pre-allocate weights once outside the chunk loop.
        with hk.name_scope("input_layer_norm"):
            ln_scale = hk.get_parameter(
                "scale", (num_channels,), act.dtype, init=jnp.ones)
            ln_offset = hk.get_parameter(
                "offset", (num_channels,), act.dtype, init=jnp.zeros)
        from residudes_engine_jax.haiku.primitives import linear_get_params
        w1, _ = linear_get_params(
            num_channels, num_intermediate * 2,
            dtype=act.dtype, use_bias=False,
            initializer="relu", name="transition1",
        )
        w2, _ = linear_get_params(
            num_intermediate, num_channels,
            dtype=act.dtype, use_bias=False,
            initializer="zeros", name="transition2",
        )

        try:
            chunk = int(os.environ.get(
                "BIO_TRT_JAX_MSA_TRANSITION_CHUNK", "256"))
        except ValueError:
            chunk = 256

        def _ffn(x):
            x = (x - x.mean(axis=-1, keepdims=True))
            var = jnp.mean(x * x, axis=-1, keepdims=True)
            x = x * jax.lax.rsqrt(var + 1e-5)
            x = x * ln_scale + ln_offset
            x = jnp.dot(x, w1)
            a, b = jnp.split(x, 2, axis=-1)
            c = jax.nn.swish(a) * b
            return jnp.dot(c, w2)

        # Single shot if S already small enough.
        if act.ndim < 3 or act.shape[-3] <= chunk:
            return _ffn(act)

        # AF3-style sharded_apply: ``hk.scan`` over fixed-size chunks
        # with ``dynamic_update_slice``. XLA cannot unroll a scan body, so
        # the per-chunk peak memory bound holds at runtime.
        S = act.shape[-3]
        pad = (-S) % chunk
        if pad:
            pad_shape = list(act.shape); pad_shape[-3] = pad
            act_p = jnp.concatenate(
                [act, jnp.zeros(pad_shape, dtype=act.dtype)], axis=-3)
        else:
            act_p = act

        out_shape = act_p.shape[:-1] + (num_channels,)
        out_buf = jnp.zeros(out_shape, dtype=act.dtype)

        def step(buf, idx):
            slc = jax.lax.dynamic_slice_in_dim(act_p, idx, chunk, axis=-3)
            slc_out = _ffn(slc)
            buf = jax.lax.dynamic_update_slice_in_dim(buf, slc_out, idx, axis=-3)
            return buf, ()

        slice_starts = jnp.arange(0, act_p.shape[-3], chunk)
        out_buf, _ = hk.scan(step, out_buf, slice_starts)
        if pad:
            out_buf = jax.lax.dynamic_slice_in_dim(out_buf, 0, S, axis=-3)
        return out_buf


class MSAStack(hk.Module):
    """One MSA self-update: pair-weighted averaging + transition."""

    def __init__(self, *, c: int = 8, n_heads: int = 8,
                 num_intermediate_factor: int = 4,
                 name: str = "msa_stack"):
        super().__init__(name=name)
        self.c = c
        self.n_heads = n_heads
        self.factor = num_intermediate_factor

    def __call__(self, m: jax.Array, z: jax.Array) -> jax.Array:
        m = m + MSAPairWeightedAveraging(c=self.c, n_heads=self.n_heads)(m, z)
        m = m + MSATransition(num_intermediate_factor=self.factor)(m)
        return m


class MSAModuleBlock(hk.Module):
    """One MSAModule block: OPM (m→z) + MSAStack (m self) + pair stack (z self)."""

    def __init__(
        self,
        *,
        variant: MsaModuleVariant,
        is_last_block: bool,
        name: str | None = None,
    ):
        super().__init__(name=name)
        self.variant = variant
        self.is_last_block = is_last_block

    def __call__(self, m: jax.Array, z: jax.Array, pair_mask: jax.Array,
                 m_mask: jax.Array | None = None):
        v = self.variant
        # 1. m → z: outer product mean
        z = z + OuterProductMean(c_z=v.c_z, c_hidden=v.opm_c_hidden,
                                 name="outer_product_mean_msa")(m, m_mask)
        # 2. m self-update (skip on last block)
        if not self.is_last_block:
            m = MSAStack(c=v.pwa_c, n_heads=v.pwa_n_heads,
                         num_intermediate_factor=v.msa_transition_factor,
                         name="msa_stack")(m, z)
        # 3. z self-update via pair stack — reuse main PairformerBlock
        # without single track. Names match the converter (same Protenix
        # tri_mul_in/out, tri_att_start/end, pair_transition layout).
        pf_variant = PairformerVariant(
            num_blocks=1,
            token_s=0,                        # unused (with_single=False)
            token_z=v.c_z,
            num_head_pair=v.pair_num_head,
            num_head_single=0,                # unused
            transition_factor=v.pair_transition_factor,
        )
        z = PairformerBlock(variant=pf_variant, with_single=False,
                            name="pair_stack")(z, pair_mask)
        return m, z


class MSAModuleStack(hk.Module):
    """N MSAModule blocks in sequence. ``use_scan`` for layer_stack."""

    def __init__(
        self,
        *,
        variant: MsaModuleVariant,
        use_scan: bool = False,
        remat: bool = True,
        name: str = "msa_module",
    ):
        super().__init__(name=name)
        self.variant = variant
        self.use_scan = use_scan
        self.remat = remat

    def __call__(self, m: jax.Array, z: jax.Array, pair_mask: jax.Array,
                 m_mask: jax.Array | None = None) -> jax.Array:
        v = self.variant
        if self.use_scan:
            return self._scan_call(m, z, pair_mask, m_mask)

        for i in range(v.num_blocks):
            block = MSAModuleBlock(
                variant=v,
                is_last_block=(i + 1 == v.num_blocks),
                name=f"msa_block_{i}",
            )
            block_fn = hk.remat(block) if self.remat else block
            m, z = block_fn(m, z, pair_mask, m_mask)
        return z

    @hk.transparent
    def _scan_call(self, m, z, pair_mask, m_mask):
        """layer_stack version. Note: last block has different behavior
        (skips MSA self-update). To use scan we still run the MSA self-
        update on the last block — its output is unused, so accuracy is
        unchanged but compute is slightly higher. The scan body must be
        homogeneous, so we trade a little compute for compile-time wins.
        """
        v = self.variant
        # All blocks treated as non-last for shape uniformity. Last block's
        # m output is discarded after scan exits.
        def block_fn(carry):
            mm, zz = carry
            mm, zz = MSAModuleBlock(
                variant=v, is_last_block=False, name="msa_block",
            )(mm, zz, pair_mask, m_mask)
            return (mm, zz)

        stack = hk.experimental.layer_stack(v.num_blocks)(block_fn)
        m, z = stack((m, z))
        return z
