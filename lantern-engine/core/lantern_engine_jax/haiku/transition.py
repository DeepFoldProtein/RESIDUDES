# SPDX-License-Identifier: Apache-2.0
"""SwiGLU TransitionBlock for the pair track.

The pair-track transition is a 4x feed-forward with a Gated Linear Unit
(SwiGLU) activation. We use the explicit-split form
``a, b = split(Linear(2N)(act)); swish(a) * b`` which XLA fuses to a
single contraction. (Tried ``tokamax.gated_linear_unit`` with weight
shape ``[in, 2, N]``; on RTX 4090 it was net-slower than XLA's fusion
even at N=644, so we keep the simpler form.)

At inference on N≳600 the [N, N, 2*intermediate] intermediate becomes
~1.6 GB. AF3 chunks this over the leading "broadcast" residue dim via
``mapping.sharded_apply``. We do the same when N×N×intermediate exceeds
``BIO_TRT_JAX_TRANSITION_SHARD_THRESHOLD`` (default 1 GB), using
``hk.scan`` + ``dynamic_update_slice`` so XLA cannot unroll the loop.
"""

from __future__ import annotations

import os

import haiku as hk
import jax
import jax.numpy as jnp

from lantern_engine_jax.haiku.primitives import LayerNorm, Linear, linear_get_params


class TransitionBlock(hk.Module):
    """Pair-track transition with SwiGLU."""

    def __init__(
        self,
        *,
        num_intermediate_factor: int = 4,
        name: str | None = None,
    ):
        super().__init__(name=name)
        self.num_intermediate_factor = num_intermediate_factor

    def __call__(self, act: jax.Array) -> jax.Array:
        num_channels = act.shape[-1]
        num_intermediate = num_channels * self.num_intermediate_factor

        # Estimate intermediate footprint and decide whether to shard along
        # the leading axis (-3, the "broadcast" dim — for pair input that's
        # the row residue axis i in [N, N, c_z]). The intermediate shape is
        # ``act.shape[:-1] + (2*num_intermediate,)`` so for [N, N, c_z] with
        # c_z=256, factor=4 → 2*1024=2048 → [N, N, 2048] = 1.58 GB at N=644.
        # Default: never shard (8 EiB threshold). Set the env var to a real
        # byte threshold (e.g. 1 GB = 1073741824) to enable AF3-style residue
        # sharding for memory at large N. Sharding adds ~50ms/cycle compile
        # + dispatch overhead at typical scales — opt in only when needed.
        try:
            shard_thresh = int(os.environ.get(
                "BIO_TRT_JAX_TRANSITION_SHARD_THRESHOLD",
                str(1 << 63)))
        except ValueError:
            shard_thresh = 1 << 63
        try:
            shard_size = int(os.environ.get(
                "BIO_TRT_JAX_TRANSITION_SHARD_SIZE", "32"))
        except ValueError:
            shard_size = 32

        # bytes_per_elt: bf16=2, fp16=2, fp32=4. Default to dtype inferred.
        bpe = jnp.dtype(act.dtype).itemsize
        # Skip sharding for low-rank (≤2D) acts (single-track transitions).
        do_shard = (
            act.ndim >= 3
            and act.shape[-3] > shard_size
            and (act.size // act.shape[-1]) * (2 * num_intermediate) * bpe
                > shard_thresh
        )

        # Fast path (default): use the original Haiku module composition.
        # This dispatches to tokamax LayerNorm when available and otherwise
        # uses the high-quality fallback in primitives.LayerNorm. Avoids any
        # regression vs the pre-shard implementation.
        if not do_shard:
            x = LayerNorm(name="input_layer_norm")(act)
            x = Linear(num_intermediate * 2, use_bias=False,
                       initializer="relu", name="transition1")(x)
            a, b = jnp.split(x, 2, axis=-1)
            c = jax.nn.swish(a) * b
            return Linear(num_channels, use_bias=False, initializer="zeros",
                          name="transition2")(c)

        # Shard path: hoist params and inline LN/GLU so the body is a pure
        # function of the activation slice (so ``hk.scan`` can carry the
        # output buffer without re-creating params each iteration).
        with hk.name_scope("input_layer_norm"):
            ln_scale = hk.get_parameter(
                "scale", (num_channels,), act.dtype, init=jnp.ones)
            ln_offset = hk.get_parameter(
                "offset", (num_channels,), act.dtype, init=jnp.zeros)
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

        def _ffn(x):
            xn = x - x.mean(axis=-1, keepdims=True)
            v = jnp.mean(xn * xn, axis=-1, keepdims=True)
            xn = xn * jax.lax.rsqrt(v + 1e-5)
            xn = xn * ln_scale + ln_offset
            xn = jnp.dot(xn, w1)
            a, b = jnp.split(xn, 2, axis=-1)
            c = jax.nn.swish(a) * b
            return jnp.dot(c, w2)

        leading = act.shape[-3]
        pad = (-leading) % shard_size
        if pad:
            pad_shape = list(act.shape); pad_shape[-3] = pad
            act_p = jnp.concatenate(
                [act, jnp.zeros(pad_shape, dtype=act.dtype)], axis=-3)
        else:
            act_p = act

        out_buf = jnp.zeros(act_p.shape, dtype=act.dtype)

        def step(buf, idx):
            slc = jax.lax.dynamic_slice_in_dim(act_p, idx, shard_size, axis=-3)
            slc_out = _ffn(slc)
            buf = jax.lax.dynamic_update_slice_in_dim(buf, slc_out, idx, axis=-3)
            return buf, ()

        slice_starts = jnp.arange(0, act_p.shape[-3], shard_size)
        out_buf, _ = hk.scan(step, out_buf, slice_starts)
        if pad:
            out_buf = jax.lax.dynamic_slice_in_dim(out_buf, 0, leading, axis=-3)
        return out_buf
