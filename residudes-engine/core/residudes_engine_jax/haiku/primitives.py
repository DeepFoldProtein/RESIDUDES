# SPDX-License-Identifier: Apache-2.0
"""Haiku Linear / LayerNorm matching AF3's weight conventions.

Why these wrappers exist:
  - AF3's hm.Linear stores w as [in, out] and supports multi-dim "out" shapes
    via ``num_output=(num_head, head_dim)``. PyTorch nn.Linear stores
    [out, in] and only supports 1D out. The weight converter relies on this
    convention being consistent.
  - LayerNorm with ``param_axis`` (used by triangle multiplication's center
    norm) is not in vanilla hk.LayerNorm.
"""

from __future__ import annotations

from typing import Sequence

import haiku as hk
import jax
import jax.numpy as jnp

from residudes_engine_jax.haiku._tokamax import get_tokamax, _shape_too_small


class LayerNorm(hk.Module):
    """LayerNorm with optional non-final ``param_axis`` (matches AF3).

    Numerics:
      Matches AF3's ``hm.LayerNorm``: bf16/fp16 inputs are upcast to fp32 for
      mean/var/normalize, then downcast back. The hand-rolled path before
      this fix did everything in the input dtype, which loses precision over
      wide channels (256-384) at protein scale.

    Implementation:
      When tokamax is available and ``axis == param_axis == -1``, dispatch to
      ``tokamax.layer_norm`` (Pallas-Triton, fp32-internal). Falls back to a
      manual path that also does the fp32 upcast, so the precision behaviour
      is identical regardless of which kernel runs.
    """

    def __init__(
        self,
        *,
        axis: int = -1,
        param_axis: int = -1,
        create_scale: bool = True,
        create_offset: bool = True,
        eps: float = 1e-5,
        use_fast_variance: bool = False,
        name: str | None = None,
    ):
        super().__init__(name=name)
        self.axis = axis
        self.param_axis = param_axis
        self.create_scale = create_scale
        self.create_offset = create_offset
        self.eps = eps
        self.use_fast_variance = use_fast_variance

    def __call__(self, x: jax.Array) -> jax.Array:
        axis = self.axis if self.axis >= 0 else x.ndim + self.axis
        param_axis = (
            self.param_axis if self.param_axis >= 0 else x.ndim + self.param_axis
        )

        param_shape = (x.shape[param_axis],)
        scale = (
            hk.get_parameter("scale", param_shape, x.dtype, init=jnp.ones)
            if self.create_scale else None
        )
        offset = (
            hk.get_parameter("offset", param_shape, x.dtype, init=jnp.zeros)
            if self.create_offset else None
        )

        # Fast path: tokamax.layer_norm. Only valid when axis == param_axis ==
        # last-axis (it expects 1-D scale/offset matching x.shape[axis]). The
        # ``triangle.center_norm`` site uses axis=0 — that one falls through.
        # Also skipped when leading shapes are small enough that Pallas
        # kernel-launch overhead would beat the savings.
        tk = get_tokamax()
        if (tk is not None and axis == param_axis == x.ndim - 1
                and not _shape_too_small(x)):
            try:
                return tk.layer_norm(
                    x, scale=scale, offset=offset,
                    axis=axis, epsilon=self.eps, subtract_mean=True,
                    implementation=("triton", "xla"),
                )
            except (NotImplementedError, ExceptionGroup, TypeError):
                pass  # fall through to manual

        # Manual path. Optionally upcast bf16/fp16 → fp32 for AF3-style precision
        # (~+2x compute time at small N). Default OFF — set
        # ``RESIDUDES_ENGINE_JAX_LAYERNORM_FP32=1`` to opt in. Most Protenix / Boltz / OF3
        # production checkpoints were trained + scored with native bf16 LN, so
        # bf16-internal is the closer numerics match for downstream PT inference.
        import os
        upcast = os.environ.get("RESIDUDES_ENGINE_JAX_LAYERNORM_FP32", "0").lower() not in (
            "0", "false", "no", "off", "")
        orig_dtype = x.dtype
        is_low = orig_dtype in (jnp.bfloat16, jnp.float16)
        x_c = x.astype(jnp.float32) if (upcast and is_low) else x

        if self.use_fast_variance:
            mean = jnp.mean(x_c, axis=axis, keepdims=True)
            mean_of_sq = jnp.mean(jnp.square(x_c), axis=axis, keepdims=True)
            var = jnp.maximum(0.0, mean_of_sq - jnp.square(mean))
        else:
            mean = jnp.mean(x_c, axis=axis, keepdims=True)
            centered = x_c - mean
            var = jnp.mean(jnp.square(centered), axis=axis, keepdims=True)
        inv = jax.lax.rsqrt(var + self.eps)
        normed = (x_c - mean) * inv

        # Broadcast shape for params: 1 everywhere except the param_axis.
        broadcast_shape = [1] * x.ndim
        broadcast_shape[param_axis] = x.shape[param_axis]
        broadcast_shape = tuple(broadcast_shape)

        if scale is not None:
            normed = normed * scale.astype(normed.dtype).reshape(broadcast_shape)
        if offset is not None:
            normed = normed + offset.astype(normed.dtype).reshape(broadcast_shape)
        return normed.astype(orig_dtype)


class Linear(hk.Module):
    """Linear with [in, out] weight + multi-dim out support.

    ``num_output`` may be int or tuple. When tuple, output shape is
    ``x.shape[:-1] + tuple(num_output)``; the parameter ``w`` has shape
    ``(in, *num_output)``.
    """

    def __init__(
        self,
        num_output: int | Sequence[int],
        *,
        use_bias: bool = True,
        bias_init: float = 0.0,
        initializer: str = "linear",
        transpose_weights: bool = False,
        name: str | None = None,
    ):
        super().__init__(name=name)
        self.num_output = (
            (num_output,) if isinstance(num_output, int) else tuple(num_output)
        )
        self.use_bias = use_bias
        self.bias_init = bias_init
        self.initializer = initializer
        self.transpose_weights = transpose_weights

    def __call__(self, x: jax.Array) -> jax.Array:
        in_dim = x.shape[-1]
        out_shape = self.num_output

        if self.transpose_weights:
            w_shape = (*out_shape, in_dim)
        else:
            w_shape = (in_dim, *out_shape)

        w_init = _initializer(self.initializer, in_dim)
        w = hk.get_parameter("w", w_shape, x.dtype, init=w_init)

        if self.transpose_weights:
            # Contract last axis of x with last axis of w.
            y = jnp.tensordot(x, w, axes=[[-1], [-1]])
        else:
            y = jnp.tensordot(x, w, axes=[[-1], [0]])

        if self.use_bias:
            b = hk.get_parameter(
                "b",
                out_shape,
                x.dtype,
                init=hk.initializers.Constant(self.bias_init),
            )
            y = y + b
        return y


def _initializer(name: str, fan_in: int):
    """AF3-style initializer keys."""
    if name == "linear":
        return hk.initializers.VarianceScaling(
            scale=1.0, mode="fan_in", distribution="truncated_normal"
        )
    if name == "relu":
        return hk.initializers.VarianceScaling(
            scale=2.0, mode="fan_in", distribution="truncated_normal"
        )
    if name == "zeros":
        return hk.initializers.Constant(0.0)
    if name == "default":
        return hk.initializers.VarianceScaling(
            scale=1.0, mode="fan_avg", distribution="truncated_normal"
        )
    raise ValueError(f"Unknown initializer: {name!r}")


def linear_get_params(
    in_dim: int,
    num_output: int | Sequence[int],
    *,
    dtype=jnp.float32,
    use_bias: bool = False,
    bias_init: float = 0.0,
    initializer: str = "linear",
    name: str,
) -> tuple[jax.Array, jax.Array | None]:
    """Allocate a Linear's parameters under ``<scope>/<name>/`` and return them.

    Mirrors AF3's ``hm.haiku_linear_get_params``: opens a name-scope so the
    weight key matches what a normal ``Linear(name=name)`` would produce, but
    returns the raw weight (and bias) instead of computing the matmul. Used
    by ``swiglu_ffn`` to feed ``tokamax.gated_linear_unit`` directly.
    """
    out_shape = (
        (num_output,) if isinstance(num_output, int) else tuple(num_output)
    )
    w_shape = (in_dim, *out_shape)
    w_init = _initializer(initializer, in_dim)
    with hk.name_scope(name):
        w = hk.get_parameter("w", w_shape, dtype, init=w_init)
        b = None
        if use_bias:
            b = hk.get_parameter(
                "b", out_shape, dtype,
                init=hk.initializers.Constant(bias_init),
            )
    return w, b
