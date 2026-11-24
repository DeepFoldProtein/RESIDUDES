# SPDX-License-Identifier: Apache-2.0
"""Optional tokamax kernels (AF3-style fused triangle/attention/SwiGLU).

When the ``.venv-jax`` (= the JAX-only environment built from
``pyproject-jax.toml``) is active, ``tokamax`` is importable and provides
fused kernels for:
  - dot product attention        (Pallas Triton / Mosaic-GPU flash attn)
  - triangle multiplication      (XLA, but with a fused softmax+LN+gating path)
  - gated linear unit (SwiGLU)
  - layer norm

In the legacy ``.venv`` (numpy<2 + jax<0.9.1), ``tokamax`` cannot be installed.
We fall back to ``jax.nn.dot_product_attention`` and the existing
hand-rolled einsum/SwiGLU paths.

Selection is also controllable via env vars:
  BIO_TRT_JAX_TOKAMAX           = "0" / "1" (default: auto = "1" if importable)
  BIO_TRT_JAX_ATTENTION_IMPL    = "auto" (default), "mosaic", "triton",
                                  "cudnn", "xla", "xla_chunked"
"""

from __future__ import annotations

import os
from typing import Any

import jax
import jax.numpy as jnp


def _env_truthy(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.lower() not in ("0", "false", "no", "off", "")


def _min_dpa_head_dim() -> int:
    """Below this head_dim *and* below DPA_MAX_N_FOR_GATE seq len, skip tokamax.

    Pallas-Triton flash attention launch overhead and the bf16 K/V reload
    don't pay off for small head_dim *if* the sequence length is also small.
    On RTX 4090 with Protenix v2 ubiquitin (N=76, head_dim 24/32/48),
    routing these shapes through ``jax.nn.dot_product_attention``
    cuts the trunk pairformer from ~32 s to ~12 s.

    But at large N (e.g. 9fm7 N=582) the materialized-softmax fallback
    OOMs the 4090 — tokamax flash attention's streaming behavior is
    *required* for memory, even at small head_dim.

    So this is a *compound* gate (see ``dot_product_attention``): tokamax
    is only skipped when BOTH head_dim < this threshold AND the q seq
    length is < ``BIO_TRT_JAX_DPA_MAX_N_FOR_GATE``.
    """
    try:
        return int(os.environ.get("BIO_TRT_JAX_DPA_MIN_HEAD_DIM", "64"))
    except ValueError:
        return 64


def _max_n_for_gate() -> int:
    """Max sequence length below which the head_dim gate may apply.

    Above this, always try tokamax (it's memory-efficient flash attention,
    needed to fit large N on consumer GPUs). Default 256 — well above the
    typical small-N regime where the launch overhead dominates, but well
    below the size where materializing [N,N,H] softmax OOMs.
    """
    try:
        return int(os.environ.get("BIO_TRT_JAX_DPA_MAX_N_FOR_GATE", "256"))
    except ValueError:
        return 256


def _min_n_for_tokamax() -> int:
    """Minimum leading-shape product below which tokamax LN/GLU are skipped.

    Measured on RTX 4090, Protenix v2:
      - N=76  ubiquitin: tokamax LN+GLU was 3x slower than manual XLA.
      - N=644 9fm7:      tokamax LN+GLU was ~60% slower (222s vs 139s).
    The Pallas-Triton LN/GLU kernels on a 4090 don't beat XLA's fused
    softmax-LN-multiply for our shapes — kernel launch + dispatch overhead
    eats the savings. Tokamax DPA is the only path that's a clear win
    (and it has its own internal threshold in ``dot_product_attention``).

    Default ``BIO_TRT_JAX_TOKAMAX_MIN_N=99999`` effectively disables tokamax
    LN/GLU. Set lower (e.g. ``256``) on B200/H100 where the kernels may win.
    """
    try:
        return int(os.environ.get("BIO_TRT_JAX_TOKAMAX_MIN_N", "99999"))
    except ValueError:
        return 99999


def _shape_too_small(*arrays: jax.Array) -> bool:
    """Heuristic: total elements along the first non-trivial axis is below the
    tokamax break-even point. Used by ``layer_norm`` / ``swiglu_ffn``
    dispatchers; ``dot_product_attention`` has its own internal threshold."""
    threshold = _min_n_for_tokamax()
    if threshold <= 0:
        return False
    for a in arrays:
        # The "row" axis tokamax-LN/GLU strides over is the last but one.
        if a.ndim >= 2 and a.shape[-2] >= threshold:
            return False
    return True


# Lazy-imported tokamax module (None if unavailable).
_tokamax: Any = None
_tokamax_tried = False


def get_tokamax():
    """Return the ``tokamax`` module if importable + enabled, else ``None``."""
    global _tokamax, _tokamax_tried
    if _tokamax_tried:
        return _tokamax
    _tokamax_tried = True
    if not _env_truthy("BIO_TRT_JAX_TOKAMAX", default=True):
        return None

    # Tokamax's _ConfigOption.value (tokamax/_src/config.py) lazily calls
    # ``absl.flags.FLAGS(sys.argv)`` on first access, which raises
    # UnrecognizedFlagError when sys.argv contains app-specific flags
    # (e.g. Protenix's --model_name). Preempt this by parsing FLAGS with an
    # empty argv right after import, so subsequent ``is_parsed()`` calls in
    # tokamax return True and never touch sys.argv.
    # Importing tokamax (0.0.17) resets TRITON_CACHE_DIR to "" through Triton's
    # knobs; every later Triton compile (cuequivariance_jax kernels, torch
    # inductor) then dies with "Could not create or locate cache dir".
    # Remember the configured value and put it back right after the import.
    _triton_cache_dir = os.environ.get("TRITON_CACHE_DIR")
    try:
        import tokamax as tk  # type: ignore
        from absl import flags as _absl_flags  # type: ignore
        if not _absl_flags.FLAGS.is_parsed():
            _absl_flags.FLAGS([__file__])
        if _triton_cache_dir:
            os.environ["TRITON_CACHE_DIR"] = _triton_cache_dir
            try:
                from triton.runtime import cache as _tcache  # type: ignore
                if hasattr(_tcache, "knobs"):
                    _tcache.knobs.cache.dir = _triton_cache_dir
            except Exception:
                pass
        elif os.environ.get("TRITON_CACHE_DIR") == "":
            del os.environ["TRITON_CACHE_DIR"]          # fall back to Triton's default ~/.triton
        _tokamax = tk
        # Pull in the autotune cache restored from a previous run so the
        # warm hot path doesn't re-microbenchmark the same shapes.
        if _env_truthy("BIO_TRT_JAX_AUTOTUNE_PERSIST", default=True):
            try:
                from residudes_engine_jax.haiku import _autotune_cache as _ac
                _ac.install()
            except Exception:
                pass
    except Exception:
        _tokamax = None
    return _tokamax


def has_tokamax() -> bool:
    return get_tokamax() is not None


def attention_impl_preference() -> tuple[str, ...]:
    """Implementations to try, in order, for ``tokamax.dot_product_attention``.

    Default 'auto' → prefer Pallas (mosaic/triton) flash-attn on GPU, then
    cudnn, then xla. Override with env BIO_TRT_JAX_ATTENTION_IMPL.
    """
    requested = os.environ.get("BIO_TRT_JAX_ATTENTION_IMPL", "auto").lower()
    if requested != "auto":
        return (requested,)
    on_gpu = False
    try:
        on_gpu = any(d.platform == "gpu" for d in jax.devices())
    except Exception:
        pass
    if on_gpu:
        # Pallas Triton/Mosaic-GPU first (real flash kernels), then cudnn,
        # then xla as last resort.
        return ("triton", "mosaic", "cudnn", "xla")
    return ("xla",)


def dot_product_attention(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    *,
    bias: jax.Array | None = None,
    mask: jax.Array | None = None,
    scale: float | None = None,
):
    """Tokamax-preferred attention with a transparent jax.nn fallback.

    Shape convention matches ``jax.nn.dot_product_attention`` /
    ``tokamax.dot_product_attention``:
      q, k, v: ``[*B, T, num_head, head_dim]``
      bias:   broadcastable to ``[*B, num_head, T, T]``
      mask:   broadcastable to ``[*B, num_head, T, T]`` (bool)
    """
    tk = get_tokamax()
    head_dim = q.shape[-1] if q.ndim >= 2 else 0
    # q can be [..., T, num_head, head_dim] (single-track self attn) or
    # [B, T, num_head, head_dim] (triangle attn with vmap-style batch on B).
    # Use the second-to-last leading axis as a proxy for sequence length.
    seq_len = q.shape[-3] if q.ndim >= 3 else q.shape[-2] if q.ndim >= 2 else 0
    skip_for_small = (
        head_dim < _min_dpa_head_dim() and seq_len < _max_n_for_gate()
    )
    if tk is not None and not skip_for_small:
        impls = attention_impl_preference()
        try:
            # Tokamax wraps its API with jaxtyping + typeguard, which fails
            # type-checks against tracers when the call is inside jax.jit.
            # Disable while we make the call so tracers are accepted.
            try:
                from tokamax._src import jaxtyping as _tk_jaxtyping  # type: ignore
                _disable_ctx = _tk_jaxtyping.disable_jaxtyping(True)
            except Exception:
                import contextlib
                _disable_ctx = contextlib.nullcontext()
            with _disable_ctx:
                return tk.dot_product_attention(
                    q, k, v,
                    bias=bias, mask=mask, scale=scale,
                    implementation=impls,
                )
        except (NotImplementedError, ExceptionGroup, TypeError):
            # Fall through to jax.nn — tokamax raises NotImplementedError
            # (or ExceptionGroup) for shapes the Pallas kernels can't tile
            # (e.g. very small head_dim, FP32 inputs on cudnn-only hardware),
            # or TypeError when typeguard rejects tracer args. jax.nn always
            # works.
            pass

    # Fallback path — same as before tokamax. (We tested an XLA-fallback
    # variant for small shapes; it regressed by ~2.5× — XLA's
    # materialized softmax is slower than cuDNN flash at N=76.)
    try:
        return jax.nn.dot_product_attention(
            q, k, v, bias=bias, mask=mask, scale=scale,
            implementation="cudnn",
        )
    except (TypeError, NotImplementedError, ValueError) as e:
        # ValueError surfaces from cuDNN's "Unsupported dtype for inputs."
        # in older jax (e.g. 0.5.3) when q/k/v are fp32 — cuDNN flash
        # attention only accepts bf16/fp16. The XLA fallback handles fp32.
        # The XLA path materialises the [B,H,T,S] logits (4·H·N³ bytes), so say
        # once, per shape class, why we are on it.
        _warn_xla_fallback(q, k, bias, mask, e)
        return jax.nn.dot_product_attention(
            q, k, v, bias=bias, mask=mask, scale=scale,
        )


_XLA_FALLBACK_SEEN: set = set()


def _warn_xla_fallback(q, k, bias, mask, exc) -> None:
    try:
        import haiku as hk
        where = hk.experimental.current_name()
    except Exception:
        where = "?"
    key = (where, tuple(q.shape), str(q.dtype), None if bias is None else (tuple(bias.shape), str(bias.dtype)),
           None if mask is None else tuple(mask.shape), type(exc).__name__)
    if key in _XLA_FALLBACK_SEEN:
        return
    _XLA_FALLBACK_SEEN.add(key)
    import logging
    logging.getLogger("residudes_engine_jax.attention").warning(
        "cuDNN attention rejected at %s: q%s %s bias%s %s mask%s → XLA fallback (materialised logits): %s: %s",
        where, tuple(q.shape), q.dtype, None if bias is None else tuple(bias.shape),
        None if bias is None else bias.dtype,
        None if mask is None else tuple(mask.shape), type(exc).__name__, str(exc).splitlines()[0][:200])


def gated_linear_unit_swiglu(x: jax.Array, w: jax.Array) -> jax.Array | None:
    """Tokamax fused SwiGLU. Returns None when tokamax is unavailable.

    ``w`` must have shape ``[in_dim, 2, out_dim]`` (matches tokamax's GLU API).
    Caller is responsible for reshaping their concat'd ``[in_dim, 2*out_dim]``
    weight if they take this path.
    """
    tk = get_tokamax()
    if tk is None:
        return None
    try:
        # tokamax.gated_linear_unit expects activation as a callable.
        return tk.gated_linear_unit(x, w, activation=jax.nn.swish)
    except Exception:
        return None


def swiglu_ffn(
    x: jax.Array,
    *,
    num_intermediate: int,
    name: str,
    initializer: str = "relu",
) -> jax.Array:
    """Linear -> SwiGLU FFN entry layer with optional tokamax fusion.

    Allocates a single weight ``[in, 2, num_intermediate]`` under
    ``<scope>/<name>/w`` (the same path a ``Linear(name=name)`` would use,
    so the converter can target it identically). Computes
    ``swish(x @ w[:, 0, :]) * (x @ w[:, 1, :])``.

    When tokamax is importable + the JIT call site allows it, dispatches to
    the Pallas-Triton fused GLU kernel. Otherwise falls back to a manual
    tensordot + slice + swish + multiply path that is bit-equivalent.
    """
    from residudes_engine_jax.haiku.primitives import linear_get_params  # local: avoid cycle

    in_dim = x.shape[-1]
    w, _ = linear_get_params(
        in_dim, (2, num_intermediate),
        dtype=x.dtype, use_bias=False,
        initializer=initializer, name=name,
    )

    tk = get_tokamax()
    # Skip tokamax on small shapes — Pallas-Triton launch overhead exceeds the
    # savings below N≈256 (measured: 3x slowdown on N=76).
    if tk is not None and not _shape_too_small(x):
        impls = ("triton", "xla")
        try:
            try:
                from tokamax._src import jaxtyping as _tk_jaxtyping  # type: ignore
                _disable_ctx = _tk_jaxtyping.disable_jaxtyping(True)
            except Exception:
                import contextlib
                _disable_ctx = contextlib.nullcontext()
            with _disable_ctx:
                return tk.gated_linear_unit(
                    x, w, activation=jax.nn.swish, implementation=impls,
                )
        except (NotImplementedError, ExceptionGroup, TypeError):
            pass  # fall through

    # Manual fallback. Shape arithmetic: x @ w[:, k, :] for k in {0, 1}.
    y = jnp.tensordot(x, w, axes=[[-1], [0]])  # [..., 2, num_intermediate]
    a = y[..., 0, :]
    b = y[..., 1, :]
    return jax.nn.swish(a) * b
