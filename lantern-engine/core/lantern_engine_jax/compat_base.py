# SPDX-License-Identifier: Apache-2.0
"""Base compat wrapper: PyTorch nn.Module that delegates forward to a JAX fn.

Mirrors the TRT compat layer (CompatPairformerTRT, CompatTokenTransformerTRT)
but routes through a JIT-compiled Haiku/JAX function instead of a TRT engine.

Subclasses provide:
  - build_apply_fn(): returns a hk.transform-able function f(params, *args)
  - forward_udf(...): per-model interface adapter (signatures differ across
    OF3 / Protenix / Boltz). Calls _run_jax(*jax_args).

Each subclass is constructed once per model with a fixed param tree (loaded
from converter output). The base class handles:
  - Param tree storage + dtype cast (BF16 by default).
  - Per-shape JIT cache: jax.jit traces are keyed on input shape, so a new
    sequence length triggers one recompile.
  - Torch <-> JAX bridge on every call (DLPack zero-copy).
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import Any, Callable

import jax
import jax.numpy as jnp
import torch
import torch.nn as nn

from lantern_engine_jax.bridge import torch_to_jax, jax_to_torch, get_jax_device


@dataclass
class JAXModuleConfig:
    """Per-module config (analogous to AcceleratedConfig for TRT)."""
    # Engine I/O dtype (matches TRT BF16 default).
    dtype: jnp.dtype = jnp.bfloat16
    # Multiplicity for the diffusion token transformer; ignored by pairformer.
    multiplicity: int = 5
    # If True, forward expects/returns tensors with a leading batch dim of 1.
    support_batch: bool = False
    # Whether to enable jax.jit caching. Disable for debugging.
    jit: bool = True


class CompatJAXBase(nn.Module):
    """Base PyTorch wrapper around a Haiku/JAX forward function.

    Concrete subclasses implement ``forward(...)`` matching the host model's
    expected interface (e.g. OF3's pairformer takes single_mask + pair_mask;
    Protenix's takes only pair_mask). They convert torch tensors via
    torch_to_jax(), dispatch through ``_call_jax(*jax_inputs)``, and return
    torch tensors via jax_to_torch().
    """

    def __init__(
        self,
        params: dict[str, Any],
        apply_fn: Callable[..., Any],
        config: JAXModuleConfig,
    ):
        super().__init__()
        self._config = config
        self._apply_fn = apply_fn
        # Cast and pin params on the chosen device (one-time, at load).
        self._params = jax.device_put(
            jax.tree_util.tree_map(
                lambda x: jnp.asarray(x, dtype=self._cast_dtype(x)),
                params,
            ),
            get_jax_device(),
        )
        if config.jit:
            self._jit_apply = jax.jit(apply_fn)
        else:
            self._jit_apply = apply_fn

    def _cast_dtype(self, x: Any) -> jnp.dtype:
        """Cast floating params to engine dtype, leave int params alone."""
        a = jnp.asarray(x)
        if jnp.issubdtype(a.dtype, jnp.floating):
            return self._config.dtype
        return a.dtype

    @property
    def config(self) -> JAXModuleConfig:
        return self._config

    def _call_jax(self, *args: jax.Array, **kwargs: jax.Array) -> Any:
        """Run the JIT-compiled Haiku apply function and block on result.

        Subclasses pass already-bridged JAX arrays (via torch_to_jax). The
        return value is left as JAX arrays; the caller bridges back via
        jax_to_torch when assembling the torch output.
        """
        out = self._jit_apply(self._params, *args, **kwargs)
        # block_until_ready() is delayed to jax_to_torch in the call site to
        # let multiple JAX calls overlap if the host model dispatches several
        # in sequence.
        return out

    # Subclasses must override.
    def forward(self, *args, **kwargs):  # pragma: no cover - abstract
        raise NotImplementedError


def make_apply_fn(
    forward_module_cls: type,
    *module_args,
    **module_kwargs,
) -> Callable[..., Any]:
    """Wrap a Haiku module class into a hk.transform-style apply function.

    Usage:
        import haiku as hk
        from lantern_engine_jax.haiku.pairformer import PairformerStack

        apply = make_apply_fn(PairformerStack, config=of3_pairformer_config)
        params = converter.load(...)  # shape-matched param tree
        wrapped = CompatPairformerJAXOF3(params, apply, JAXModuleConfig(...))
    """
    import haiku as hk

    def fn(*args, **kwargs):
        mod = forward_module_cls(*module_args, **module_kwargs)
        return mod(*args, **kwargs)

    return hk.transform(fn).apply


# Per-model setter base — keeps the same shape as the TRT side
# (OpenFold3AcceleratedModules etc.) so the existing InferenceRunner._inject
# code paths can register a JAX backend with a one-line swap.
@dataclass
class JAXAcceleratedModule:
    """A single replacement: where to slot the JAX module into the host model."""
    name: str
    compat_cls: type
    setter: Callable[[nn.Module, nn.Module], nn.Module]
    config: JAXModuleConfig = field(default_factory=JAXModuleConfig)
