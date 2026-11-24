# SPDX-License-Identifier: Apache-2.0
"""JAX hot-module backend for Bio-TRT.

Mirrors the TRT injection scope: replaces `pairformer` and `token_transformer`
PyTorch modules inside OpenFold3, Protenix, and Boltz-2 with JAX (Haiku)
implementations, using DLPack zero-copy for the host/device boundary.

The PyTorch model otherwise runs unchanged — input embedder, template/MSA,
diffusion sampler shell, and confidence heads stay in PyTorch.
"""

from residudes_engine_jax.bridge import torch_to_jax, jax_to_torch, get_jax_device
from residudes_engine_jax.compat_base import (
    JAXModuleConfig,
    CompatJAXBase,
)

__all__ = [
    "torch_to_jax",
    "jax_to_torch",
    "get_jax_device",
    "JAXModuleConfig",
    "CompatJAXBase",
]
