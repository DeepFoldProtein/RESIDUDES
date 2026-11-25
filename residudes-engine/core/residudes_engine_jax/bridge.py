# SPDX-License-Identifier: Apache-2.0
"""DLPack bridge between PyTorch CUDA tensors and JAX device arrays.

The hot-swap design keeps the surrounding model in PyTorch and calls a JAX
forward function for pairformer / token transformer. Each call must move
inputs/outputs across the framework boundary without H2D copies.

Strategy:
  - torch.cuda.Tensor -> DLPack capsule -> jax.dlpack.from_dlpack
  - jax.Array -> DLPack capsule -> torch.utils.dlpack.from_dlpack

Both directions share the same CUDA pointer; the cost was historically a
host-blocking ``stream.synchronize()`` to guarantee write-after-read
ordering. Profiling on RTX 4090 showed that for Protenix v2 inference
(4 trunk cycles + 200 diffusion steps × 5 transfers each = ~1000 syncs)
those barriers serialize host execution and dominate JAX hot-swap cost.

The default now skips the host-side syncs and trusts the DLPack stream
hand-off. Set ``RESIDUDES_ENGINE_JAX_BRIDGE_SYNC=1`` to restore the old behaviour
when running mixed CUDA streams or debugging a race.
"""

from __future__ import annotations

import os
from typing import Any

import jax
import jax.dlpack
import jax.numpy as jnp
import torch
import torch.utils.dlpack


def _bridge_sync() -> bool:
    """When true, host-block on stream sync at every transfer.

    Default false — saves ~50 % of JAX hot-swap time on Protenix v2.
    Enable for debugging if you suspect a race between PT and JAX
    streams.
    """
    return os.environ.get("RESIDUDES_ENGINE_JAX_BRIDGE_SYNC", "0").lower() in (
        "1", "true", "yes", "on")


_JAX_DEVICE: Any = None


def get_jax_device() -> Any:
    """Return the JAX device used for hot-swapped modules.

    Defaults to the first CUDA device. Override with set_jax_device() if the
    PyTorch model lives on a non-default GPU.
    """
    global _JAX_DEVICE
    if _JAX_DEVICE is None:
        try:
            gpus = jax.devices("gpu")
            if gpus:
                _JAX_DEVICE = gpus[0]
                return _JAX_DEVICE
        except RuntimeError:
            pass
        # Fall back to whatever default device JAX has (CPU on this venv).
        _JAX_DEVICE = jax.devices()[0]
    return _JAX_DEVICE


def set_jax_device(device: Any) -> None:
    global _JAX_DEVICE
    _JAX_DEVICE = device


def torch_to_jax(t: torch.Tensor) -> jax.Array:
    """Move a PyTorch CUDA tensor to a JAX array, zero-copy when possible.

    By default skips the explicit stream sync and trusts the DLPack
    hand-off (set ``RESIDUDES_ENGINE_JAX_BRIDGE_SYNC=1`` to opt back in).
    """
    if not t.is_cuda:
        return jnp.asarray(t.detach().cpu().numpy())

    if _bridge_sync():
        torch.cuda.current_stream(t.device).synchronize()
    t = t.contiguous()
    # JAX 0.10+ dlpack accepts any object implementing __dlpack__ /
    # __dlpack_device__ — torch.Tensor does. The legacy PyCapsule API
    # (torch.utils.dlpack.to_dlpack) is rejected with TypeError on 0.10+.
    return jax.dlpack.from_dlpack(t)


def jax_to_torch(j: jax.Array, *, dtype: torch.dtype | None = None) -> torch.Tensor:
    """Move a JAX array back to PyTorch. Zero-copy when on the same GPU.

    By default skips ``block_until_ready()`` (PyTorch's next op on this
    tensor will see the same stream the JAX kernel queued on, so a host
    barrier is not required for correctness on a single GPU). Set
    ``RESIDUDES_ENGINE_JAX_BRIDGE_SYNC=1`` to opt back into the old host-blocking
    behaviour.
    """
    if _bridge_sync():
        j.block_until_ready()
    # torch >=2.6 accepts a producer object with __dlpack__ — j is a
    # jax.Array which implements that. The PyCapsule path is deprecated.
    t = torch.utils.dlpack.from_dlpack(j)
    if dtype is not None and t.dtype != dtype:
        t = t.to(dtype)
    return t
