# Copyright 2024 DeepMind Technologies Limited
# SPDX-License-Identifier: Apache-2.0
"""Optional context-parallelism (CP) scaffolding for single-node multi-GPU.

When CP is enabled via :func:`enable_cp`, AF3's heavy pair-track ops
(TriangleMultiplication, GridSelfAttention, OuterProductMean) switch to
``shard_map``'d variants over a √cp × √cp 2-D mesh. Otherwise the modules
behave identically to the upstream serial implementation — every helper in
this module is a no-op when CP is disabled, so the same module classes
work in both regimes.

Design follows boltz-cp (NVIDIA-Digital-Bio/boltz-cp):

* pair ``[N, N, c]`` sharded 2-D over ``(cp_i, cp_j)``;
* single ``[N, c_s]`` and MSA depth axis sharded 1-D over the matching CP axis;
* params fully replicated;
* triangle ops use Cannon-style ring (``jax.lax.ppermute``) along the
  contracted axis;
* triangle attention uses Ring Attention (online softmax over rotating K/V/bias).

To enable from a process entry point::

    from alphafold3.model.network import parallelism
    parallelism.enable_cp(size_cp=4)         # 2×2 grid on first 4 GPUs
    # … now jit'd model picks up sharding constraints automatically.

Env overrides:

* ``AF3_CP_SIZE=4`` — sets the world size when :func:`enable_cp` is called
  with ``size_cp=None``.
* ``AF3_CP_DEVICES=0,1,2,3`` — pin the mesh to specific JAX device ids.
  Useful for NUMA-aware placement when NVLink is absent.

NCCL / NVLink: JAX collectives map directly to NCCL on CUDA. NCCL
detects NVLink at startup and uses it transparently; PCIe P2P is the
fallback (cross-NUMA pairs fall back further to host staging). On
NVLink-less consumer cards (RTX 40-series) ``AF3_CP_DEVICES`` lets you
keep the mesh inside a single NUMA node — log on import warns when this
is violated. The usual NCCL knobs (``NCCL_P2P_LEVEL``, ``NCCL_DEBUG=INFO``,
``NCCL_SHM_DISABLE``) work as documented since we don't intercept them.
"""

from __future__ import annotations

import logging
import math
import os
import subprocess
from typing import Iterable, Optional, Sequence

import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

_AXIS_I = "cp_i"
_AXIS_J = "cp_j"

_MESH: Optional[Mesh] = None

_logger = logging.getLogger("af3.parallelism")


# ─────────────────────────── Mesh lifecycle ────────────────────────────────


def enable_cp(
    size_cp: Optional[int] = None,
    device_ids: Optional[Sequence[int]] = None,
) -> Optional[Mesh]:
    """Initialise the CP mesh.

    Args:
      size_cp: total CP world size; must be a perfect square (1, 4, 9, 16, …).
        If ``None``, reads ``AF3_CP_SIZE`` env var (default ``1`` → disabled).
        ``size_cp <= 1`` disables CP and the helpers below become no-ops.
      device_ids: explicit JAX device ids to populate the mesh with. Useful
        for NUMA-aware placement on PCIe-only systems (e.g. RTX 4090 nodes
        have no NVLink, so cross-NUMA GPU pairs add 2–3× latency to ring
        collectives). If ``None``, reads ``AF3_CP_DEVICES`` env var as a
        comma-separated id list; otherwise uses the first ``size_cp``
        devices visible to JAX.

    Returns:
      The created :class:`jax.sharding.Mesh`, or ``None`` when CP is disabled.

    Raises:
      ValueError: if ``size_cp`` is not a perfect square, or if
        ``device_ids`` has a wrong length.
      RuntimeError: if fewer than ``size_cp`` JAX devices are visible.

    Notes:
      Collectives (``jax.lax.ppermute`` / ``all_to_all`` / ``psum``) under
      the returned mesh are routed by XLA through NCCL on CUDA. NCCL picks
      NVLink P2P when available and falls back to PCIe P2P (then host
      staging) otherwise. The only "NVLink-aware" knob we have at this
      layer is *which device ids end up on the mesh* — see ``device_ids``.
    """
    global _MESH
    if size_cp is None:
        size_cp = int(os.environ.get("AF3_CP_SIZE", "1"))
    if size_cp <= 1:
        _MESH = None
        return None
    side = int(math.isqrt(size_cp))
    if side * side != size_cp:
        raise ValueError(f"size_cp must be a perfect square, got {size_cp}")
    devs = jax.devices()
    if len(devs) < size_cp:
        raise RuntimeError(
            f"AF3_CP_SIZE={size_cp} but only {len(devs)} JAX devices visible."
        )
    if device_ids is None:
        env_ids = os.environ.get("AF3_CP_DEVICES", "").strip()
        if env_ids:
            device_ids = [int(x) for x in env_ids.split(",") if x.strip()]
    if device_ids is not None:
        if len(device_ids) != size_cp:
            raise ValueError(
                f"device_ids has {len(device_ids)} entries but size_cp={size_cp}"
            )
        id_to_dev = {d.id: d for d in devs}
        try:
            chosen = [id_to_dev[i] for i in device_ids]
        except KeyError as e:
            raise ValueError(f"device id {e} not visible to JAX") from None
    else:
        chosen = list(devs[:size_cp])
    arr = np.array(chosen).reshape(side, side)
    _MESH = Mesh(arr, axis_names=(_AXIS_I, _AXIS_J))
    _log_topology(chosen)
    return _MESH


def _log_topology(chosen_devs: Iterable[jax.Device]) -> None:
    """Best-effort log of NVLink / PCIe layout of the chosen devices.

    Inspects ``nvidia-smi topo -m`` and ``nvidia-smi nvlink -s``; logs a
    warning when chosen devices lack NVLink or cross NUMA boundaries so the
    operator can re-pick ``AF3_CP_DEVICES``. Silent on non-CUDA backends.
    """
    try:
        local_ids = [d.local_hardware_id for d in chosen_devs]
    except AttributeError:
        local_ids = [d.id for d in chosen_devs]
    _logger.info(
        "CP mesh: %d device(s) (local_ids=%s)", len(local_ids), local_ids
    )
    if jax.default_backend() != "gpu":
        return
    try:
        nvl = subprocess.run(
            ["nvidia-smi", "nvlink", "-s"],
            check=False, capture_output=True, text=True, timeout=5,
        ).stdout
        has_nvlink = ("Link" in nvl) and ("does not have" not in nvl)
    except Exception:
        has_nvlink = None
    if has_nvlink is False:
        _logger.warning(
            "NVLink absent on selected GPUs — ring collectives will use PCIe"
            " P2P (and host staging for cross-NUMA pairs). Set"
            " AF3_CP_DEVICES to keep the mesh within one NUMA node when"
            " possible."
        )
    try:
        topo = subprocess.run(
            ["nvidia-smi", "topo", "-m"],
            check=False, capture_output=True, text=True, timeout=5,
        ).stdout
        if topo and any("SYS" in line for line in topo.splitlines()):
            _logger.info(
                "GPU topology reports cross-NUMA (SYS) links — collectives"
                " between such pairs traverse the CPU interconnect; expect"
                " higher latency."
            )
    except Exception:
        pass


def disable_cp() -> None:
    """Tear down the CP mesh (mostly for tests)."""
    global _MESH
    _MESH = None


def get_mesh() -> Optional[Mesh]:
    """Active CP mesh, or ``None`` if CP is disabled."""
    return _MESH


def is_enabled() -> bool:
    return _MESH is not None


def cp_size() -> int:
    """Total CP world size (``1`` when disabled)."""
    return 1 if _MESH is None else _MESH.size


def cp_side() -> int:
    """Side length of the CP grid (√cp; ``1`` when disabled)."""
    return 1 if _MESH is None else _MESH.shape[_AXIS_I]


def axis_i() -> str:
    """Name of the row axis of the CP mesh."""
    return _AXIS_I


def axis_j() -> str:
    """Name of the column axis of the CP mesh."""
    return _AXIS_J


# ─────────────────────────── Sharding specs ────────────────────────────────
#
# Each helper returns a NamedSharding when CP is enabled and ``None`` otherwise.
# Trailing channel dims and any leading batch dims (specified by the caller)
# are replicated. Shapes referenced below are written without a batch leading
# axis — AF3 modules typically operate on un-batched tensors.


def pair_sharding(extra_leading_replicated: int = 0):
    """Sharding for pair tensor ``[*B, N, N, c]`` — 2-D shard on (cp_i, cp_j)."""
    if _MESH is None:
        return None
    leading = (None,) * extra_leading_replicated
    return NamedSharding(_MESH, P(*leading, _AXIS_I, _AXIS_J, None))


def single_sharding(extra_leading_replicated: int = 0):
    """Sharding for single tensor ``[*B, N, c]`` — N axis on cp_i, cp_j replicated."""
    if _MESH is None:
        return None
    leading = (None,) * extra_leading_replicated
    return NamedSharding(_MESH, P(*leading, _AXIS_I, None))


def msa_sharding(extra_leading_replicated: int = 0):
    """Sharding for MSA tensor ``[*B, S, N, c]`` — S→cp_j, N→cp_i.

    This pairs naturally with :func:`pair_sharding`: in OuterProductMean
    the MSA depth (``S``) is reduced along cp_j and the output pair lands
    sharded over (cp_i, cp_j).
    """
    if _MESH is None:
        return None
    leading = (None,) * extra_leading_replicated
    return NamedSharding(_MESH, P(*leading, _AXIS_J, _AXIS_I, None))


def replicated():
    """Fully replicated sharding — for params and small scalars."""
    if _MESH is None:
        return None
    return NamedSharding(_MESH, P())


# ─────────────────────────── Constraint wrappers ───────────────────────────
#
# Use these inside Haiku modules to nudge XLA into reshapes/transposes at the
# right boundaries. All become identity when CP is disabled.


def _ensure_ndim(x, expected_min):
    if x.ndim < expected_min:
        raise ValueError(
            f"expected at least {expected_min} dims, got shape {x.shape}"
        )


def constrain_pair(x):
    """``jax.lax.with_sharding_constraint`` for a pair-track tensor.

    The two trailing N axes (before the last channel axis) are sharded on
    (cp_i, cp_j); everything else (including any leading batch dims) is
    replicated. No-op when CP is disabled.
    """
    if _MESH is None:
        return x
    _ensure_ndim(x, 3)
    return jax.lax.with_sharding_constraint(
        x, pair_sharding(extra_leading_replicated=x.ndim - 3)
    )


def constrain_single(x):
    """``with_sharding_constraint`` for a single-track tensor ``[*, N, c]``."""
    if _MESH is None:
        return x
    _ensure_ndim(x, 2)
    return jax.lax.with_sharding_constraint(
        x, single_sharding(extra_leading_replicated=x.ndim - 2)
    )


def constrain_msa(x):
    """``with_sharding_constraint`` for an MSA tensor ``[*, S, N, c]``."""
    if _MESH is None:
        return x
    _ensure_ndim(x, 3)
    return jax.lax.with_sharding_constraint(
        x, msa_sharding(extra_leading_replicated=x.ndim - 3)
    )


def constrain_replicated(x):
    """Mark a tensor as fully replicated (no-op when CP is disabled)."""
    if _MESH is None:
        return x
    return jax.lax.with_sharding_constraint(x, replicated())


# ─────────────────────────── Ring primitives ───────────────────────────────
#
# Thin wrappers around ``jax.lax.ppermute`` that produce the boltz-cp
# "ring shift by one" pattern. They're intended for use inside ``shard_map``
# blocks — outside ``shard_map`` the axis names are not in scope.


def _axis_size_static(axis_name: str) -> int:
    """Resolve the size of an axis at trace time (works on JAX 0.4 and 0.5).

    ``jax.lax.axis_size`` only appears in newer JAX. We grab the size off
    the active mesh — required to be set via :func:`enable_cp` for any code
    path that calls into the ring primitives.
    """
    if _MESH is None:
        raise RuntimeError(
            "ring primitives called without an active CP mesh; "
            "enable_cp(...) must run first."
        )
    return _MESH.shape[axis_name]


def ring_shift(x, axis_name: str):
    """Ring-shift ``x`` by one position along ``axis_name`` (rank ``r`` → ``r+1``).

    Equivalent to boltz-cp's ``One2OneComm`` with ``send_rank = r + 1`` and
    ``recv_rank = r - 1``. ``jax.lax.ppermute`` schedules send/recv via XLA's
    collective scheduler — no parity/deadlock handling needed.
    """
    n = _axis_size_static(axis_name)
    return jax.lax.ppermute(
        x, axis_name, perm=[(i, (i + 1) % n) for i in range(n)]
    )


def ring_shift_reverse(x, axis_name: str):
    """Ring-shift the opposite direction (rank ``r`` → ``r - 1``)."""
    n = _axis_size_static(axis_name)
    return jax.lax.ppermute(
        x, axis_name, perm=[(i, (i - 1) % n) for i in range(n)]
    )


# ─────────────────────────── Comm benchmark ────────────────────────────────
#
# Measures end-to-end (NCCL + XLA) latency / bandwidth of the collective
# patterns CP actually uses, so the operator can see — at startup — whether
# the chosen device layout is going to bottleneck inference.


def bench_comm(
    payload_mb: float = 64.0,
    warmup_iters: int = 3,
    timed_iters: int = 5,
    log: bool = True,
):
    """Time ppermute / all_gather / all_to_all / psum on the current CP mesh.

    Args:
      payload_mb: per-rank payload size (MiB) for the timing tensor. The
        bf16 default tile is ``[N, N, c]`` where N is picked to match this
        size. 64 MiB ≈ 6 MiB per ring shift hop on a 2×2 grid.
      warmup_iters: untimed warmup repetitions (lets XLA compile + NCCL
        cache its plan).
      timed_iters: repetitions averaged for the final timing.
      log: emit a one-line per-pattern timing summary at INFO level.

    Returns:
      Dict mapping pattern name → median seconds per call. Empty when CP
      is disabled.
    """
    import time
    import jax.numpy as jnp
    from functools import partial
    from jax.experimental.shard_map import shard_map

    if _MESH is None:
        if log:
            _logger.info("bench_comm: CP disabled, nothing to measure.")
        return {}

    P_side = cp_side()
    elems = int(payload_mb * 1024 * 1024 // 2)              # bf16 = 2 B
    n = max(P_side * P_side, int(math.sqrt(elems / 128)))   # → [n, n, 128]
    # psum_scatter on cp_j (after sharding by cp_i) needs (n/P_side) divisible
    # by P_side too, so the tile-on-tile scatter is valid: round n to a
    # multiple of P_side^2.
    step = P_side * P_side
    n = ((n + step - 1) // step) * step
    c = 128
    shape = (n, n, c)
    rep_pair = P(_AXIS_I, _AXIS_J, None)

    x = jnp.zeros(shape, dtype=jnp.bfloat16)

    def _time(fn, label: str) -> float:
        for _ in range(warmup_iters):
            y = fn(x); y.block_until_ready()
        ts = []
        for _ in range(timed_iters):
            t0 = time.perf_counter()
            y = fn(x); y.block_until_ready()
            ts.append(time.perf_counter() - t0)
        ts.sort()
        med = ts[len(ts) // 2]
        if log:
            tile_mb = (n // P_side) * (n // P_side) * c * 2 / (1024 * 1024)
            _logger.info(
                "bench_comm[%-22s] median=%.2f ms  (tile=%.1f MiB → %.1f GiB/s)",
                label, med * 1000, tile_mb,
                tile_mb / 1024 / med if med > 0 else 0.0,
            )
        return med

    @partial(jax.jit, in_shardings=NamedSharding(_MESH, rep_pair),
             out_shardings=NamedSharding(_MESH, rep_pair))
    @partial(shard_map, mesh=_MESH, in_specs=rep_pair, out_specs=rep_pair,
             check_rep=False)
    def ppermute_j(t):
        return ring_shift(t, _AXIS_J)

    @partial(jax.jit, in_shardings=NamedSharding(_MESH, rep_pair),
             out_shardings=NamedSharding(_MESH, rep_pair))
    @partial(shard_map, mesh=_MESH, in_specs=rep_pair, out_specs=rep_pair,
             check_rep=False)
    def ppermute_i(t):
        return ring_shift(t, _AXIS_I)

    @partial(jax.jit, in_shardings=NamedSharding(_MESH, rep_pair),
             out_shardings=NamedSharding(_MESH, P(None, _AXIS_J, None)))
    @partial(shard_map, mesh=_MESH, in_specs=rep_pair,
             out_specs=P(None, _AXIS_J, None), check_rep=False)
    def all_gather_i(t):
        return jax.lax.all_gather(t, _AXIS_I, axis=0, tiled=True)

    @partial(jax.jit, in_shardings=NamedSharding(_MESH, rep_pair),
             out_shardings=NamedSharding(_MESH, rep_pair))
    @partial(shard_map, mesh=_MESH, in_specs=rep_pair, out_specs=rep_pair,
             check_rep=False)
    def psum_scatter_j(t):
        return jax.lax.psum_scatter(t, _AXIS_J, scatter_dimension=1, tiled=True)

    @partial(jax.jit, in_shardings=NamedSharding(_MESH, rep_pair),
             out_shardings=NamedSharding(_MESH, P()))
    @partial(shard_map, mesh=_MESH, in_specs=rep_pair, out_specs=P(),
             check_rep=False)
    def all_reduce_full(t):
        return jax.lax.psum(t, (_AXIS_I, _AXIS_J))

    if log:
        _logger.info(
            "bench_comm: payload=%.0f MiB per rank; tile %dx%dx%d (bf16) on"
            " %d-rank mesh", payload_mb, n // P_side, n // P_side, c, cp_size(),
        )
    results = {
        "ring_shift cp_j":     _time(ppermute_j,    "ring_shift cp_j"),
        "ring_shift cp_i":     _time(ppermute_i,    "ring_shift cp_i"),
        "all_gather cp_i":     _time(all_gather_i,  "all_gather cp_i"),
        "psum_scatter cp_j":   _time(psum_scatter_j, "psum_scatter cp_j"),
        "psum (i,j)":          _time(all_reduce_full, "psum (i,j)"),
    }
    return results
