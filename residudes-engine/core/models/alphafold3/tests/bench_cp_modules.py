# Copyright 2024 DeepMind Technologies Limited
# SPDX-License-Identifier: Apache-2.0
"""Wall-clock bench: per-module CP-off vs CP-on across N.

Measures the break-even N where 2-D CP starts to win against single-GPU
on this hardware. Runs:

  * ``parallelism.bench_comm`` — raw collective bandwidth (NCCL routing).
  * For each ``N`` and each CP module: median forward time, CP-off vs
    CP-on, plus speedup.

Use the AF3 venv (``/gpfs/deepfold/casp/casp17/alphafold3/.venv``) for
JAX 0.9.1 + tokamax 0.0.11. The script assumes ≥4 visible GPUs.

Run::

    PYTHONPATH=src .venv/bin/python tests/bench_cp_modules.py
"""

from __future__ import annotations

import logging
import os
import statistics
import time
from typing import Callable

import haiku as hk
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P

from alphafold3.model import model_config
from alphafold3.model.network import parallelism, parallelism_modules as pm

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

_NUM_ITERS = 5
_WARMUP = 2


def _time_apply(apply_fn, *args, num_iters: int = _NUM_ITERS, warmup: int = _WARMUP) -> float:
    """Median wall time per call (ms), after warmup."""
    for _ in range(warmup):
        out = apply_fn(*args)
        jax.block_until_ready(out)
    samples = []
    for _ in range(num_iters):
        t0 = time.perf_counter()
        out = apply_fn(*args)
        jax.block_until_ready(out)
        samples.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(samples)


def _bench_module(
    name: str,
    sizes,
    make_fn: Callable,
    make_inputs: Callable,
    input_pair_axes,
    rng_seed: int = 0,
):
    """Run a CP module across sizes; report off/on/speedup.

    ``input_pair_axes`` lists, per input tensor, which axes carry (cp_i, cp_j).
    Use ``None`` for replicated axes. Example for ``act=[N, N, c]`` →
    ``(('cp_i', 'cp_j', None),)``. For ``act=[S, N, c]`` (MSA) →
    ``(('cp_j', 'cp_i', None), ('cp_j', 'cp_i'))`` — depth on cp_j, N on cp_i.

    For CP-on timing we ``jax.device_put`` each input under the matching
    ``NamedSharding`` and jit ``apply`` with ``in_shardings`` so XLA actually
    distributes the work. Without this, jit would broadcast unsharded inputs
    and shard_map ends up running its body sequentially on device 0 — which
    is exactly what produced the bogus 1.00× speedup in the first run.
    """
    print(f"\n=== {name} ===")
    print(f"{'N':>6}  {'cp_off (ms)':>12}  {'cp_on (ms)':>12}  {'speedup':>9}")
    for N in sizes:
        key = jax.random.PRNGKey(rng_seed)
        inputs = make_inputs(N, key)

        net = hk.transform(make_fn())
        parallelism.disable_cp()
        params = net.init(key, *inputs)

        # CP off
        apply_off = jax.jit(net.apply)
        try:
            t_off_str = f"{_time_apply(apply_off, params, None, *inputs):>12.2f}"
            t_off_val = float(t_off_str)
        except Exception as e:  # noqa: BLE001
            t_off_str = f"{type(e).__name__:>12.12}"
            t_off_val = float("nan")
        del apply_off

        # CP on: build mesh, shard the inputs, jit with in_shardings.
        mesh = parallelism.enable_cp(size_cp=4, device_ids=[0, 1, 2, 3])
        in_shardings = tuple(NamedSharding(mesh, P(*spec)) for spec in input_pair_axes)
        inputs_sh = tuple(jax.device_put(x, sh) for x, sh in zip(inputs, in_shardings))
        params_sh = jax.device_put(params, NamedSharding(mesh, P()))
        apply_on = jax.jit(
            net.apply,
            in_shardings=(NamedSharding(mesh, P()), None, *in_shardings),
            out_shardings=NamedSharding(mesh, P("cp_i", "cp_j", None)),
        )
        try:
            t_on_str = f"{_time_apply(apply_on, params_sh, None, *inputs_sh):>12.2f}"
            t_on_val = float(t_on_str)
        except Exception as e:  # noqa: BLE001
            t_on_str = f"{type(e).__name__:>12.12}"
            t_on_val = float("nan")
        parallelism.disable_cp()
        del apply_on

        if t_off_val == t_off_val and t_on_val == t_on_val:  # both not NaN
            speedup = f"{t_off_val/t_on_val:>8.2f}×"
        else:
            speedup = f"{'-':>9}"
        print(f"{N:>6}  {t_off_str}  {t_on_str}  {speedup}")


def bench_triangle_mul(sizes, c: int = 128, eqn: str = "ikc,jkc->ijc",
                       module_cls=None, label_suffix: str = ""):
    """Bench TriangleMultiplication CP variants.

    ``module_cls`` defaults to the central-only ``TriangleMultiplicationCP``;
    pass ``TriangleMultiplicationCPWhole`` to time the whole-module shard_map
    variant under the same shapes.
    """
    if module_cls is None:
        module_cls = pm.TriangleMultiplicationCP
    gc = model_config.GlobalConfig(
        bfloat16="all", final_init="linear",
        flash_attention_implementation="xla",
    )

    def make_fn():
        def fn(act, mask):
            return module_cls(
                module_cls.Config(equation=eqn, use_glu_kernel=False),
                gc, name="tri",
            )(act, mask)
        return fn

    def make_inputs(N, key):
        act = jax.random.normal(key, (N, N, c), dtype=jnp.bfloat16)
        mask = jnp.ones((N, N), dtype=jnp.float32)
        return (act, mask)

    _bench_module(
        f"{module_cls.__name__} (c={c}){label_suffix}",
        sizes, make_fn, make_inputs,
        input_pair_axes=(("cp_i", "cp_j", None), ("cp_i", "cp_j")),
    )


def bench_grid_attention(sizes, c: int = 128, num_head: int = 4):
    gc = model_config.GlobalConfig(
        bfloat16="all", final_init="linear",
        flash_attention_implementation="xla",
    )

    def make_fn():
        def fn(act, mask):
            return pm.GridSelfAttentionCP(
                pm.GridSelfAttentionCP.Config(num_head=num_head),
                gc, transpose=False, name="grid",
            )(act, mask)
        return fn

    def make_inputs(N, key):
        act = jax.random.normal(key, (N, N, c), dtype=jnp.bfloat16)
        mask = jnp.ones((N, N), dtype=jnp.float32)
        return (act, mask)

    _bench_module(
        f"GridSelfAttentionCP (c={c}, H={num_head})", sizes, make_fn, make_inputs,
        input_pair_axes=(("cp_i", "cp_j", None), ("cp_i", "cp_j")),
    )


def bench_opm(sizes, S: int = 64, c_msa: int = 64, c_outer: int = 32, c_out: int = 128):
    gc = model_config.GlobalConfig(bfloat16="all", final_init="linear")

    def make_fn():
        def fn(act, mask):
            return pm.OuterProductMeanCP(
                pm.OuterProductMeanCP.Config(num_outer_channel=c_outer),
                gc, num_output_channel=c_out, name="opm",
            )(act, mask)
        return fn

    def make_inputs(N, key):
        act = jax.random.normal(key, (S, N, c_msa), dtype=jnp.bfloat16)
        mask = jnp.ones((S, N), dtype=jnp.float32)
        return (act, mask)

    # OPM output is pair-sharded; inputs are MSA-sharded.
    _bench_module(
        f"OuterProductMeanCP (S={S}, c_msa={c_msa}, c_out={c_out})",
        sizes, make_fn, make_inputs,
        input_pair_axes=(("cp_j", "cp_i", None), ("cp_j", "cp_i")),
    )


def bench_collectives():
    """Raw collective bandwidth via parallelism.bench_comm."""
    print("\n=== raw collective bandwidth ===")
    mesh = parallelism.enable_cp(size_cp=4, device_ids=[0, 1, 2, 3])
    assert mesh is not None
    for mb in (1, 16, 64, 256):
        print(f"\npayload = {mb} MiB")
        parallelism.bench_comm(payload_mb=mb, warmup_iters=2, timed_iters=5, log=print)
    parallelism.disable_cp()


def main():
    devs = jax.devices()
    print(f"JAX backend: {jax.default_backend()}  devices={len(devs)}")
    for d in devs[:8]:
        print(f"  {d}")
    assert len(devs) >= 4, "need 4+ GPUs"

    # Raw NCCL/PCIe characterization first.
    bench_collectives()

    # Per-module compute+comm. Trim N at the top so we don't hit single-GPU
    # OOM during the cp_off init pass (init runs the serial path).
    tri_sizes = [256, 512, 768, 1024]
    bench_triangle_mul(tri_sizes, module_cls=pm.TriangleMultiplicationCP)
    bench_triangle_mul(tri_sizes, module_cls=pm.TriangleMultiplicationCPWhole)
    bench_grid_attention([256, 512, 768, 1024])
    bench_opm([256, 512, 768, 1024])


if __name__ == "__main__":
    main()
