# Copyright 2024 DeepMind Technologies Limited
# SPDX-License-Identifier: Apache-2.0
"""Peak-memory bench: does CP let us run bigger N?

Critical question: at what N does the **serial** (single-GPU) path OOM, and
does CP survive there?

Implementation: spawn one subprocess per (N, mode) cell so the GPU arena is
empty before each measurement. Each cell records peak ``bytes_in_use`` from
``jax.devices()[*].memory_stats()`` and emits a single line of ``key=value``
output that the driver parses. OOMs are caught and reported.

Run::

    PYTHONPATH=src AF3_CP_DEVICES=0,1,2,3 \\
      .venv/bin/python tests/bench_cp_memory.py
"""

from __future__ import annotations

import os
import subprocess
import sys


def _child(mode: str, N: int, c: int = 128):
    """One worker process: build params, run the apply, print result."""
    import jax, jax.numpy as jnp, haiku as hk  # noqa: E402
    from jax.sharding import NamedSharding, PartitionSpec as P  # noqa: E402
    from alphafold3.model import model_config  # noqa: E402
    from alphafold3.model.network import parallelism, parallelism_modules as pm  # noqa: E402

    gc = model_config.GlobalConfig(
        bfloat16="all", final_init="linear",
        flash_attention_implementation="xla",
    )
    eqn = "ikc,jkc->ijc"

    def _emit(status, peak_per_dev=None, err=""):
        import json as _json
        payload = {
            "mode": mode, "N": N, "status": status,
            "peak_mb": peak_per_dev or [], "err": err,
        }
        print("RESULT " + _json.dumps(payload))

    try:
        if mode == "cp_off":
            module_cls = pm.TriangleMultiplicationCP
            parallelism.disable_cp()
        elif mode == "cp_on":
            module_cls = pm.TriangleMultiplicationCP
        elif mode == "cp_whole":
            module_cls = pm.TriangleMultiplicationCPWhole
        else:
            raise ValueError(f"unknown mode {mode}")

        cfg = module_cls.Config(equation=eqn, use_glu_kernel=False)

        def fn(a, m):
            return module_cls(cfg, gc, name="tri")(a, m)

        net = hk.transform(fn)
        key = jax.random.PRNGKey(0)

        if mode == "cp_off":
            devs = jax.devices()[:1]
            act = jax.random.normal(key, (N, N, c), dtype=jnp.bfloat16)
            mask = jnp.ones((N, N), dtype=jnp.float32)
            parallelism.disable_cp()
            params = net.init(key, act, mask)
            apply = jax.jit(net.apply)
            args = (params, None, act, mask)
        else:
            mesh = parallelism.enable_cp(size_cp=4, device_ids=[0, 1, 2, 3])
            devs = list(mesh.devices.flatten())
            pair_sh = NamedSharding(mesh, P("cp_i", "cp_j", None))
            mask_sh = NamedSharding(mesh, P("cp_i", "cp_j"))
            repl_sh = NamedSharding(mesh, P())

            # Generate the random input *directly* into the sharded layout —
            # without this, ``jax.random.normal`` allocates the full tensor on
            # device 0 and the peak counter inflates by N²×c bytes before the
            # apply even runs (and worse, OOMs at large N).
            make_act = jax.jit(
                lambda k: jax.random.normal(k, (N, N, c), dtype=jnp.bfloat16),
                out_shardings=pair_sh,
            )
            make_mask = jax.jit(
                lambda: jnp.ones((N, N), dtype=jnp.float32),
                out_shardings=mask_sh,
            )
            act = make_act(key)
            mask = make_mask()

            # Init params using a tiny stand-in (full N²) is what would OOM —
            # but the modules' params depend only on the channel size, so we
            # can init with a smaller act of the SAME channel size.
            parallelism.disable_cp()
            init_act = jnp.zeros((4, 4, c), dtype=jnp.bfloat16)
            init_mask = jnp.zeros((4, 4), dtype=jnp.float32)
            params = net.init(key, init_act, init_mask)
            parallelism.enable_cp(size_cp=4, device_ids=[0, 1, 2, 3])
            params = jax.device_put(params, repl_sh)

            apply = jax.jit(
                net.apply,
                in_shardings=(repl_sh, None, pair_sh, mask_sh),
                out_shardings=pair_sh,
            )
            args = (params, None, act, mask)

        # Warmup + measure.
        out = apply(*args)
        jax.block_until_ready(out)

        peak = []
        for d in devs:
            s = d.memory_stats() or {}
            peak.append(s.get("peak_bytes_in_use", 0) / (1024 * 1024))
        _emit("ok", peak)
    except Exception as e:  # noqa: BLE001
        _emit("fail", None, f"{type(e).__name__}:{str(e)[:80]}")


def _driver():
    sizes = [1024, 1536, 2048, 2560, 3072, 4096]
    modes = ["cp_off", "cp_on", "cp_whole"]
    results = {}
    print(f"{'N':>5}  {'cp_off':>20}  {'cp_on (per-dev)':>30}  {'cp_whole':>30}")

    for N in sizes:
        row = {}
        for mode in modes:
            env = dict(os.environ)
            env.setdefault("AF3_CP_DEVICES", "0,1,2,3")
            try:
                proc = subprocess.run(
                    [sys.executable, "-u", __file__, "_child", mode, str(N)],
                    capture_output=True, text=True, timeout=600,
                    env=env, cwd=os.path.dirname(os.path.dirname(__file__)),
                )
                import json as _json
                line = next(
                    (ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")),
                    None,
                )
                if line is None:
                    err = proc.stderr.splitlines()[-1] if proc.stderr else "no output"
                    row[mode] = f"crash:{err[:40]}"
                else:
                    parts = _json.loads(line[len("RESULT "):])
                    if parts["status"] == "ok":
                        row[mode] = ",".join(f"{p:.0f}" for p in parts["peak_mb"]) + " MB"
                    else:
                        row[mode] = parts["err"][:40]
            except subprocess.TimeoutExpired:
                row[mode] = "timeout"
        results[N] = row
        print(f"{N:>5}  {row['cp_off']:>20}  {row['cp_on']:>30}  {row['cp_whole']:>30}")


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "_child":
        _child(sys.argv[2], int(sys.argv[3]))
    else:
        _driver()
