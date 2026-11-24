# Copyright 2024 DeepMind Technologies Limited
# SPDX-License-Identifier: Apache-2.0
"""Tighter parity test: TriangleMultiplicationCP under jit + sharded inputs.

Closer to the production AF3 setup than ``test_parallelism_triangle_mul.py``:

  * ``jax.jit`` with explicit ``in_shardings`` (so GSPMD actually distributes
    work rather than the shard_map handling it on a single-device input).
  * ``bf16`` math (production AF3 runs in bf16).
  * Multiple chained TriMul invocations to expose any compounding error.
  * Compares numerical output against a serial single-device run.

If this test fails, the bug we're seeing in T1313/A1 end-to-end is reproduced
at the smallest possible scope.

Run::

    PYTHONPATH=src AF3_CP_DEVICES=0,1,2,3 \\
      .venv/bin/python tests/test_trimul_jit_parity.py
"""

from __future__ import annotations

import logging
import sys
import traceback

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P

from alphafold3.model import model_config
from alphafold3.model.network import parallelism, parallelism_modules as pm

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("test")


def _check_close(label, ref, got, atol=5e-2, rtol=5e-2):
    ref = np.asarray(jax.device_get(ref))
    got = np.asarray(jax.device_get(got))
    diff = np.abs(ref - got)
    rel = diff / (np.abs(ref) + 1e-6)
    log.info(
        "[%s] shape=%s  max|Δ|=%.3e  max|ref|=%.3e  median|Δ/ref|=%.3e",
        label, ref.shape, diff.max(), np.abs(ref).max(),
        float(np.median(rel)),
    )
    np.testing.assert_allclose(got, ref, atol=atol, rtol=rtol, err_msg=label)


def test_trimul_jit_parity(N: int = 128, c: int = 128, n_chain: int = 4,
                           use_glu_kernel: bool = False):
    """Chained TriMul under jit, sharded vs serial.

    ``n_chain`` repeats: out = TriMul(out) loop, so errors compound across
    multiple invocations (closer to how PairFormerIteration uses TriMul).
    """
    log.info("== test_trimul_jit_parity N=%d c=%d n_chain=%d glu=%s ==",
             N, c, n_chain, use_glu_kernel)
    gc = model_config.GlobalConfig(
        bfloat16="all", final_init="linear",
        flash_attention_implementation="xla",
    )

    def fn(act, mask):
        # Chained calls — exactly the pattern PairFormerIteration uses
        # (act += TriMul(act, mask) twice + ...).  We use += chaining like
        # the upstream module does.
        cfg_o = pm.TriangleMultiplicationCP.Config(
            equation="ikc,jkc->ijc", use_glu_kernel=use_glu_kernel,
        )
        cfg_i = pm.TriangleMultiplicationCP.Config(
            equation="kjc,kic->ijc", use_glu_kernel=use_glu_kernel,
        )
        for k in range(n_chain):
            act = act + pm.TriangleMultiplicationCP(
                cfg_o, gc, name=f"trimul_out_{k}",
            )(act, mask)
            act = act + pm.TriangleMultiplicationCP(
                cfg_i, gc, name=f"trimul_in_{k}",
            )(act, mask)
        return act

    net = hk.transform(fn)
    key = jax.random.PRNGKey(0)
    act = jax.random.normal(key, (N, N, c), dtype=jnp.bfloat16) * 0.1
    mask = jnp.ones((N, N), dtype=jnp.float32)

    parallelism.disable_cp()
    params = net.init(key, act, mask)
    # Serial reference via jit (no CP) — matches production's compile path.
    apply_serial = jax.jit(net.apply)
    ref = apply_serial(params, None, act, mask)
    jax.block_until_ready(ref)

    # CP path via jit + in_shardings, exactly like ModelRunner._model uses.
    mesh = parallelism.enable_cp(size_cp=4, device_ids=[0, 1, 2, 3])
    pair_sh = NamedSharding(mesh, P("cp_i", "cp_j", None))
    mask_sh = NamedSharding(mesh, P("cp_i", "cp_j"))
    repl_sh = NamedSharding(mesh, P())
    act_s = jax.device_put(act, pair_sh)
    mask_s = jax.device_put(mask, mask_sh)
    params_s = jax.device_put(params, repl_sh)
    apply_cp = jax.jit(
        net.apply,
        in_shardings=(repl_sh, None, pair_sh, mask_sh),
        out_shardings=pair_sh,
    )
    got = apply_cp(params_s, None, act_s, mask_s)
    jax.block_until_ready(got)
    parallelism.disable_cp()

    _check_close(
        f"chained-trimul n={n_chain} glu={use_glu_kernel}",
        ref, got,
    )


def main():
    failures = []
    cases = [
        ("n=1_glu=False", dict(n_chain=1, use_glu_kernel=False)),
        ("n=1_glu=True",  dict(n_chain=1, use_glu_kernel=True)),
        ("n=4_glu=False", dict(n_chain=4, use_glu_kernel=False)),
        ("n=4_glu=True",  dict(n_chain=4, use_glu_kernel=True)),
    ]
    for name, kw in cases:
        try:
            test_trimul_jit_parity(**kw)
            log.info("OK  %s", name)
        except Exception:                       # noqa: BLE001
            log.error("FAIL %s\n%s", name, traceback.format_exc())
            failures.append(name)
    if failures:
        log.error("%d test(s) failed: %s", len(failures), failures)
        sys.exit(1)
    log.info("all tests passed")


if __name__ == "__main__":
    main()
