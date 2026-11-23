# Copyright 2024 DeepMind Technologies Limited
# SPDX-License-Identifier: Apache-2.0
"""Numerical-parity check: GridSelfAttentionCP vs serial reference.

Verifies that ``parallelism_modules.GridSelfAttentionCP`` produces the same
output as the upstream serial path for both ``transpose=False`` (row /
outgoing) and ``transpose=True`` (column / incoming) cases when initialised
with the same params.

Run::

    PYTHONPATH=src .venv/bin/python tests/test_parallelism_grid_attention.py

Exits 0 on success, non-zero on failure.
"""

from __future__ import annotations

import logging
import sys
import traceback

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

from alphafold3.model import model_config
from alphafold3.model.network import parallelism, parallelism_modules

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("test")


def _check_close(label, ref, got, atol=1e-2, rtol=1e-2):
    ref = jax.device_get(ref)
    got = jax.device_get(got)
    diff = np.abs(ref - got).max()
    log.info(
        "[%s] shape=%s  max|Δ|=%.2e  max|ref|=%.2e  max|got|=%.2e",
        label, ref.shape, diff, np.abs(ref).max(), np.abs(got).max(),
    )
    np.testing.assert_allclose(got, ref, atol=atol, rtol=rtol, err_msg=label)


def test_grid_attention_parity(N: int = 32, c: int = 16, num_head: int = 4):
    """Init params once with CP off, run with CP off and CP enabled,
    expect matching output for both transpose flags."""
    log.info("== test_grid_attention_parity  N=%d c=%d ==", N, c)
    gc = model_config.GlobalConfig(
        bfloat16="none",
        final_init="linear",  # non-zero init so output isn't trivially zero
        # Force a vanilla XLA attention path: tokamax's Triton kernels are
        # particular about shapes and dtypes, and the CP body's `b_g` ends
        # up fully replicated which trips its dispatch heuristic on small N.
        flash_attention_implementation="xla",
    )

    for transpose, label in [(False, "row"), (True, "col")]:

        def fn(act, mask):
            return parallelism_modules.GridSelfAttentionCP(
                parallelism_modules.GridSelfAttentionCP.Config(num_head=num_head),
                gc,
                transpose=transpose,
                name="grid",
            )(act, mask)

        net = hk.transform(fn)
        key = jax.random.PRNGKey(42)
        act = jax.random.normal(key, (N, N, c), dtype=jnp.float32)
        mask = jnp.ones((N, N), dtype=jnp.float32)

        parallelism.disable_cp()
        params = net.init(key, act, mask)

        # Reference: CP disabled.
        ref = net.apply(params, None, act, mask)

        # CP: 2×2 mesh.
        parallelism.enable_cp(size_cp=4, device_ids=[0, 1, 2, 3])
        got = net.apply(params, None, act, mask)
        parallelism.disable_cp()

        _check_close(f"grid-attn {label}", ref, got)


def main():
    failures = []
    for name in ("test_grid_attention_parity",):
        try:
            globals()[name]()
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
