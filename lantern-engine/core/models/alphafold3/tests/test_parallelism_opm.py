# Copyright 2024 DeepMind Technologies Limited
# SPDX-License-Identifier: Apache-2.0
"""Numerical-parity check: OuterProductMeanCP vs serial reference.

Verifies that ``parallelism_modules.OuterProductMeanCP`` produces the same
output as its CP-disabled path when initialised with the same params.

Run::

    PYTHONPATH=src .venv/bin/python tests/test_parallelism_opm.py

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


def test_opm_parity(S: int = 16, N: int = 32, c: int = 16,
                    c_outer: int = 8, c_out: int = 12):
    """Init params with CP off; run with CP off and CP on; compare."""
    log.info("== test_opm_parity  S=%d N=%d c=%d ==", S, N, c)
    gc = model_config.GlobalConfig(
        bfloat16="none",
        final_init="linear",  # non-zero output_w init
    )

    def fn(act, mask):
        return parallelism_modules.OuterProductMeanCP(
            parallelism_modules.OuterProductMeanCP.Config(num_outer_channel=c_outer),
            gc,
            num_output_channel=c_out,
            name="opm",
        )(act, mask)

    net = hk.transform(fn)
    key = jax.random.PRNGKey(7)
    act = jax.random.normal(key, (S, N, c), dtype=jnp.float32)
    mask = jnp.ones((S, N), dtype=jnp.float32)

    parallelism.disable_cp()
    params = net.init(key, act, mask)
    ref = net.apply(params, None, act, mask)

    parallelism.enable_cp(size_cp=4, device_ids=[0, 1, 2, 3])
    got = net.apply(params, None, act, mask)
    parallelism.disable_cp()

    _check_close("opm", ref, got)


def main():
    failures = []
    for name in ("test_opm_parity",):
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
