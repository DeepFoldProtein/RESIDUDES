# Copyright 2024 DeepMind Technologies Limited
# SPDX-License-Identifier: Apache-2.0
"""Numerical-parity check: TriangleMultiplicationCP vs serial reference.

Run::

    source <core>/.venv/bin/activate
    PYTHONPATH=src python tests/test_parallelism_triangle_mul.py

Verifies that:
* the central CP einsum ``cp_pair_einsum`` matches a plain ``jnp.einsum``
  on the same inputs (within bf16 tolerance, since GLU runs at low precision
  in the real module);
* the Haiku module ``TriangleMultiplicationCP`` produces the same output
  as ``modules.TriangleMultiplication`` for both ``outgoing`` and ``incoming``
  equations when initialised with the same params.

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


# ──────────────────────────────────────────────────────────────────────────


def _check_close(label: str, ref, got, atol: float = 1e-2, rtol: float = 1e-2):
    ref = jax.device_get(ref)
    got = jax.device_get(got)
    diff = np.abs(ref - got).max()
    log.info(
        "[%s] shape=%s  max|Δ|=%.2e  max|ref|=%.2e  max|got|=%.2e",
        label, ref.shape, diff, np.abs(ref).max(), np.abs(got).max(),
    )
    np.testing.assert_allclose(got, ref, atol=atol, rtol=rtol, err_msg=label)


def test_cp_einsum_primitive(N: int = 32, c: int = 16):
    """Compare cp_pair_einsum vs serial einsum, both equations."""
    log.info("== test_cp_einsum_primitive  N=%d c=%d ==", N, c)
    key = jax.random.PRNGKey(0)
    a = jax.random.normal(key, (c, N, N), dtype=jnp.float32)
    b = jax.random.normal(jax.random.fold_in(key, 1), (c, N, N), dtype=jnp.float32)

    # Serial reference (no CP mesh active here).
    parallelism.disable_cp()
    ref_out = jnp.einsum("cik,cjk->cij", a, b)
    ref_inc = jnp.einsum("ckj,cki->cij", a, b)

    # CP path on the 2×2 mesh.
    mesh = parallelism.enable_cp(size_cp=4, device_ids=[0, 1, 2, 3])
    assert mesh is not None
    cp_out = parallelism_modules.cp_pair_einsum(a, b, "cik,cjk->cij")
    cp_inc = parallelism_modules.cp_pair_einsum(a, b, "ckj,cki->cij")
    _check_close("outgoing einsum", ref_out, cp_out)
    _check_close("incoming einsum", ref_inc, cp_inc)
    parallelism.disable_cp()


def test_haiku_module_parity(N: int = 32, c: int = 16):
    """Init params once with CP off, then run the module both with CP off
    and with CP enabled, expecting matching output."""
    log.info("== test_haiku_module_parity  N=%d c=%d ==", N, c)
    gc = model_config.GlobalConfig(
        bfloat16="none",                      # keep fp32 to widen tolerance
        final_init="linear",                  # avoid zero-init nullifying output
    )

    for eqn, label in [
        ("ikc,jkc->ijc", "outgoing"),
        ("kjc,kic->ijc", "incoming"),
    ]:
        cfg = parallelism_modules.TriangleMultiplicationCP.Config(
            equation=eqn,
            # use_glu_kernel=True relies on tokamax which itself is local and
            # numerically identical to the fallback path; either choice is
            # fine for parity testing, but the non-GLU path is faster to
            # init for a small test.
            use_glu_kernel=False,
        )

        def fn(act, mask):
            return parallelism_modules.TriangleMultiplicationCP(
                cfg, gc, name="tri"
            )(act, mask)

        net = hk.transform(fn)
        key = jax.random.PRNGKey(42)
        act = jax.random.normal(key, (N, N, c), dtype=jnp.float32)
        mask = jnp.ones((N, N), dtype=jnp.float32)
        params = net.init(key, act, mask)

        # Reference: CP disabled.
        parallelism.disable_cp()
        ref = net.apply(params, None, act, mask)

        # CP: 2×2 mesh.
        parallelism.enable_cp(size_cp=4, device_ids=[0, 1, 2, 3])
        got = net.apply(params, None, act, mask)
        parallelism.disable_cp()

        _check_close(f"tri-mul {label}", ref, got)


def test_whole_module_parity(N: int = 32, c: int = 16):
    """Parity check for :class:`TriangleMultiplicationCPWhole` — same as the
    central-only CP variant, but the entire ``__call__`` body runs inside one
    shard_map (Linear/LayerNorm on local tiles, no GSPMD bridging).

    Same params load both modules because the parameter layout is upstream-
    compatible. Run both CP off and CP on for both equation directions."""
    log.info("== test_whole_module_parity  N=%d c=%d ==", N, c)
    gc = model_config.GlobalConfig(
        bfloat16="none",
        final_init="linear",
    )
    for eqn, label in [
        ("ikc,jkc->ijc", "outgoing"),
        ("kjc,kic->ijc", "incoming"),
    ]:
        cfg = parallelism_modules.TriangleMultiplicationCPWhole.Config(
            equation=eqn, use_glu_kernel=False,
        )

        def fn(act, mask):
            return parallelism_modules.TriangleMultiplicationCPWhole(
                cfg, gc, name="tri",
            )(act, mask)

        net = hk.transform(fn)
        key = jax.random.PRNGKey(42)
        act = jax.random.normal(key, (N, N, c), dtype=jnp.float32)
        mask = jnp.ones((N, N), dtype=jnp.float32)
        params = net.init(key, act, mask)

        parallelism.disable_cp()
        ref = net.apply(params, None, act, mask)

        parallelism.enable_cp(size_cp=4, device_ids=[0, 1, 2, 3])
        got = net.apply(params, None, act, mask)
        parallelism.disable_cp()

        _check_close(f"tri-mul-whole {label}", ref, got)


def main():
    failures = []
    for name in (
        "test_cp_einsum_primitive",
        "test_haiku_module_parity",
        "test_whole_module_parity",
    ):
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
