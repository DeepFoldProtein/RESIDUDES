# Copyright 2024 DeepMind Technologies Limited
# SPDX-License-Identifier: Apache-2.0
"""End-to-end parity check: chained TriMul → GridAttn → OPM via CP.

Composes the three CP modules in the same sequence as
``modules.EvoformerIteration`` exercises them (OPM produces pair update from
MSA → TriMul outgoing → TriMul incoming → GridAttn row → GridAttn col) and
verifies that the chained output matches the CP-disabled reference.

This is the smallest test that catches sharding-composition bugs — e.g., a
module returning the pair tensor with axis 0 / axis 1 swapped, or with an
unexpected replication on cp_j, would propagate as a numerical mismatch when
the next module's shard_map slices the input.

Run::

    PYTHONPATH=src .venv/bin/python tests/test_parallelism_iteration.py
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
from alphafold3.model.network import parallelism, parallelism_modules as pm

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("test")


def _check_close(label, ref, got, atol=2e-2, rtol=2e-2):
    ref = jax.device_get(ref)
    got = jax.device_get(got)
    diff = np.abs(ref - got).max()
    log.info(
        "[%s] shape=%s  max|Δ|=%.2e  max|ref|=%.2e  max|got|=%.2e",
        label, ref.shape, diff, np.abs(ref).max(), np.abs(got).max(),
    )
    np.testing.assert_allclose(got, ref, atol=atol, rtol=rtol, err_msg=label)


def test_chain_parity(S: int = 12, N: int = 32, c_pair: int = 16,
                      c_msa: int = 16, c_outer: int = 8, num_head: int = 4):
    """Compose OPM + 2× TriMul + 2× GridAttn in sequence; compare CP off vs on."""
    log.info("== test_chain_parity  S=%d N=%d c_pair=%d ==", S, N, c_pair)
    gc = model_config.GlobalConfig(
        bfloat16="none",
        final_init="linear",
        flash_attention_implementation="xla",
    )

    def fn(pair_act, msa_act, pair_mask, msa_mask):
        # OPM: msa → Δpair
        pair_act = pair_act + pm.OuterProductMeanCP(
            pm.OuterProductMeanCP.Config(num_outer_channel=c_outer),
            gc, num_output_channel=c_pair, name="opm",
        )(msa_act, msa_mask)
        # TriMul outgoing
        pair_act = pair_act + pm.TriangleMultiplicationCP(
            pm.TriangleMultiplicationCP.Config(
                equation="ikc,jkc->ijc", use_glu_kernel=False,
            ),
            gc, name="trimul_outgoing",
        )(pair_act, pair_mask)
        # TriMul incoming
        pair_act = pair_act + pm.TriangleMultiplicationCP(
            pm.TriangleMultiplicationCP.Config(
                equation="kjc,kic->ijc", use_glu_kernel=False,
            ),
            gc, name="trimul_incoming",
        )(pair_act, pair_mask)
        # GridAttn row
        pair_act = pair_act + pm.GridSelfAttentionCP(
            pm.GridSelfAttentionCP.Config(num_head=num_head),
            gc, transpose=False, name="grid_row",
        )(pair_act, pair_mask)
        # GridAttn col
        pair_act = pair_act + pm.GridSelfAttentionCP(
            pm.GridSelfAttentionCP.Config(num_head=num_head),
            gc, transpose=True, name="grid_col",
        )(pair_act, pair_mask)
        return pair_act

    net = hk.transform(fn)
    key = jax.random.PRNGKey(123)
    pair_init = jax.random.normal(key, (N, N, c_pair), dtype=jnp.float32) * 0.1
    msa_init = jax.random.normal(jax.random.fold_in(key, 1), (S, N, c_msa),
                                 dtype=jnp.float32)
    pair_mask = jnp.ones((N, N), dtype=jnp.float32)
    msa_mask = jnp.ones((S, N), dtype=jnp.float32)

    parallelism.disable_cp()
    params = net.init(key, pair_init, msa_init, pair_mask, msa_mask)
    ref = net.apply(params, None, pair_init, msa_init, pair_mask, msa_mask)

    parallelism.enable_cp(size_cp=4, device_ids=[0, 1, 2, 3])
    got = net.apply(params, None, pair_init, msa_init, pair_mask, msa_mask)
    parallelism.disable_cp()

    _check_close("chain", ref, got)


def main():
    failures = []
    for name in ("test_chain_parity",):
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
