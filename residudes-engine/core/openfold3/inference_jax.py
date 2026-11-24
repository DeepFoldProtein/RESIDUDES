#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""OpenFold3 inference with JAX hot-swap.

Replaces ``model.pairformer_stack`` and
``model.sample_diffusion.diffusion_module.diffusion_transformer`` with
Haiku/JAX modules that share weights from the same checkpoint, then defers
to OpenFold3's native inference flow.

Usage::

    python openfold3/inference_jax.py \\
        --query examples/openfold3/ubiquitin.json \\
        --checkpoint /gpfs/deepfold/casp/casp17/checkpoints/of3-p2-155k.pt \\
        --output-dir results/of3_jax_ubiquitin

JAX-specific flags (consumed before OF3's parser):
    --no_jax           Run vanilla PyTorch (skip the swap; for baselines).
    --jax_dtype fp32|bf16   Engine dtype for the JAX modules (default bf16).
    --jax_cache_dir PATH    Persistent XLA compilation cache directory
                            (default: $JAX_COMPILATION_CACHE_DIR or
                            ~/.cache/jax_openfold3).

All other flags are forwarded to ``openfold3/inference.py``. The TRT engines
path (``--engines-dir``) is implicitly ignored when JAX is active — the
hot-swap takes the same modules TRT would have replaced.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger("openfold3-jax")

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_BIO_TRT_ROOT = os.path.dirname(_SCRIPT_DIR)

# ── Pre-parse JAX-only flags (so OF3's parser doesn't see them) ───────────
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--no_jax", action="store_true")
_pre.add_argument("--jax_dtype", choices=["fp32", "bf16"], default="bf16")
_pre.add_argument("--jax_cache_dir", default=None)
_jax_args, _remaining = _pre.parse_known_args()
sys.argv = [sys.argv[0]] + _remaining

# Force --no-trt so OF3's TRT injection path is skipped (we replace those
# modules with JAX instead). Idempotent if already present.
if not _jax_args.no_jax and "--no-trt" not in sys.argv:
    sys.argv.append("--no-trt")

# ── Configure JAX BEFORE first jax import (compilation cache) ─────────────
_cache_dir = _jax_args.jax_cache_dir or os.environ.get(
    "JAX_COMPILATION_CACHE_DIR",
    os.path.expanduser("~/.cache/jax_openfold3"),
)
os.makedirs(_cache_dir, exist_ok=True)
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", _cache_dir)

import jax  # noqa: E402
jax.config.update("jax_compilation_cache_dir", _cache_dir)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)
jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)

# ── Import paths (must match openfold3/inference.py) ──────────────────────
sys.path.insert(0, _BIO_TRT_ROOT)                        # residudes_engine_jax
sys.path.insert(0, str(Path(_BIO_TRT_ROOT) / "thirdparty"))
sys.path.insert(0, _BIO_TRT_ROOT)                        # openfold3.compat


def _inject_jax(lightning_module, ckpt_path: str) -> None:
    """Replace pairformer_stack + diffusion_transformer with JAX wrappers."""
    import jax.numpy as jnp
    from residudes_engine_jax.openfold3 import OpenFold3JAX, OpenFold3JAXConfig

    cfg = OpenFold3JAXConfig.from_checkpoint(ckpt_path)
    cfg.dtype = jnp.bfloat16 if _jax_args.jax_dtype == "bf16" else jnp.float32
    logger.info(
        "JAX hot-swap: dtype=%s pairformer=%d blocks  token_transformer=%d blocks",
        _jax_args.jax_dtype,
        cfg.pairformer_num_blocks, cfg.token_transformer_num_blocks,
    )
    wrapper = OpenFold3JAX(lightning_module.model, cfg, ckpt_path=ckpt_path)
    wrapper.optimize()


# ── Monkey-patch InferenceExperimentRunner.setup to inject after load ─────
if not _jax_args.no_jax:
    # Defer the import until OF3's package is on path (sys.path has been
    # set up above).
    from openfold3.entry_points.experiment_runner import (  # type: ignore
        InferenceExperimentRunner,
    )

    _orig_setup = InferenceExperimentRunner.setup

    # We grab the checkpoint path from CLI argv (parse_known_args left it).
    _ckpt_path = None
    for i, a in enumerate(sys.argv):
        if a in ("--checkpoint",) and i + 1 < len(sys.argv):
            _ckpt_path = sys.argv[i + 1]
            break
        if a.startswith("--checkpoint="):
            _ckpt_path = a.split("=", 1)[1]
            break

    def _patched_setup(self, *args, **kwargs):
        out = _orig_setup(self, *args, **kwargs)
        if _ckpt_path is None:
            raise RuntimeError(
                "JAX hot-swap requires --checkpoint to be passed in CLI."
            )
        logger.info("Injecting JAX from %s", _ckpt_path)
        _inject_jax(self.lightning_module, _ckpt_path)
        return out

    InferenceExperimentRunner.setup = _patched_setup
    logger.info("Compilation cache: %s", _cache_dir)


# ── Defer to OF3's inference entry point ─────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )
    # Reuse openfold3/inference.py's main()
    import importlib.util
    _of3_inf = os.path.join(_SCRIPT_DIR, "inference.py")
    spec = importlib.util.spec_from_file_location("openfold3_inference", _of3_inf)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main()
