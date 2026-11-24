# SPDX-License-Identifier: Apache-2.0
"""Phase-by-phase timing profiler for Protenix inference (PT + JAX modes).

Wraps the major sub-modules with CUDA-synchronized timers and prints a
breakdown at the end. Specifically times:
  - PairformerStack forward (per call x N_recycle)
  - sample_diffusion (whole loop)
  - DiffusionModule.forward (per diffusion step)
  - DiffusionTransformer (token transformer) forward (per call)
  - AtomAttentionEncoder.forward
  - AtomAttentionDecoder.forward
  - DLPack bridge (torch_to_jax / jax_to_torch) call counts
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from collections import defaultdict
from contextlib import contextmanager

logger = logging.getLogger("biotrt-profile")

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_BIO_TRT_ROOT = os.path.dirname(_SCRIPT_DIR)
_PROTENIX_ROOT = os.path.join(_BIO_TRT_ROOT, "models/protenix")

_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--no_jax", action="store_true")
_pre.add_argument("--jax_full_diffusion", action="store_true")
_pre.add_argument("--jax_dtype", choices=["fp32", "bf16"], default="bf16")
_pre.add_argument("--jax_variant", choices=["v1", "v2"], default="v2")
_jax_args, _remaining = _pre.parse_known_args()
sys.argv = [sys.argv[0]] + _remaining

# Configure JAX cache before importing
import jax
cache_dir = os.path.expanduser("~/.cache/jax_protenix")
os.makedirs(cache_dir, exist_ok=True)
jax.config.update("jax_compilation_cache_dir", cache_dir)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)
jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)

sys.path.insert(0, _BIO_TRT_ROOT)
sys.path.insert(0, _PROTENIX_ROOT)
os.chdir(_PROTENIX_ROOT)

import importlib.util
_fi = importlib.util.spec_from_file_location("biotrt_fast_init",
    os.path.join(_SCRIPT_DIR, "_fast_init.py"))
_fim = importlib.util.module_from_spec(_fi)
_fi.loader.exec_module(_fim); _fim.patch_protenix_trunc_normal()

import torch

# ── Timing infra ──────────────────────────────────────────────────────────
_times = defaultdict(list)
_counts = defaultdict(int)


@contextmanager
def time_block(label):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    yield
    # Force any pending JAX async work to finish on the same device.
    try:
        import jax
        for d in jax.devices():
            if d.platform == "gpu":
                d.synchronize_all_activity()
                break
    except Exception:
        pass
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    _times[label].append(time.perf_counter() - t0)
    _counts[label] += 1


def wrap_class_method(module_path: str, class_name: str, label: str, attr: str = "forward"):
    """Replace ``module.class_name.attr`` with a timed version."""
    import importlib
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    orig = getattr(cls, attr)

    def wrapped(self, *args, **kwargs):
        with time_block(label):
            return orig(self, *args, **kwargs)
    setattr(cls, attr, wrapped)


def wrap_function(module_path: str, fn_name: str, label: str):
    parts = module_path.split(".")
    import importlib
    mod = importlib.import_module(module_path)
    orig = getattr(mod, fn_name)

    def wrapped(*args, **kwargs):
        with time_block(label):
            return orig(*args, **kwargs)
    setattr(mod, fn_name, wrapped)


# Patch PT modules with timing
import protenix.model.modules.pairformer as _pf
import protenix.model.modules.diffusion as _df
import protenix.model.modules.transformer as _tr
import protenix.model.generator as _gen

wrap_class_method("protenix.model.protenix", "Protenix", "Protenix.forward")
wrap_class_method("protenix.model.modules.pairformer", "PairformerStack", "PairformerStack")
wrap_class_method("protenix.model.modules.pairformer", "MSAModule", "MSAModule")
wrap_class_method("protenix.model.modules.pairformer", "MSAStack", "MSAStack")
wrap_class_method("protenix.model.modules.embedders", "InputFeatureEmbedder", "InputFeatureEmbedder")
# Try to wrap TemplateEmbedder if it exists
wrap_class_method("protenix.model.modules.pairformer", "TemplateEmbedder", "TemplateEmbedder")
wrap_class_method("protenix.model.modules.confidence", "ConfidenceHead", "ConfidenceHead")
wrap_class_method("protenix.model.modules.head", "DistogramHead", "DistogramHead")
wrap_class_method("protenix.model.modules.diffusion", "DiffusionModule", "DiffusionModule.forward")
wrap_class_method("protenix.model.modules.diffusion", "DiffusionModule", "DiffusionModule.f_forward", attr="f_forward")
wrap_class_method("protenix.model.modules.diffusion", "DiffusionConditioning", "DiffusionConditioning")
wrap_class_method("protenix.model.modules.diffusion", "DiffusionConditioning", "DiffusionConditioning.prepare_cache", attr="prepare_cache")
wrap_class_method("protenix.model.modules.transformer", "DiffusionTransformer", "DiffusionTransformer")
wrap_class_method("protenix.model.modules.transformer", "AtomAttentionEncoder", "AtomAttentionEncoder")
wrap_class_method("protenix.model.modules.transformer", "AtomAttentionEncoder", "AtomAttentionEncoder.prepare_cache", attr="prepare_cache")
wrap_class_method("protenix.model.modules.transformer", "AtomAttentionDecoder", "AtomAttentionDecoder")
wrap_class_method("protenix.model.modules.transformer", "AtomTransformer", "AtomTransformer")
# Wrap JAX hot-swap wrappers
try:
    wrap_class_method("residudes_engine_jax.protenix", "_CompatPairformerJAXProtenix", "JAX/PairformerWrapper.forward")
    wrap_class_method("residudes_engine_jax.protenix", "_CompatTokenTransformerJAXProtenix", "JAX/TTWrapper.forward")
except Exception as e:
    print(f"could not wrap JAX wrappers: {e}")
wrap_function("protenix.model.generator", "sample_diffusion", "sample_diffusion")

# Bridge counter
import residudes_engine_jax.bridge as _br
_orig_t2j = _br.torch_to_jax
_orig_j2t = _br.jax_to_torch


def _counting_t2j(t):
    with time_block("BRIDGE/torch_to_jax"):
        return _orig_t2j(t)


def _counting_j2t(j, **kw):
    with time_block("BRIDGE/jax_to_torch"):
        return _orig_j2t(j, **kw)


_br.torch_to_jax = _counting_t2j
_br.jax_to_torch = _counting_j2t

# JAX-side hot-swap (mirror inference_jax.py)
if not _jax_args.no_jax:
    from runner.inference import InferenceRunner
    _orig_load = InferenceRunner.load_checkpoint

    def _patched_load(self):
        _orig_load(self)
        ckpt_dir = self.configs.load_checkpoint_dir
        model_name = self.configs.model_name
        ckpt_path = os.path.join(ckpt_dir, f"{model_name}.pt")
        from residudes_engine_jax.protenix import ProtenixJAX, ProtenixJAXConfig
        import jax.numpy as jnp
        cfg = ProtenixJAXConfig.from_checkpoint(ckpt_path, variant=_jax_args.jax_variant)
        cfg.dtype = jnp.bfloat16 if _jax_args.jax_dtype == "bf16" else jnp.float32
        ProtenixJAX(self.model, cfg, ckpt_path=ckpt_path).optimize()
        logger.info(f"JAX hot-swap done (variant={cfg.variant}, dtype={_jax_args.jax_dtype})")
    InferenceRunner.load_checkpoint = _patched_load

if not _jax_args.no_jax and _jax_args.jax_full_diffusion:
    import protenix.model.generator as _gen2
    from residudes_engine_jax.haiku.diffusion_module import DiffusionModuleConfig
    from residudes_engine_jax.protenix_sampler import make_jax_sample_diffusion
    _jax_sd = None

    def _patched_sd(*args, **kwargs):
        global _jax_sd
        if _jax_sd is None:
            import inspect
            ckpt_path = None
            for fr in inspect.stack():
                fl = fr.frame.f_locals
                if "self" in fl and hasattr(fl["self"], "configs"):
                    cdir = fl["self"].configs.load_checkpoint_dir
                    mname = fl["self"].configs.model_name
                    if cdir and mname:
                        ckpt_path = os.path.join(cdir, f"{mname}.pt")
                        break
            import jax.numpy as jnp
            dt = jnp.bfloat16 if _jax_args.jax_dtype == "bf16" else jnp.float32
            _jax_sd = make_jax_sample_diffusion(
                ckpt_path=ckpt_path, config=DiffusionModuleConfig(c_z=256), dtype=dt)
        with time_block("sample_diffusion"):
            return _jax_sd(*args, **kwargs)
    _gen2.sample_diffusion = _patched_sd


def _print_summary():
    print("\n" + "="*78)
    print("PHASE TIMING BREAKDOWN")
    print("="*78)
    print(f"{'Label':<48} {'Count':>6} {'Total(s)':>10} {'Avg(ms)':>10}")
    print("-"*78)
    rows = []
    for label, ts in _times.items():
        rows.append((label, len(ts), sum(ts), 1000 * sum(ts) / len(ts) if ts else 0))
    rows.sort(key=lambda r: -r[2])
    for label, cnt, total, avg in rows:
        print(f"{label:<48} {cnt:>6d} {total:>10.3f} {avg:>10.2f}")
    print("="*78)
    # Print per-call distribution for the most-suspect labels
    for label in ("JAX/PairformerWrapper.forward", "JAX/TTWrapper.forward"):
        ts = _times.get(label)
        if ts and len(ts) >= 2:
            print(f"\n{label} per-call (ms):")
            print("  " + ", ".join(f"{1000*t:.0f}" for t in ts[:15]))
            print(f"  min={1000*min(ts):.0f}, max={1000*max(ts):.0f}, median={1000*sorted(ts)[len(ts)//2]:.0f}")


import atexit
atexit.register(_print_summary)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S")
    from runner.inference import run
    run()
