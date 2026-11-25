# SPDX-License-Identifier: Apache-2.0
"""Protenix inference with JAX hot-swap.

Replaces ``model.pairformer_stack`` and
``model.diffusion_module.diffusion_transformer`` with Haiku/JAX modules
that share weights from the same checkpoint, then defers to Protenix's
native ``runner.inference.run()`` for everything else (data, embedding,
diffusion sampler, dumping).

Usage::

    python protenix/inference_jax.py \\
        --input_json_path /abs/path/input.json \\
        --dump_dir /abs/path/out \\
        --load_checkpoint_dir /abs/path/ckpt_dir \\
        --model_name protenix-v2

JAX-specific flags (consumed before Protenix's parser):
    --no_jax           Run vanilla PyTorch (skip the swap; for baselines).
    --jax_dtype fp32|bf16   Engine dtype for the JAX modules (default fp32).
    --jax_variant v1|v2     Protenix model variant (default v2).
    --jax_cache_dir PATH    Persistent XLA compilation cache directory
                            (default: $JAX_COMPILATION_CACHE_DIR or
                            ~/.cache/jax_protenix).

Note on env vars: this entry point does NOT require ``LAYERNORM_TYPE`` or
``DS_BUILD_OPS`` — those are OF3-specific. Protenix's native FusedLayerNorm
remains in use for the unswapped modules (input_embedder, msa_module,
diffusion_module shell, confidence head); only pairformer + token
transformer get the JAX backend.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger("protenix-jax")

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_CORE_ROOT = os.path.dirname(_SCRIPT_DIR)
_PROTENIX_ROOT = os.path.join(_CORE_ROOT, "models/protenix")

# ── Pre-parse JAX-only flags (so Protenix's parser doesn't see them) ──────
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--no_jax", action="store_true")
_pre.add_argument("--jax_dtype", choices=["fp32", "bf16"], default="bf16")
_pre.add_argument("--jax_variant", choices=["v1", "v2"], default="v2")
_pre.add_argument("--jax_cache_dir", default=None)
_pre.add_argument("--jax_full_diffusion", action="store_true",
                  help="Replace the entire sample_diffusion outer loop with "
                  "JAX (in addition to hot-swapped pairformer/transformer). "
                  "Requires variant=v2.")
_jax_args, _remaining = _pre.parse_known_args()
sys.argv = [sys.argv[0]] + _remaining

# ── Configure JAX BEFORE first jax import (compilation cache) ─────────────
_cache_dir = _jax_args.jax_cache_dir or os.environ.get(
    "JAX_COMPILATION_CACHE_DIR",
    os.path.expanduser("~/.cache/jax_protenix"),
)
os.makedirs(_cache_dir, exist_ok=True)
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", _cache_dir)

import jax
jax.config.update("jax_compilation_cache_dir", _cache_dir)
# Reuse cached binaries even when the source HLO has changed cosmetically.
jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)
jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)

# ── Import paths ──────────────────────────────────────────────────────────
sys.path.insert(0, _CORE_ROOT)        # residudes_engine_jax
sys.path.insert(0, _PROTENIX_ROOT)        # protenix native
_USER_CWD = os.getcwd()
os.chdir(_PROTENIX_ROOT)


# ── AF3-dialect → Protenix-native conversion (parity with protenix/inference.py) ─
def _maybe_convert_af3_input(json_path):
    """If ``json_path`` is an AF3-dialect JSON (or a directory of such),
    convert via scripts/af3_input_adapter.af3_to_pt2 and return the new path
    pointing at a Protenix-native list-of-jobs JSON. Else return unchanged.
    """
    import json as _json, tempfile, importlib.util as _ilu
    from pathlib import Path as _Path
    if not json_path:
        return json_path
    is_dir = os.path.isdir(json_path)
    if not is_dir and not os.path.isfile(json_path):
        return json_path

    def _is_af3(p):
        if not p.endswith(".json") or not os.path.isfile(p):
            return False
        try:
            with open(p) as f:
                d = _json.load(f)
        except (OSError, _json.JSONDecodeError):
            return False
        return isinstance(d, dict) and d.get("dialect") == "alphafold3"

    if is_dir:
        files = sorted(str(p) for p in _Path(json_path).iterdir() if _is_af3(str(p)))
        if not files:
            return json_path
    else:
        if not _is_af3(json_path):
            return json_path
        files = [json_path]

    spec = _ilu.spec_from_file_location(
        "residudes_af3_adapter",
        os.path.join(_CORE_ROOT, "scripts", "af3_input_adapter.py"),
    )
    adapter = _ilu.module_from_spec(spec)
    spec.loader.exec_module(adapter)

    sidecar = _Path(tempfile.mkdtemp(prefix="residudes_pt2_jax_af3_"))
    merged_jobs = []
    for fp in files:
        with open(fp) as f:
            data = _json.load(f)
        jobs = adapter.af3_to_pt2(data, sidecar_dir=sidecar)
        if isinstance(jobs, list):
            merged_jobs.extend(jobs)
        else:
            merged_jobs.append(jobs)
    out = sidecar / "pt2_input.json"
    out.write_text(_json.dumps(merged_jobs, indent=2))
    logger.info(
        "AF3 dialect → pt2 native: %s (%d input file(s) → %d job(s))",
        out, len(files), len(merged_jobs),
    )
    return str(out)


def _resolve_json_paths(json_path):
    """Resolve relative MSA / template paths in a Protenix native JSON to
    absolute (relative to caller's CWD), since this script chdir'd into
    models/protenix/. Returns a possibly-new path."""
    import json as _json, tempfile
    if not json_path or not os.path.isfile(json_path):
        return json_path
    with open(json_path) as f:
        data = _json.load(f)
    _PATH_KEYS = ("pairedMsaPath", "unpairedMsaPath", "templatesPath")
    changed = False
    for entry in (data if isinstance(data, list) else [data]):
        for seq in entry.get("sequences", []):
            for chain_type in ("proteinChain", "rnaSequence"):
                chain = seq.get(chain_type, {})
                for key in _PATH_KEYS:
                    val = chain.get(key)
                    if val and not os.path.isabs(val):
                        chain[key] = os.path.normpath(os.path.join(_USER_CWD, val))
                        changed = True
    if not changed:
        return json_path
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", prefix="residudes_pt2_jax_", delete=False)
    _json.dump(data, tmp, indent=2)
    tmp.close()
    return tmp.name


# Apply both conversions to whatever --input_json_path is in argv.
for _i, _arg in enumerate(sys.argv[:-1]):
    if _arg == "--input_json_path":
        _converted = _maybe_convert_af3_input(sys.argv[_i + 1])
        if _converted != sys.argv[_i + 1]:
            sys.argv[_i + 1] = _converted
        _resolved = _resolve_json_paths(sys.argv[_i + 1])
        if _resolved != sys.argv[_i + 1]:
            logger.info("Resolved JSON paths → temp file: %s", _resolved)
            sys.argv[_i + 1] = _resolved
        break

# Reuse the fast-init monkeypatch (~265s -> ~2s init time).
import importlib.util
_fi_spec = importlib.util.spec_from_file_location(
    "residudes_fast_init", os.path.join(_SCRIPT_DIR, "_fast_init.py"))
_fi_mod = importlib.util.module_from_spec(_fi_spec)
_fi_spec.loader.exec_module(_fi_mod)
_fi_mod.patch_protenix_trunc_normal()


def _inject_jax(model, ckpt_path: str) -> None:
    """Replace pairformer_stack + diffusion_transformer with JAX wrappers."""
    import jax.numpy as jnp
    from residudes_engine_jax.protenix import ProtenixJAX, ProtenixJAXConfig

    cfg = ProtenixJAXConfig.from_checkpoint(ckpt_path, variant=_jax_args.jax_variant)
    cfg.dtype = jnp.bfloat16 if _jax_args.jax_dtype == "bf16" else jnp.float32
    logger.info(
        "JAX hot-swap: variant=%s dtype=%s pairformer=%d blocks  token_transformer=%d blocks",
        cfg.variant, _jax_args.jax_dtype,
        cfg.pairformer_num_blocks, cfg.token_transformer_num_blocks,
    )
    wrapper = ProtenixJAX(model, cfg, ckpt_path=ckpt_path)
    wrapper.optimize()


# ── Monkey-patch InferenceRunner.load_checkpoint to inject after weights load ─
if not _jax_args.no_jax:
    from runner.inference import InferenceRunner

    _orig_load = InferenceRunner.load_checkpoint

    def _patched_load(self):
        _orig_load(self)
        ckpt_dir = getattr(self.configs, "load_checkpoint_dir", None)
        model_name = getattr(self.configs, "model_name", None)
        if ckpt_dir is None or model_name is None:
            raise RuntimeError(
                "JAX hot-swap requires --load_checkpoint_dir + --model_name."
            )
        ckpt_path = os.path.join(ckpt_dir, f"{model_name}.pt")
        logger.info("Injecting JAX from %s", ckpt_path)
        _inject_jax(self.model, ckpt_path)

    InferenceRunner.load_checkpoint = _patched_load
    logger.info("Compilation cache: %s", _cache_dir)

    # Persist tokamax autotune cache after the first forward — atexit can be
    # unreliable when JAX background threads outlive the main thread.
    from runner.inference import infer_predict as _orig_infer_predict_mod
    import runner.inference as _ri

    def _save_tokamax_after(orig_fn):
        def _wrapped(*a, **kw):
            try:
                return orig_fn(*a, **kw)
            finally:
                try:
                    from residudes_engine_jax.haiku import _autotune_cache as _ac
                    _ac.save_now()
                except Exception as _e:
                    logger.warning("autotune cache save failed: %s", _e)
        return _wrapped

    _ri.infer_predict = _save_tokamax_after(_ri.infer_predict)

# ── Optional stage timing (RESIDUDES_ENGINE_JAX_PROFILE=1) ─────────────────────────
if not _jax_args.no_jax and os.environ.get("RESIDUDES_ENGINE_JAX_PROFILE", "0") == "1":
    import time as _time
    import torch as _torch
    from protenix.model.protenix import Protenix as _Protenix

    _orig_main_loop = _Protenix._main_inference_loop

    def _patched_main_loop(self, *args, **kwargs):
        out = _orig_main_loop(self, *args, **kwargs)
        if isinstance(out, tuple) and len(out) == 3:
            _, _, tt = out
            if isinstance(tt, dict):
                logger.info("[STAGE] %s",
                            ", ".join(f"{k}={v:.2f}s" for k, v in tt.items()))
        return out

    _Protenix._main_inference_loop = _patched_main_loop

    # Trunk sub-stage timing — wrap the three big modules so we can see
    # input_embedder vs msa_module vs pairformer_stack splits.
    def _wrap_module_timer(mod, label, sums):
        import torch as _t
        _orig_call = mod.__class__.__call__

        def _patched_call(self, *args, **kwargs):
            if _t.cuda.is_available():
                _t.cuda.synchronize()
            t0 = _time.perf_counter()
            out = _orig_call(self, *args, **kwargs)
            if _t.cuda.is_available():
                _t.cuda.synchronize()
            elapsed = _time.perf_counter() - t0
            sums[label] = sums.get(label, 0.0) + elapsed
            sums.setdefault(label + "_calls", []).append(elapsed)
            return out
        mod.__class__.__call__ = _patched_call

    _trunk_sums = {}

    def _attach_trunk_timers(model):
        if getattr(model, "_residudes_trunk_timers_attached", False):
            return
        _wrap_module_timer(model.input_embedder, "input_embedder", _trunk_sums)
        _wrap_module_timer(model.msa_module, "msa_module", _trunk_sums)
        _wrap_module_timer(model.pairformer_stack, "pairformer_stack", _trunk_sums)
        if hasattr(model, "diffusion_module") and hasattr(model.diffusion_module, "diffusion_transformer"):
            _wrap_module_timer(
                model.diffusion_module.diffusion_transformer,
                "diff_transformer", _trunk_sums,
            )
        if hasattr(model, "confidence_head"):
            _wrap_module_timer(model.confidence_head, "confidence_head", _trunk_sums)
        model._residudes_trunk_timers_attached = True

    _orig_get_pf = _Protenix.get_pairformer_output

    def _patched_get_pf(self, *args, **kwargs):
        _attach_trunk_timers(self)
        _trunk_sums.clear()
        out = _orig_get_pf(self, *args, **kwargs)
        if _trunk_sums:
            logger.info("[TRUNK] %s",
                        ", ".join(f"{k}={v:.2f}s"
                                  for k, v in sorted(_trunk_sums.items())
                                  if not k.endswith("_calls")))
            for k, v in sorted(_trunk_sums.items()):
                if k.endswith("_calls") and v:
                    per = ", ".join(f"{x*1000:.0f}" for x in v)
                    logger.info("[TRUNK] %s per-call ms: [%s]", k[:-6], per)
        return out

    _Protenix.get_pairformer_output = _patched_get_pf
    logger.info("Stage profiler enabled (RESIDUDES_ENGINE_JAX_PROFILE=1)")


# ── (optional) full diffusion sampler in JAX ─────────────────────────────
if not _jax_args.no_jax and _jax_args.jax_full_diffusion:
    if _jax_args.jax_variant != "v2":
        raise SystemExit(
            "--jax_full_diffusion currently only supports variant=v2."
        )
    import protenix.model.generator as _gen
    from residudes_engine_jax.haiku.diffusion_module import DiffusionModuleConfig
    from residudes_engine_jax.protenix_sampler import make_jax_sample_diffusion

    _orig_sample_diffusion = _gen.sample_diffusion
    _jax_sample_diffusion = None

    def _patched_sample_diffusion(*args, **kwargs):
        global _jax_sample_diffusion
        if _jax_sample_diffusion is None:
            # Lazy build using ckpt_path captured from runner.configs at first call.
            # Find ckpt path from the calling frame's runner.
            import inspect
            ckpt_path = None
            for frame_info in inspect.stack():
                f_locals = frame_info.frame.f_locals
                if "self" in f_locals and hasattr(f_locals["self"], "configs"):
                    cfg = f_locals["self"].configs
                    cdir = getattr(cfg, "load_checkpoint_dir", None)
                    mname = getattr(cfg, "model_name", None)
                    if cdir and mname:
                        ckpt_path = os.path.join(cdir, f"{mname}.pt")
                        break
            if ckpt_path is None:
                raise RuntimeError("Could not infer checkpoint path for JAX sampler.")
            cfg = DiffusionModuleConfig(c_z=256)  # v2
            import jax.numpy as jnp
            dt = jnp.bfloat16 if _jax_args.jax_dtype == "bf16" else jnp.float32
            _jax_sample_diffusion = make_jax_sample_diffusion(
                ckpt_path=ckpt_path, config=cfg, dtype=dt,
            )
            logger.info("JAX sample_diffusion ready.")
        return _jax_sample_diffusion(*args, **kwargs)

    _gen.sample_diffusion = _patched_sample_diffusion
    logger.info("Patched sample_diffusion -> JAX (full diffusion loop on JAX)")


# ── Run Protenix's native entry point ─────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    from runner.inference import run
    run()
