#!/usr/bin/env python3
"""
Protenix inference with residudes-engine TRT acceleration.

Monkey-patches Protenix's InferenceRunner to inject TRT after model loading.
Runs from models/protenix/ CWD so Protenix config imports work correctly.

Extra args (parsed before Protenix sees them):
    --engines_dir PATH    Path to TRT engines (pairformer_engine/ + token_transformer_engine/)
    --no_trt              Run pure PyTorch (no TRT injection)
    --pairformer_only     Only accelerate pairformer, not token transformer
"""

import sys
import os
import logging

logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
_THIRDPARTY = os.path.join(_REPO_ROOT, "thirdparty")
_PROTENIX_ROOT = os.path.join(_REPO_ROOT, "models", "protenix")

# Pre-parse TRT args before anything else
import argparse
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--engines_dir", type=str, default=None)
_pre.add_argument("--no_trt", action="store_true")
_pre.add_argument("--pairformer_only", action="store_true")
# ipSAE post-hook flags
_pre.add_argument("--ipsae", action="store_true",
                  help="Compute ipSAE interface scores on all dumped samples "
                       "(forces --need_atom_confidence true).")
_pre.add_argument("--ipsae_pae_cutoff", type=float, default=10.0)
_pre.add_argument("--ipsae_dist_cutoff", type=float, default=15.0)
_pre.add_argument("--no_write_ipsae_files", action="store_true",
                  help="Compute ipSAE but don't write .txt/_byres.txt/.pml files")
_trt_args, _remaining = _pre.parse_known_args()

# If --ipsae is set, force Protenix's --need_atom_confidence true so the
# full_data_sample_*.json files (containing token_pair_pae + atom_plddt) are
# dumped. Without them there's nothing for ipSAE to read.
if _trt_args.ipsae:
    _has_nac = False
    for i, a in enumerate(_remaining):
        if a == "--need_atom_confidence" and i + 1 < len(_remaining):
            _remaining[i + 1] = "true"
            _has_nac = True
            break
    if not _has_nac:
        _remaining += ["--need_atom_confidence", "true"]

sys.argv = [sys.argv[0]] + _remaining

# Resolve relative paths in argv BEFORE chdir, so user's CWD is respected.
# Protenix config parser reads these from sys.argv later.
_PATH_ARGS = {"--engines_dir", "--dump_dir", "--input_json_path",
              "--load_checkpoint_dir", "--load_checkpoint_path"}
_USER_CWD = os.getcwd()
for i, arg in enumerate(sys.argv[:-1]):
    if arg in _PATH_ARGS:
        val = sys.argv[i + 1]
        if val and not os.path.isabs(val):
            sys.argv[i + 1] = os.path.abspath(val)

# Also resolve pre-parsed TRT args
if _trt_args.engines_dir and not os.path.isabs(_trt_args.engines_dir):
    _trt_args.engines_dir = os.path.abspath(_trt_args.engines_dir)

# Resolve relative paths inside input JSON (MSA/template paths).
# Protenix reads these after chdir, so relative paths would break.
# AF3 input.json passthrough: if --input_json_path is an AF3-dialect file
# (or a directory of such files), convert each to pt2 native (list-of-jobs)
# at runtime, MERGE all jobs into one big list, and rewrite argv. Single
# model load processes the whole batch.
def _maybe_convert_af3_input(json_path):
    import json as _json, tempfile, importlib.util
    from pathlib import Path
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
        files = sorted(str(p) for p in Path(json_path).iterdir() if _is_af3(str(p)))
        if not files:
            return json_path
    else:
        if not _is_af3(json_path):
            return json_path
        files = [json_path]

    spec = importlib.util.spec_from_file_location(
        "residudes_af3_adapter",
        os.path.join(_REPO_ROOT, "scripts", "af3_input_adapter.py"),
    )
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)

    sidecar = Path(tempfile.mkdtemp(prefix="residudes_pt2_af3_"))
    merged_jobs: list = []
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
    """Resolve relative paths in input JSON to absolute (relative to JSON's directory).

    Writes a temp copy if changes are needed and updates sys.argv to point to it.
    Returns the (possibly new) json path.
    """
    import json, tempfile, shutil
    if not json_path or not os.path.isfile(json_path):
        return json_path
    json_dir = _USER_CWD
    with open(json_path) as f:
        data = json.load(f)
    _JSON_PATH_KEYS = ("pairedMsaPath", "unpairedMsaPath", "templatesPath")
    changed = False
    for entry in (data if isinstance(data, list) else [data]):
        for seq in entry.get("sequences", []):
            for chain_type in ("proteinChain", "rnaSequence"):
                chain = seq.get(chain_type, {})
                for key in _JSON_PATH_KEYS:
                    val = chain.get(key)
                    if val and not os.path.isabs(val):
                        chain[key] = os.path.normpath(os.path.join(json_dir, val))
                        changed = True
    if changed:
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", prefix="residudes_", delete=False)
        json.dump(data, tmp, indent=2)
        tmp.close()
        return tmp.name
    return json_path

# Find input_json_path from argv (already resolved to absolute), apply AF3
# adapter if dialect=alphafold3, then resolve JSON-internal relative paths.
for i, arg in enumerate(sys.argv[:-1]):
    if arg == "--input_json_path":
        converted = _maybe_convert_af3_input(sys.argv[i + 1])
        if converted != sys.argv[i + 1]:
            sys.argv[i + 1] = converted
        resolved = _resolve_json_paths(sys.argv[i + 1])
        if resolved != sys.argv[i + 1]:
            logger.info("Resolved JSON paths → temp file: %s", resolved)
            sys.argv[i + 1] = resolved
        break

# Set up paths
sys.path.insert(0, _THIRDPARTY)
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, _PROTENIX_ROOT)
os.chdir(_PROTENIX_ROOT)

# Monkey-patch Protenix's scipy-based weight init (~267s) with torch native (~2s).
# Checkpoint loading overwrites these weights anyway, so exact init values don't matter.
import importlib.util
_fi_spec = importlib.util.spec_from_file_location(
    "residudes_fast_init", os.path.join(_SCRIPT_DIR, "_fast_init.py"))
_fi_mod = importlib.util.module_from_spec(_fi_spec)
_fi_spec.loader.exec_module(_fi_mod)
_fi_mod.patch_protenix_trunc_normal()


def _inject_trt(model, engines_dir: str, pairformer_only: bool = False):
    """Replace pairformer_stack and diffusion_transformer with TRT engines."""
    from pathlib import Path

    sys.path.insert(0, _THIRDPARTY)
    from tensorrt_bionemo.configs import AcceleratedConfig
    from tensorrt_bionemo.runtime import OnDemandContextMemoryManager

    # Load compat via file path to avoid protenix namespace collision
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "residudes_compat", os.path.join(_SCRIPT_DIR, "compat.py"))
    compat = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(compat)

    engines_dir = Path(engines_dir)
    pf_dir = engines_dir / "pairformer_engine" / "trt"
    tt_dir = engines_dir / "token_transformer_engine" / "trt"

    if not pf_dir.exists():
        raise FileNotFoundError(f"Pairformer engine not found: {pf_dir}")

    configs = {"structure_pairformer": AcceleratedConfig(
        checkpoint=str(pf_dir), backend="trt")}

    if not pairformer_only and tt_dir.exists():
        configs["token_transformer"] = AcceleratedConfig(
            checkpoint=str(tt_dir), backend="trt")
    elif not pairformer_only:
        logger.warning("Token transformer engine not found, using PyTorch")

    allocator = OnDemandContextMemoryManager()
    wrapped = compat.ProtenixTRT(model)
    wrapped.optimize(configs, context_memory_allocator=allocator)
    logger.info("TRT injected: %s", list(configs.keys()))


# ── Monkey-patch InferenceRunner to inject TRT after load_checkpoint ──────
if _trt_args.engines_dir and not _trt_args.no_trt:
    from runner.inference import InferenceRunner

    _orig_load = InferenceRunner.load_checkpoint

    def _patched_load(self):
        _orig_load(self)
        logger.info("Injecting TRT engines from %s", _trt_args.engines_dir)
        _inject_trt(self.model, _trt_args.engines_dir, _trt_args.pairformer_only)

    InferenceRunner.load_checkpoint = _patched_load


# ── Run Protenix's native entry point ─────────────────────────────────────
def _find_dump_dir_from_argv() -> str | None:
    """Extract --dump_dir value from sys.argv (already resolved to absolute)."""
    for i, a in enumerate(sys.argv[:-1]):
        if a == "--dump_dir":
            return sys.argv[i + 1]
    return None


def _run_ipsae_hook():
    dump_dir = _find_dump_dir_from_argv()
    if not dump_dir or not os.path.isdir(dump_dir):
        logger.warning("[ipsae] dump_dir not found; skipping")
        return
    # Load via file path to avoid collision with models/protenix/scripts/
    # (both have a `scripts/` on sys.path, Protenix's has __init__.py so it wins).
    import importlib.util
    _hook_path = os.path.join(
        _REPO_ROOT, "scripts", "postprocess", "protenix_ipsae.py")
    spec = importlib.util.spec_from_file_location(
        "residudes_protenix_ipsae", _hook_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    logger.info(
        "[ipsae] scanning %s (pae<%.1f, dist<%.1f)",
        dump_dir, _trt_args.ipsae_pae_cutoff, _trt_args.ipsae_dist_cutoff,
    )
    mod.run_ipsae_on_dump(
        dump_dir=dump_dir,
        pae_cutoff=_trt_args.ipsae_pae_cutoff,
        dist_cutoff=_trt_args.ipsae_dist_cutoff,
        write_files=not _trt_args.no_write_ipsae_files,
        verbose=True,
    )


if __name__ == "__main__":
    # Optional stage profiler — share the same env knob the JAX path uses.
    if os.environ.get("RESIDUDES_ENGINE_JAX_PROFILE", "0") == "1":
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
        logger.info("Stage profiler enabled (RESIDUDES_ENGINE_JAX_PROFILE=1)")

    from runner.inference import run
    run()
    if _trt_args.ipsae:
        _run_ipsae_hook()
