#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
OpenFold3 inference with Bio-TRT acceleration.

Drops in TRT-accelerated pairformer + token transformer in place of the
standard PyTorch modules, then runs the standard OF3 prediction pipeline.

Usage:
    python openfold3/inference.py \\
        --query examples/query_ubiquitin.json \\
        --checkpoint /home/jovyan/.openfold3/of3-p2-155k.pt \\
        --engines-dir /workspace/engines/openfold3 \\
        --output-dir results/ubiquitin_trt

If engines don't exist yet, build them first:
    bash openfold3/trt/build_engines.sh \\
        --checkpoint /home/jovyan/.openfold3/of3-p2-155k.pt \\
        --engines-dir /workspace/engines/openfold3

Flags:
    --no-trt         Run with pure PyTorch (for baseline comparison)
    --pairformer-only  Only accelerate pairformer (skip token transformer TRT)
"""

import argparse
import json
import logging
import os
import sys
import tempfile

from pathlib import Path

logger = logging.getLogger(__name__)

# ── Bio-TRT thirdparty ──────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT / "thirdparty"))
sys.path.insert(0, str(_REPO_ROOT))   # for openfold3.compat


def parse_args():
    p = argparse.ArgumentParser(
        description="OpenFold3 inference with Bio-TRT TRT acceleration"
    )
    p.add_argument("--query", required=True, type=Path,
                   help="Path to input query JSON file")
    p.add_argument("--checkpoint", required=True, type=Path,
                   help="Path to OF3 checkpoint (.pt)")
    p.add_argument("--engines-dir", type=Path, default=None,
                   help="Directory containing pairformer_engine/ and "
                        "token_transformer_engine/ subdirs. "
                        "If None, falls back to --no-trt mode.")
    p.add_argument("--output-dir", type=Path, default=Path("results"),
                   help="Output directory for predictions")
    p.add_argument("--num-diffusion-samples", type=int, default=None,
                   help="Override number of diffusion samples")
    p.add_argument("--num-seeds", type=int, default=None,
                   help="Override number of model seeds")
    # ColabFold MSA / template fetching is OFF by default. With these flags
    # off, OF3 uses the *_msa_file_paths and template_alignment_file_path
    # supplied in the query JSON (e.g. files prepared by gen_inputs/openfold3_p2/).
    # Pass --colabfold-msa / --colabfold-templates to opt back in.
    p.add_argument("--colabfold-msa", "--use-msa-server",
                   action="store_true", dest="colabfold_msa",
                   help="Opt in to fetching MSAs from the ColabFold server. "
                        "Off by default — falls back to user-provided "
                        "main_msa_file_paths / paired_msa_file_paths.")
    p.add_argument("--colabfold-templates", "--use-templates",
                   action="store_true", dest="colabfold_templates",
                   help="Opt in to fetching templates from the ColabFold "
                        "server. Off by default — uses the user-provided "
                        "template_alignment_file_path.")
    # Legacy --no-msa / --no-templates kept as accepted-but-no-op aliases
    # for backward compatibility (they were the previous opt-out flags).
    p.add_argument("--no-msa", action="store_true",
                   help="(deprecated, no-op — colabfold is off by default)")
    p.add_argument("--no-templates", action="store_true",
                   help="(deprecated, no-op — colabfold is off by default)")
    p.add_argument("--no-trt", action="store_true",
                   help="Run without TRT acceleration (pure PyTorch baseline)")
    p.add_argument("--pairformer-only", action="store_true",
                   help="Only accelerate pairformer, not token transformer")
    p.add_argument("--runner-yaml", type=Path, default=None,
                   help="Optional OF3 runner YAML overrides")
    # ── MSA / template file injection ─────────────────────────────────────
    p.add_argument("--msa-a3m", type=Path, nargs="+", metavar="PATH",
                   help="Pre-computed MSA file(s) in .a3m format. "
                        "Injected into main_msa_file_paths for every protein chain "
                        "in the query (overrides ColabFold MSA fetch).")
    p.add_argument("--template-cif", type=Path, nargs="+", metavar="PATH",
                   help="Template structure file(s) in mmCIF (.cif) format. "
                        "Copied to a temp dir used as the template structures directory.")
    p.add_argument("--template-pdb", type=Path, nargs="+", metavar="PATH",
                   help="Template structure file(s) in PDB (.pdb) format. "
                        "Copied to a temp dir used as the template structures directory.")
    return p.parse_args()


def _maybe_convert_af3_query(query_path: Path) -> Path:
    """Accept either a single AF3-dialect input.json or a directory of them.
    Convert each to OF3 native, then merge into one query JSON whose
    `queries` dict contains every target's chains. Single model load
    processes the whole batch. If `query_path` is not AF3 dialect (single
    file), returns unchanged."""
    import importlib.util
    import tempfile
    p = Path(query_path)
    if not p.exists():
        return p

    def _is_af3(fp: Path) -> bool:
        if fp.suffix != ".json" or not fp.is_file():
            return False
        try:
            d = json.loads(fp.read_text())
        except (OSError, json.JSONDecodeError):
            return False
        return isinstance(d, dict) and d.get("dialect") == "alphafold3"

    if p.is_dir():
        files = sorted(fp for fp in p.iterdir() if _is_af3(fp))
        if not files:
            return p
    elif p.is_file():
        if not _is_af3(p):
            return p
        files = [p]
    else:
        return p

    spec = importlib.util.spec_from_file_location(
        "biotrt_af3_adapter",
        str(_REPO_ROOT / "scripts" / "af3_input_adapter.py"),
    )
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)

    sidecar = Path(tempfile.mkdtemp(prefix="biotrt_of3_af3_"))
    pairing_mode = os.environ.get("OF3_PAIRING_MODE", "")
    if pairing_mode not in ("precomputed", "online"):
        raise SystemExit(
            f"OF3_PAIRING_MODE must be set to 'precomputed' or 'online' "
            f"(got {pairing_mode!r}). Pass --pairing-mode to the OF3 submit "
            f"script, or export the env var manually."
        )
    merged: dict = {"seeds": None, "queries": {}}
    for fp in files:
        data = json.loads(fp.read_text())
        of3_native = adapter.af3_to_of3(data, sidecar_dir=sidecar,
                                        pairing_mode=pairing_mode)
        if merged["seeds"] is None:
            merged["seeds"] = of3_native.get("seeds")
        merged["queries"].update(of3_native.get("queries", {}))
    if merged["seeds"] is None:
        merged["seeds"] = [42]
    out = sidecar / "of3_query.json"
    out.write_text(json.dumps(merged, indent=2))
    logger.info(
        "AF3 dialect → OF3 native: %s (%d input file(s) → %d query/queries)",
        out, len(files), len(merged["queries"]),
    )
    return out


def _inject_trt(lightning_module, engines_dir: Path, pairformer_only: bool = False):
    """Replace pairformer_stack (and optionally diffusion_transformer) with TRT engines."""
    from tensorrt_bionemo.configs import AcceleratedConfig
    from tensorrt_bionemo.runtime import OnDemandContextMemoryManager

    # Import compat — re-exported from protenix.compat with OF3 names
    from openfold3.compat import OpenFold3TRT

    pairformer_engine_dir = engines_dir / "pairformer_engine" / "trt"
    token_transformer_engine_dir = engines_dir / "token_transformer_engine" / "trt"

    if not pairformer_engine_dir.exists():
        raise FileNotFoundError(
            f"Pairformer engine not found at {pairformer_engine_dir}. "
            f"Run build_engines.sh first."
        )

    accelerated_configs = {
        "structure_pairformer": AcceleratedConfig(
            checkpoint=str(pairformer_engine_dir),
            backend="trt",
        ),
    }

    if not pairformer_only:
        if not token_transformer_engine_dir.exists():
            logger.warning(
                f"Token transformer engine not found at {token_transformer_engine_dir}. "
                "Falling back to PyTorch token transformer."
            )
        else:
            accelerated_configs["token_transformer"] = AcceleratedConfig(
                checkpoint=str(token_transformer_engine_dir),
                backend="trt",
            )

    allocator = OnDemandContextMemoryManager()

    # Wrap the underlying OpenFold3 model and inject TRT modules in-place
    wrapped = OpenFold3TRT(lightning_module.model)
    wrapped.optimize(accelerated_configs, context_memory_allocator=allocator)

    logger.info(
        "TRT acceleration injected: %s", list(accelerated_configs.keys())
    )


def _patch_query_set(query_set, args, tmp_dir: Path):
    """Inject MSA / template file paths into an in-memory InferenceQuerySet.

    Args:
        query_set: InferenceQuerySet loaded from JSON.
        args:      Parsed CLI args (may have .msa_a3m, .template_cif, .template_pdb).
        tmp_dir:   Writable temp directory for staging copied template files.

    Returns:

        The same query_set object, mutated in-place.
    """
    import shutil

    # ── MSA injection ────────────────────────────────────────────────────────
    if args.msa_a3m:
        # OF3's MSA parser filters by filename stem against max_seq_counts keys
        # (colabfold_main, uniref90_hits, ...).  Arbitrary filenames are skipped.
        # Stage each provided .a3m into a per-chain temp dir as "colabfold_main.a3m"
        # and pass the *directory* to main_msa_file_paths so OF3 picks it up.
        msa_dirs = []
        for i, src in enumerate(args.msa_a3m):
            src = src.resolve()
            if not src.exists():
                raise FileNotFoundError(f"--msa-a3m path not found: {src}")
            msa_dir = tmp_dir / f"msa_{i}"
            msa_dir.mkdir(parents=True, exist_ok=True)
            dst = msa_dir / "colabfold_main.a3m"
            shutil.copy2(src, dst)
            msa_dirs.append(msa_dir)
            logger.info("Staged MSA %s → %s/colabfold_main.a3m", src.name, msa_dir)
        for query in query_set.queries.values():
            for chain in query.chains:
                # Only inject for protein / RNA / DNA chains (not ligands)
                if chain.sequence is not None:
                    existing = list(chain.main_msa_file_paths or [])
                    # Prepend provided dirs; keep any already set in JSON
                    chain.main_msa_file_paths = msa_dirs + existing
        logger.info("Injected %d MSA dir(s) into all sequence chains", len(msa_dirs))

    # ── Template structure injection ─────────────────────────────────────────
    template_files = list(args.template_cif or []) + list(args.template_pdb or [])
    if template_files:
        template_dir = tmp_dir / "templates"
        template_dir.mkdir(parents=True, exist_ok=True)
        for src in template_files:
            src = src.resolve()
            if not src.exists():
                raise FileNotFoundError(f"Template file not found: {src}")
            dst = template_dir / src.name
            shutil.copy2(src, dst)
            logger.info("Staged template %s → %s", src.name, dst)

        # Inject per-chain template_entry_chain_ids if not set, using filename stem
        # as the template ID (OF3 convention: stem = PDB code or custom ID).
        template_ids = [Path(f).stem for f in template_files]
        for query in query_set.queries.values():
            for chain in query.chains:
                if chain.sequence is not None and chain.template_entry_chain_ids is None:
                    chain.template_entry_chain_ids = template_ids
        logger.info(
            "Staged %d template file(s) to %s; template IDs: %s",
            len(template_files), template_dir, template_ids,
        )

        # Return template_dir so the caller can configure the runner
        return template_dir
    return None


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )
    args = parse_args()

    # ── OpenFold3 imports (requires OF3 repo on PYTHONPATH) ─────────────────
    try:
        from openfold3.entry_points.experiment_runner import InferenceExperimentRunner
        from openfold3.entry_points.import_utils import _torch_gpu_setup
        from openfold3.entry_points.validator import InferenceExperimentConfig
        from openfold3.projects.of3_all_atom.config.inference_query_format import (
            InferenceQuerySet,
        )
        from openfold3.core.config import config_utils
    except ImportError as e:
        sys.exit(
            f"Cannot import openfold3: {e}\n"
            "Set PYTHONPATH to the OpenFold3 repo:\n"
            "  export PYTHONPATH=/path/to/openfold-3:$PYTHONPATH"
        )

    _torch_gpu_setup()

    # RESIDUDES_ENGINE_PROFILE=1 → per-stage timing / peak memory of OpenFold3.forward
    # (core/scripts/stage_profiler.py; no-op otherwise). Works for torch, TRT and JAX.
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        from stage_profiler import install_class_profiler
        from openfold3.projects.of3_all_atom.model import OpenFold3 as _OF3Model
        install_class_profiler(
            _OF3Model,
            ("input_embedder", "template_embedder", "msa_module_embedder", "msa_module",
             "pairformer_stack", "sample_diffusion", "aux_heads"),
            label="of3",
        )
    except Exception as _e:  # pragma: no cover
        logging.getLogger("of3-inference").warning(f"stage profiler not installed: {_e}")

    # AF3 input.json passthrough: if --query is AF3 dialect, convert to OF3
    # native at runtime and point args.query at the converted file.
    args.query = _maybe_convert_af3_query(args.query)

    # ── Build OF3 experiment runner ──────────────────────────────────────────
    runner_args = config_utils.load_yaml(args.runner_yaml) if args.runner_yaml else {}

    # Surface the input query path so OF3OutputWriter (and any cross-model
    # writer) can attach MSA / template provenance to summary.json.
    os.environ["BIO_TRT_INPUT_PATH"] = str(Path(args.query).resolve())

    # Honour the `seeds` list inside the query JSON as the model inference
    # seed list. Without this override, OF3 uses experiment_settings.seeds
    # (default [42]) and treats the JSON's `seeds` field as data-module-only,
    # so the output dir is always seed_42 regardless of the caller's choice.
    try:
        _qs = json.loads(Path(args.query).read_text())
        _json_seeds = _qs.get("seeds")
        if isinstance(_json_seeds, list) and _json_seeds:
            runner_args.setdefault("experiment_settings", {})
            runner_args["experiment_settings"]["seeds"] = [int(s) for s in _json_seeds]
            logger.info(f"Using seeds from query JSON: {_json_seeds}")
    except Exception as _e:
        logger.warning(f"could not inspect seeds in {args.query}: {_e}")

    # Disable deepspeed evoformer_attn (not compatible with CUDA 13 JIT compile).
    # Use standard attention instead — no accuracy impact.
    from openfold3.projects.of3_all_atom.project_entry import ModelUpdate
    import os as _os
    # OF3's native num_recycles default is 3; we override to 10 to match
    # AF3 / Protenix v2 defaults so cross-model comparisons share the same
    # refinement depth. Override via OF3_NUM_RECYCLES env var if needed.
    _num_recycles = int(_os.environ.get("OF3_NUM_RECYCLES", "10"))
    logger.info(f"OF3 num_recycles = {_num_recycles}")
    _custom = {
        "settings": {
            "memory": {
                "eval": {"use_deepspeed_evo_attention": False},
            },
        },
        "architecture": {
            "shared": {"num_recycles": _num_recycles},
        },
    }
    # Optional override: drop aux_heads.pairformer down to sequential
    # per-sample processing for small-VRAM GPUs. Default behaviour keeps
    # OF3's 750-token cutoff (full-batch aux head above that); setting
    # OF3_PER_SAMPLE_TOKEN_CUTOFF=0 forces per-sample processing at every
    # length, which fits 24 GB at seq_len ≈ 700 with ~10× slower aux head.
    _cutoff = _os.environ.get("OF3_PER_SAMPLE_TOKEN_CUTOFF")
    if _cutoff is not None:
        _custom["settings"]["memory"]["eval"]["per_sample_token_cutoff"] = int(_cutoff)
    model_update = ModelUpdate(presets=["predict"], custom=_custom)
    expt_config = InferenceExperimentConfig(
        inference_ckpt_path=args.checkpoint,
        model_update=model_update,
        **runner_args,
    )
    # Surface output_dir to the dataset featurizer so it can drop a
    # post-load `inputs_runtime.json` (post-cap MSA depth + selected
    # template hits) — the unified writer reads it back into summary.json.
    import os as _os
    _os.environ["OF3_OUTPUT_DIR"] = str(args.output_dir)
    # Pairing stats cache: DataLoader workers fork from main; setting the
    # tempfile in the parent env BEFORE Lightning spawns workers ensures all
    # forks inherit the same path, so worker-side find_pairing_indices and
    # create_paired_from_precomputed write to a single file the main-side
    # writer can read back.
    import tempfile as _tf
    _os.environ["OF3_PAIRING_JSON"] = _tf.mkstemp(prefix="of3_pairing_", suffix=".json")[1]

    expt_runner = InferenceExperimentRunner(
        expt_config,
        num_diffusion_samples=args.num_diffusion_samples,
        num_model_seeds=args.num_seeds,
        use_msa_server=args.colabfold_msa,
        use_templates=args.colabfold_templates,
        output_dir=args.output_dir,
    )

    # ── Load checkpoint ──────────────────────────────────────────────────────
    logger.info("Setting up OF3 runner and loading checkpoint…")
    expt_runner.setup()

    # ── Inject TRT ───────────────────────────────────────────────────────────
    use_trt = not args.no_trt and args.engines_dir is not None
    if use_trt:
        logger.info("Injecting TRT acceleration from %s", args.engines_dir)
        _inject_trt(
            expt_runner.lightning_module,
            args.engines_dir,
            pairformer_only=args.pairformer_only,
        )
    else:
        if not args.no_trt and args.engines_dir is None:
            logger.warning("--engines-dir not provided; running without TRT.")
        else:
            logger.info("TRT disabled (--no-trt). Running pure PyTorch baseline.")

    # ── Load query and inject MSA / template files ──────────────────────────
    query_set = InferenceQuerySet.from_json(args.query)

    has_extra_inputs = args.msa_a3m or args.template_cif or args.template_pdb
    _tmp_ctx = tempfile.TemporaryDirectory(prefix="biotrt_") if has_extra_inputs else None
    try:
        if has_extra_inputs:
            tmp_dir = Path(_tmp_ctx.name)
            template_dir = _patch_query_set(query_set, args, tmp_dir)
            if template_dir is not None:
                # Configure runner to use the staged template directory.
                # OF3 exposes this via dataset_config template_preprocessor_settings.
                # Inject via runner_args override if not already set.
                logger.info(
                    "Template structures directory: %s\n"
                    "  Configure OF3 template_preprocessor_settings.structure_directory "
                    "via --runner-yaml if the default template search path differs.",
                    template_dir,
                )

        logger.info("Running inference on %d queries…", len(query_set.queries))
        expt_runner.run(query_set)
    finally:
        if _tmp_ctx is not None:
            _tmp_ctx.cleanup()

    expt_runner.cleanup()
    logger.info("Done. Results in %s", expt_runner.output_dir)


if __name__ == "__main__":
    main()
