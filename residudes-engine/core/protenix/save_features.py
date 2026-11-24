#!/usr/bin/env python3
"""
Save pre-computed Protenix features to .pt files.

Preprocesses JSON input(s) through Protenix's featurization pipeline
(sequence parsing, MSA, templates) and saves the result so that
inference can skip the expensive preprocessing step.

Supports bulk processing:
    --input_json_path   Single JSON file
    --input_dir         Directory of JSON files (processes all *.json)
    --skip_existing     Skip samples whose .pt already exists
    --num_workers       Parallel feature computation (CPU-bound)

Output:
    <output_dir>/<sample_name>.pt   (contains data dict + atom_array)
"""

import sys
import os
import logging
import pickle
import time
import glob
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch

logger = logging.getLogger(__name__)

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
_THIRDPARTY = os.path.join(_REPO_ROOT, "thirdparty")
_PROTENIX_ROOT = os.path.join(_REPO_ROOT, "models", "protenix")

# Pre-parse our extra args
import argparse
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--output_dir", type=str, required=True,
                  help="Directory to save feature .pt files")
_pre.add_argument("--input_dir", type=str, default=None,
                  help="Directory of JSON files (processes all *.json)")
_pre.add_argument("--skip_existing", action="store_true",
                  help="Skip samples whose .pt already exists in output_dir")
_pre.add_argument("--num_workers", type=int, default=0,
                  help="Parallel workers for feature computation (0=sequential)")
_extra_args, _remaining = _pre.parse_known_args()
sys.argv = [sys.argv[0]] + _remaining

# Capture user's CWD BEFORE chdir, so we can resolve relative paths inside
# input JSONs (unpairedMsaPath, pairedMsaPath, templatesPath) later.
_USER_CWD = os.getcwd()

# Resolve relative paths BEFORE chdir so user's CWD is respected.
for attr in ("output_dir", "input_dir"):
    val = getattr(_extra_args, attr, None)
    if val and not os.path.isabs(val):
        setattr(_extra_args, attr, os.path.abspath(val))

_PATH_ARGS = {"--dump_dir", "--input_json_path", "--load_checkpoint_dir"}
for i, arg in enumerate(sys.argv[:-1]):
    if arg in _PATH_ARGS:
        val = sys.argv[i + 1]
        if val and not os.path.isabs(val):
            sys.argv[i + 1] = os.path.abspath(val)

sys.path.insert(0, _THIRDPARTY)
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, _PROTENIX_ROOT)
os.chdir(_PROTENIX_ROOT)

# Monkey-patch Protenix's scipy-based weight init (~267s) with torch native (~2s).
import importlib.util as _importlib_util
_fi_spec = _importlib_util.spec_from_file_location(
    "biotrt_fast_init", os.path.join(_SCRIPT_DIR, "_fast_init.py"))
_fi_mod = _importlib_util.module_from_spec(_fi_spec)
_fi_spec.loader.exec_module(_fi_mod)
_fi_mod.patch_protenix_trunc_normal()


def _resolve_json_paths(json_path):
    """Resolve MSA/template paths inside input JSON relative to user's CWD.

    Writes a temp copy if changes are needed; returns the path to use.
    Mirrors the logic in protenix/inference.py so bulk save_features
    works with relative JSON paths when CWD != repo root.
    """
    import json as _json
    import tempfile
    if not json_path or not os.path.isfile(json_path):
        return json_path
    with open(json_path) as f:
        data = _json.load(f)
    _JSON_PATH_KEYS = ("pairedMsaPath", "unpairedMsaPath", "templatesPath")
    changed = False
    for entry in (data if isinstance(data, list) else [data]):
        for seq in entry.get("sequences", []):
            for chain_type in ("proteinChain", "rnaSequence"):
                chain = seq.get(chain_type, {})
                for key in _JSON_PATH_KEYS:
                    val = chain.get(key)
                    if val and not os.path.isabs(val):
                        chain[key] = os.path.normpath(os.path.join(_USER_CWD, val))
                        changed = True
    if changed:
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", prefix="biotrt_savefeat_", delete=False)
        _json.dump(data, tmp, indent=2)
        tmp.close()
        return tmp.name
    return json_path


def _collect_json_files(input_json_path, input_dir):
    """Collect all JSON files to process."""
    json_files = []
    if input_dir:
        json_files = sorted(glob.glob(os.path.join(input_dir, "*.json")))
        if not json_files:
            raise FileNotFoundError(f"No JSON files found in {input_dir}")
    if input_json_path and input_json_path != "null":
        if input_json_path not in json_files:
            json_files.append(input_json_path)
    return json_files


def _collect_sample_names(json_files):
    """Pre-scan JSON files to get all sample names for skip_existing check."""
    import json
    samples = []  # (json_path, sample_index, sample_name)
    for jf in json_files:
        with open(jf, "r") as f:
            data = json.load(f)
        if not isinstance(data, list):
            data = [data]
        for i, entry in enumerate(data):
            name = entry.get("name", f"sample_{i}")
            samples.append((jf, i, name))
    return samples


def _process_single_json(json_path, configs, output_dir, skip_existing):
    """Process all samples in a single JSON file."""
    import json
    from protenix.data.inference.infer_dataloader import InferenceDataset

    # Resolve relative MSA/template paths inside the JSON (may return temp copy)
    json_path = _resolve_json_paths(json_path)
    # Override input_json_path for this file
    configs.input_json_path = json_path

    with open(json_path, "r") as f:
        entries = json.load(f)
    if not isinstance(entries, list):
        entries = [entries]

    # Check which samples to skip
    if skip_existing:
        to_process = []
        for i, entry in enumerate(entries):
            name = entry.get("name", f"sample_{i}")
            pt_path = os.path.join(output_dir, f"{name}.pt")
            if os.path.exists(pt_path):
                logger.info(f"  Skipping {name} (already exists)")
            else:
                to_process.append(i)
        if not to_process:
            return 0
    else:
        to_process = list(range(len(entries)))

    dataset = InferenceDataset(configs=configs)
    saved = 0

    for idx in to_process:
        data, atom_array, error_message = dataset[idx]
        sample_name = data.get("sample_name", f"sample_{idx}")

        if error_message:
            logger.error(f"  Failed {sample_name}: {error_message}")
            continue

        save_path = os.path.join(output_dir, f"{sample_name}.pt")
        torch.save({
            "data": data,
            "atom_array": pickle.dumps(atom_array),
        }, save_path)
        saved += 1
        logger.info(
            f"  Saved {sample_name}: N_token={data['N_token'].item()}, "
            f"N_atom={data['N_atom'].item()}, N_msa={data['N_msa'].item()}"
        )

    return saved


def _worker_process_json(args):
    """Worker function for multiprocessing."""
    json_path, configs_dict, output_dir, skip_existing = args

    # Re-setup logging in worker
    logging.basicConfig(
        format="%(asctime)s,%(msecs)-3d %(levelname)-8s [%(filename)s:%(lineno)s] %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Re-setup paths in worker
    sys.path.insert(0, _THIRDPARTY)
    sys.path.insert(0, _REPO_ROOT)
    sys.path.insert(0, _PROTENIX_ROOT)
    os.chdir(_PROTENIX_ROOT)

    from protenix.config.config import parse_configs

    # Reconstruct configs from dict
    configs = parse_configs(
        configs=configs_dict,
        arg_str="",
        fill_required_with_null=True,
    )
    basename = os.path.basename(json_path)
    logger.info(f"[Worker] Processing {basename}")
    try:
        saved = _process_single_json(json_path, configs, output_dir, skip_existing)
        return basename, saved, None
    except Exception as e:
        import traceback
        return basename, 0, traceback.format_exc()


def main():
    log_format = (
        "%(asctime)s,%(msecs)-3d %(levelname)-8s "
        "[%(filename)s:%(lineno)s %(funcName)s] %(message)s"
    )
    logging.basicConfig(
        format=log_format,
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        filemode="w",
    )

    from configs.configs_base import configs as configs_base
    from configs.configs_data import data_configs
    from configs.configs_inference import inference_configs
    from configs.configs_model_type import model_configs
    from protenix.config.config import parse_configs, parse_sys_args
    from typing import Mapping

    arg_str = parse_sys_args()
    configs = {**configs_base, **{"data": data_configs}, **inference_configs}
    configs = parse_configs(configs=configs, arg_str=arg_str, fill_required_with_null=True)
    model_name = configs.model_name

    base_configs = {**configs_base, **{"data": data_configs}, **inference_configs}
    model_specifics = model_configs[model_name]

    def deep_update(d, u):
        for k, v in u.items():
            if isinstance(v, Mapping) and k in d and isinstance(d[k], Mapping):
                deep_update(d[k], v)
            else:
                d[k] = v
        return d

    deep_update(base_configs, model_specifics)
    configs = parse_configs(configs=base_configs, arg_str=arg_str, fill_required_with_null=True)

    from runner.inference import download_inference_cache
    download_inference_cache(configs)

    output_dir = _extra_args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Collect JSON files
    input_json_path = getattr(configs, "input_json_path", None)
    json_files = _collect_json_files(input_json_path, _extra_args.input_dir)
    total_samples = _collect_sample_names(json_files)

    if _extra_args.skip_existing:
        existing = set(
            os.path.splitext(f)[0]
            for f in os.listdir(output_dir) if f.endswith(".pt")
        )
        pending = [s for s in total_samples if s[2] not in existing]
        logger.info(
            f"Found {len(json_files)} JSON files, {len(total_samples)} total samples, "
            f"{len(existing)} already saved, {len(pending)} pending"
        )
    else:
        pending = total_samples
        logger.info(
            f"Found {len(json_files)} JSON files, {len(total_samples)} total samples"
        )

    if not pending:
        logger.info("Nothing to process. All features already exist.")
        return

    t_start = time.time()

    num_workers = _extra_args.num_workers
    skip_existing = _extra_args.skip_existing

    # Deduplicate JSON files that have pending samples
    pending_jsons = sorted(set(s[0] for s in pending))

    if num_workers > 1 and len(pending_jsons) > 1:
        # Parallel processing across JSON files
        logger.info(f"Processing {len(pending_jsons)} JSON files with {num_workers} workers")
        # We can't pickle configs directly, so serialize the base dict
        configs_raw = {**base_configs}
        worker_args = [
            (jf, configs_raw, output_dir, skip_existing)
            for jf in pending_jsons
        ]
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(_worker_process_json, a): a[0] for a in worker_args}
            total_saved = 0
            for future in as_completed(futures):
                basename, saved, error = future.result()
                if error:
                    logger.error(f"[Worker] {basename} failed:\n{error}")
                else:
                    total_saved += saved
                    logger.info(f"[Worker] {basename}: saved {saved} features")
    else:
        # Sequential processing
        total_saved = 0
        for jf in pending_jsons:
            basename = os.path.basename(jf)
            logger.info(f"Processing {basename}")
            configs.input_json_path = jf
            saved = _process_single_json(jf, configs, output_dir, skip_existing)
            total_saved += saved

    elapsed = time.time() - t_start
    total_pt = len([f for f in os.listdir(output_dir) if f.endswith(".pt")])
    logger.info(
        f"Done. Saved {total_saved} new features in {elapsed:.1f}s. "
        f"Total .pt files in {output_dir}: {total_pt}"
    )


if __name__ == "__main__":
    main()
