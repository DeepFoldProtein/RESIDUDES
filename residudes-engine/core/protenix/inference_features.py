#!/usr/bin/env python3
"""
Protenix inference from pre-computed feature files with TRT acceleration.

Loads .pt feature files saved by save_features.py and runs model inference,
skipping the expensive JSON->MSA->template->feature preprocessing pipeline.

Bulk-optimized:
    - Model + TRT loaded once, then iterates all .pt files
    - Background prefetch: next features loaded/transferred while GPU runs inference
    - --skip_existing skips samples already in dump_dir
    - --prefetch N controls how many samples to pre-load (default: 2)
    - Progress logging with ETA

Extra args (parsed before Protenix sees them):
    --features_dir PATH   Directory containing .pt feature files (required)
    --engines_dir PATH    Path to TRT engines
    --no_trt              Run pure PyTorch (no TRT injection)
    --pairformer_only     Only accelerate pairformer, not token transformer
    --seeds SEEDS         Comma-separated seeds (default: 101)
    --skip_existing       Skip samples already in dump_dir
    --prefetch N          Number of features to prefetch in background (default: 2)
"""

import sys
import os
import logging
import pickle
import time
import glob
import threading
import queue

import torch

logger = logging.getLogger(__name__)

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
_THIRDPARTY = os.path.join(_REPO_ROOT, "thirdparty")
_PROTENIX_ROOT = os.path.join(_REPO_ROOT, "models", "protenix")

# Pre-parse our extra args
import argparse
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--features_dir", type=str, required=True,
                  help="Directory containing .pt feature files from save_features.py")
_pre.add_argument("--engines_dir", type=str, default=None)
_pre.add_argument("--no_trt", action="store_true")
_pre.add_argument("--pairformer_only", action="store_true")
_pre.add_argument("--seeds", type=str, default="101")
_pre.add_argument("--skip_existing", action="store_true",
                  help="Skip samples whose output already exists in dump_dir")
_pre.add_argument("--prefetch", type=int, default=2,
                  help="Number of features to prefetch in background (default: 2)")
_extra_args, _remaining = _pre.parse_known_args()
sys.argv = [sys.argv[0]] + _remaining

# Resolve relative paths BEFORE chdir so user's CWD is respected.
for attr in ("features_dir", "engines_dir"):
    val = getattr(_extra_args, attr, None)
    if val and not os.path.isabs(val):
        setattr(_extra_args, attr, os.path.abspath(val))

_PATH_ARGS = {"--dump_dir", "--load_checkpoint_dir", "--load_checkpoint_path"}
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


def _get_existing_samples(dump_dir, seeds):
    """Scan dump_dir to find already-completed (sample, seed) pairs."""
    existing = set()
    if not os.path.isdir(dump_dir):
        return existing
    for sample_dir in os.listdir(dump_dir):
        sample_path = os.path.join(dump_dir, sample_dir)
        if not os.path.isdir(sample_path) or sample_dir == "ERR":
            continue
        for seed in seeds:
            seed_dir = os.path.join(sample_path, f"seed_{seed}", "predictions")
            if os.path.isdir(seed_dir) and any(
                f.endswith(".cif") for f in os.listdir(seed_dir)
            ):
                existing.add((sample_dir, seed))
    return existing


def _format_eta(elapsed, done, total):
    """Format ETA string."""
    if done == 0:
        return "?"
    rate = elapsed / done
    remaining = rate * (total - done)
    if remaining < 60:
        return f"{remaining:.0f}s"
    elif remaining < 3600:
        return f"{remaining / 60:.1f}m"
    else:
        return f"{remaining / 3600:.1f}h"


_SENTINEL = None  # Signals end of prefetch queue


class FeaturePrefetcher:
    """Background prefetcher that loads .pt features and transfers to GPU ahead of time.

    Producer thread: reads .pt from disk → unpickle → to_device(GPU) → put in queue
    Consumer (main thread): get from queue → immediately run inference (no I/O wait)
    """

    def __init__(self, work_plan, device, prefetch_count=2):
        """
        Args:
            work_plan: list of (feat_path, seed, sample_name) in execution order
            device: torch device to transfer data to
            prefetch_count: queue size = how many items loaded ahead
        """
        self._work_plan = work_plan
        self._device = device
        self._queue = queue.Queue(maxsize=prefetch_count)
        self._thread = threading.Thread(target=self._producer, daemon=True)
        self._error = None

    def start(self):
        self._thread.start()

    def _load_one(self, feat_path):
        """Load a single feature file: disk read → unpickle → to_device."""
        from protenix.utils.torch_utils import to_device
        saved = torch.load(feat_path, map_location="cpu", weights_only=False)
        data = saved["data"]
        atom_array = pickle.loads(saved["atom_array"])
        # Transfer tensors to GPU on a separate CUDA stream to avoid blocking
        # the default stream where inference runs.
        data = to_device(data, self._device)
        return data, atom_array

    def _producer(self):
        """Background thread: sequentially load features into the queue."""
        try:
            for feat_path, seed, sample_name in self._work_plan:
                data, atom_array = self._load_one(feat_path)
                # Blocks if queue is full (back-pressure when GPU is busy)
                self._queue.put((feat_path, seed, sample_name, data, atom_array))
        except Exception as e:
            self._error = e
        finally:
            self._queue.put(_SENTINEL)

    def __iter__(self):
        """Yields (feat_path, seed, sample_name, data, atom_array) items."""
        while True:
            item = self._queue.get()
            if item is _SENTINEL:
                if self._error is not None:
                    raise self._error
                return
            yield item


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

    from contextlib import nullcontext
    from typing import Mapping

    from configs.configs_base import configs as configs_base
    from configs.configs_data import data_configs
    from configs.configs_inference import inference_configs
    from configs.configs_model_type import model_configs
    from protenix.config.config import parse_configs, parse_sys_args
    from protenix.model.protenix import Protenix
    from protenix.utils.seed import seed_everything
    from runner.dumper import DataDumper
    from runner.inference import update_inference_configs

    # Parse Protenix configs
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

    seeds = [int(s) for s in _extra_args.seeds.split(",")]

    # Collect feature files
    features_dir = _extra_args.features_dir
    feature_files = sorted(glob.glob(os.path.join(features_dir, "*.pt")))
    if not feature_files:
        raise FileNotFoundError(f"No .pt feature files found in {features_dir}")

    # Skip existing
    dump_dir = configs.dump_dir
    os.makedirs(dump_dir, exist_ok=True)

    if _extra_args.skip_existing:
        existing = _get_existing_samples(dump_dir, seeds)
        work_items = []
        for feat_path in feature_files:
            sample_name = os.path.splitext(os.path.basename(feat_path))[0]
            for seed in seeds:
                if (sample_name, seed) not in existing:
                    work_items.append((feat_path, seed))
                else:
                    logger.info(f"Skipping {sample_name} seed={seed} (already exists)")
        logger.info(
            f"Found {len(feature_files)} features x {len(seeds)} seeds = "
            f"{len(feature_files) * len(seeds)} total. "
            f"Skipping {len(existing)}, processing {len(work_items)}"
        )
    else:
        work_items = [
            (feat_path, seed)
            for seed in seeds
            for feat_path in feature_files
        ]
        logger.info(
            f"Processing {len(feature_files)} features x {len(seeds)} seeds = "
            f"{len(work_items)} total inference runs"
        )

    if not work_items:
        logger.info("Nothing to process. All results already exist.")
        return

    # Build ordered work plan: grouped by seed, then by feature file
    # This ensures seed_everything() is called once per seed block.
    from collections import defaultdict
    work_by_seed = defaultdict(list)
    for feat_path, seed in work_items:
        sample_name = os.path.splitext(os.path.basename(feat_path))[0]
        work_by_seed[seed].append((feat_path, seed, sample_name))

    # Flatten into execution order: seed1 samples, seed2 samples, ...
    work_plan = []
    for seed in seeds:
        if seed in work_by_seed:
            work_plan.extend(work_by_seed[seed])

    # Setup device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    # Init model (once)
    logger.info("Initializing Protenix model...")
    model = Protenix(configs).to(device)

    checkpoint_path = os.path.join(configs.load_checkpoint_dir, f"{configs.model_name}.pt")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint["model"]
    if list(state_dict.keys())[0].startswith("module."):
        state_dict = {k[len("module."):]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict=state_dict, strict=configs.load_strict)
    model.eval()
    del checkpoint
    torch.cuda.empty_cache()
    logger.info(f"Model loaded: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params")

    # Inject TRT (once)
    if _extra_args.engines_dir and not _extra_args.no_trt:
        logger.info(f"Injecting TRT engines from {_extra_args.engines_dir}")
        _inject_trt(model, _extra_args.engines_dir, _extra_args.pairformer_only)

    # Init dumper
    dumper = DataDumper(
        base_dir=dump_dir,
        need_atom_confidence=configs.need_atom_confidence,
        sorted_by_ranking_score=configs.sorted_by_ranking_score,
    )

    eval_precision = {
        "fp32": torch.float32,
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }[configs.dtype]

    # Start background prefetcher
    prefetch_count = max(1, _extra_args.prefetch)
    logger.info(f"Starting prefetcher (prefetch={prefetch_count})")
    prefetcher = FeaturePrefetcher(work_plan, device, prefetch_count=prefetch_count)
    prefetcher.start()

    t0_start = time.time()
    done_count = 0
    total_count = len(work_plan)
    success_count = 0
    fail_count = 0
    current_seed = None

    for feat_path, seed, sample_name, data, atom_array in prefetcher:
        # Seed management: call seed_everything when seed changes
        if seed != current_seed:
            current_seed = seed
            seed_everything(seed=seed, deterministic=configs.deterministic)
            seed_count = sum(1 for _, s, _ in work_plan if s == seed)
            logger.info(f"=== Seed {seed}: {seed_count} samples ===")

        try:
            t2_start = time.time()

            n_token = data["N_token"].item()
            logger.info(
                f"[{done_count + 1}/{total_count}] {sample_name} [seed:{seed}]: "
                f"N_token {n_token}, N_atom {data['N_atom'].item()}, "
                f"N_msa {data['N_msa'].item()}"
            )

            new_configs = update_inference_configs(configs, n_token)
            model.configs = new_configs

            enable_amp = (
                torch.autocast(device_type="cuda", dtype=eval_precision)
                if torch.cuda.is_available()
                else nullcontext()
            )

            with torch.no_grad(), enable_amp:
                prediction, _, _ = model(
                    input_feature_dict=data["input_feature_dict"],
                    label_full_dict=None,
                    label_dict=None,
                    mode="inference",
                    mc_dropout_apply_rate=configs.mc_dropout_apply_rate,
                )

            dumper.dump(
                dataset_name="",
                pdb_id=sample_name,
                seed=seed,
                pred_dict=prediction,
                atom_array=atom_array,
                entity_poly_type={
                    k: v
                    for k, v in data["entity_poly_type"].items()
                    if v != "non-polymer"
                },
            )

            t2_end = time.time()
            done_count += 1
            success_count += 1
            elapsed = t2_end - t0_start
            eta = _format_eta(elapsed, done_count, total_count)
            logger.info(
                f"{sample_name} [seed:{seed}] done in {t2_end - t2_start:.2f}s "
                f"({done_count}/{total_count}, ETA: {eta})"
            )
            # Release prediction tensors; prefetcher manages feature memory
            del prediction
            torch.cuda.empty_cache()

        except Exception as e:
            import traceback
            error_message = f"{sample_name} failed: {e}\n{traceback.format_exc()}"
            logger.error(error_message)
            error_dir = os.path.join(dump_dir, "ERR")
            os.makedirs(error_dir, exist_ok=True)
            with open(os.path.join(error_dir, f"{sample_name}.txt"), "a") as f:
                f.write(error_message)
            done_count += 1
            fail_count += 1
            torch.cuda.empty_cache()

    t0_end = time.time()
    elapsed = t0_end - t0_start
    logger.info(
        f"=== Bulk inference complete ===\n"
        f"  Total: {total_count}, Success: {success_count}, Failed: {fail_count}\n"
        f"  Time: {elapsed:.1f}s"
        + (f" ({elapsed / success_count:.1f}s/sample avg)" if success_count else "")
        + f"\n  Results: {dump_dir}"
    )


if __name__ == "__main__":
    main()
