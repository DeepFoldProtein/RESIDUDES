# SPDX-License-Identifier: CC-BY-NC-SA-4.0
"""A drop-in for run_alphafold.ModelRunner that runs the PyTorch port.

The fork's main loop featurises, calls `run_inference(example, rng_key)`, then
`extract_inference_results` / `extract_embeddings` / `extract_distogram`, and
writes the unified outputs. Only the forward pass changes."""
from __future__ import annotations

import functools
import os
import pathlib

import numpy as np
import torch

from alphafold3.model import features, model
from alphafold3.model.components import utils

from . import model as torch_model
from .params import Params


def _seed_from_key(rng_key) -> int:
    try:
        import jax
        data = np.asarray(jax.random.key_data(rng_key)) if hasattr(jax.random, "key_data") and jax.dtypes.issubdtype(
            getattr(rng_key, "dtype", np.uint32), jax.dtypes.prng_key) else np.asarray(rng_key)
    except Exception:  # noqa: BLE001
        data = np.asarray(rng_key)
    return int(data.reshape(-1)[-1])


class TorchModelRunner:
    """Same constructor and methods as run_alphafold.ModelRunner."""

    def __init__(self, config, device, model_dir: pathlib.Path):
        self._model_config = config
        self._device = torch.device(os.environ.get("LANTERN_ENGINE_AF3_TORCH_DEVICE", "cuda"))
        self._model_dir = pathlib.Path(model_dir)
        self._cp_enabled = False
        self.cfg = torch_model.Config(
            num_recycles=int(config.num_recycles),
            num_diffusion_samples=int(config.heads.diffusion.eval.num_samples),
            diffusion_steps=int(os.environ.get("LANTERN_ENGINE_AF3_STEPS") or config.heads.diffusion.eval.steps),   # override for smoke tests only
            return_distogram=bool(config.return_distogram),
            return_embeddings=bool(config.return_embeddings),
            trunk_dtype=torch.float32 if os.environ.get("LANTERN_ENGINE_AF3_TORCH_DTYPE", "bf16") == "fp32" else torch.bfloat16,
            sample_batch=int(os.environ["LANTERN_ENGINE_AF3_SAMPLE_BATCH"]) if os.environ.get("LANTERN_ENGINE_AF3_SAMPLE_BATCH") else None,
        )

    @functools.cached_property
    def model_params(self) -> Params:
        print(f"[af3 torch] loading parameters from {self._model_dir} onto {self._device}", flush=True)
        params = Params(self._model_dir, self._device)
        engines_dir = os.environ.get("LANTERN_ENGINE_AF3_ENGINES_DIR")
        if engines_dir:
            from .trt import Engines
            self.cfg.engines = Engines(pathlib.Path(engines_dir))
        return params

    def run_inference(self, featurised_example: features.BatchDict, rng_key) -> model.ModelResult:
        example = utils.remove_invalidly_typed_feats(featurised_example)
        seed = _seed_from_key(rng_key)
        return torch_model.predict(example, self.model_params, self.cfg, seed)

    def extract_inference_results(self, batch, result, target_name: str):
        return list(model.Model.get_inference_result(batch=batch, result=result, target_name=target_name))

    def extract_embeddings(self, result, num_tokens: int, dtype="float16"):
        out = {}
        if "single_embeddings" in result:
            out["single_embeddings"] = result["single_embeddings"][:num_tokens].astype(dtype)
        if "pair_embeddings" in result:
            out["pair_embeddings"] = result["pair_embeddings"][:num_tokens, :num_tokens].astype(dtype)
        return out or None

    def extract_distogram(self, result, num_tokens: int):
        if "distogram" not in result["distogram"]:
            return None
        return result["distogram"]["distogram"][:num_tokens, :num_tokens, :]
