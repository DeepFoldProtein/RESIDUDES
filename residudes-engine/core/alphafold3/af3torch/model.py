# SPDX-License-Identifier: CC-BY-NC-SA-4.0
"""AlphaFold 3 forward pass in PyTorch: trunk with recycling, diffusion sampling,
confidence and distogram heads. Produces the same result dictionary as
alphafold3.model.Model so the published post-processing can be reused as is."""
from __future__ import annotations

import dataclasses
import time

import numpy as np
import torch

from . import nn
from .diffusion_head import sample
from .evoformer import PAIR_CHANNEL, SEQ_CHANNEL, create_target_feat_embedding, evoformer
from .featurization import batch_to_device
from .heads import confidence_head, distogram_head
from .params import Params


@dataclasses.dataclass
class Config:
    num_recycles: int = 10
    num_diffusion_samples: int = 5
    diffusion_steps: int = 200
    return_distogram: bool = False
    return_embeddings: bool = False
    trunk_dtype: torch.dtype = torch.bfloat16
    sample_batch: int | None = None          # denoise this many samples at once (None = all)
    engines: object = None                   # af3torch.trt.Engines, or None for pure PyTorch
    log=print


def _np(t):
    if isinstance(t, torch.Tensor):
        t = t.detach()
        if t.dtype == torch.bfloat16:
            t = t.float()
        return t.cpu().numpy()
    return t


def predict(batch_dict: dict, params: Params, cfg: Config, seed: int) -> dict:
    """Run the model on one featurised example; returns numpy arrays keyed like the JAX model's output."""
    device = params.device
    t0 = time.time()
    with torch.inference_mode():
        batch = batch_to_device(batch_dict, device)
        gen = torch.Generator(device=device)
        gen.manual_seed(int(seed) & 0x7FFFFFFF)
        root = params.root().child("diffuser")
        dtype = cfg.trunk_dtype
        n = batch.num_res

        target_feat = create_target_feat_embedding(batch, root, dtype)
        prev = {"pair": torch.zeros((n, n, PAIR_CHANNEL), device=device), "single": torch.zeros((n, SEQ_CHANNEL), device=device)}
        ev = root.child("evoformer")
        for i in range(cfg.num_recycles + 1):
            out = evoformer(batch, prev, target_feat, ev, gen=gen, dtype=dtype, engines=cfg.engines)
            prev = {"pair": out["pair"].float(), "single": out["single"].float()}
            cfg.log(f"[af3 torch] recycle {i}/{cfg.num_recycles} done at {time.time() - t0:.1f} s")
        embeddings = {"pair": prev["pair"], "single": prev["single"], "target_feat": target_feat}

        dh = root.child("~").child("diffusion_head")
        samples = sample(batch, embeddings, dh, num_samples=cfg.num_diffusion_samples, steps=cfg.diffusion_steps, gen=gen,
                         sample_batch=cfg.sample_batch, engines=cfg.engines)
        cfg.log(f"[af3 torch] {cfg.num_diffusion_samples} samples × {cfg.diffusion_steps} steps done at {time.time() - t0:.1f} s")

        ch = root.child("confidence_head")
        conf_list = [confidence_head(samples["atom_positions"][k], embeddings, batch.token_features.mask,
                                     batch.pseudo_beta_info.token_atoms_to_pseudo_beta, batch.token_features.asym_id, ch, dtype=dtype)
                     for k in range(cfg.num_diffusion_samples)]
        confidence = {k: torch.stack([c[k] for c in conf_list]) for k in conf_list[0]}
        distogram = distogram_head(batch, embeddings, root.child("distogram_head"), return_distogram=cfg.return_distogram)
        cfg.log(f"[af3 torch] heads done at {time.time() - t0:.1f} s")

    result = {"diffusion_samples": {k: _np(v) for k, v in samples.items()},
              "distogram": {k: _np(v) for k, v in distogram.items()},
              **{k: _np(v) for k, v in confidence.items()}}
    if cfg.return_embeddings:
        result["single_embeddings"] = _np(embeddings["single"])
        result["pair_embeddings"] = _np(embeddings["pair"])
    result["__identifier__"] = params.identifier
    return result
