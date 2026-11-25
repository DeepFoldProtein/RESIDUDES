#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Check the engine's torch reference models (they need a CUDA device to build) (the graphs the TensorRT engines
implement) against the AlphaFold 3 port, using the OpenFold 3-named state dict
that af3_to_openfold3.py produces. CPU, float32, a few blocks.

    core/.venv/bin/python core/alphafold3/tests/compare_trt_reference.py [--blocks 2] [--n 32]
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

import numpy as np
import torch

CORE = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CORE / "thirdparty"))
sys.path.insert(0, str(CORE / "alphafold3"))

from tensorrt_bionemo.configs import DiffusionTransformerConfig, PairformerConfig  # noqa: E402
from tensorrt_bionemo.models.openfold3.convert import (convert_hf_diffusion_transformer_torch,  # noqa: E402
                                                      convert_hf_pairformer_torch)
from tensorrt_bionemo._torch.layers.transformers.diffusion_transformer import OpenFold3DiffusionTransformer  # noqa: E402
from tensorrt_bionemo._torch.layers.transformers.pairformer import PairformerModule  # noqa: E402

from af3torch import diffusion_transformer as tdt  # noqa: E402
from af3torch import modules as tm  # noqa: E402
from af3torch.params import Params  # noqa: E402
from trt.af3_to_openfold3 import pairformer_state_dict, token_transformer_state_dict  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model-dir", default="/data/databases/af3")
ap.add_argument("--blocks", type=int, default=2)
ap.add_argument("--n", type=int, default=32)
args = ap.parse_args()
torch.manual_seed(0)
rng = np.random.default_rng(0)
N, NB = args.n, args.blocks
DEV = "cuda" if torch.cuda.is_available() else "cpu"
params = Params(pathlib.Path(args.model_dir), device=DEV)
root = params.root().child("diffuser")


def report(name, ref, port, t0, valid=None):
    ref, port = ref.detach().float(), port.detach().float()
    if valid is not None:                      # padded rows differ by construction (engine keeps the pair bias there)
        nd = 2 if ref.ndim == 3 and ref.shape[0] == ref.shape[1] else 1
        ref, port = ref[(slice(0, valid),) * nd], port[(slice(0, valid),) * nd]
    rel = (ref - port).norm() / max(ref.norm().item(), 1e-6)
    flag = "ok" if rel < 1e-3 else ("CHECK" if rel < 1e-2 else "MISMATCH")
    print(f"{name:34s} max|Δ| {(ref - port).abs().max():.3e}  rel {rel:.2e}  |ref| {ref.abs().max():.2e}  {flag}  ({time.time() - t0:.1f}s)", flush=True)


# ── pairformer ────────────────────────────────────────────────────────────
t0 = time.time()
cfg = PairformerConfig(token_s=384, token_z=128, pairwise_head_width=32, pairwise_num_heads=4, num_blocks=NB, num_heads=16,
                       dtype="float32", norm_epsilon=1e-5, support_batch=False, attention_initial_norm=True, version="v1",
                       trimul_high_precision=False, tri_attn_end_bias_before_transpose=True, backend="torch")
sd = pairformer_state_dict(params, num_blocks=NB)
ref = PairformerModule(cfg)
ref.load_weights(convert_hf_pairformer_torch(cfg, weights=sd))
ref.eval().to(DEV)
s = torch.from_numpy(rng.standard_normal((N, 384)).astype(np.float32)).to(DEV)
z = torch.from_numpy(rng.standard_normal((N, N, 128)).astype(np.float32)).to(DEV)
mask = torch.ones(N, device=DEV); mask[-3:] = 0
pair_mask = mask[:, None] * mask[None, :]
with torch.no_grad():
    rs, rz = ref(s[None], z[None], mask[None], pair_mask[None])   # the reference model takes a batch axis
    rs, rz = rs[0], rz[0]
    ps, pz = s, z
    for i in range(NB):
        sc = root.child("evoformer").stack("__layer_stack_no_per_layer_1", i).child("trunk_pairformer")
        pz, ps = tm.pairformer_iteration(pz, pair_mask, sc, single_act=ps, seq_mask=mask)
report(f"pairformer ×{NB}: pair (valid tokens)", rz, pz, t0, valid=N - 3)
report(f"pairformer ×{NB}: single (valid tokens)", rs, ps, t0, valid=N - 3)

# ── diffusion token transformer ───────────────────────────────────────────
t0 = time.time()
tcfg = DiffusionTransformerConfig(num_blocks=4, num_heads=16, dim=768, dim_single_cond=384, dim_pairwise=128, expansion_factor=2,
                                  bias_proj=True, conditioned_transition_using_silu=True, attention_initial_norm=False,
                                  post_layer_norm=False, version="v1", dtype="float32", norm_epsilon=1e-5, support_batch=False,
                                  backend="torch")
tsd = token_transformer_state_dict(params, num_blocks=4)
tref = OpenFold3DiffusionTransformer(tcfg)
tref.load_weights(convert_hf_diffusion_transformer_torch(tcfg, weights=tsd, prefix="sample_diffusion.diffusion_module.diffusion_transformer.blocks"))
tref.eval().to(DEV)
a = torch.from_numpy(rng.standard_normal((N, 768)).astype(np.float32)).to(DEV)
cond = torch.from_numpy(rng.standard_normal((N, 384)).astype(np.float32)).to(DEV)
with torch.no_grad():
    ra = tref(a=a[None], s=cond[None], z=z[None], mask=mask[None])[0]
    pa = tdt.transformer(a, mask, cond, z, root.child("~").child("diffusion_head").child("transformer"), num_blocks=4)
report("token transformer ×4 (valid tokens)", ra, pa, t0, valid=N - 3)
