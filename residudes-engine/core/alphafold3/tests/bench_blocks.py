#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Micro-benchmark of the port's pair-stack blocks at a given token count (GPU, bf16):
grid attention (both orientations), triangle multiplication, pair transition, one
full pairformer block, one MSA-stack block. Prints seconds per call.

    core/.venv/bin/python core/alphafold3/tests/bench_blocks.py --n 2048 [--model-dir /gpfs/database/af3]
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from af3torch import modules as tm  # noqa: E402
from af3torch.params import Params  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=2048)
ap.add_argument("--model-dir", default="/gpfs/database/af3")
ap.add_argument("--reps", type=int, default=3)
args = ap.parse_args()
dev = "cuda"
params = Params(pathlib.Path(args.model_dir), device=dev)
root = params.root().child("diffuser")
pf = root.child("evoformer").stack("__layer_stack_no_per_layer_1", 0).child("trunk_pairformer")
ms = root.child("evoformer").stack("__layer_stack_no_per_layer", 0).child("msa_stack")
N, dt = args.n, torch.bfloat16
torch.manual_seed(0)
pair = torch.randn(N, N, 128, device=dev, dtype=dt)
single = torch.randn(N, 384, device=dev, dtype=dt)
msa = torch.randn(1024, N, 64, device=dev, dtype=dt)
mask = torch.ones(N, device=dev, dtype=dt)
pair_mask = mask[:, None] * mask[None, :]
msa_mask = torch.ones(1024, N, device=dev, dtype=dt)


def timeit(name, fn):
    with torch.inference_mode():
        fn(); torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(args.reps):
            fn()
        torch.cuda.synchronize()
    dt_ = (time.time() - t0) / args.reps
    print(f"{name:34s} {dt_:8.3f} s   (peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB)", flush=True)
    torch.cuda.reset_peak_memory_stats()


print(f"N={N}  chunk={tm.nn.shard_size(N, tm.PAIR_ATTENTION_CHUNK)}  {torch.cuda.get_device_name()}")
timeit("grid attention (rows)", lambda: tm.grid_self_attention(pair, pair_mask, pf.child("pair_attention1"), transpose=False))
timeit("grid attention (columns)", lambda: tm.grid_self_attention(pair, pair_mask, pf.child("pair_attention2"), transpose=True))
timeit("triangle mult (outgoing)", lambda: tm.triangle_multiplication(pair, pair_mask, pf.child("triangle_multiplication_outgoing"), outgoing=True))
timeit("triangle mult (incoming)", lambda: tm.triangle_multiplication(pair, pair_mask, pf.child("triangle_multiplication_incoming"), outgoing=False))
timeit("pair transition", lambda: tm.transition_block(pair, pf.child("pair_transition")))
timeit("pairformer block (pair + single)", lambda: tm.pairformer_iteration(pair, pair_mask, pf, single_act=single, seq_mask=mask))
timeit("msa stack block", lambda: tm.evoformer_iteration(msa, pair, msa_mask, pair_mask, ms))
