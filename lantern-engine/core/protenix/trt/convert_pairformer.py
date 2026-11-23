#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
Convert Protenix pairformer checkpoint to safetensors format for TRT engine building.
Adapted from NIM's convert_pairformer_checkpoint.py.

Protenix IS OpenFold3 — weight keys are identical.
The _patched_load_of3_state_dict handles Protenix checkpoint format.

Usage:
    python convert_pairformer.py \\
        --local_checkpoint /path/to/protenix.pt \\
        --output_dir /workspace/pairformer_safetensors \\
        --triangle_attention_backend VANILLA \\
        --triangle_attn_node_chunk_size 32
"""

import sys
import os

# Prepend thirdparty to path so tensorrt_bionemo and tensorrt_llm_lite are importable
_THIRDPARTY = os.path.join(os.path.dirname(__file__), '..', '..', 'thirdparty')
sys.path.insert(0, os.path.abspath(_THIRDPARTY))

import argparse
import json
import time
from pathlib import Path

import torch
import safetensors
import safetensors.torch
from tensorrt_llm_lite import logger

from tensorrt_bionemo.configs import BackendType
from tensorrt_bionemo.mapping import Mapping
from tensorrt_bionemo.models.protenix import ProtenixConfig
from tensorrt_bionemo.models.protenix.convert import convert_hf_pairformer


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Convert Protenix pairformer checkpoint to TRT-BNM safetensors"
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Protenix model name (e.g. protenix-v2). "
             "When set, overrides pairformer dimensions from model_configs.",
    )
    parser.add_argument(
        "--tp_size", type=int, default=1, help="N-way tensor parallelism size"
    )
    parser.add_argument(
        "--dcp_size", type=int, default=1, help="N-way data-context parallelism size"
    )
    parser.add_argument(
        "--max_transition_tp_size",
        type=bool,
        default=True,
        help="Whether to use the max transition tp size",
    )
    parser.add_argument(
        "--max_attention_pairwise_tp_size",
        type=bool,
        default=True,
        help="Whether to use the max attention pairwise tp size",
    )
    parser.add_argument(
        "--max_tri_mul_tp_size",
        type=bool,
        default=True,
        help="Whether to use the max tri mul tp size",
    )
    parser.add_argument(
        "--triangle_attn_node_chunk_size",
        type=int,
        default=32,  # Protenix default: 32 for large sequence support
        help="The chunk size for the triangle attention node",
    )
    parser.add_argument(
        "--triangle_attn_cueq_fallback_threshold",
        type=int,
        default=0,
        help="Threshold to fall back from CUEQUIV to VANILLA for triangle attention",
    )
    parser.add_argument(
        "--max_batch_size",
        type=int,
        default=1,
        help="The max batch size for the pairformer",
    )
    parser.add_argument(
        "--dtype", type=str, default="float32", choices=["bfloat16", "float32"]
    )
    parser.add_argument(
        "--norm_epsilon",
        type=float,
        default=0.001,  # Protenix uses 1e-3, not 1e-5
        help="Layer norm epsilon (Protenix default: 0.001)",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default="pairformer_checkpoint",
        help="The path to save the TensorRT-BNM checkpoint",
    )
    parser.add_argument(
        "--triangle_attention_backend",
        type=str,
        default="VANILLA",  # Protenix default: VANILLA (B200 cuEquivariance issue)
        choices=["VANILLA", "TRIFAST", "CUEQUIV"],
        help="The backend of triangle attention",
    )
    parser.add_argument(
        "--no_trimul_high_precision",
        action="store_true",
        help="Force trimul_high_precision=False even for protenix-v2. "
             "Halves tri_mul activation memory (FP32→bf16) at the cost of "
             "numerical stability for c_z=256.",
    )
    parser.add_argument(
        "--local_checkpoint",
        type=Path,
        default=None,
        help="The path to the local checkpoint",
    )
    parser.add_argument(
        "--support_batch", type=bool, default=False, help="Whether to support batch"
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="trt",
        choices=["trt"],
        help="The backend to convert",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="The number of workers for converting checkpoint in parallel",
    )

    args = parser.parse_args()

    # Allow PROTENIX_CKPT env var as fallback if --local_checkpoint not provided
    if args.local_checkpoint is None:
        env_ckpt = os.environ.get("PROTENIX_CKPT")
        if env_ckpt:
            args.local_checkpoint = Path(env_ckpt)
        else:
            parser.error(
                "--local_checkpoint is required (or set PROTENIX_CKPT env var)"
            )

    return args


def convert(worker_rank, world_size, configs, args):
    # Dump TRT config
    if args.backend == BackendType.TRT:
        (args.output_dir / f"{BackendType.TRT}").mkdir(parents=True, exist_ok=True)
        with (args.output_dir / f"{BackendType.TRT}/config.json").open("w") as f:
            json.dump(configs[BackendType.TRT].model_dump(), f, indent=4)

    for rank in range(worker_rank, world_size, args.workers):
        mapping = Mapping(
            world_size=world_size,
            tp_size=args.tp_size,
            dcp_size=args.dcp_size,
            rank=rank,
        )
        if args.backend == "all" or args.backend == BackendType.TRT:
            weights = convert_hf_pairformer(
                configs[BackendType.TRT],
                mapping,
                local_checkpoint=str(args.local_checkpoint),
                model_name="protenix",
            )
            safetensors.torch.save_file(
                weights,
                args.output_dir / f"{BackendType.TRT}/rank{rank}.safetensors",
            )


def main():
    args = parse_arguments()
    world_size = args.tp_size * args.dcp_size

    args.output_dir.mkdir(exist_ok=True, parents=True)

    tik = time.time()
    protenix_config = ProtenixConfig()
    pairformer_config = protenix_config.trunk.pairformer

    # Always use support_batch=True — required for v2 (token_z=256) where TRT
    # can't find implementations for non-batched layout at seqlen > 1024.
    args.support_batch = True

    # Auto-detect dimensions from checkpoint for non-default models (e.g. protenix-v2)
    token_s = pairformer_config.token_s
    token_z = pairformer_config.token_z
    pairwise_head_width = pairformer_config.pairwise_head_width
    pairwise_num_heads = pairformer_config.pairwise_num_heads
    num_blocks = pairformer_config.num_blocks
    num_heads = pairformer_config.num_heads

    if args.model_name:
        import torch
        ckpt = torch.load(str(args.local_checkpoint), map_location="cpu", weights_only=False)
        state = ckpt["model"] if "model" in ckpt else ckpt
        _pf = "module.pairformer_stack.blocks.0"
        # Detect token_z from pair_transition layernorm
        for k, v in state.items():
            if k == f"{_pf}.pair_transition.layernorm1.weight":
                token_z = v.shape[0]
                break
        # Detect pairwise_num_heads (triangle attention bias) from tri_att_start.linear
        for k, v in state.items():
            if k == f"{_pf}.tri_att_start.linear.weight":
                pairwise_num_heads = v.shape[0]
                break
        # Detect num_heads (self-attention pair bias) from linear_nobias_z
        for k, v in state.items():
            if k == f"{_pf}.attention_pair_bias.linear_nobias_z.weight":
                num_heads = v.shape[0]
                break
        pairwise_head_width = token_z // pairwise_num_heads
        del ckpt, state
        print(f"[{args.model_name}] Detected: token_z={token_z}, "
              f"num_heads={num_heads}, "
              f"pairwise_num_heads={pairwise_num_heads}, "
              f"pairwise_head_width={pairwise_head_width}")

    config = {
        "max_batch_size": args.max_batch_size,
        "max_transition_tp_size": args.max_transition_tp_size,
        "max_attention_pairwise_tp_size": args.max_attention_pairwise_tp_size,
        "max_tri_mul_tp_size": args.max_tri_mul_tp_size,
        "triangle_attn_node_chunk_size": args.triangle_attn_node_chunk_size,
        "triangle_attn_cueq_fallback_threshold": args.triangle_attn_cueq_fallback_threshold,
        "no_update_s": False,
        "no_update_z": False,
        "backend": "trt",
        "token_s": token_s,
        "token_z": token_z,
        "pairwise_head_width": pairwise_head_width,
        "pairwise_num_heads": pairwise_num_heads,
        "num_blocks": num_blocks,
        "num_heads": num_heads,
        "dtype": args.dtype,
        "norm_epsilon": args.norm_epsilon,
        "architecture": "pairformer",
        "mapping": {
            "world_size": world_size,
            "tp_size": args.tp_size,
            "dcp_size": args.dcp_size,
        },
        "disable_custom_all_reduce": (
            args.max_transition_tp_size
            or args.max_attention_pairwise_tp_size
            or args.max_tri_mul_tp_size
        ),
        "triangle_attention_backend": args.triangle_attention_backend,
        "post_layer_norm": pairformer_config.post_layer_norm,
        # v2 has c_z=256 which needs FP32 triangle multiplication for numerical
        # stability. v1 (c_z=128) is stable in BF16.
        # `--no_trimul_high_precision` forces False to halve tri_mul memory
        # at the cost of stability — useful for very large N_token where the
        # FP32 pair tensor doesn't fit.
        "trimul_high_precision": (args.model_name == "protenix-v2") and not args.no_trimul_high_precision,
        "version": "v1",
        "support_batch": args.support_batch,
    }
    trt_pairformer_config = pairformer_config.model_copy(update=config)

    configs = {
        BackendType.TRT: trt_pairformer_config,
    }

    if args.workers == 1:
        convert(0, world_size, configs, args)
    else:
        if args.workers > world_size:
            args.workers = world_size
        logger.info(f"Convert checkpoint using {args.workers} workers.")
        import torch.multiprocessing as mp
        mp.spawn(convert, nprocs=args.workers, args=(world_size, configs, args))

    tok = time.time()
    t = time.strftime("%H:%M:%S", time.gmtime(tok - tik))
    print(f"Total time of converting checkpoints: {t}")


if __name__ == "__main__":
    main()
