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
Convert OpenFold3 pairformer checkpoint to safetensors for TRT engine building.

OpenFold3 checkpoint format differs from Protenix:
  - Uses EMA weights: ckpt["ema"]["params"]
  - Or deepspeed format: directory with bf16_zero_pp_rank_0_mp_rank_00_model_states.pt

Usage:
    python convert_pairformer.py \\
        --local_checkpoint /path/to/openfold3.pt \\
        --output_dir /workspace/pairformer_safetensors \\
        --triangle_attention_backend VANILLA \\
        --triangle_attn_node_chunk_size 32
"""

import sys
import os

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
from tensorrt_bionemo.models.openfold3 import OpenFold3Config
from tensorrt_bionemo.models.openfold3.convert import convert_hf_pairformer
import tensorrt_bionemo.hubs.local as _trtbnm_local


def _patched_load_of3_state_dict(local_checkpoint: str):
    """Load OpenFold3 checkpoint into state dict.

    Handles:
      - EMA format: ckpt["ema"]["params"]
      - DeepSpeed format: directory
      - Plain dict
    """
    ckpt_path = Path(local_checkpoint)

    # DeepSpeed checkpoint directory
    if ckpt_path.is_dir():
        ds_files = list(ckpt_path.glob("bf16_zero_pp_rank_*_model_states.pt"))
        if ds_files:
            raw = torch.load(ds_files[0], map_location="cpu", weights_only=False)
            # DeepSpeed zero stage 2/3: state is in "module"
            state = raw.get("module", raw)
            if isinstance(state, dict):
                return {k.replace("model.", ""): v for k, v in state.items()}

    ckpt = torch.load(local_checkpoint, map_location="cpu", weights_only=False)

    if isinstance(ckpt, dict):
        if "ema" in ckpt and "params" in ckpt.get("ema", {}):
            return ckpt["ema"]["params"]
        if "state_dict" in ckpt:
            return {k.replace("model.", ""): v
                    for k, v in ckpt["state_dict"].items()}
        if "model" in ckpt:
            state = ckpt["model"]
            sample_key = next(iter(state))
            if sample_key.startswith("module."):
                state = {k[len("module."):]: v for k, v in state.items()}
            return state
        # plain weight dict
        return ckpt

    return ckpt


_trtbnm_local._load_of3_state_dict = _patched_load_of3_state_dict


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Convert OpenFold3 pairformer checkpoint to TRT-BNM safetensors"
    )
    parser.add_argument("--tp_size", type=int, default=1)
    parser.add_argument("--dcp_size", type=int, default=1)
    parser.add_argument(
        "--triangle_attn_node_chunk_size", type=int, default=32,
        help="Chunk size for triangle attention (enables large N support)"
    )
    parser.add_argument(
        "--triangle_attention_backend", type=str, default="VANILLA",
        choices=["VANILLA", "TRIFAST", "CUEQUIV"],
    )
    parser.add_argument("--max_batch_size", type=int, default=1)
    parser.add_argument(
        "--dtype", type=str, default="float32", choices=["bfloat16", "float32"]
    )
    parser.add_argument(
        "--norm_epsilon", type=float, default=0.001,
        help="Layer norm epsilon (OpenFold3 default: 0.001)"
    )
    parser.add_argument("--output_dir", type=Path, default="pairformer_checkpoint")
    parser.add_argument("--local_checkpoint", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--backend", type=str, default="trt", choices=["trt"])

    args = parser.parse_args()

    if args.local_checkpoint is None:
        env_ckpt = os.environ.get("OPENFOLD3_CKPT")
        if env_ckpt:
            args.local_checkpoint = Path(env_ckpt)
        else:
            parser.error(
                "--local_checkpoint is required (or set OPENFOLD3_CKPT env var)"
            )
    return args


def convert(worker_rank, world_size, configs, args):
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
        weights = convert_hf_pairformer(
            configs[BackendType.TRT],
            mapping,
            local_checkpoint=str(args.local_checkpoint),
            model_name="openfold3",
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
    openfold3_config = OpenFold3Config()
    pairformer_config = openfold3_config.trunk.pairformer

    config = {
        "max_batch_size": args.max_batch_size,
        "triangle_attn_node_chunk_size": args.triangle_attn_node_chunk_size,
        "triangle_attention_backend": args.triangle_attention_backend,
        "triangle_attn_cueq_fallback_threshold": 0,
        "max_transition_tp_size": True,
        "max_attention_pairwise_tp_size": True,
        "max_tri_mul_tp_size": True,
        "no_update_s": False,
        "no_update_z": False,
        "backend": "trt",
        "token_s": pairformer_config.token_s,
        "token_z": pairformer_config.token_z,
        "pairwise_head_width": pairformer_config.pairwise_head_width,
        "pairwise_num_heads": pairformer_config.pairwise_num_heads,
        "num_blocks": pairformer_config.num_blocks,
        "num_heads": pairformer_config.num_heads,
        "dtype": args.dtype,
        "norm_epsilon": args.norm_epsilon,
        "architecture": "pairformer",
        "mapping": {
            "world_size": world_size,
            "tp_size": args.tp_size,
            "dcp_size": args.dcp_size,
        },
        "disable_custom_all_reduce": True,
        "triangle_attention_backend": args.triangle_attention_backend,
        "post_layer_norm": pairformer_config.post_layer_norm,
        "trimul_high_precision": False,
        "version": "v1",
        "support_batch": False,
    }
    trt_config = pairformer_config.model_copy(update=config)
    configs = {BackendType.TRT: trt_config}

    if args.workers == 1:
        convert(0, world_size, configs, args)
    else:
        if args.workers > world_size:
            args.workers = world_size
        import torch.multiprocessing as mp
        mp.spawn(convert, nprocs=args.workers, args=(world_size, configs, args))

    tok = time.time()
    t = time.strftime("%H:%M:%S", time.gmtime(tok - tik))
    print(f"Total time of converting checkpoints: {t}")


if __name__ == "__main__":
    main()
