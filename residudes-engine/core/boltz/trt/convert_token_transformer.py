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
Convert Boltz2 token transformer checkpoint to safetensors for TRT engine building.

Usage:
    python convert_token_transformer.py \\
        --local_checkpoint /path/to/boltz2.ckpt \\
        --output_dir /workspace/token_transformer_safetensors \\
        --multiplicity 5
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
from tensorrt_bionemo.models.boltz2 import Boltz2Config
from tensorrt_bionemo.models.boltz2.convert import convert_hf_diffusion_transformer


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Convert Boltz2 token transformer checkpoint to TRT-BNM safetensors"
    )
    parser.add_argument("--tp_size", type=int, default=1)
    parser.add_argument("--dcp_size", type=int, default=1)
    parser.add_argument(
        "--multiplicity",
        type=int,
        default=5,
        help="Diffusion multiplicity (number of samples per forward pass)",
    )
    parser.add_argument(
        "--dtype", type=str, default="float32", choices=["bfloat16", "float32"]
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default="token_transformer_checkpoint",
    )
    parser.add_argument(
        "--pairwise_attn_backend",
        type=str,
        default="VANILLA",
        choices=["VANILLA"],
    )
    parser.add_argument("--local_checkpoint", type=Path, default=None)
    parser.add_argument("--disable_custom_all_reduce", type=bool, default=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--backend", type=str, default="trt", choices=["trt"])

    args = parser.parse_args()

    if args.local_checkpoint is None:
        env_ckpt = os.environ.get("BOLTZ2_CKPT")
        if env_ckpt:
            args.local_checkpoint = Path(env_ckpt)
        else:
            parser.error(
                "--local_checkpoint is required (or set BOLTZ2_CKPT env var)"
            )
    return args


def convert(worker_rank, world_size, tt_config, args):
    if args.backend == BackendType.TRT:
        out = args.output_dir / f"{BackendType.TRT}"
        out.mkdir(parents=True, exist_ok=True)
        with (out / "config.json").open("w") as f:
            json.dump(tt_config.model_dump(), f, indent=4)

    for rank in range(worker_rank, world_size, args.workers):
        mapping = Mapping(
            world_size=world_size,
            tp_size=args.tp_size,
            dcp_size=args.dcp_size,
            rank=rank,
        )
        weights = convert_hf_diffusion_transformer(
            tt_config,
            mapping,
            local_checkpoint=str(args.local_checkpoint),
            model_name="boltz-2",
        )
        safetensors.torch.save_file(
            weights,
            args.output_dir / f"{BackendType.TRT}/rank{rank}.safetensors",
        )


def main():
    args = parse_arguments()
    assert args.dcp_size == 1, "DCP not supported for token transformer"
    world_size = args.tp_size * args.dcp_size

    args.output_dir.mkdir(exist_ok=True, parents=True)

    tik = time.time()
    boltz2_config = Boltz2Config()
    tt_config = boltz2_config.structure_module.score_model.token_transformer

    config_update = {
        "multiplicity": args.multiplicity,
        "backend": "trt",
        "dtype": args.dtype,
        "architecture": "token_transformer",
        "mapping": {
            "world_size": world_size,
            "tp_size": args.tp_size,
            "dcp_size": 1,
        },
        "disable_custom_all_reduce": args.disable_custom_all_reduce,
        "pairwise_attn_backend": args.pairwise_attn_backend,
    }
    tt_config = tt_config.model_copy(update=config_update)

    if args.workers == 1:
        convert(0, world_size, tt_config, args)
    else:
        if args.workers > world_size:
            args.workers = world_size
        import torch.multiprocessing as mp
        mp.spawn(convert, nprocs=args.workers, args=(world_size, tt_config, args))

    tok = time.time()
    t = time.strftime("%H:%M:%S", time.gmtime(tok - tik))
    print(f"Total time of converting checkpoints: {t}")


if __name__ == "__main__":
    main()
