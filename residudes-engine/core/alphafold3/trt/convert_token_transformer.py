#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Convert the AlphaFold 3 diffusion token transformer (24 blocks) to the
safetensors the TensorRT engine builder consumes (see convert_pairformer.py).

    python convert_token_transformer.py --openfold3-state af3_as_openfold3.pt --output_dir DIR --multiplicity 5
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
CORE = HERE.parents[1]
sys.path.insert(0, str(CORE / "thirdparty"))
sys.path.insert(0, str(CORE))

import openfold3.trt.convert_pairformer as of3  # noqa: E402,F401  (patches the checkpoint loader)
import safetensors.torch  # noqa: E402
from tensorrt_bionemo.configs import BackendType  # noqa: E402
from tensorrt_bionemo.mapping import Mapping  # noqa: E402
from tensorrt_bionemo.models.openfold3 import OpenFold3Config  # noqa: E402
from tensorrt_bionemo.models.openfold3.convert import convert_hf_diffusion_transformer  # noqa: E402

NORM_EPSILON = 1e-5


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--openfold3-state", type=pathlib.Path, required=True)
    ap.add_argument("--output_dir", type=pathlib.Path, required=True)
    ap.add_argument("--multiplicity", type=int, default=5, help="diffusion samples per engine call")
    ap.add_argument("--dtype", default="float32", choices=["bfloat16", "float32"])
    args = ap.parse_args()
    t0 = time.time()
    tt = OpenFold3Config().structure_module.score_model.token_transformer
    config = tt.model_copy(update={
        "multiplicity": args.multiplicity, "backend": "trt", "dtype": args.dtype, "architecture": "token_transformer",
        "mapping": {"world_size": 1, "tp_size": 1, "dcp_size": 1}, "disable_custom_all_reduce": True,
        "pairwise_attn_backend": "VANILLA", "version": "v1", "norm_epsilon": NORM_EPSILON,
    })
    out = args.output_dir / BackendType.TRT
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(config.model_dump(), indent=4))
    weights = convert_hf_diffusion_transformer(config, Mapping(world_size=1, tp_size=1, dcp_size=1, rank=0),
                                               local_checkpoint=str(args.openfold3_state), model_name="openfold3")
    safetensors.torch.save_file(weights, out / "rank0.safetensors")
    print(f"token transformer: {len(weights)} tensors → {out} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
