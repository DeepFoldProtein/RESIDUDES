#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Convert the AlphaFold 3 trunk pairformer (48 blocks) to the safetensors the
TensorRT engine builder consumes.

AlphaFold 3's parameters are first written in OpenFold 3 names
(af3_to_openfold3.py); the OpenFold 3 converter then packs them. Two settings
differ from OpenFold 3: LayerNorm epsilon 1e-5 and the ending-node attention
bias taken before the transpose.

    python convert_pairformer.py --openfold3-state af3_as_openfold3.pt --output_dir DIR
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

import openfold3.trt.convert_pairformer as of3  # noqa: E402  (patches the checkpoint loader to accept a plain .pt)
import safetensors.torch  # noqa: E402
from tensorrt_bionemo.configs import BackendType  # noqa: E402
from tensorrt_bionemo.mapping import Mapping  # noqa: E402
from tensorrt_bionemo.models.openfold3 import OpenFold3Config  # noqa: E402
from tensorrt_bionemo.models.openfold3.convert import convert_hf_pairformer  # noqa: E402

NORM_EPSILON = 1e-5


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--openfold3-state", type=pathlib.Path, required=True, help="output of af3_to_openfold3.py")
    ap.add_argument("--output_dir", type=pathlib.Path, required=True)
    ap.add_argument("--triangle_attention_backend", default="VANILLA", choices=["VANILLA", "TRIFAST", "CUEQUIV"])
    ap.add_argument("--triangle_attn_node_chunk_size", type=int, default=32)
    ap.add_argument("--dtype", default="float32", choices=["bfloat16", "float32"])
    args = ap.parse_args()
    t0 = time.time()
    pf = OpenFold3Config().trunk.pairformer
    config = pf.model_copy(update={
        "max_batch_size": 1, "triangle_attn_node_chunk_size": args.triangle_attn_node_chunk_size,
        "triangle_attention_backend": args.triangle_attention_backend, "triangle_attn_cueq_fallback_threshold": 0,
        "max_transition_tp_size": True, "max_attention_pairwise_tp_size": True, "max_tri_mul_tp_size": True,
        "no_update_s": False, "no_update_z": False, "backend": "trt",
        "token_s": pf.token_s, "token_z": pf.token_z, "pairwise_head_width": pf.pairwise_head_width,
        "pairwise_num_heads": pf.pairwise_num_heads, "num_blocks": pf.num_blocks, "num_heads": pf.num_heads,
        "dtype": args.dtype, "norm_epsilon": NORM_EPSILON, "architecture": "pairformer",
        "mapping": {"world_size": 1, "tp_size": 1, "dcp_size": 1}, "disable_custom_all_reduce": True,
        "post_layer_norm": pf.post_layer_norm, "trimul_high_precision": False, "version": "v1", "support_batch": False,
        "tri_attn_end_bias_before_transpose": True,       # AlphaFold 3 convention (see tensorrt_bionemo configs)
    })
    out = args.output_dir / BackendType.TRT
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(config.model_dump(), indent=4))
    weights = convert_hf_pairformer(config, Mapping(world_size=1, tp_size=1, dcp_size=1, rank=0),
                                    local_checkpoint=str(args.openfold3_state), model_name="openfold3")
    safetensors.torch.save_file(weights, out / "rank0.safetensors")
    print(f"pairformer: {len(weights)} tensors → {out} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
