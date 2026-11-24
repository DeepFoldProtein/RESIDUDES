#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""AlphaFold 3 on the PyTorch and TensorRT backends.

Same command line as core/models/alphafold3/run_alphafold.py (the JAX entry
point) plus `--engines_dir`; the featurisation, seed loop and unified writer are
the fork's, only the forward pass runs the PyTorch port in core/alphafold3/af3torch,
with the trunk pairformer and the diffusion transformer on TensorRT engines when
`--engines_dir` points at a set built by core/alphafold3/trt/build_engines.sh.

    python core/alphafold3/inference.py --json_path input.json --model_dir <dir with af3.bin> \
        --output_dir <out> --run_data_pipeline=false --run_inference=true ...
"""
from __future__ import annotations

import os
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
FORK = HERE.parent / "models" / "alphafold3"
sys.path.insert(0, str(FORK))
sys.path.insert(0, str(HERE))

# JAX is still imported by the fork's main (device check); keep it off the card.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_CLIENT_MEM_FRACTION", "0.02")
os.environ.pop("XLA_PYTHON_CLIENT_MEM_FRACTION", None)     # deprecated name; JAX 0.11 rejects it next to the new one

import run_alphafold as ra  # noqa: E402
from absl import app, flags  # noqa: E402
from af3torch.runner import TorchModelRunner  # noqa: E402

ra.ModelRunner = TorchModelRunner
_ENGINES_DIR = flags.DEFINE_string("engines_dir", None, "TensorRT engines for the trunk pairformer and the diffusion transformer "
                                   "(pairformer_engine/, token_transformer_engine/); PyTorch for everything when unset.")


def main(argv):
    if _ENGINES_DIR.value:
        os.environ["RESIDUDES_ENGINE_AF3_ENGINES_DIR"] = _ENGINES_DIR.value
    return ra.main(argv)


if __name__ == "__main__":
    flags.mark_flags_as_required(["output_dir"])
    app.run(main)
