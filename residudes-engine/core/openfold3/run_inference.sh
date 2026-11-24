#!/bin/bash
# Run OpenFold3 inference with Bio-TRT TRT acceleration.
#
# This script handles the Python environment setup (PYTHONPATH for
# both the OF3 repo and Bio-TRT thirdparty) then calls inference.py.
#
# Step 1 (one-time): Build TRT engines
#   bash openfold3/trt/build_engines.sh \
#       --checkpoint /home/jovyan/.openfold3/of3-p2-155k.pt \
#       --engines-dir /workspace/engines/openfold3
#
# Step 2: Run inference
#   bash openfold3/run_inference.sh \
#       --query  examples/query_ubiquitin.json \
#       --checkpoint /home/jovyan/.openfold3/of3-p2-155k.pt \
#       --engines-dir /workspace/engines/openfold3 \
#       --output-dir results/ubiquitin_trt
#
# All extra flags are forwarded to inference.py:
#   --no-trt            Pure PyTorch baseline (no engines needed)
#   --no-msa            Offline mode (skip ColabFold MSA)
#   --pairformer-only   Only accelerate pairformer
#   --num-diffusion-samples N

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"

export PYTHONPATH="${REPO_ROOT}/thirdparty:${REPO_ROOT}:${PYTHONPATH:-}"
export LAYERNORM_TYPE="${LAYERNORM_TYPE:-torch_layernorm}"
# Disable deepspeed JIT custom ops (evoformer_attn not compatible with CUDA 13 JIT)
export DS_BUILD_OPS="${DS_BUILD_OPS:-0}"

# LD_LIBRARY_PATH: pip-installed native libs (libcue_ops, libcublas, libnvinfer)
# live in non-standard site-packages paths that the linker can't find.
_VENV_SITE="$REPO_ROOT/.venv/lib/python3.11/site-packages"
if [[ -d "$_VENV_SITE" ]]; then
    _LD_EXTRA="$_VENV_SITE/cuequivariance_ops/lib:$_VENV_SITE/nvidia/cu13/lib:$_VENV_SITE/tensorrt_libs"
else
    _LD_EXTRA=""
fi

# LD_PRELOAD: C++ IdentitySZ plugin needs libstdc++ with GLIBCXX_3.4.32.
_STDCXX_DIR=""
for _STDCXX in \
    "$REPO_ROOT/thirdparty/tensorrt_bionemo/libs/libstdc++.so.6" \
    "/opt/conda/lib/libstdc++.so.6" \
    "/usr/lib/gcc/x86_64-linux-gnu/13/libstdc++.so" \
    "/usr/lib/gcc/x86_64-linux-gnu/12/libstdc++.so"; do
    if [[ -f "$_STDCXX" ]] && strings "$_STDCXX" 2>/dev/null | grep "GLIBCXX_3.4.32" >/dev/null 2>&1; then
        export LD_PRELOAD="$_STDCXX"
        _STDCXX_DIR="$(dirname "$_STDCXX")"
        break
    fi
done
export LD_LIBRARY_PATH="${_LD_EXTRA}${_STDCXX_DIR:+:$_STDCXX_DIR}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

exec "$PYTHON" "$SCRIPT_DIR/inference.py" "$@"

