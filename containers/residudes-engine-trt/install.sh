#!/bin/sh
# The prediction runtime: Protenix v2, OpenFold 3 and Boltz-2 on PyTorch or TensorRT.
#
# Everything comes from core/uv.lock, the same file `residudes-engine setup venv` uses, so the
# image and a site venv are the same environment. It already pins tensorrt 10.16 and the
# nvidia-* wheels, so the CUDA runtime base only has to supply the driver-adjacent pieces.
#
# Not in here, by necessity: the model weights, and the TensorRT engines, which are built on the
# card they run on (`residudes-engine engines build`) and belong to that machine, not to an image.
set -eu

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends \
    ca-certificates curl git python3.12 python3.12-venv libpython3.12 \
    libgomp1 libstdc++6
rm -rf /var/lib/apt/lists/*

# No wheel cache: for the CUDA stacks it is 8-10 GB sitting on disk beside the venv it
# produced, which is what exhausts a CI runner during the build rather than the image size.
export UV_NO_CACHE=1

cd /opt/core
UV_PROJECT_ENVIRONMENT=/opt/venv uv sync --frozen --no-install-project --python /usr/bin/python3.12
rm -rf /root/.cache/uv

/opt/venv/bin/python - <<'EOF'
import torch
print("torch", torch.__version__, "built for cuda", torch.version.cuda)
import tensorrt
print("tensorrt", tensorrt.__version__)
import cuequivariance_torch                      # the fused kernels the engines were built with
print("cuequivariance-torch ok")
EOF
