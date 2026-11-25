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
# No wheel cache: for the CUDA stacks it is 8-10 GB sitting on disk beside the venv it
# produced, which is what exhausts a CI runner during the build rather than the image size.
export UV_NO_CACHE=1
stage=${1:-all}

if [ "$stage" = all ] || [ "$stage" = system ]; then
    apt-get update
    apt-get install -y --no-install-recommends \
        ca-certificates curl git python3.12 python3.12-venv libpython3.12 \
        libgomp1 libstdc++6
    rm -rf /var/lib/apt/lists/*
fi

cd /opt/core

# The venv is 9 GB, and a single `uv sync` makes it a single layer that size - which does not
# survive every network: measured, a 9 GB blob dies partway with PROTOCOL_ERROR or unexpected EOF
# while the 3 GB images pull cleanly on the same link. So the heaviest wheels go in first, one
# group per call, and each call is its own RUN in the Dockerfile. Versions are read from the
# lockfile, and the closing `uv sync --frozen` is what guarantees the result still matches it.

heavy() {        # the pinned version of a package, from uv.lock
    /usr/bin/python3.12 - "$1" <<'EOF'
import sys, tomllib
name = sys.argv[1]
lock = tomllib.load(open("/opt/core/uv.lock", "rb"))
for p in lock.get("package", []):
    if p["name"] == name:
        print(f"{name}=={p['version']}")
        break
EOF
}

nvidia_pins() {  # every nvidia-* package the lock pins, as install arguments
    /usr/bin/python3.12 - <<'EOF'
import tomllib
lock = tomllib.load(open("/opt/core/uv.lock", "rb"))
print(" ".join(f"{p['name']}=={p['version']}" for p in lock.get("package", [])
                if p["name"].startswith(("nvidia-", "tensorrt"))))
EOF
}

if [ "$stage" = all ] || [ "$stage" = venv ]; then
    uv venv /opt/venv --python /usr/bin/python3.12
fi
if [ "$stage" = all ] || [ "$stage" = torch ]; then
    VIRTUAL_ENV=/opt/venv uv pip install --no-deps "$(heavy torch)"
fi
if [ "$stage" = all ] || [ "$stage" = cuda ]; then
    # shellcheck disable=SC2046
    VIRTUAL_ENV=/opt/venv uv pip install --no-deps $(nvidia_pins)
fi
if [ "$stage" = all ] || [ "$stage" = rest ]; then
    UV_PROJECT_ENVIRONMENT=/opt/venv uv sync --frozen --no-install-project --python /usr/bin/python3.12
    rm -rf /root/.cache/uv
fi
[ "$stage" = all ] || [ "$stage" = check ] || exit 0

/opt/venv/bin/python - <<'EOF'
import torch
print("torch", torch.__version__, "built for cuda", torch.version.cuda)
import tensorrt
print("tensorrt", tensorrt.__version__)
import cuequivariance_torch                      # the fused kernels the engines were built with
print("cuequivariance-torch ok")
EOF
