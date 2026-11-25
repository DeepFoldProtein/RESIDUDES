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

# The venv is 9 GB, and one `uv sync` makes one layer that size - which does not survive every
# network: measured, a 9 GB blob dies partway with PROTOCOL_ERROR or unexpected EOF while the 3 GB
# images pull cleanly on the same link. So the sync is done in passes, each its own RUN in the
# Dockerfile, holding the heaviest packages back until last. Every pass is `uv sync --frozen`, so
# the project's indexes are used (torch comes from the PyTorch index, as a +cu128 local version)
# and each pass is verified against the lockfile rather than hand-pinned.
heavy_args() {      # --no-install-package for each package whose wheel is over the given size
    /usr/bin/python3.12 - "$1" <<'EOF'
import sys, tomllib
limit = float(sys.argv[1]) * 1e9
lock = tomllib.load(open("/opt/core/uv.lock", "rb"))
names = [p["name"] for p in lock.get("package", [])
         if max((w.get("size", 0) for w in p.get("wheels", [])), default=0) > limit]
print(" ".join(f"--no-install-package {n}" for n in sorted(names)))
EOF
}

sync() {            # one pass, excluding what the caller holds back
    # shellcheck disable=SC2086
    UV_PROJECT_ENVIRONMENT=/opt/venv uv sync --frozen --no-install-project \
        --python /usr/bin/python3.12 $1
}

if [ "$stage" = all ] || [ "$stage" = base ]; then
    sync "$(heavy_args 0.2) --no-install-package torch"       # everything small
fi
if [ "$stage" = all ] || [ "$stage" = cuda ]; then
    sync "--no-install-package torch"                         # the CUDA wheels
fi
if [ "$stage" = all ] || [ "$stage" = torch ]; then
    sync ""                                                   # torch last, and the lock is met
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
