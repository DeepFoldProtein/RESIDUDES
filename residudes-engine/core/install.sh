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

# The venv is 9 GB, and one `uv sync` makes one layer that size - which does not survive every
# network: measured, a 9 GB blob dies partway with PROTOCOL_ERROR or unexpected EOF while the 3 GB
# images pull cleanly on the same link. So the sync is done in passes, each its own RUN in the
# Dockerfile, holding the heaviest packages back until last. Every pass is `uv sync --frozen`, so
# the project's indexes are used (torch comes from the PyTorch index, as a +cu128 local version)
# and each pass is verified against the lockfile rather than hand-pinned.
if [ "$stage" = all ] || [ "$stage" = sync ]; then
    cd /opt/core                              # only this stage needs the lockfile beside it
    UV_PROJECT_ENVIRONMENT=/opt/venv uv sync --frozen --no-install-project --python /usr/bin/python3.12
    rm -rf /root/.cache/uv
fi

# The venv is ~9 GB. As one layer that does not survive every network: measured, a 9 GB blob dies
# partway with PROTOCOL_ERROR or unexpected EOF while the 3 GB images pull cleanly on the same
# link. Splitting the *install* does not help - uv rebuilds the environment when an earlier pass
# left it incomplete - so the split happens after it: site-packages is bin-packed into buckets of
# similar size, and the Dockerfile copies one bucket per layer. The environment is still exactly
# what `uv sync --frozen` produced.
if [ "$stage" = split ]; then
    /opt/venv/bin/python - "${2:-4}" <<'EOF'
import os, shutil, sys
from pathlib import Path
site = next(Path("/opt/venv/lib").glob("python3.*/site-packages"))
entries = sorted(site.iterdir())
def weight(p):
    if p.is_file():
        return p.stat().st_size
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
sized = sorted(((weight(e), e) for e in entries), reverse=True)
bins = [[0, []] for _ in range(int(sys.argv[1]))]
for size, path in sized:                      # heaviest first into the lightest bucket
    b = min(bins, key=lambda x: x[0])
    b[0] += size
    b[1].append(path)
for i, (total, paths) in enumerate(bins, 1):
    out = Path(f"/split/{i}{site}")
    out.mkdir(parents=True, exist_ok=True)
    for path in paths:
        shutil.move(str(path), str(out / path.name))
    print(f"bucket {i}: {total / 1e9:.1f} GB, {len(paths)} entries", file=sys.stderr)
rest = Path("/split/rest")                    # bin/, pyvenv.cfg and the empty site-packages tree
rest.mkdir(parents=True, exist_ok=True)
shutil.move("/opt/venv", str(rest / "venv"))
EOF
    exit 0
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
