#!/bin/sh
# residudes-qa with its CUDA stack: the same lockfile as the CPU image plus the cu130 extra.
#
# NVIDIA's CUDA runtime image is the base, and the cu130 extra now pins the nvidia-* wheels torch
# loads at import (libnvrtc, libcupti and the rest). Before that pinning, a clean install of the
# extra died on `import torch` outside this cluster, where /usr/local/cuda-13.2 happens to exist -
# the container is how that came to light.
set -eu

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends \
    ca-certificates curl python3.12 python3.12-venv libpython3.12
rm -rf /var/lib/apt/lists/*

# No wheel cache: for the CUDA stacks it is 8-10 GB sitting on disk beside the venv it
# produced, which is what exhausts a CI runner during the build rather than the image size.
export UV_NO_CACHE=1

cd /opt/qa
# dev group too (pytest), and the cu130 extra: torch 2.11 with the CUDA 13 wheels, the exact
# build CASP17 scored with (DFIRE sums differ at 1e-5 against another one)
UV_PROJECT_ENVIRONMENT=/opt/venv uv sync --frozen --no-install-project --extra cu130 \
    --python /usr/bin/python3.12
rm -rf /root/.cache/uv

/opt/venv/bin/python - <<'EOF'
import importlib.util
mods = ["numpy", "pandas", "scipy", "sklearn", "xgboost", "gemmi", "matplotlib", "seaborn",
        "umap", "numba", "networkx", "igraph", "leidenalg", "tqdm", "Bio", "ot", "click", "pyarrow"]
missing = [m for m in mods if not importlib.util.find_spec(m)]
print("dependency set:", len(mods) - len(missing), "of", len(mods), "importable")
assert not missing, f"missing: {missing}"
import torch
print("torch", torch.__version__, "built for cuda", torch.version.cuda)
assert torch.__version__.startswith("2.11"), f"wrong torch: {torch.__version__}"
EOF
