#!/bin/sh
# The QA stack's dependencies, straight from its lockfile: `uv sync` with the project itself left
# out, because the checkout is bind-mounted at run time rather than baked in. torch is an opt-in
# extra of that package (cu130 / cu124), so a plain sync gives the CPU set and no CUDA.
set -eu

apt-get update
apt-get install -y --no-install-recommends ca-certificates
rm -rf /var/lib/apt/lists/*

cd /opt/qa
# dev group too: it is pytest, and an image that can run its own suite is worth 2 MB
UV_PROJECT_ENVIRONMENT=/opt/venv uv sync --frozen --no-install-project
rm -rf /root/.cache/uv

/opt/venv/bin/python - <<'PY'
import importlib.util
mods = ["numpy", "pandas", "scipy", "sklearn", "xgboost", "gemmi", "matplotlib", "seaborn",
        "umap", "numba", "networkx", "igraph", "leidenalg", "tqdm", "Bio", "ot", "click", "pyarrow"]
missing = [m for m in mods if not importlib.util.find_spec(m)]
print("dependency set:", len(mods) - len(missing), "of", len(mods), "importable")
assert not missing, f"missing: {missing}"
assert importlib.util.find_spec("torch") is None, "torch must stay out of the CPU image"
PY
