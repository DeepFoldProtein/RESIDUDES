#!/usr/bin/env bash
# Set up Bio-TRT/.venv-jax — a parallel uv-managed environment with
# JAX >= 0.9.2 + tokamax + numpy >= 2.
#
# This venv lives alongside the main .venv (TRT path) — they are entirely
# independent. Use this one when running JAX inference (`protenix/inference_jax.py`)
# and the main .venv for TRT-engine inference.
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"
JAX_VENV="$ROOT/.venv-jax"
PYPROJECT="$ROOT/pyproject-jax.toml"

if [[ ! -f "$PYPROJECT" ]]; then
    echo "error: $PYPROJECT not found" >&2
    exit 1
fi

echo "[setup_jax_venv] root=$ROOT"
echo "[setup_jax_venv] target venv=$JAX_VENV"
echo "[setup_jax_venv] using pyproject=$PYPROJECT"

uv venv "$JAX_VENV" --python 3.11

# Install all deps from pyproject-jax.toml into the JAX venv.
# We use `uv pip install` (not `uv sync`) because uv sync would expect
# the file to be named pyproject.toml. Reading the [project].dependencies
# block manually keeps both pyproject files independent.
PYBIN="$JAX_VENV/bin/python"

# Extract deps from pyproject-jax.toml [project] table.
"$PYBIN" - <<'PY' > /tmp/jax_deps.txt
import tomllib, pathlib
data = tomllib.loads(pathlib.Path("pyproject-jax.toml").read_text())
for d in data["project"]["dependencies"]:
    print(d)
PY

echo "[setup_jax_venv] resolved deps:"
sed 's/^/    /' /tmp/jax_deps.txt

# Install with PyTorch CUDA 12.8 wheel index.
uv pip install \
    --python "$PYBIN" \
    --index-strategy=unsafe-best-match \
    --extra-index-url https://download.pytorch.org/whl/cu128 \
    -r /tmp/jax_deps.txt

# Sanity check.
"$PYBIN" - <<'PY'
import jax, numpy
print("jax:", jax.__version__, "devices:", jax.devices())
print("numpy:", numpy.__version__)
try:
    import tokamax
    print("tokamax:", tokamax.__version__)
except Exception as e:
    print("tokamax: FAILED ->", e)
import torch
print("torch:", torch.__version__, "cuda:", torch.cuda.is_available())
PY

echo
echo "[setup_jax_venv] OK"
echo "  Activate:  source $JAX_VENV/bin/activate"
echo "  Or invoke: $JAX_VENV/bin/python ..."
