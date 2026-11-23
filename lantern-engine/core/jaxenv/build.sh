#!/usr/bin/env bash
# Build core/jaxenv/.venv with uv (Python 3.12). Needs network and a C++20 compiler
# (AlphaFold 3's extension; on RHEL 9 `export CC=/opt/rh/gcc-toolset-14/root/usr/bin/gcc CXX=.../g++`). ~15 min.
#   bash core/jaxenv/build.sh            (locally)
#   sbatch -c 8 -t 03:00:00 --wrap "bash core/jaxenv/build.sh"   (on a cluster)
set -euo pipefail
cd "$(dirname "$0")"
export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR="${LANTERN_ENGINE_UV_CACHE_DIR:-$PWD/../../runs/_setup/uv-cache}"
unset VIRTUAL_ENV
export DS_BUILD_OPS=0
uv python install 3.12 >/dev/null 2>&1 || true
uv sync --python 3.12 --no-progress
# AlphaFold 3 reads two CCD-derived pickles at import time; build them once from
# the components.cif that the libcifpp build installed next to the package.
if ! ls .venv/lib/python3.*/site-packages/alphafold3/constants/converters/ccd.pickle >/dev/null 2>&1; then
    echo "--- build_data (AlphaFold 3 CCD tables) ---"
    .venv/bin/build_data
fi
echo "--- smoke test ---"
.venv/bin/python - <<'PY'
import importlib, numpy, scipy
print("numpy", numpy.__version__, "scipy", scipy.__version__)
import jax; print("jax", jax.__version__, jax.devices() if False else "(devices checked on GPU node)")
import torch; print("torch", torch.__version__)
for m in ("tokamax", "cuequivariance_jax", "cuequivariance_torch", "haiku", "boltz", "openfold3", "alphafold3", "alphafold3.cpp"):
    try:
        mod = importlib.import_module(m); print(m, getattr(mod, "__version__", "ok"))
    except Exception as e:
        print(m, "FAILED:", type(e).__name__, str(e)[:120])
PY
echo "--- done: $(du -sh .venv | cut -f1) ---"
