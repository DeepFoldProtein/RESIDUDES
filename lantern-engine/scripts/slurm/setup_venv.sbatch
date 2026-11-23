#!/bin/bash
#SBATCH -J fmx-setup-venv
#SBATCH -c 8
#SBATCH -t 03:00:00
# Build the runtime venv at core/.venv on a compute node (needs network to
# PyPI). Run once per checkout; the site file's `venv` points at it.
#
#   sbatch --export=ALL,LANTERN_ENGINE_ROOT=/path/to/lantern-engine \
#       -o runs/_setup/venv.%j.out -e runs/_setup/venv.%j.err \
#       scripts/slurm/setup_venv.sbatch [cu13|cu12]
#
# or simply:  lantern-engine setup venv --slurm
#
# cu13: Blackwell (B200, RTX PRO 6000) and Ada (RTX 4090) via torch cu128 wheels;
#       also installs the in-venv JAX stack used by --backend jax.
# cu12: Hopper/Ampere hosts without a CUDA 13 driver (TRT and torch backends only).
set -euo pipefail
: "${LANTERN_ENGINE_ROOT:?export LANTERN_ENGINE_ROOT (sbatch copies this script, so it cannot locate itself)}"
CUDA=${1:-cu13}
EXTRAS=(--extra "$CUDA" --extra boltz --extra openfold3)
[ "$CUDA" = "cu13" ] && EXTRAS+=(--extra jax-cu13)
export PATH="$HOME/.local/bin:$PATH"
export DS_BUILD_OPS=0                       # never JIT-build deepspeed ops during install
# uv's cache must live on a filesystem with room (a quota-limited cache dir fails
# mid-build). Keep it next to the venv so uv can hardlink wheels.
export UV_CACHE_DIR="${LANTERN_ENGINE_UV_CACHE_DIR:-$LANTERN_ENGINE_ROOT/runs/_setup/uv-cache}"
mkdir -p "$UV_CACHE_DIR"
unset VIRTUAL_ENV                           # an activated unrelated venv confuses uv
cd "$LANTERN_ENGINE_ROOT/core"
echo "▶ host $(hostname)  uv $(uv --version 2>/dev/null || echo MISSING)  extras: ${EXTRAS[*]}"
echo "▶ project $PWD   uv cache $UV_CACHE_DIR"
date
if uv sync --frozen "${EXTRAS[@]}"; then
    echo "▶ uv sync --frozen (lockfile honoured) OK"
else
    echo "▶ frozen sync failed — re-resolving against pyproject.toml"
    uv sync "${EXTRAS[@]}"
fi
# The lock's jax extra resolves to a jax version without a CUDA PJRT plugin, so JAX
# would silently run on CPU. Add the cuda12 plugin at the locked jax version.
JAXV=$(.venv/bin/python -c 'import jax; print(jax.__version__)' 2>/dev/null || true)
if [ -n "$JAXV" ] && ! .venv/bin/python -c 'import jax_cuda12_plugin' 2>/dev/null; then
    echo "▶ jax $JAXV has no CUDA plugin — uv pip install jax[cuda12]==$JAXV"
    uv pip install --python .venv/bin/python "jax[cuda12]==$JAXV"
fi
date
.venv/bin/python - <<'PY'
import sys, numpy, torch, tensorrt as trt
print("▶ python", sys.version.split()[0], "| torch", torch.__version__, "| tensorrt", trt.__version__, "| numpy", numpy.__version__)
import boltz, openfold3, cuequivariance_ops_torch  # noqa: F401
print("▶ boltz / openfold3 / cuequivariance_ops_torch import OK")
try:
    import jax, importlib.util as u
    plug = "cuda12 plugin" if u.find_spec("jax_cuda12_plugin") else "NO CUDA PLUGIN"
    print("▶ jax", jax.__version__, "with", plug)
except ImportError:
    print("▶ jax not installed (cu12 flavour): --backend jax unavailable in this venv")
PY
du -sh .venv "$UV_CACHE_DIR"
echo "▶ done. Next: bin/lantern-engine doctor"
