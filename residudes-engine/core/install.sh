#!/bin/sh
# AlphaFold 3 and the JAX backends, from core/jaxenv/uv.lock - the same file
# `residudes-engine setup venv` builds the site's JAX environment from.
#
# jax[cuda13] brings its own CUDA libraries as wheels, so the base only supplies the system
# pieces around them. AlphaFold 3's parameters are licensed per user and are never in an image:
# they are bind-mounted and pointed at with [site] af3_weights.
set -eu

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends \
    ca-certificates curl git python3.12 python3.12-venv libpython3.12 \
    libgomp1 libstdc++6 zstd
rm -rf /var/lib/apt/lists/*

# jaxenv's lock has two path dependencies, so the lockfile alone is not enough:
#   alphafold3   ../models/alphafold3   non-editable, scikit-build-core - a C++ extension gets
#                                       compiled here, which is work the image exists to save
#   openfold3    ../models/openfold-3   editable, so it resolves to whatever is mounted
# Both source trees are copied in at the paths the lock names; `uv sync` fails otherwise with
# "Distribution not found at: file:///opt/models/alphafold3".
apt-get update
# python3.12-dev because alphafold3's CMakeLists does
#   find_package(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)
# and the Development component is the headers, which python3.12-venv does not carry; zlib for
# libcifpp. The five FetchContent repositories (abseil, pybind11, pybind11_abseil, libcifpp,
# dssp) are cloned during the configure step, so this build needs network.
apt-get install -y --no-install-recommends \
    build-essential cmake ninja-build python3.12-dev zlib1g-dev
rm -rf /var/lib/apt/lists/*

# No wheel cache: for the CUDA stacks it is 8-10 GB sitting on disk beside the venv it
# produced, which is what exhausts a CI runner during the build rather than the image size.
export UV_NO_CACHE=1

cd /opt/jaxenv
UV_PROJECT_ENVIRONMENT=/opt/venv uv sync --frozen --no-install-project --python /usr/bin/python3.12
rm -rf /root/.cache/uv

/opt/venv/bin/python - <<'EOF'
import jax
print("jax", jax.__version__)
import haiku
print("dm-haiku ok")
import jax_triton                                  # tokamax calls triton_call(out_type=...)
print("jax-triton ok")
import alphafold3
from alphafold3.cpp import cif_dict                 # the compiled extension, not just the python
print("alphafold3 + compiled cpp ok")
EOF
