#!/bin/sh
# ESMFold2's environment. The model is EvolutionaryScale's `esm` 3.3.0, which pins python
# >=3.12,<3.13 and pulls torch and a Biohub fork of transformers — which is why it cannot share
# the engine's venv and why `residudes-engine predict --model ef2` calls a separate interpreter.
#
# The CASP17 runner scripts (build_input.py, run_inference.py, write_unified.py) live in a private
# fork and are bind-mounted at /opt/esm; everything they import is installed here from public
# sources. On the cluster this env is a conda prefix, only because the local pyenv 3.12 was built
# without ssl; a container's python has it, so plain pip is enough.
set -eu

# The exact upstream commit the CASP17 fork sits on: esm's pyproject says 3.3.0 but no v3.3.0 tag
# was published, so this is pinned by sha (confirmed present in evolutionaryscale/esm).
ESM_REF=${ESM_REF:-975722d72d20104261b33e2e9cce98f30c031037}

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends \
    ca-certificates curl git python3.12 python3.12-venv python3-pip libpython3.12
rm -rf /var/lib/apt/lists/*

python3.12 -m venv /opt/venv
/opt/venv/bin/python -m pip install --no-cache-dir -U pip wheel setuptools

# esm's pyproject asks for `transformers @ git+https://github.com/Biohub/transformers.git@main`,
# and that repository is gone (404). The commit the CASP17 environment actually installed,
# f9a5a374, carries src/transformers/models/esmfold2/ and is reachable in
# evolutionaryscale/transformers, so it is pinned by sha here - a moving `main` would not have
# been reproducible anyway. Installed first so pip keeps it instead of resolving esm's dead URL.
TRANSFORMERS_REF=${TRANSFORMERS_REF:-f9a5a374be135f63b3019c1cefb91ea9e2d27e10}
/opt/venv/bin/python -m pip install --no-cache-dir \
    "transformers @ git+https://github.com/evolutionaryscale/transformers.git@${TRANSFORMERS_REF}"

# esm's own dependency list, minus the two handled above, plus gemmi for the unified writer.
# torch is pinned to the version the CASP17 environment ran (esm only asks for >=2.2.0, which
# would otherwise drift to whatever is newest the day the image is built).
/opt/venv/bin/python -m pip install --no-cache-dir --no-deps \
    "esm @ git+https://github.com/evolutionaryscale/esm.git@${ESM_REF}"
/opt/venv/bin/python -m pip install --no-cache-dir \
    "torch==${TORCH_VERSION:-2.12.0}" "ipython" "einops" "biotite>=1.0.0" "rdkit" "msgpack-numpy" "biopython" "scikit-learn" "brotli" "attrs" "pandas" "cloudpathlib" "httpx" "tenacity" "zstd" "ipywidgets" "py3dmol" "pydssp" "boto3" "pygtrie" "dna_features_viewer" "accelerate" gemmi

/opt/venv/bin/python -m pip check
/opt/venv/bin/python - <<'EOF'
import torch, esm, gemmi
print("python ok | torch", torch.__version__, "| esm", esm.__version__ if hasattr(esm, "__version__") else "3.3.0",
      "| gemmi", gemmi.__version__)
from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model     # the Biohub fork
print("ESMFold2Model importable:", ESMFold2Model.__name__)
EOF
