#!/bin/sh
# AlphaFold 3 and the JAX backends, from core/jaxenv/uv.lock - the same file
# `residudes-engine setup venv` builds the site's JAX environment from.
#
# jax[cuda13] brings its own CUDA libraries as wheels, so the base only supplies the system
# pieces around them. AlphaFold 3's parameters are licensed per user and are never in an image:
# they are bind-mounted and pointed at with [site] af3_weights.
set -eu

export DEBIAN_FRONTEND=noninteractive
stage=${1:-all}

# gcc, libc6-dev and the Python headers: Triton compiles a small C launcher the first time a kernel runs, so
# bt2 died mid-prediction with "Failed to find C compiler" on a runtime image that had none. libxrender1 and
# libxext6: RDKit's drawing module, which openfold3 imports, links them; without them of3 failed at import.
if [ "$stage" = all ] || [ "$stage" = system ]; then
apt-get update
apt-get install -y --no-install-recommends \
    ca-certificates curl git python3.12 python3.12-venv libpython3.12 \
    libgomp1 libstdc++6 zstd \
    gcc libc6-dev python3.12-dev libxrender1 libxext6
rm -rf /var/lib/apt/lists/*
fi

# jaxenv's lock has two path dependencies, so the lockfile alone is not enough:
#   alphafold3   ../models/alphafold3   non-editable, scikit-build-core - a C++ extension gets
#                                       compiled here, which is work the image exists to save
#   openfold3    ../models/openfold-3   editable, so it resolves to whatever is mounted
# Both source trees are copied in at the paths the lock names; `uv sync` fails otherwise with
# "Distribution not found at: file:///opt/models/alphafold3".
# The compiler toolchain belongs to the build and not to the image, so it is installed in the
# stage that compiles and never reaches the final one.
#   python3.12-dev because alphafold3's CMakeLists does
#     find_package(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)
#   and the Development component is the headers, which python3.12-venv does not carry; zlib for
#   libcifpp. The five FetchContent repositories (abseil, pybind11, pybind11_abseil, libcifpp,
#   dssp) are cloned during the configure step, so this build needs network.
if [ "$stage" = all ] || [ "$stage" = sync ]; then
    apt-get update
    apt-get install -y --no-install-recommends \
        build-essential cmake ninja-build python3.12-dev zlib1g-dev
    rm -rf /var/lib/apt/lists/*
fi

# No wheel cache: for the CUDA stacks it is 8-10 GB sitting on disk beside the venv it
# produced, which is what exhausts a CI runner during the build rather than the image size.
export UV_NO_CACHE=1

if [ "$stage" = all ] || [ "$stage" = sync ]; then
    cd /opt/jaxenv                              # only this stage needs the lockfile beside it
    # --no-editable: openfold3 is an editable path dependency, which here would point at /opt/models, a
    # tree the final stage does not copy; installed, it lives in site-packages with everything else.
    UV_PROJECT_ENVIRONMENT=/opt/venv uv sync --frozen --no-install-project --no-editable --python /usr/bin/python3.12
    rm -rf /root/.cache/uv
    # AlphaFold 3 reads two CCD-derived pickles at import time, built from the components.cif its libcifpp
    # build installed (core/jaxenv/build.sh does the same for the self-built venv). Without them the image
    # passed its import check and failed on the first prediction:
    #   FileNotFoundError: .../alphafold3/constants/converters/chemical_component_sets.pickle
    /opt/venv/bin/build_data
fi

# HMMER 3.4 with AlphaFold 3's --seq_limit patch, built as AlphaFold 3's own image builds it. The af3 MSA source is
# AlphaFold 3's data pipeline (jackhmmer, nhmmer, hmmsearch), and the only source that searches RNA; without HMMER
# here a clean image-only install could predict but not build those MSAs - a protein-RNA run's MSA job died in 1 s.
# Built in the stage that has the compiler; the final stage copies /opt/hmmer.
if [ "$stage" = all ] || [ "$stage" = hmmer ]; then
    mkdir -p /tmp/hmmer
    (cd /tmp/hmmer \
     && curl -fsSLO http://eddylab.org/software/hmmer/hmmer-3.4.tar.gz \
     && echo "ca70d94fd0cf271bd7063423aabb116d42de533117343a9b27a65c17ff06fbf3  hmmer-3.4.tar.gz" | sha256sum --check \
     && tar zxf hmmer-3.4.tar.gz \
     && patch -p0 < /opt/models/alphafold3/docker/jackhmmer_seq_limit.patch \
     && cd hmmer-3.4 && ./configure --prefix /opt/hmmer && make -j"$(nproc)" && make install \
     && cd easel && make install)
    rm -rf /tmp/hmmer
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
import jax
print("jax", jax.__version__)
import haiku
print("dm-haiku ok")
import jax_triton                                  # tokamax calls triton_call(out_type=...)
print("jax-triton ok")
import alphafold3
from alphafold3.cpp import cif_dict                 # the compiled extension, not just the python
print("alphafold3 + compiled cpp ok")
from pathlib import Path
conv = Path(alphafold3.__file__).parent / "constants" / "converters"
for pk in ("ccd.pickle", "chemical_component_sets.pickle"):  # read at import by the model, not by `import alphafold3`
    assert (conv / pk).is_file(), f"missing {pk}"
from alphafold3.constants import chemical_component_sets    # the module that failed in a prediction
print("CCD tables ok")
import openfold3
assert "site-packages" in openfold3.__file__, openfold3.__file__
from openfold3.entry_points.experiment_runner import InferenceExperimentRunner   # what the of3 runner imports
import shutil
assert shutil.which("gcc"), "no C compiler: Triton cannot build its launcher"
print("openfold3 installed")
import os, subprocess
os.environ["PATH"] = "/opt/hmmer/bin:" + os.environ.get("PATH", "")
for b in ("jackhmmer", "nhmmer", "hmmsearch", "hmmbuild", "hmmalign"):   # what the af3 MSA source runs
    assert shutil.which(b), f"no {b}: the af3 MSA source cannot run in this image"
from alphafold3.data.tools.subprocess_utils import jackhmmer_seq_limit_supported
assert jackhmmer_seq_limit_supported(shutil.which("jackhmmer")), "jackhmmer without AlphaFold 3's --seq_limit patch"
print("hmmer 3.4 (--seq_limit) ok")
EOF
