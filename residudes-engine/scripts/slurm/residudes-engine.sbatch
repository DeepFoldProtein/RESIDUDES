#!/bin/bash
#SBATCH -J residudes-engine
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH -t 02:00:00
# Generic Slurm wrapper: forwards every argument to `residudes-engine` (predict, engines build, …).
# Normally submitted by `residudes-engine submit`, which exports RESIDUDES_ENGINE_ROOT and
# RESIDUDES_ENGINE_SITE_FILE and sets -o/-e into the output directory. Direct use:
#   sbatch --export=ALL,RESIDUDES_ENGINE_ROOT=/path/to/residudes-engine scripts/slurm/residudes-engine.sbatch \
#       predict --model pt2 --input input.json --out out/seed_1 --seed 1
set -euo pipefail
: "${RESIDUDES_ENGINE_ROOT:?RESIDUDES_ENGINE_ROOT must be exported (sbatch copies this script, so it cannot locate itself)}"
echo "▶ slurm job ${SLURM_JOB_ID:-<local>} on $(hostname)  gpu: $(nvidia-smi -L 2>/dev/null | head -1 || echo none)"

# With RESIDUDES_ENGINE_IMAGES naming a directory that holds residudes-engine-trt.sif and residudes-engine-jax.sif, the
# job runs inside the image for its model - AlphaFold 3 in the jax image, the others in the trt image - with the image's
# own interpreter, so nothing from the checkout's venvs is used. `submit` exports the environment it was called in, so
# setting the variable once where the agent runs moves every prediction it submits into the images. Bound: the checkout
# (at the image's path and its own), and the top-level directory of every absolute path in the site file and in the
# arguments, which covers the weights, engines and databases and the job's input and output.
if [ -n "${RESIDUDES_ENGINE_IMAGES:-}" ]; then
    model=""; prev=""
    for a in "$@"; do
        if [ "$prev" = "--model" ] || [ "$prev" = "-m" ]; then model=$a; fi
        prev=$a
    done
    image="$RESIDUDES_ENGINE_IMAGES/residudes-engine-trt.sif"
    [ "$model" = af3 ] && image="$RESIDUDES_ENGINE_IMAGES/residudes-engine-jax.sif"
    # ESMFold2 has its own image (python 3.12, torch and the Biohub transformers fork). Sent to the trt image it ran
    # the host's esm conda env instead and died there; residudes-ef2.sif was pulled and never used. Its runscript is
    # plain python, so the engine CLI is started explicitly, and the runner is pointed at the image's interpreter.
    mode=run; cli=""; ef2_env=""
    if [ "$model" = ef2 ] && [ -f "$RESIDUDES_ENGINE_IMAGES/residudes-ef2.sif" ]; then
        image="$RESIDUDES_ENGINE_IMAGES/residudes-ef2.sif"; mode=exec; cli=/opt/residudes-engine/bin/residudes-engine
        ef2_env=",RESIDUDES_ENGINE_ESM_PYTHON=/opt/venv/bin/python,RESIDUDES_ENGINE_PYTHON=/opt/venv/bin/python,RESIDUDES_ENGINE_IN_IMAGE=1"
    fi
    binds=$(python3 - "${RESIDUDES_ENGINE_SITE_FILE:-}" "$RESIDUDES_ENGINE_ROOT" "$PWD" "$@" <<'PY'
import re, sys
from pathlib import Path
tops = set()
def add(x):
    if isinstance(x, str) and x.startswith("/") and len(x) > 1:
        top = "/" + x.split("/")[1]
        if Path(top).exists():               # a bind whose source is missing stops the container from starting
            tops.add(top)
def uncomment(line):                         # the template's commented examples ("/data/af3") are not paths in use
    out, quote = [], None
    for ch in line:
        if quote:
            quote = None if ch == quote else quote
        elif ch in "\"'":
            quote = ch
        elif ch == "#":
            break
        out.append(ch)
    return "".join(out)
site = sys.argv[1]
if site and Path(site).is_file():            # every quoted absolute path; no TOML parser on a 3.9 system python
    text = "\n".join(uncomment(ln) for ln in Path(site).read_text().splitlines())
    for x in re.findall(r'"(/[^"]+)"', text):
        add(x)
for a in sys.argv[2:]:
    add(a.split("=", 1)[-1])
print(",".join(sorted(tops - {"/opt", "/usr", "/proc", "/sys", "/dev"})))
PY
)
    echo "▶ in image $(basename "$image") for model ${model:-?}; bound: ${binds:-none}"
    exec apptainer "$mode" --nv ${binds:+-B "$binds"} -B "$RESIDUDES_ENGINE_ROOT:/opt/residudes-engine" \
        --env "RESIDUDES_ENGINE_VENV=/opt/venv,RESIDUDES_ENGINE_VENV_JAX=/opt/venv$ef2_env" "$image" ${cli:+"$cli"} "$@"
fi
exec "$RESIDUDES_ENGINE_ROOT/bin/residudes-engine" "$@"
