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
exec "$RESIDUDES_ENGINE_ROOT/bin/residudes-engine" "$@"
