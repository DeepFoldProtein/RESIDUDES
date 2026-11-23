#!/bin/bash
#SBATCH -J lantern-engine
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH -t 02:00:00
# Generic Slurm wrapper: forwards every argument to `lantern-engine` (predict, engines build, …).
# Normally submitted by `lantern-engine submit`, which exports LANTERN_ENGINE_ROOT and
# LANTERN_ENGINE_SITE_FILE and sets -o/-e into the output directory. Direct use:
#   sbatch --export=ALL,LANTERN_ENGINE_ROOT=/path/to/lantern-engine scripts/slurm/lantern-engine.sbatch \
#       predict --model pt2 --input input.json --out out/seed_1 --seed 1
set -euo pipefail
: "${LANTERN_ENGINE_ROOT:?LANTERN_ENGINE_ROOT must be exported (sbatch copies this script, so it cannot locate itself)}"
echo "▶ slurm job ${SLURM_JOB_ID:-<local>} on $(hostname)  gpu: $(nvidia-smi -L 2>/dev/null | head -1 || echo none)"
exec "$LANTERN_ENGINE_ROOT/bin/lantern-engine" "$@"
