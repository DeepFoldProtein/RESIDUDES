#!/bin/bash
#SBATCH --job-name=lqa-interface_cpu
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=6:00:00
#SBATCH --array=0-5
#SBATCH --output=<OUTPUT>/logs/cpu_%A_%a.out
#SBATCH --error=<OUTPUT>/logs/cpu_%A_%a.err
#
# CPU feature extraction shard (hbond + bbdep_chi + ndrd).
#
# Substitute <TARGET_DIR>, <STO>, <OUTPUT>, <INDEX> before submitting, e.g.
#   sed -e 's|<TARGET_DIR>|/path/to/H1335|g' \
#       -e 's|<STO>|A1B1C1D1E1|g'             \
#       -e 's|<OUTPUT>|/path/to/H1335_scored|g' \
#       -e 's|<INDEX>|/path/to/H1335_scored/index.parquet|g' \
#       cpu.sbatch | sbatch
set -euo pipefail
# residudes-qa: submit with --export=ALL,LQA_ROOT=<residudes-qa checkout>
source "${LQA_ROOT:?set LQA_ROOT}/scripts/env.sh"

mkdir -p <OUTPUT>/logs
SHARD="${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT}"
echo "=== residudes-qa interface cpu shard $SHARD on $(hostname) ==="
date
# CLI does not yet shard, but the driver can be sharded via --variants / --methods.
# For full-target run, drop --array=0-5 and run once with default sbatch params.
$VENV_PY -m residudes_qa.interface.cli features \
    --index "<INDEX>" \
    --feature cpu \
    --workers 8 \
    -o "<OUTPUT>"
date
echo "=== done ==="
