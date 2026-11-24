#!/bin/bash
#SBATCH --job-name=lqa-interface_mpnn
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=8:00:00
#SBATCH --array=0-7
#SBATCH --output=<OUTPUT>/logs/mpnn_%A_%a.out
#SBATCH --error=<OUTPUT>/logs/mpnn_%A_%a.err
#
# ProteinMPNN scoring shard.  GPU-bound, sequential within a shard.
# Substitute <OUTPUT>, <INDEX>, <AB_CHAINS>, <AG_CHAINS>, <HEAVY>, <LIGHT>.
set -euo pipefail
# residudes-qa: submit with --export=ALL,LQA_ROOT=<residudes-qa checkout>
source "${LQA_ROOT:?set LQA_ROOT}/scripts/env.sh"

mkdir -p <OUTPUT>/logs
SHARD="${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT}"
echo "=== residudes-qa interface mpnn shard $SHARD on $(hostname) ==="
date
$VENV_PY -m residudes_qa.interface.cli features \
    --index "<INDEX>" \
    --feature mpnn \
    --ab-chains "<AB_CHAINS>" \
    --ag-chains "<AG_CHAINS>" \
    --heavy "<HEAVY>" \
    --light "<LIGHT>" \
    -o "<OUTPUT>"
date
echo "=== done ==="
