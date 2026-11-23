#!/bin/bash
#SBATCH --job-name=lqa-interface_dfire
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=4:00:00
#SBATCH --array=0-5
#SBATCH --output=<OUTPUT>/logs/dfire_%A_%a.out
#SBATCH --error=<OUTPUT>/logs/dfire_%A_%a.err
#
# DFIRE v2 GPU shard.  --workers=2 to avoid OOM on shared GPU.
# Substitute <OUTPUT>, <INDEX> before submitting.
set -euo pipefail
# lantern-qa: submit with --export=ALL,LQA_ROOT=<lantern-qa checkout>
source "${LQA_ROOT:?set LQA_ROOT}/scripts/env.sh"

mkdir -p <OUTPUT>/logs
SHARD="${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT}"
echo "=== lantern-qa interface dfire shard $SHARD on $(hostname) ==="
date
$VENV_PY -m lantern_qa.interface.cli features \
    --index "<INDEX>" \
    --feature dfire \
    --workers 2 \
    -o "<OUTPUT>"
date
echo "=== done ==="
