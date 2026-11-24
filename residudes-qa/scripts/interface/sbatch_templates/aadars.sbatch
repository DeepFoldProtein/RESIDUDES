#!/bin/bash
#SBATCH --job-name=lqa-interface_aadars
#SBATCH --partition=normal
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --array=0-7
#SBATCH --output=<OUTPUT>/logs/aadars_%A_%a.out
#SBATCH --error=<OUTPUT>/logs/aadars_%A_%a.err
#
# aADARS v3 (Fab CDR energy).  Ab mode only.  Requires HMMER on PATH for
# ANARCI — sourced from analysis/.venv-cdr.
# Substitute <OUTPUT>, <INDEX>, <AG_CHAINS>, <HEAVY>, <LIGHT>.
set -euo pipefail
# residudes-qa: submit with --export=ALL,LQA_ROOT=<residudes-qa checkout>
source "${LQA_ROOT:?set LQA_ROOT}/scripts/env.sh"
export PATH=$LQA_CDR_BIN:$PATH

mkdir -p <OUTPUT>/logs
SHARD="${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT}"
echo "=== residudes-qa interface aadars shard $SHARD on $(hostname) ==="
date
$VENV_PY -m residudes_qa.interface.cli features \
    --index "<INDEX>" \
    --feature aadars \
    --ag-chains "<AG_CHAINS>" \
    --heavy "<HEAVY>" \
    --light "<LIGHT>" \
    --workers 8 \
    -o "<OUTPUT>"
date
echo "=== done ==="
