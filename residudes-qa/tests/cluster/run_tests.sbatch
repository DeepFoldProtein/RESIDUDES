#!/bin/bash
#SBATCH -J residudes-qa-tests
#SBATCH -p normal
#SBATCH -c 8
#SBATCH -t 1:00:00
#SBATCH --gres=gpu:1
#SBATCH -o %x.%j.out
#SBATCH -e %x.%j.err
#
# Run the residudes-qa suite on a compute node with BOUNDED threads: the clustering,
# UMAP and torch tests otherwise spawn one worker per core and hog the shared login
# node. A GPU is requested so the GPU-backed tests run for real rather than skipping.
#
#   sbatch residudes-qa/tests/cluster/run_tests.sbatch        # from anywhere in the checkout
#   PY=/path/to/python sbatch …                               # another interpreter
set -uo pipefail

# cap every parallel backend to the allocated cores
CORES=${SLURM_CPUS_PER_TASK:-8}
export OMP_NUM_THREADS=$CORES OPENBLAS_NUM_THREADS=$CORES MKL_NUM_THREADS=$CORES \
       NUMBA_NUM_THREADS=$CORES NUMEXPR_NUM_THREADS=$CORES VECLIB_MAXIMUM_THREADS=$CORES

QA=$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../.." && pwd)   # the residudes-qa checkout
PY=${PY:-${RESIDUDES_QA_PYTHON:-$QA/.venv/bin/python}}
[ -x "$PY" ] || { echo "no interpreter at $PY (cd $QA && uv sync --extra cu130)" >&2; exit 2; }
cd "$QA"

date; echo "node=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-?} cores=$CORES"
echo "=== $PY -m pytest tests ==="
"$PY" -m pytest tests -q -p no:cacheprovider -rA --durations=15
rc=$?
echo "=== exit=$rc ==="
date
exit $rc
