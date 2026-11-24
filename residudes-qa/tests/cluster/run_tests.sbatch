#!/bin/bash
#SBATCH -J mc_pytest
#SBATCH -p normal
#SBATCH -c 8
#SBATCH -t 1:00:00
#SBATCH --gres=gpu:1
#SBATCH -o %x.%j.out
#SBATCH -e %x.%j.err
#
# Run the MassiveClustering pytest suite on a compute node with BOUNDED threads
# (the suite's UMAP/clustering/torch tests otherwise spawn ~all cores and hog
# the shared login node). GPU requested so the GPU-backed tests run for real.
set -uo pipefail

# cap every parallel backend to the 8 allocated cores
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 \
       NUMBA_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 VECLIB_MAXIMUM_THREADS=8
TOOL=/gpfs/deepfold/casp/casp17/analysis/MassiveClustering_tool
PY=/gpfs/deepfold/casp/casp17/analysis/.venv/bin/python
export PYTHONPATH=$TOOL
cd "$TOOL"

date; echo "node=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-?} cores=${SLURM_CPUS_PER_TASK:-?}"
echo "=== pytest tests/ ==="
$PY -m pytest tests/ -q -p no:cacheprovider -rA --durations=15
echo "=== exit=$? ==="
date
