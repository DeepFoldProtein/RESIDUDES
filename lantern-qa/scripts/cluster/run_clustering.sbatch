#!/bin/bash
#SBATCH -J cluster
#SBATCH -p normal
#SBATCH -c 8
#SBATCH -t 8:00:00
#SBATCH --gres=gpu:1
#SBATCH --gres-flags=enforce-binding
#SBATCH -o %x.%j.out
#SBATCH -e %x.%j.err
#
# MassiveClustering on a CASP17 target — Slurm-side body.
#
# Submitted by analysis/MassiveClustering_tool/run_clustering.sh; you
# normally don't invoke this directly. All positional args are forwarded
# verbatim to run_clustering.py (via analysis/.venv).
#
# Env overrides:
#   LQA_ROOT      lantern-qa checkout (exported by run_clustering.sh)
#   GPU           default: 0 (the local GPU id seen by CUDA after Slurm
#                 binds --gres=gpu:1 — i.e. always 0 inside the job)

set -euo pipefail

# lantern-qa: LQA_ROOT is exported by run_clustering.sh (Slurm runs a spool copy of this file)
source "${LQA_ROOT:?set LQA_ROOT to the lantern-qa checkout}/scripts/env.sh"
VENV_PYTHON="$VENV_PY"
DRIVER="$LQA_ROOT/src/lantern_qa/cluster/run_clustering.py"

[ -x "$VENV_PYTHON" ] || { echo "error: $VENV_PYTHON not found. Run uv sync in $LQA_ROOT first." >&2; exit 2; }
[ -f "$DRIVER" ]      || { echo "error: $DRIVER not found." >&2; exit 2; }

date
echo "▶ Job ID:    ${SLURM_JOB_ID:-<local>}"
echo "▶ Node:      ${SLURMD_NODENAME:-$(hostname)}"
echo "▶ GPU:       ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "▶ args:      $*"

# Force gpu_id=0 inside the Slurm job (Slurm pins the requested GPU as
# device 0 inside CUDA_VISIBLE_DEVICES). Strip any user-supplied --gpu
# from the forwarded args and re-append --gpu 0 so we always target the
# Slurm-assigned device.
clean_args=()
skip=0
TGT=""; OUTROOT=""; want_root=0
for a in "$@"; do
    if [ "$skip" = "1" ]; then skip=0; continue; fi
    if [ "$want_root" = "1" ]; then OUTROOT="$a"; want_root=0; fi
    case "$a" in
        --gpu) skip=1 ;;
        --gpu=*) ;;
        --out-root) want_root=1; clean_args+=("$a") ;;
        --out-root=*) OUTROOT="${a#--out-root=}"; clean_args+=("$a") ;;
        *) clean_args+=("$a"); [ -z "$TGT" ] && [ -d "$a" ] && TGT="$a" ;;
    esac
done

# Marker to scope post-run cleanup to _pdb_view dirs THIS run created.
MARKER="$(mktemp)"

set +e
"$VENV_PYTHON" "$DRIVER" "${clean_args[@]}" --gpu 0
rc=$?
set -e

# NOTE (2026-06-20): the global post-run _pdb_view sweep below is DISABLED.
# It did `find "$base"/cluster_results -name _pdb_view -newer MARKER -rm`, but a
# CONCURRENT sibling job (started after this job's MARKER) creates a _pdb_view
# that is ALSO newer than MARKER → this job's cleanup deleted the sibling's
# cache mid-medoid-export → FileNotFoundError race. The driver (run_clustering.py
# ~L2444) already removes ITS OWN per-method _pdb_view at end of run (unless
# --keep-pdb-view), so the global sweep is both redundant and racy. Disabled.
# (Opt back into the old behaviour with MC_SWEEP_PDB_VIEW=1 if ever needed.)
if [ "$rc" = "0" ] && [ "${MC_SWEEP_PDB_VIEW:-0}" = "1" ] && [ "${MC_KEEP_PDB_VIEW:-0}" != "1" ]; then
    base="${OUTROOT:-${TGT:+$TGT/cluster_results}}"
    if [ -n "$base" ] && [ -d "$base" ]; then
        echo "▶ cleanup: removing regenerable _pdb_view caches under $base"
        find "$base" -depth -type d -name _pdb_view -newer "$MARKER" -exec rm -rf {} + 2>/dev/null || true
    fi
fi
rm -f "$MARKER"
exit "$rc"
