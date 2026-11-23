#!/bin/bash
# run_all_worker.sh — one attempt of a run_scoring.sh Slurm job, with
# self-healing on a bad GPU node.
#
# A GPU is requested by type-agnostic `gpu:1` (any model — RTX4090/RTX6000/...),
# never pinned to a specific card. If that GPU turns out unusable, the job
# re-places itself onto a DIFFERENT node (excluding the current one) instead of
# silently crawling on CPU or dying. Two triggers, both capped at $MAXTRY:
#   1. GPU not visible at startup  (the known ada-002 binding issue / lost GPU)
#   2. GPU out-of-memory mid-run   (a card shared with another job, etc.)
# Once the retry budget is spent it stops re-placing — trigger (1) falls back to
# CPU, trigger (2) exits with the real error — so a flaky cluster can never spin
# up an unbounded chain of jobs.
#
# Every decision is logged with a "[worker:<stage>]" prefix so the run log shows
# exactly where each stage landed, which GPU it got, and — on a re-placement —
# the precise problem and where it is being moved.
#
# Invoked by run_scoring.sh via `sbatch --wrap="bash .../run_all_worker.sh"`.
# All state is passed through the environment (sbatch --export):
#   Required: VENV_PY CMD GPU_REQUESTED PARTITION TIME GRES MAXTRY ATTEMPT
#             EXCLUDE JOBNAME LOG_DIR
set -uo pipefail

WORKER="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_all_worker.sh"
node="${SLURMD_NODENAME:-$(hostname -s)}"
stage="${JOBNAME##*_}"          # e.g. xgbdimer_H1382_dfire -> dfire
log() { echo "[worker:$stage] $*"; }

# Resubmit this job onto a different node, excluding the current one (plus any
# already-excluded), with the attempt counter bumped. $1 = human reason.
# Returns the resubmitted job's exit code when WAIT_CHAIN=1 (so a downstream
# afterok dependency on THIS job only clears once the re-placed work truly
# finishes); otherwise fire-and-forget and return 0.
replace() {
  local reason="$1"
  local newexcl="$node"; [ -n "${EXCLUDE:-}" ] && newexcl="$EXCLUDE,$node"
  log "ACTION: re-allocating to a different node — excluding [$newexcl], attempt $((ATTEMPT + 1))/$MAXTRY (reason: $reason)"
  local GRES_ARG=(); [ -n "$GRES" ] && GRES_ARG=(--gres="$GRES")
  # Deliberately drop any --nodelist pin here — it is what stuck us on the bad
  # node — and steer purely with the growing --exclude list.
  local EXPORTS="ALL,VENV_PY=$VENV_PY,CMD=$CMD,GPU_REQUESTED=$GPU_REQUESTED,PARTITION=$PARTITION,TIME=$TIME,GRES=$GRES,MAXTRY=$MAXTRY,ATTEMPT=$((ATTEMPT + 1)),EXCLUDE=$newexcl,JOBNAME=$JOBNAME,LOG_DIR=$LOG_DIR,WAIT_CHAIN=${WAIT_CHAIN:-0}"
  local WAIT_ARG=(); [ "${WAIT_CHAIN:-0}" = 1 ] && WAIT_ARG=(--wait)
  local rj
  if [ "${WAIT_CHAIN:-0}" = 1 ]; then
    # --parsable prints the new job id up front, then --wait blocks until it
    # finishes; capture id + propagate its exit code.
    rj=$(sbatch --parsable "${WAIT_ARG[@]}" --partition="$PARTITION" --time="$TIME" \
      --job-name="$JOBNAME" --cpus-per-task=8 "${GRES_ARG[@]}" --exclude="$newexcl" \
      --output="$LOG_DIR/lqa-interface_%j.out" --error="$LOG_DIR/lqa-interface_%j.err" \
      --export="$EXPORTS" --wrap="bash $WORKER")
    local rc=$?
    log "  -> resubmitted as job ${rj:-?} (waited; exit=$rc)"
    return $rc
  fi
  rj=$(sbatch --parsable --partition="$PARTITION" --time="$TIME" \
    --job-name="$JOBNAME" --cpus-per-task=8 "${GRES_ARG[@]}" --exclude="$newexcl" \
    --output="$LOG_DIR/lqa-interface_%j.out" --error="$LOG_DIR/lqa-interface_%j.err" \
    --export="$EXPORTS" --wrap="bash $WORKER")
  log "  -> resubmitted as job ${rj:-?} (fire-and-forget)"
  return 0
}

log "===== stage start on $node — attempt $ATTEMPT/$MAXTRY ====="
log "  GPU_REQUESTED=${GPU_REQUESTED:-0}  GRES=${GRES:-<none>}  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}  excluded-so-far=[${EXCLUDE:-none}]"

# ---- trigger 1: GPU not visible at startup ---------------------------------
gpu_ok=1
if [ "${GPU_REQUESTED:-0}" = 1 ]; then
  "$VENV_PY" -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' 2>/dev/null || gpu_ok=0
fi
if [ "${GPU_REQUESTED:-0}" = 1 ] && [ "$gpu_ok" -eq 0 ]; then
  log "PROBLEM: GPU requested (gpu:1) but torch.cuda.is_available()=False on $node — driver/binding issue or no usable GPU on this node."
  if [ "$ATTEMPT" -lt "$MAXTRY" ]; then
    replace "no GPU visible"; exit $?   # hand off (WAIT_CHAIN=1 → wait for it)
  fi
  log "WARNING: out of GPU retries ($MAXTRY) -> CPU FALLBACK (much slower)"
elif [ "${GPU_REQUESTED:-0}" = 1 ]; then
  gpuname=$("$VENV_PY" -c 'import torch;print(torch.cuda.get_device_name(0))' 2>/dev/null || echo "?")
  log "GPU OK: '$gpuname' (cuda:0, CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-?}) on $node -> running on CUDA"
else
  log "CPU stage (no GPU requested) on $node"
fi

# ---- run, watching for trigger 2: GPU out-of-memory ------------------------
# Capture combined output so we can detect a CUDA OOM and re-place on a fresh
# GPU. Output still streams to the Slurm log via tee.
out_log="$(mktemp "${TMPDIR:-/tmp}/xgbworker.XXXXXX")"
set +e
bash -c "$CMD" 2>&1 | tee "$out_log"
rc=${PIPESTATUS[0]}
set -e

if [ "$rc" -ne 0 ] && [ "${GPU_REQUESTED:-0}" = 1 ] && [ "$gpu_ok" -eq 1 ] \
   && grep -qiE 'out of memory|CUDA error|CUBLAS_STATUS|cuda runtime error|c10::OutOfMemoryError' "$out_log"; then
  errline=$(grep -iE 'out of memory|CUDA error|CUBLAS_STATUS|cuda runtime error|c10::OutOfMemoryError' "$out_log" | head -1 | sed 's/^[[:space:]]*//')
  log "PROBLEM: GPU memory/CUDA error on $node (rc=$rc): ${errline:-<matched OOM pattern>}"
  if [ "$ATTEMPT" -lt "$MAXTRY" ]; then
    replace "GPU OOM/CUDA error"; rc2=$?; rm -f "$out_log"; exit $rc2
  fi
  log "WARNING: out of GPU retries ($MAXTRY) after OOM -> giving up (propagating rc=$rc)"
fi

rm -f "$out_log"
log "stage '$stage' on $node finished rc=$rc"
exit "$rc"
