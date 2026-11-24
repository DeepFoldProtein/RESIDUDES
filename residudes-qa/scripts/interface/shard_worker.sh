#!/bin/bash
# shard_worker.sh — one shard of a sharded run-ppi-v4, with self-healing on a
# bad GPU node. If --device cuda but torch can't see a GPU (the known ada-002
# binding issue, or a node that lost its GPU), this resubmits ITSELF onto a
# different node (excluding the current one) instead of silently crawling on CPU.
#
# Invoked by shard_ppi.sh via sbatch --wrap. All state passed as env (--export).
# Required env: TOOL_DIR VENV_PY OUT SHARD TARGET CHAIN_A CHAIN_B DEVICE
#               PARTITION TIME MAXTRY ATTEMPT EXCLUDE GRES NODELIST
set -uo pipefail
export PYTHONPATH="${LQA_ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"   # residudes-qa
node="${SLURMD_NODENAME:-$(hostname -s)}"
parquet="$OUT/shard_$SHARD/parquet/xgb_scores_${CHAIN_A}_${CHAIN_B}.parquet"

# Already produced (e.g. a previous attempt finished)? Nothing to do.
[ -f "$parquet" ] && { echo "[worker $SHARD] output exists, skip"; exit 0; }

gpu_ok=1
if [ "$DEVICE" = cuda ]; then
  "$VENV_PY" -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' 2>/dev/null || gpu_ok=0
fi

if [ "$DEVICE" = cuda ] && [ "$gpu_ok" -eq 0 ]; then
  echo "[worker $SHARD] GPU NOT visible on $node (attempt $ATTEMPT/$MAXTRY)"
  if [ "$ATTEMPT" -lt "$MAXTRY" ]; then
    newexcl="$node"; [ -n "$EXCLUDE" ] && newexcl="$EXCLUDE,$node"
    echo "[worker $SHARD] re-placing: excluding $newexcl, attempt $((ATTEMPT+1))"
    EXPORTS="ALL,LQA_ROOT=$LQA_ROOT,TOOL_DIR=$TOOL_DIR,VENV_PY=$VENV_PY,OUT=$OUT,SHARD=$SHARD,TARGET=$TARGET,CHAIN_A=$CHAIN_A,CHAIN_B=$CHAIN_B,DEVICE=$DEVICE,PARTITION=$PARTITION,TIME=$TIME,MAXTRY=$MAXTRY,ATTEMPT=$((ATTEMPT+1)),EXCLUDE=$newexcl,GRES=$GRES,NODELIST=$NODELIST"
    NODE_ARG=(); [ -n "$NODELIST" ] && NODE_ARG+=(--nodelist="$NODELIST")
    NODE_ARG+=(--exclude="$newexcl")
    GRES_ARG=(); [ -n "$GRES" ] && GRES_ARG=(--gres="$GRES")
    sbatch --partition="$PARTITION" --time="$TIME" --job-name="ppishard_${SHARD}" \
      --cpus-per-task=8 "${GRES_ARG[@]}" "${NODE_ARG[@]}" \
      --output="$OUT/logs/shard_${SHARD}_%j.out" --error="$OUT/logs/shard_${SHARD}_%j.err" \
      --export="$EXPORTS" --wrap="bash $TOOL_DIR/shard_worker.sh"
    exit 0   # this attempt cleanly hands off to the re-placed job
  fi
  echo "[worker $SHARD] WARNING: out of retries -> running on CPU fallback"
fi

echo "[worker $SHARD] GPU=$gpu_ok on $node -> scoring"
exec "$VENV_PY" -m residudes_qa.interface.cli run-ppi-v4 \
  --index "$OUT/shards/shard_$SHARD.parquet" \
  ${TARGET:+--target "$TARGET"} --chain-a "$CHAIN_A" --chain-b "$CHAIN_B" \
  --device "$DEVICE" -o "$OUT/shard_$SHARD"
