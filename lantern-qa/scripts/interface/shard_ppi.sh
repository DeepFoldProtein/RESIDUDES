#!/bin/bash
# shard_ppi.sh — parallelize run-ppi-v4 across N sbatch jobs (one shard each).
#
# run-ppi-v4 iterates cifs in a SERIAL python loop (each cif's DFIRE/MPNN already
# run on GPU, but one cif at a time). For large pools this is the bottleneck.
# This splits the discovered index into N shards and fans them out as N
# independent sbatch jobs (GPU each), then a dependency job merges the per-shard
# xgb_scores parquet into one ranked table.
#
# Usage:
#   shard_ppi.sh --target-dir DIR --sto A2 --target NAME \
#       --variants af3_msa --methods af3,bt2,bt2_jax \
#       --chain-a A --chain-b B --shards 5 -o OUT \
#       [--nodelist ada] [--device cuda] [--partition normal] [--time 4:00:00]
set -euo pipefail
TOOL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$TOOL_DIR/../env.sh"     # lantern-qa: VENV_PY, LQA_ROOT, PYTHONPATH

TARGET_DIR= STO= TARGET= VARIANTS= METHODS= CHAIN_A=A CHAIN_B=B SHARDS=5 OUT=
NODELIST= EXCLUDE= DEVICE=cuda PARTITION=normal TIME=4:00:00
while [ $# -gt 0 ]; do
  case "$1" in
    --target-dir) TARGET_DIR=$2; shift 2;;
    --sto) STO=$2; shift 2;;
    --target) TARGET=$2; shift 2;;
    --variants) VARIANTS=$2; shift 2;;
    --methods) METHODS=$2; shift 2;;
    --chain-a) CHAIN_A=$2; shift 2;;
    --chain-b) CHAIN_B=$2; shift 2;;
    --shards) SHARDS=$2; shift 2;;
    -o|--output) OUT=$2; shift 2;;
    --nodelist) NODELIST=$2; shift 2;;   # comma list e.g. 'ada,ada-001,ada-003' -> shards distribute across them
    --exclude) EXCLUDE=$2; shift 2;;     # comma list of nodes to avoid (e.g. ada-002 GPU-binding issue)
    --device) DEVICE=$2; shift 2;;
    --partition) PARTITION=$2; shift 2;;
    --time) TIME=$2; shift 2;;
    *) echo "unknown arg: $1" >&2; exit 1;;
  esac
done
[ -n "$TARGET_DIR" ] && [ -n "$STO" ] && [ -n "$OUT" ] || { echo "need --target-dir --sto -o" >&2; exit 1; }
mkdir -p "$OUT"/{shards,logs}

# 1. Discover full index once (skip if already present).
IDX="$OUT/index.parquet"
if [ ! -f "$IDX" ]; then
  echo "[shard] discovering index -> $IDX"
  "$VENV_PY" -m lantern_qa.interface.cli discover \
    --target-dir "$TARGET_DIR" --sto "$STO" \
    ${TARGET:+--target "$TARGET"} \
    ${VARIANTS:+--variants "$VARIANTS"} \
    ${METHODS:+--methods "$METHODS"} \
    -o "$OUT"
fi

# 2. Split index into $SHARDS round-robin parquet shards.
echo "[shard] splitting into $SHARDS shards"
"$VENV_PY" - "$IDX" "$OUT/shards" "$SHARDS" <<'PY'
import sys, pandas as pd
idx, outdir, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
df = pd.read_parquet(idx); import os; os.makedirs(outdir, exist_ok=True)
for i in range(n):
    sub = df.iloc[i::n].reset_index(drop=True)          # round-robin = balanced
    sub.to_parquet(f"{outdir}/shard_{i}.parquet", index=False)
    print(f"  shard_{i}: {len(sub)} rows")
PY

# 3. Fan out one self-healing worker per shard (shard_worker.sh re-places itself
#    off a node whose GPU isn't visible; see that script).
GRES="gpu:1"; [ "$DEVICE" = cpu ] && GRES=""
GRES_ARG=(); [ -n "$GRES" ] && GRES_ARG=(--gres="$GRES")
NODE_ARG=()
[ -n "$NODELIST" ] && NODE_ARG+=(--nodelist="$NODELIST")
[ -n "$EXCLUDE" ]  && NODE_ARG+=(--exclude="$EXCLUDE")
MAXTRY=4
for i in $(seq 0 $((SHARDS-1))); do
  EXPORTS="ALL,LQA_ROOT=$LQA_ROOT,TOOL_DIR=$TOOL_DIR,VENV_PY=$VENV_PY,OUT=$OUT,SHARD=$i,TARGET=${TARGET:-},CHAIN_A=$CHAIN_A,CHAIN_B=$CHAIN_B,DEVICE=$DEVICE,PARTITION=$PARTITION,TIME=$TIME,MAXTRY=$MAXTRY,ATTEMPT=1,EXCLUDE=${EXCLUDE:-},GRES=$GRES,NODELIST=${NODELIST:-}"
  jid=$(sbatch --parsable --partition="$PARTITION" --time="$TIME" \
    --job-name="ppishard_${i}" --cpus-per-task=8 "${GRES_ARG[@]}" "${NODE_ARG[@]}" \
    --output="$OUT/logs/shard_${i}_%j.out" --error="$OUT/logs/shard_${i}_%j.err" \
    --export="$EXPORTS" --wrap="bash $TOOL_DIR/shard_worker.sh")
  echo "  shard $i -> job $jid"
done

# 4. Merge job: poll for all shard MASTER parquets (robust to re-placed shards),
#    then score ONCE on the full pool. z_target is pool-relative, so per-shard
#    scores are NOT comparable — we must (re)predict on the merged master.
TAG="${CHAIN_A}-${CHAIN_B}"
MERGE="$VENV_PY - '$OUT' '$TAG' '$SHARDS' <<'PY'
import sys, glob, time, pandas as pd
from lantern_qa.interface.score.xgb_ppi import ppi_model_triple, prep_xgb_ppi_input, predict_xgb_ppi, PPI_TARGET_COL
from lantern_qa.interface.core.master import METHOD_MASTER_MAP
out, tag, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
pat = f'{out}/shard_*/parquet/master_{tag}.parquet'
for _ in range(360):                       # poll up to ~2h
    fs = sorted(glob.glob(pat))
    if len(fs) >= n: break
    time.sleep(20)
fs = sorted(glob.glob(pat))
if not fs: sys.exit('no shard master parquets found')
m = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
if 'method' in m.columns:                  # canonical method_master on full pool (bt2->b2 etc.)
    mapped = m['method'].map(METHOD_MASTER_MAP)
    m['method_master'] = mapped.where(mapped.notna(), m.get('method_master'))
for tgt in ('dockq',):                      # pool-wide single-pass scoring
    mp, mt, nm = ppi_model_triple(tgt)
    X, fcols = prep_xgb_ppi_input(m, mt, nm)
    m[PPI_TARGET_COL[tgt]] = predict_xgb_ppi(X, fcols, mp)
sc = PPI_TARGET_COL['dockq']
m = m.sort_values(sc, ascending=False)
m.to_parquet(f'{out}/xgb_scores_{tag}_merged.parquet', index=False)
keep = [c for c in ('model_id','method','variant','seed','sample',sc) if c in m.columns]
m[keep].to_csv(f'{out}/xgb_scores_{tag}_merged.csv', index=False)
print(f'scored {len(m)} models from {len(fs)} shards on full pool -> {out}/xgb_scores_{tag}_merged.csv')
PY"
mjid=$(sbatch --parsable --partition="$PARTITION" --time=2:30:00 \
  --job-name="ppimerge" --cpus-per-task=4 \
  --output="$OUT/logs/merge_%j.out" --error="$OUT/logs/merge_%j.err" \
  --wrap="set -uo pipefail; ${MERGE}")
echo "[shard] merge/score job $mjid (polls for $SHARDS masters, then full-pool score)"
echo "[shard] done. ranked result -> $OUT/xgb_scores_${TAG}_merged.csv"
