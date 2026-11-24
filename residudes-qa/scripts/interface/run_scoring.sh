#!/bin/bash
# One-command Slurm entry point for the residudes-qa interface scorer (ex-XGBDimerScorer).
#
# Usage:
#   run_scoring.sh <target_dir> --sto SPEC --mode {ab|ppi} -o OUT [...flags]
#
# Required:
#   <target_dir>          CASP target dir, e.g. /path/to/targets/H1335
#   --sto SPEC            Stoichiometry: A1B1, A1B1C1D1E1, ...
#   --mode {ab|ppi}       Scoring mode (xgb_v8 vs xgb_ppi_v3)
#   -o, --output DIR      Output root
#
# Common Ab-mode flags:
#   --ab-chains DE        Antibody chain IDs
#   --ag-chains ABC       Antigen chain IDs
#   --heavy E             Fab heavy chain (single id)
#   --light D             Fab light chain (single id)
#   --ref-cif PATH        Reference structure for COM alignment / regional RMSD
#   --anchor-chains AB    Anchor (e.g. gH) for Kabsch frame alignment
#
# PPI-mode flags:
#   --probe-chains BD     Chains used for COM clustering
#
# Other:
#   --variants V1,V2      Whitelist variants
#   --methods M1,M2       Whitelist methods
#   --workers N           CPU workers (default 8)
#   --n-final N           Number of final picks (default 5)
#   --local               Run directly in this shell (no Slurm submission)
#   --partition NAME      Slurm partition (default normal)
#   --time HH:MM:SS       Walltime (default 12:00:00)
#   --gres SPEC           GPU request (default gpu:1; e.g. gpu:2). NONE = no GPU.
#   --no-gpu / --cpu      Submit without a GPU (CPU-only run)
#   --serial              run-all as ONE sequential job (no fan-out)
#   --shards N            Split each heavy stage (base/cpu/dfire/mpnn) into N
#                         parallel jobs (default: auto from model count, ≥2 for
#                         non-trivial sets). N=1 disables sharding.
#
# Env knobs:
#   XGB_SHARDS N|auto     Default shard count (default auto). XGB_SHARD_PER (~models
#                         per shard, 5000), XGB_SHARD_MAX (cap, 3), XGB_SHARD_MIN
#                         (don't shard below this many models, 300).
#   XGB_MAXTRY N          Max GPU re-placement attempts before CPU fallback
#                         (default 3). If a GPU was requested but the node can't
#                         see one, the job resubmits onto a different node up to
#                         N times, then runs on CPU. Caps endless re-placing.
#   XGB_NODELIST NODE     Pin the FIRST attempt to a node (self-heal drops it).
set -euo pipefail

TOOL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # scripts/xgb (worker + ppi_merge live here)
# residudes-qa: venv python, LQA_ROOT, PYTHONPATH=src, LQA_CDR_BIN
source "$TOOL_DIR/../env.sh"

# PYTHONPATH so the package imports without `pip install -e`.  Set early so the
# `--help` short-circuit (which execs the CLI before the main env block) works too.

PARTITION="${PARTITION:-normal}"
TIME="${TIME:-12:00:00}"
# Heavy submits request a GPU by default (DFIRE/MPNN/ppi_v4 use CUDA). Disable
# with --no-gpu (or --gres NONE); override count/type with --gres gpu:2 etc.
# If a GPU is requested but none is visible at runtime, the job still runs on
# CPU and prints a loud banner (see the --wrap preflight below).
GRES="${GRES:-gpu:1}"
LOCAL=0
# run-all fans the heavy stages out across separate Slurm jobs by default so
# dfire and mpnn run on TWO GPUs in parallel (not one job hogging a single GPU
# for both). Disable with --serial (single sequential job). Env XGB_PARALLEL=0
# flips the default.
PARALLEL="${XGB_PARALLEL:-1}"
# Sharding: split each heavy stage (base/cpu/dfire/mpnn) into N jobs that each
# score a disjoint model slice, then master globs+concats. Default = auto from
# model count (≥2 for anything non-trivial). Override with --shards N (1 = off).
SHARDS="${XGB_SHARDS:-auto}"
SHARD_PER="${XGB_SHARD_PER:-5000}"   # ~models per shard when auto
SHARD_MAX="${XGB_SHARD_MAX:-3}"      # cap (GPUs are scarce: N shards × 2 GPU stages)
SHARD_MIN_MODELS="${XGB_SHARD_MIN:-300}"  # below this, don't shard (overhead)

if [ $# -lt 1 ]; then
    sed -n '2,/^set -/p' "$0" | sed 's/^# \{0,1\}//; $d'
    exit 1
fi

# Detect: is $1 a subcommand keyword, or a positional <target_dir> (legacy)?
SUBCMDS="discover features master score filter cluster pick run-all run-ppi-v4"
SUB=""
for s in $SUBCMDS; do
    if [ "$1" = "$s" ]; then SUB="$1"; shift; break; fi
done

forward=()
MODE=""
TARGET_DIR=""
OUTPUT=""

if [ -z "$SUB" ]; then
    # Legacy positional form: <target_dir> [flags]  →  call run-all
    if [ ! -d "$1" ]; then
        echo "ERROR: '$1' is neither a subcommand nor a directory." >&2
        echo "  Subcommands: $SUBCMDS" >&2
        exit 2
    fi
    SUB="run-all"
    TARGET_DIR="$1"; shift
    forward+=(--target-dir "$TARGET_DIR")
fi

# Strip Slurm-control flags out of $@ before forwarding to the driver.
while [ $# -gt 0 ]; do
    case "$1" in
        -h|--help)
            # delegate help to python CLI for the chosen subcommand
            exec "$VENV_PY" -m residudes_qa.interface.cli "$SUB" --help
            ;;
        --local)     LOCAL=1; shift ;;
        --serial)    PARALLEL=0; shift ;;
        --parallel)  PARALLEL=1; shift ;;
        --shards)    SHARDS=$2; shift 2 ;;
        --partition) PARTITION=$2; shift 2 ;;
        --time)      TIME=$2; shift 2 ;;
        --no-gpu|--cpu) GRES=""; shift ;;
        --gres)      GRES=$2; [ "$GRES" = "NONE" ] && GRES=""; shift 2 ;;
        --target-dir) TARGET_DIR=$2; forward+=("$1" "$2"); shift 2 ;;
        --mode)      MODE=$2; forward+=("$1" "$2"); shift 2 ;;
        -o|--output) OUTPUT=$2; forward+=("$1" "$2"); shift 2 ;;
        *)           forward+=("$1"); shift ;;
    esac
done

# Default mode = ab for run-all only (other subcommands may not need --mode).
if [ "$SUB" = "run-all" ] && [ -z "$MODE" ]; then
    MODE="ab"
    forward+=(--mode ab)
fi

# Light-weight subcommands always run locally (no Slurm submit).
case "$SUB" in
    discover|master|score|filter|cluster|pick) LOCAL=1 ;;
esac

# aADARS requires HMMER from .venv-cdr; export the PATH only in Ab mode.
if [ "$MODE" = "ab" ] && [ -d "$LQA_CDR_BIN" ]; then
    export PATH="$LQA_CDR_BIN:${PATH}"
fi
# (EMA scorers are residudes_qa.ema; their data paths come from residudes_qa.paths.)
# (PYTHONPATH already exported near the top so --help works before this block.)

cmd=("$VENV_PY" -m residudes_qa.interface.cli "$SUB" "${forward[@]}")

echo "=== residudes-qa interface ==="
echo "  subcommand: $SUB"
echo "  target_dir: ${TARGET_DIR:-(n/a)}"
echo "  mode:       ${MODE:-(unset)}"
echo "  command:    ${cmd[*]}"
echo "  residudes-qa: $LQA_ROOT"
echo "  PYTHONPATH (head): ${PYTHONPATH%%:*}"
echo "  gres:       ${GRES:-(none — CPU only)}"

if [ $LOCAL -eq 1 ]; then
    exec "${cmd[@]}"
fi

# Decide where the wrapper job's Slurm logs land.  Mirror the feature-step
# jobs: drop them in <OUTPUT>/logs/ so all logs for a run live together.
# SLURM_LOG_DIR (if set) overrides as an explicit log directory.
if [ -n "${SLURM_LOG_DIR:-}" ]; then
    LOG_DIR="${SLURM_LOG_DIR%/}"
elif [ -n "$OUTPUT" ]; then
    LOG_DIR="${OUTPUT%/}/logs"
else
    LOG_DIR="."
fi
mkdir -p "$LOG_DIR"

# Shared Slurm/worker config.
#
# GPU by default (GRES, opt out with --no-gpu). If a GPU was requested but the
# node we land on can't see one (or it OOMs mid-run), run_all_worker.sh
# resubmits onto a DIFFERENT node (excluding the bad one), up to MAXTRY attempts;
# after that it falls back to CPU (no-GPU case) or fails (OOM case). MAXTRY caps
# the chain so a flaky cluster can't trigger endless re-placing.
MAXTRY="${XGB_MAXTRY:-3}"
GPU_REQUESTED=0
[ -n "$GRES" ] && GPU_REQUESTED=1
GRES_ARG=()
[ -n "$GRES" ] && GRES_ARG=(--gres="$GRES")
# Optional node pinning (env-gated; no effect unless XGB_NODELIST set). Only the
# FIRST attempt honours the pin; the self-heal path drops it (see worker).
NODELIST_ARG=()
[ -n "${XGB_NODELIST:-}" ] && NODELIST_ARG=(--nodelist="$XGB_NODELIST")

WORKER="$TOOL_DIR/run_all_worker.sh"
JOBNAME="lqa-interface_$([ -n "$TARGET_DIR" ] && basename "$TARGET_DIR" || echo "$SUB")"

# Write a self-contained, safely-quoted stage script (so commas/spaces in args
# never break sbatch --export / --wrap). Sets STAGE_SCRIPT to its path.
write_stage_script() {  # $1=tag  $2=stages-spec  [$3=shard $4=nshards]
    STAGE_SCRIPT="$LOG_DIR/stage_$1.sh"
    {
        printf '#!/bin/bash\nset -euo pipefail\nexec'
        for a in "${cmd[@]}"; do printf ' %q' "$a"; done
        printf ' --stages %q' "$2"
        if [ "${4:-1}" -gt 1 ]; then printf ' --shard %q --nshards %q' "$3" "$4"; fi
        printf '\n'
    } > "$STAGE_SCRIPT"
}

# A plain CPU stage job. $3 = dependency or "".  [$4=shard $5=nshards] optional.
submit_cpu_stage() {  # $1=tag  $2=stages  $3=dep  [$4=shard $5=nshards]
    write_stage_script "$1" "$2" "${4:-0}" "${5:-1}"
    local DEP=(); [ -n "$3" ] && DEP=(--dependency="$3")
    sbatch --parsable --partition="$PARTITION" --time="$TIME" \
        --job-name="${JOBNAME}_$1" --cpus-per-task=8 "${DEP[@]}" \
        --output="${LOG_DIR}/lqa-interface_$1_%j.out" \
        --error="${LOG_DIR}/lqa-interface_$1_%j.err" \
        --export=ALL --wrap="bash $STAGE_SCRIPT"
}

# A GPU stage job (dfire/mpnn) routed through the self-healing worker. The worker
# waits on its own re-placements (WAIT_CHAIN=1) so the merge job's afterok stays
# valid even when a stage hops nodes.  [$4=shard $5=nshards] optional.
submit_gpu_stage() {  # $1=tag  $2=stages  $3=dep  [$4=shard $5=nshards]
    write_stage_script "$1" "$2" "${4:-0}" "${5:-1}"
    local DEP=(); [ -n "$3" ] && DEP=(--dependency="$3")
    local JN="${JOBNAME}_$1"
    local EXPORTS="ALL,VENV_PY=$VENV_PY,CMD=bash $STAGE_SCRIPT,GPU_REQUESTED=$GPU_REQUESTED,PARTITION=$PARTITION,TIME=$TIME,GRES=$GRES,MAXTRY=$MAXTRY,ATTEMPT=1,EXCLUDE=,JOBNAME=$JN,LOG_DIR=$LOG_DIR,WAIT_CHAIN=1"
    sbatch --parsable --partition="$PARTITION" --time="$TIME" \
        --job-name="$JN" --cpus-per-task=8 "${GRES_ARG[@]}" "${DEP[@]}" \
        --output="${LOG_DIR}/lqa-interface_$1_%j.out" \
        --error="${LOG_DIR}/lqa-interface_$1_%j.err" \
        --export="$EXPORTS" --wrap="bash $WORKER"
}

# ---- run-all fan-out (sharded): each heavy stage split into N parallel jobs --
# Only the ab/abag staged pipeline fans out this way. mode=ppi dispatches to the
# ppi_v4 interface engine (one-pass, whole-complex work shared across interfaces)
# → submitted as a single GPU job below; shard large PPI pools with shard_ppi.sh.
if [ "$SUB" = "run-all" ] && [ "$PARALLEL" = 1 ] && [ "$MODE" != "ppi" ]; then
    echo ""
    echo "[run_scoring] ===== PARALLEL fan-out ====="
    # discover is light + must finish before any feature stage — run it inline.
    echo "[run_scoring] discover (inline on $(hostname -s)) ..."
    "${cmd[@]}" --stages discover || { echo "ERROR: discover failed" >&2; exit 1; }

    # Decide shard count N (auto from model count unless --shards/XGB_SHARDS set).
    NMODELS=$("$VENV_PY" -c "import pandas as pd; print(len(pd.read_parquet('${OUTPUT%/}/index.parquet')))" 2>/dev/null || echo 0)
    if [ "$SHARDS" = auto ]; then
        if [ "$NMODELS" -lt "$SHARD_MIN_MODELS" ]; then
            N=1
        else
            N=$(( (NMODELS + SHARD_PER - 1) / SHARD_PER ))
            [ "$N" -lt 2 ] && N=2
            [ "$N" -gt "$SHARD_MAX" ] && N=$SHARD_MAX
        fi
    else
        N=$SHARDS
    fi
    [ "$N" -lt 1 ] && N=1
    echo "[run_scoring]   models=$NMODELS  shards(N)=$N  GPU/stage=${GRES:-<none>}  MAXTRY=$MAXTRY"
    echo "[run_scoring]   flow: discover -> { base×N | cpu×N | dfire×N(GPU) | mpnn×N(GPU) | misc } parallel -> merge(master+score)"

    allj=()
    if [ "$N" -le 1 ]; then
        # No sharding: bundle CPU features into one job; dfire/mpnn one GPU job each.
        jb=$(submit_cpu_stage cpufeat "feat:base,feat:cpu,feat:clash,feat:aadars" ""); allj+=("$jb")
        jd=$(submit_gpu_stage dfire "feat:dfire" ""); allj+=("$jd")
        jm=$(submit_gpu_stage mpnn  "feat:mpnn"  ""); allj+=("$jm")
        echo "[run_scoring]   jobs: cpufeat=$jb dfire=$jd mpnn=$jm"
    else
        # Sharded: base/cpu (CPU) + dfire/mpnn (GPU) each split into N slices;
        # clash+aadars are cheap → one whole 'misc' job.
        for ((k=0; k<N; k++)); do
            jb=$(submit_cpu_stage "base$k"  "feat:base"  "" "$k" "$N"); allj+=("$jb")
            jc=$(submit_cpu_stage "cpu$k"   "feat:cpu"   "" "$k" "$N"); allj+=("$jc")
            jd=$(submit_gpu_stage "dfire$k" "feat:dfire" "" "$k" "$N"); allj+=("$jd")
            jm=$(submit_gpu_stage "mpnn$k"  "feat:mpnn"  "" "$k" "$N"); allj+=("$jm")
            echo "[run_scoring]   shard $k: base=$jb cpu=$jc dfire=$jd(GPU) mpnn=$jm(GPU)"
        done
        jx=$(submit_cpu_stage misc "feat:clash,feat:aadars" ""); allj+=("$jx")
        echo "[run_scoring]   misc(clash+aadars)=$jx"
    fi

    dep="afterok:$(IFS=:; echo "${allj[*]}")"
    jmerge=$(submit_cpu_stage merge "master,score" "$dep")

    echo ""
    echo "[run_scoring] submitted ${#allj[@]} feature jobs + merge=$jmerge"
    echo "  GPU jobs running concurrently: $([ "$N" -le 1 ] && echo 2 || echo $((2*N)))  (dfire+mpnn × N shards, each --gres=${GRES:-none})"
    echo "  merge waits afterok on all ${#allj[@]} feature jobs; master globs the shard parquets and concats."
    echo "  per-stage node+GPU logged in: ${LOG_DIR}/lqa-interface_<stage>_<jobid>.out"
    echo "  monitor: squeue -u \$USER | grep ${JOBNAME}"
    exit 0
fi

# ---- run-all PPI fan-out: shard the interface engine across N GPU jobs -------
# mode=ppi uses the ppi_v4 interface engine (per-cif serial loop on GPU). For
# large pools we split the index into N disjoint slices, fan them out as N GPU
# jobs (self-healing worker), and merge the per-shard xgb_dockq.csv.
if [ "$SUB" = "run-all" ] && [ "$PARALLEL" = 1 ] && [ "$MODE" = "ppi" ]; then
    echo ""
    echo "[run_scoring] ===== PPI PARALLEL fan-out ====="
    echo "[run_scoring] discover (inline on $(hostname -s)) ..."
    "${cmd[@]}" --stages discover || { echo "ERROR: ppi discover failed" >&2; exit 1; }
    IDX="${OUTPUT%/}/parquet/index.parquet"
    NMODELS=$("$VENV_PY" -c "import pandas as pd; print(len(pd.read_parquet('$IDX')))" 2>/dev/null || echo 0)
    if [ "$SHARDS" = auto ]; then
        if [ "$NMODELS" -lt "$SHARD_MIN_MODELS" ]; then N=1
        else N=$(( (NMODELS + SHARD_PER - 1) / SHARD_PER )); [ "$N" -lt 2 ] && N=2; [ "$N" -gt "$SHARD_MAX" ] && N=$SHARD_MAX; fi
    else N=$SHARDS; fi
    [ "$N" -lt 1 ] && N=1
    echo "[run_scoring]   models=$NMODELS  shards(N)=$N  GPU/shard=${GRES:-<none>}  MAXTRY=$MAXTRY"

    # one ppi shard job (GPU, self-heal). Reuses the shared index; slices [k::N].
    submit_ppi_shard() {  # $1=tag  $2=shard  $3=nshards
        local sp="$LOG_DIR/stage_$1.sh"
        {
            printf '#!/bin/bash\nset -euo pipefail\nexec'
            for a in "${cmd[@]}"; do printf ' %q' "$a"; done
            printf ' --index %q --shard %q --nshards %q\n' "$IDX" "$2" "$3"
        } > "$sp"
        local JN="${JOBNAME}_$1"
        local EXPORTS="ALL,VENV_PY=$VENV_PY,CMD=bash $sp,GPU_REQUESTED=$GPU_REQUESTED,PARTITION=$PARTITION,TIME=$TIME,GRES=$GRES,MAXTRY=$MAXTRY,ATTEMPT=1,EXCLUDE=,JOBNAME=$JN,LOG_DIR=$LOG_DIR,WAIT_CHAIN=1"
        sbatch --parsable --partition="$PARTITION" --time="$TIME" --job-name="$JN" \
            --cpus-per-task=8 "${GRES_ARG[@]}" \
            --output="${LOG_DIR}/lqa-interface_$1_%j.out" --error="${LOG_DIR}/lqa-interface_$1_%j.err" \
            --export="$EXPORTS" --wrap="bash $WORKER"
    }

    if [ "$N" -le 1 ]; then
        # one whole-pool ppi GPU job (still self-healing) — no merge needed.
        jp=$(submit_ppi_shard ppi 0 1)
        echo "[run_scoring] submitted single ppi job: $jp (GPU, self-heal). xgb_dockq.csv in shard dir? no — N=1 writes <OUT> directly."
        exit 0
    fi
    allj=()
    for ((k=0; k<N; k++)); do
        jp=$(submit_ppi_shard "ppi$k" "$k" "$N"); allj+=("$jp")
        echo "[run_scoring]   ppi shard $k: $jp (GPU, self-heal)"
    done
    dep="afterok:$(IFS=:; echo "${allj[*]}")"
    # merge: concat shards/*/xgb_dockq.csv → <OUT>/xgb_dockq.csv
    msp="$LOG_DIR/stage_ppimerge.sh"
    printf '#!/bin/bash\nset -euo pipefail\nexec %q %q %q\n' \
        "$VENV_PY" "$TOOL_DIR/ppi_merge.py" "$OUTPUT" > "$msp"
    jmerge=$(sbatch --parsable --partition="$PARTITION" --time="$TIME" \
        --job-name="${JOBNAME}_ppimerge" --cpus-per-task=2 --dependency="$dep" \
        --output="${LOG_DIR}/lqa-interface_ppimerge_%j.out" --error="${LOG_DIR}/lqa-interface_ppimerge_%j.err" \
        --export=ALL --wrap="bash $msp")
    echo ""
    echo "[run_scoring] submitted ${#allj[@]} ppi shard jobs + merge=$jmerge"
    echo "  GPU jobs concurrent: $N (each --gres=${GRES:-none}, self-heal on bad GPU/OOM)"
    echo "  merge (afterok) concats shard xgb_dockq.csv → ${OUTPUT%/}/xgb_dockq.csv"
    echo "  monitor: squeue -u \$USER | grep ${JOBNAME}"
    exit 0
fi

# ---- single-job path (--serial, or non-run-all heavy submit) ---------------
EXPORTS="ALL,VENV_PY=$VENV_PY,CMD=${cmd[*]},GPU_REQUESTED=$GPU_REQUESTED,PARTITION=$PARTITION,TIME=$TIME,GRES=$GRES,MAXTRY=$MAXTRY,ATTEMPT=1,EXCLUDE=,JOBNAME=$JOBNAME,LOG_DIR=$LOG_DIR"
SBATCH_OUT=$(sbatch \
    --partition="$PARTITION" \
    --time="$TIME" \
    --job-name="$JOBNAME" \
    --cpus-per-task=8 \
    "${GRES_ARG[@]}" \
    "${NODELIST_ARG[@]}" \
    --output="${LOG_DIR}/lqa-interface_%j.out" \
    --error="${LOG_DIR}/lqa-interface_%j.err" \
    --export="$EXPORTS" \
    --wrap="bash $WORKER")
echo "$SBATCH_OUT"
