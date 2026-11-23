#!/bin/bash
# Click-click entry point for MassiveClustering on a CASP17 target.
# Submits a Slurm job by default (so you don't hog a login node's GPU).
#
# Usage:
#   run_clustering.sh <target_dir> --sto A1B1C1D1E1 --method tm
#   run_clustering.sh <target_dir> --sto A1B1C1D1E1 --method itm --cdr
#   run_clustering.sh <target_dir> --sto C1D1 --method drmsd
#   run_clustering.sh --help
#
# Required positional / flag args:
#   <target_dir>            CASP target dir, e.g. targets/H1311
#   --sto SPEC              Stoichiometry: A1B1, A1B1C1D1E1, C1D1, ...
#   --method tm|itm|drmsd|anchored-rmsd|anchored-tm|com|tcr-angle|
#            jaccard-epitope|jaccard-paratope|jaccard-contact|all
#                           Distance metric.
#                             tm/itm/drmsd: standard (alignment region = measurement region).
#                             anchored-rmsd / anchored-tm: Kabsch on --anchor-chains,
#                               then RMSD or TM-score on --probe-chains in that frame.
#                               Requires --anchor-chains and --probe-chains.
#                             jaccard-epitope/-paratope/-contact: binding-mode
#                               distance = 1 - Jaccard over interface residue sets
#                               (epitope=Ag side, paratope=Ab side) or contact
#                               pairs (-contact = Ag+Ab simultaneously). Requires
#                               --ag-chains + --ab-chains. Computed in core
#                               (iface_metrics); already a distance (no 1-x).
#                             all: monomer => tm + drmsd; multimer => tm + itm + drmsd.
#                                  --anchor-chains/--probe-chains 같이 주면
#                                  anchored-rmsd + anchored-tm도 추가됨.
#
# Optional run flags (forwarded to the driver):
#   --ag-chains SPEC        Antigen (epitope) chains for jaccard-* (e.g. 'AB').
#   --ab-chains SPEC        Antibody (paratope) chains for jaccard-* (e.g. 'CD').
#   --iface-cutoff Å        Heavy-atom contact cutoff for jaccard-* (default 5.0).
#   --anchor-chains SPEC    Chain set used for Kabsch (e.g. 'CD' or 'C,D').
#                           Required for --method anchored-*.
#   --probe-chains SPEC     Chain set whose distance is measured after alignment
#                           (e.g. 'E'). May reuse --anchor-chains when
#                           --anchor-mode/--probe-mode select disjoint residue
#                           subsets (e.g. framework vs cdr).
#                           Required for --method anchored-*.
#   --anchor-mode MODE      all|framework|cdr|cdr1|cdr2|cdr3 — residue subset
#                           within --anchor-chains. 'framework' = chain minus
#                           IMGT CDR (ANARCI). 'cdr' = all IMGT CDR. cdr1/2/3
#                           = individual loops. Default 'all'.
#                           Non-'all' requires the ANARCI venv ($LANTERN_QA_CDR_BIN).
#   --probe-mode MODE       Same options, applied to --probe-chains.
#   --anchor-spec SPEC      Mixed-mode replacement for --anchor-chains/-mode:
#                           'AB:cdr+E' = AB CDR + E all chain.
#                           'A:cdr1+B:cdr3+E' = A CDR1 + B CDR3 + E all.
#                           Mutually exclusive with --anchor-chains.
#   --probe-spec SPEC       Same syntax, applied to probe.
#   --cdr                   Drop IMGT CDR1/2/3 residues per chain (ANARCI).
#                           Chains where ANARCI finds no Ig/TCR domain are
#                           kept intact. Requires the ANARCI venv ($LANTERN_QA_CDR_BIN).
#                           Not allowed with anchored-* methods (use --anchor-mode/--probe-mode).
#   --variant NAME          MSA variant subdir (default: af3_msa)
#   --cluster-method NAME   fuzzy|kmeans|hierarchical|leiden (default: fuzzy)
#   --skip-existing         Reuse cached matrix on re-run
#   --no-medoid-pdbs        Skip CIF->PDB export for consensus/medoid output
#   --keep-pdb-view         Retain the intermediate _pdb_view/ folder
#                           (default: deleted at end of run to save disk).
#   --allow-mismatch        Continue even when input shapes differ; keeps the
#                           dominant shape only. Default: abort with diagnostic.
#   --target-name NAME      Override target label in output filenames.
#   --list                  Discover structures and exit (no work, no Slurm).
#
# Slurm behaviour (set on this wrapper, not the driver):
#   --partition NAME        Slurm partition (default: normal)
#   --time HH:MM:SS         Walltime (default: 8:00:00)
#   --local                 Run directly here, do NOT submit to Slurm.
#                           Useful for a quick `--list` check or debugging.
#
# Output (mirrors inferences/visualization):
#   <target>/cluster_results/<sto>/<variant>/<method>[_cdr]/
#     matrices/        <method>_matrix.npy + structure_names.npy
#     ranking.csv      summary.json -> per-prediction confidence
#     results/         massivecluster output: clustering CSV, t-SNE/UMAP PNGs,
#                      consensus CIF, medoid PDBs, per-cluster confidence stats
#     slurm.out / .err Slurm logs (when run via sbatch)

set -euo pipefail

TOOL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # scripts/cluster
source "$TOOL_DIR/../env.sh"                                # lantern-qa: VENV_PY, LQA_ROOT, PYTHONPATH
VENV_PYTHON="$VENV_PY"
DRIVER="$LQA_ROOT/src/lantern_qa/cluster/run_clustering.py"
SBATCH_FILE="$TOOL_DIR/run_clustering.sbatch"

PARTITION="${PARTITION:-normal}"
TIME="${TIME:-8:00:00}"
LOCAL=0
LIST=0
LOG_DIR=""
JOB_NAME_OVERRIDE=""

# Pull Slurm-controlling flags out of $@ before forwarding the rest.
forward=()
while [ $# -gt 0 ]; do
    case "$1" in
        -h|--help)
            sed -n '2,/^set -/p' "$0" | sed 's/^# \{0,1\}//; $d'
            exit 0
            ;;
        --local)     LOCAL=1; shift ;;
        --partition) PARTITION=$2; shift 2 ;;
        --time)      TIME=$2; shift 2 ;;
        --log-dir)   LOG_DIR=$2; shift 2 ;;
        --job-name)  JOB_NAME_OVERRIDE=$2; shift 2 ;;
        --list)      LIST=1; forward+=("$1"); shift ;;
        *)           forward+=("$1"); shift ;;
    esac
done

if [ ! -x "$VENV_PYTHON" ]; then
    cat >&2 <<EOF
error: $VENV_PYTHON not found.
       Run one-time setup first:
           cd $LQA_ROOT && uv sync --extra cu130
       (or with CDR support:)
           cd $LQA_ROOT && uv sync --extra cu130   # + CDR venv: see README
EOF
    exit 1
fi

[ -f "$DRIVER" ]      || { echo "error: driver missing: $DRIVER" >&2; exit 1; }
[ -f "$SBATCH_FILE" ] || { echo "error: sbatch file missing: $SBATCH_FILE" >&2; exit 1; }

# --method all: expand into one sbatch per metric. Monomers (single chain
# total — ignoring copies, e.g. A2 = 2 chains) get tm + drmsd; multimers
# additionally get itm. Done before --list so that listing under
# --method all just shows discovery once (per first method).
method_idx=-1
for ((i=0; i<${#forward[@]}; i++)); do
    if [ "${forward[i]}" = "--method" ]; then method_idx=$i; break; fi
done
if [ "$method_idx" -ge 0 ] && [ "${forward[$((method_idx+1))]}" = "all" ]; then
    sto_val=""
    has_anchor=0; has_probe=0; has_tcr_csv=0
    for ((i=0; i<${#forward[@]}; i++)); do
        if [ "${forward[i]}" = "--sto" ]; then sto_val="${forward[$((i+1))]}"; fi
        if [ "${forward[i]}" = "--anchor-chains" ] || [ "${forward[i]}" = "--anchor-spec" ]; then has_anchor=1; fi
        if [ "${forward[i]}" = "--probe-chains"  ] || [ "${forward[i]}" = "--probe-spec"  ]; then has_probe=1;  fi
        if [ "${forward[i]}" = "--tcr-angle-csv" ]; then has_tcr_csv=1; fi
    done
    chain_count=$(echo "$sto_val" | grep -oE '[A-Z]([0-9]+)' | awk -F'[A-Z]' '{s+=$2} END {print s+0}')
    if [ -z "$chain_count" ] || [ "$chain_count" -le 1 ]; then
        methods=(tm drmsd)
    else
        methods=(tm itm drmsd)
    fi
    # anchored-* + com only when anchor/probe both supplied (com is also
    # anchored — uses the same anchor for Kabsch, probe chains for COM L2)
    if [ "$has_anchor" = "1" ] && [ "$has_probe" = "1" ]; then
        methods+=(anchored-rmsd anchored-tm com)
    fi
    # tcr-angle opt-in via --tcr-angle-csv (consumes pre-computed Pierce
    # CSV, or auto-runs Pierce via .venv-cdr/bin/python).
    if [ "$has_tcr_csv" = "1" ]; then
        methods+=(tcr-angle)
    fi
    if [ "$LIST" = "1" ]; then
        # discovery is method-independent — show it once with the first method
        forward[$((method_idx+1))]="${methods[0]}"
        echo "(--method all → would expand to: ${methods[*]})"
        exec "$VENV_PYTHON" "$DRIVER" "${forward[@]}"
    fi
    echo "expanding --method all → [${methods[*]}] (sto=$sto_val, chains=$chain_count)"
    for m in "${methods[@]}"; do
        forward[$((method_idx+1))]="$m"
        bash "$0" "${forward[@]}" \
            ${PARTITION:+--partition "$PARTITION"} \
            ${TIME:+--time "$TIME"} \
            ${LOG_DIR:+--log-dir "$LOG_DIR"}
    done
    exit 0
fi

# --list / --local: run directly (no GPU work for --list anyway).
if [ "$LIST" = "1" ] || [ "$LOCAL" = "1" ]; then
    exec "$VENV_PYTHON" "$DRIVER" "${forward[@]}"
fi

# Build a job name from target + sto + method (best-effort, falls back to "cluster")
JOB_NAME="cluster"
if [ -n "$JOB_NAME_OVERRIDE" ]; then
    JOB_NAME="$JOB_NAME_OVERRIDE"
else
    target_arg=""; sto_arg=""; method_arg=""; cdr_arg=""
    anchor_arg=""; probe_arg=""; anchor_mode=""; probe_mode=""
    anchor_spec=""; probe_spec=""
    prev=""
    for a in "${forward[@]}"; do
        case "$prev" in
            --sto)            sto_arg=$a ;;
            --method)         method_arg=$a ;;
            --anchor-chains)  anchor_arg=$a ;;
            --probe-chains)   probe_arg=$a ;;
            --anchor-mode)    anchor_mode=$a ;;
            --probe-mode)     probe_mode=$a ;;
            --anchor-spec)    anchor_spec=$a ;;
            --probe-spec)     probe_spec=$a ;;
        esac
        case "$a" in
            --cdr) cdr_arg="cdr" ;;
            --*)   ;;
            *)     [ -z "$target_arg" ] && target_arg=$a ;;
        esac
        prev=$a
    done
    target_base=$(basename "${target_arg:-}")
    # spec-to-tag for jobname: '+'→'_', ':'→'-', strip nothing else.
    # Bash mini-version of run_clustering.py's spec_to_tag — kept simple,
    # the driver writes the canonical tag into the matrices/anchor_probe.json.
    sanitise_spec() { echo "$1" | sed -e 's/+/_/g' -e 's/:/-/g'; }
    if [[ "$method_arg" == anchored-* ]]; then
        if [ -n "$anchor_spec" ]; then
            a_part=$(sanitise_spec "$anchor_spec")
        else
            a_part="$anchor_arg"
            [ -n "$anchor_mode" ] && [ "$anchor_mode" != "all" ] && a_part="$anchor_arg-$anchor_mode"
        fi
        if [ -n "$probe_spec" ]; then
            p_part=$(sanitise_spec "$probe_spec")
        else
            p_part="$probe_arg"
            [ -n "$probe_mode" ] && [ "$probe_mode" != "all" ] && p_part="$probe_arg-$probe_mode"
        fi
        JOB_NAME="cl-${target_base:-tgt}-${sto_arg:-sto}-${method_arg}-${a_part}-${p_part}"
    else
        JOB_NAME="cl-${target_base:-tgt}-${sto_arg:-sto}-${method_arg:-m}${cdr_arg:+-$cdr_arg}"
    fi
fi

# Default log directory: alongside outputs so logs travel with the run.
# We need target_dir resolved for that — fallback to $PWD if we couldn't parse.
if [ -z "$LOG_DIR" ]; then
    if [ -n "${target_arg:-}" ] && [ -d "$target_arg" ]; then
        LOG_DIR="$(readlink -f "$target_arg")/cluster_results/_slurm_logs"
    else
        LOG_DIR="$PWD/cluster_logs"
    fi
fi
mkdir -p "$LOG_DIR"

JID=$(sbatch --parsable \
    -J "$JOB_NAME" \
    -p "$PARTITION" \
    -t "$TIME" \
    -o "$LOG_DIR/%x.%j.out" \
    -e "$LOG_DIR/%x.%j.err" \
    --export=ALL,LQA_ROOT="$LQA_ROOT" \
    "$SBATCH_FILE" "${forward[@]}")

echo "submitted: jobid=$JID  name=$JOB_NAME  partition=$PARTITION  time=$TIME"
echo "  logs:    $LOG_DIR/$JOB_NAME.$JID.{out,err}"
echo "  watch:   squeue -u \$USER -o '%.10i %.30j %.2t %.10M %R'  |  tail -f $LOG_DIR/$JOB_NAME.$JID.out"
