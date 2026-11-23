#!/usr/bin/env bash
# Build the TensorRT engines for AlphaFold 3: the trunk pairformer (48 blocks) and
# the diffusion token transformer (24 blocks). Same recipe as openfold3/trt/build_engines.sh;
# the weights are first expressed in OpenFold 3 names (af3_to_openfold3.py).
#
#   bash core/alphafold3/trt/build_engines.sh --checkpoint <dir with af3.bin> --engines-dir <out> \
#        [--min-seqlen 4] [--max-seqlen 2048] [--multiplicity 5]
#
# Environment: PYTHON (default python), TRIANGLE_ATTN_BACKEND (VANILLA | CUEQUIV), BUILDER_FORCE_NUM_PROFILES.
set -euo pipefail
log() { echo "$(date '+%Y-%m-%d %H:%M:%S') INFO $*"; }
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
THIRDPARTY="$REPO_ROOT/thirdparty"
MIN_SEQLEN="${MIN_SEQLEN:-4}"
MAX_SEQLEN="${MAX_SEQLEN:-2048}"
ENGINES_DIR="${ENGINES_DIR:-$REPO_ROOT/engines/alphafold3}"
MULTIPLICITY="${MULTIPLICITY:-5}"
CKPT=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --min-seqlen) MIN_SEQLEN="$2"; shift 2 ;;
        --max-seqlen) MAX_SEQLEN="$2"; shift 2 ;;
        --engines-dir) ENGINES_DIR="$2"; shift 2 ;;
        --checkpoint) CKPT="$2"; shift 2 ;;
        --multiplicity) MULTIPLICITY="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done
[[ -n "$CKPT" ]] || { echo "ERROR: --checkpoint <dir with af3.bin> is required"; exit 1; }
[[ -d "$CKPT" ]] || CKPT="$(dirname "$CKPT")"      # a path to af3.bin itself is accepted too
export PYTHONPATH="$THIRDPARTY:$REPO_ROOT:${PYTHONPATH:-}"
PYTHON="${PYTHON:-python}"
TRIANGLE_ATTN_BACKEND="${TRIANGLE_ATTN_BACKEND:-VANILLA}"

log "######################################"
log "# AlphaFold 3 TRT engine build START #"
log "######################################"
log "Weights:      $CKPT"
log "Engines dir:  $ENGINES_DIR"
log "Seq range:    [$MIN_SEQLEN, $MAX_SEQLEN]"
log "Multiplicity: $MULTIPLICITY"
mkdir -p "$ENGINES_DIR"

STATE="$ENGINES_DIR/af3_as_openfold3.pt"
log "0/4 Writing the parameters in OpenFold 3 names → $STATE"
$PYTHON "$SCRIPT_DIR/af3_to_openfold3.py" --model-dir "$CKPT" --out "$STATE"

PAIRFORMER_CKPT_DIR="$ENGINES_DIR/pairformer_safetensors"
log "1/4 Converting pairformer → $PAIRFORMER_CKPT_DIR"
$PYTHON "$SCRIPT_DIR/convert_pairformer.py" --openfold3-state "$STATE" --output_dir "$PAIRFORMER_CKPT_DIR" \
    --triangle_attention_backend "$TRIANGLE_ATTN_BACKEND" --triangle_attn_node_chunk_size 32

TT_CKPT_DIR="$ENGINES_DIR/token_transformer_safetensors"
log "2/4 Converting token transformer → $TT_CKPT_DIR"
$PYTHON "$SCRIPT_DIR/convert_token_transformer.py" --openfold3-state "$STATE" --output_dir "$TT_CKPT_DIR" \
    --multiplicity "$MULTIPLICITY"

PAIRFORMER_ENGINE_DIR="$ENGINES_DIR/pairformer_engine"
log "3/4 Building pairformer TRT engine → $PAIRFORMER_ENGINE_DIR"
BUILDER_FORCE_NUM_PROFILES="${BUILDER_FORCE_NUM_PROFILES:-2}" $PYTHON "$SCRIPT_DIR/build_engine.py" \
    --model openfold3 --module pairformer \
    --checkpoint_dir "$PAIRFORMER_CKPT_DIR" --output_dir "$PAIRFORMER_ENGINE_DIR" \
    --min_seqlen "$MIN_SEQLEN" --max_seqlen "$MAX_SEQLEN" --weakly_dtype bfloat16 --norm_epsilon 0.00001

TT_ENGINE_DIR="$ENGINES_DIR/token_transformer_engine"
log "4/4 Building token transformer TRT engine → $TT_ENGINE_DIR"
BUILDER_FORCE_NUM_PROFILES="${BUILDER_FORCE_NUM_PROFILES:-2}" $PYTHON "$SCRIPT_DIR/build_engine.py" \
    --model openfold3 --module token_transformer \
    --checkpoint_dir "$TT_CKPT_DIR" --output_dir "$TT_ENGINE_DIR" \
    --min_seqlen "$MIN_SEQLEN" --max_seqlen "$MAX_SEQLEN" --weakly_dtype bfloat16 --norm_epsilon 0.00001

rm -f "$STATE"
log "AlphaFold 3 engines ready under $ENGINES_DIR"
