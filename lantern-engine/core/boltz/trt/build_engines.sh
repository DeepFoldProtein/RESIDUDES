#!/bin/bash
# Build TRT engines for Boltz2 (structure pairformer + token transformer).
#
# Usage:
#   export BOLTZ2_CKPT=/path/to/boltz2.ckpt
#   export ENGINES_DIR=/workspace/engines
#   bash build_engines.sh [--min-seqlen 4] [--max-seqlen 2048]
#
# Environment variables:
#   BOLTZ2_CKPT      Path to Boltz2 checkpoint
#   ENGINES_DIR      Output directory (default: ./engines)
#   MIN_SEQLEN       Minimum sequence length (default: 4)
#   MAX_SEQLEN       Maximum sequence length (default: 2048)
#   MULTIPLICITY     Diffusion multiplicity (default: 5)
#
# Note: confidence pairformer is built separately if INCLUDE_CONFIDENCE=1 is set.

set -euo pipefail

log() { echo "$(date '+%Y-%m-%d %H:%M:%S') INFO $*"; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
THIRDPARTY="$REPO_ROOT/thirdparty"

MIN_SEQLEN="${MIN_SEQLEN:-4}"
MAX_SEQLEN="${MAX_SEQLEN:-2048}"
ENGINES_DIR="${ENGINES_DIR:-$REPO_ROOT/engines/boltz}"
MULTIPLICITY="${MULTIPLICITY:-5}"
INCLUDE_CONFIDENCE="${INCLUDE_CONFIDENCE:-0}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --min-seqlen) MIN_SEQLEN="$2"; shift 2 ;;
        --max-seqlen) MAX_SEQLEN="$2"; shift 2 ;;
        --engines-dir) ENGINES_DIR="$2"; shift 2 ;;
        --checkpoint) BOLTZ2_CKPT="$2"; shift 2 ;;
        --multiplicity) MULTIPLICITY="$2"; shift 2 ;;
        --with-confidence) INCLUDE_CONFIDENCE=1; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [[ -z "${BOLTZ2_CKPT:-}" ]]; then
    echo "ERROR: Set BOLTZ2_CKPT=/path/to/boltz2.ckpt or use --checkpoint"
    exit 1
fi

export PYTHONPATH="$THIRDPARTY:${PYTHONPATH:-}"
PYTHON="${PYTHON:-/opt/conda/bin/python3.11}"
TRIANGLE_ATTN_BACKEND="${TRIANGLE_ATTN_BACKEND:-VANILLA}"

log "##################################"
log "# Boltz2 TRT engine build START  #"
log "##################################"
log "Checkpoint:  $BOLTZ2_CKPT"
log "Engines dir: $ENGINES_DIR"
log "Seq range:   [$MIN_SEQLEN, $MAX_SEQLEN]"
log "Multiplicity: $MULTIPLICITY"

mkdir -p "$ENGINES_DIR"

# ──────────────────────────────────────────────
# Step 1: Convert structure pairformer
# ──────────────────────────────────────────────
PAIRFORMER_CKPT_DIR="$ENGINES_DIR/pairformer_safetensors"
log "1/4 Converting structure pairformer → $PAIRFORMER_CKPT_DIR"
$PYTHON "$SCRIPT_DIR/convert_pairformer.py" \
    --local_checkpoint "$BOLTZ2_CKPT" \
    --output_dir "$PAIRFORMER_CKPT_DIR" \
    --pairformer_type structure \
    --triangle_attention_backend "${TRIANGLE_ATTN_BACKEND:-VANILLA}" \
    --triangle_attn_node_chunk_size 32
log "✓ Structure pairformer converted"

# ──────────────────────────────────────────────
# Step 2: Convert token transformer
# ──────────────────────────────────────────────
TT_CKPT_DIR="$ENGINES_DIR/token_transformer_safetensors"
log "2/4 Converting token transformer → $TT_CKPT_DIR"
$PYTHON "$SCRIPT_DIR/convert_token_transformer.py" \
    --local_checkpoint "$BOLTZ2_CKPT" \
    --output_dir "$TT_CKPT_DIR" \
    --multiplicity "$MULTIPLICITY"
log "✓ Token transformer converted"

# ──────────────────────────────────────────────
# Step 3: Build structure pairformer engine
# ──────────────────────────────────────────────
PAIRFORMER_ENGINE_DIR="$ENGINES_DIR/pairformer_engine"
log "3/4 Building structure pairformer TRT engine → $PAIRFORMER_ENGINE_DIR"
BUILDER_FORCE_NUM_PROFILES="${BUILDER_FORCE_NUM_PROFILES:-2}" $PYTHON "$SCRIPT_DIR/build_engine.py" \
    --model boltz-2 \
    --module structure_pairformer \
    --checkpoint_dir "$PAIRFORMER_CKPT_DIR" \
    --output_dir "$PAIRFORMER_ENGINE_DIR" \
    --min_seqlen "$MIN_SEQLEN" \
    --max_seqlen "$MAX_SEQLEN" \
    --weakly_dtype bfloat16
log "✓ Structure pairformer engine built"

# ──────────────────────────────────────────────
# Step 4: Build token transformer engine
# ──────────────────────────────────────────────
TT_ENGINE_DIR="$ENGINES_DIR/token_transformer_engine"
log "4/4 Building token transformer TRT engine → $TT_ENGINE_DIR"
BUILDER_FORCE_NUM_PROFILES="${BUILDER_FORCE_NUM_PROFILES:-2}" $PYTHON "$SCRIPT_DIR/build_engine.py" \
    --model boltz-2 \
    --module token_transformer \
    --checkpoint_dir "$TT_CKPT_DIR" \
    --output_dir "$TT_ENGINE_DIR" \
    --min_seqlen "$MIN_SEQLEN" \
    --max_seqlen "$MAX_SEQLEN" \
    --weakly_dtype bfloat16
log "✓ Token transformer engine built"

# ──────────────────────────────────────────────
# Optional: Build confidence pairformer engine
# ──────────────────────────────────────────────
if [[ "$INCLUDE_CONFIDENCE" == "1" ]]; then
    CONF_CKPT_DIR="$ENGINES_DIR/confidence_pairformer_safetensors"
    log "5/5 Converting + building confidence pairformer (optional)"
    $PYTHON "$SCRIPT_DIR/convert_pairformer.py" \
        --local_checkpoint "$BOLTZ2_CKPT" \
        --output_dir "$CONF_CKPT_DIR" \
        --pairformer_type confidence \
        --triangle_attention_backend "${TRIANGLE_ATTN_BACKEND:-VANILLA}" \
        --triangle_attn_node_chunk_size 32

    CONF_ENGINE_DIR="$ENGINES_DIR/confidence_pairformer_engine"
    BUILDER_FORCE_NUM_PROFILES="${BUILDER_FORCE_NUM_PROFILES:-2}" $PYTHON "$SCRIPT_DIR/build_engine.py" \
        --model boltz-2 \
        --module confidence_pairformer \
        --checkpoint_dir "$CONF_CKPT_DIR" \
        --output_dir "$CONF_ENGINE_DIR" \
        --min_seqlen "$MIN_SEQLEN" \
        --max_seqlen "$MAX_SEQLEN" \
        --weakly_dtype bfloat16
    log "✓ Confidence pairformer engine built"
fi

log ""
log "Engine directory contents:"
find "$ENGINES_DIR" -name "*.engine" -o -name "config.json" | sort | sed 's|^|  |'

log "##################################"
log "# Boltz2 TRT engine build DONE   #"
log "##################################"
