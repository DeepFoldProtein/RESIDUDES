#!/bin/bash
# Build TRT engines for OpenFold3 (pairformer + token transformer).
#
# Usage:
#   export OPENFOLD3_CKPT=/path/to/openfold3.pt
#   export ENGINES_DIR=/workspace/engines
#   bash build_engines.sh [--min-seqlen 4] [--max-seqlen 2048]
#
# Environment variables:
#   OPENFOLD3_CKPT   Path to OpenFold3 checkpoint
#   ENGINES_DIR      Output directory (default: ./engines/openfold3)
#   MIN_SEQLEN       Minimum sequence length (default: 4)
#   MAX_SEQLEN       Maximum sequence length (default: 2048)
#   MULTIPLICITY     Diffusion multiplicity (default: 5)

set -euo pipefail

log() { echo "$(date '+%Y-%m-%d %H:%M:%S') INFO $*"; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
THIRDPARTY="$REPO_ROOT/thirdparty"

MIN_SEQLEN="${MIN_SEQLEN:-4}"
MAX_SEQLEN="${MAX_SEQLEN:-2048}"
ENGINES_DIR="${ENGINES_DIR:-$REPO_ROOT/engines/openfold3}"
MULTIPLICITY="${MULTIPLICITY:-5}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --min-seqlen) MIN_SEQLEN="$2"; shift 2 ;;
        --max-seqlen) MAX_SEQLEN="$2"; shift 2 ;;
        --engines-dir) ENGINES_DIR="$2"; shift 2 ;;
        --checkpoint) OPENFOLD3_CKPT="$2"; shift 2 ;;
        --multiplicity) MULTIPLICITY="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [[ -z "${OPENFOLD3_CKPT:-}" ]]; then
    echo "ERROR: Set OPENFOLD3_CKPT=/path/to/openfold3.pt or use --checkpoint"
    exit 1
fi

export PYTHONPATH="$THIRDPARTY:$REPO_ROOT:${PYTHONPATH:-}"
#PYTHON="${PYTHON:-/opt/conda/bin/python3.11}"
PYTHON=python

TRIANGLE_ATTN_BACKEND="${TRIANGLE_ATTN_BACKEND:-VANILLA}"

# ── backend validation ────────────────────────────────────────────────────
# VANILLA: no extra deps, works without TRT plugin .so
# CUEQUIV: uses cuequivariance Python TRT plugins (cuequivariance-ops-torch)
#   - No pre-built .so needed; plugins register via tensorrt.plugin API

if [[ "$TRIANGLE_ATTN_BACKEND" == "CUEQUIV" ]]; then
    CUDA_MAJOR=$("$PYTHON" -c "import torch; v=torch.version.cuda; print(v.split('.')[0])" 2>/dev/null || echo "12")
    CUEQ_PKG="cuequivariance-ops-torch-cu${CUDA_MAJOR}"
    log "CUEQUIV backend: checking ${CUEQ_PKG}…"
    if ! "$PYTHON" -c "import cuequivariance_ops_torch" &>/dev/null 2>&1; then
        log "Installing ${CUEQ_PKG}…"
        "$PYTHON" -m pip install "${CUEQ_PKG}" -q
    fi
fi

log "####################################"
log "# OpenFold3 TRT engine build START #"
log "####################################"
log "Checkpoint:   $OPENFOLD3_CKPT"
log "Engines dir:  $ENGINES_DIR"
log "Seq range:    [$MIN_SEQLEN, $MAX_SEQLEN]"
log "Multiplicity: $MULTIPLICITY"

mkdir -p "$ENGINES_DIR"

PAIRFORMER_CKPT_DIR="$ENGINES_DIR/pairformer_safetensors"
log "1/4 Converting pairformer → $PAIRFORMER_CKPT_DIR"
$PYTHON "$SCRIPT_DIR/convert_pairformer.py" \
    --local_checkpoint "$OPENFOLD3_CKPT" \
    --output_dir "$PAIRFORMER_CKPT_DIR" \
    --triangle_attention_backend "${TRIANGLE_ATTN_BACKEND:-VANILLA}" \
    --triangle_attn_node_chunk_size 32
log "✓ Pairformer converted"

TT_CKPT_DIR="$ENGINES_DIR/token_transformer_safetensors"
log "2/4 Converting token transformer → $TT_CKPT_DIR"
$PYTHON "$SCRIPT_DIR/convert_token_transformer.py" \
    --local_checkpoint "$OPENFOLD3_CKPT" \
    --output_dir "$TT_CKPT_DIR" \
    --multiplicity "$MULTIPLICITY"
log "✓ Token transformer converted"

PAIRFORMER_ENGINE_DIR="$ENGINES_DIR/pairformer_engine"
log "3/4 Building pairformer TRT engine → $PAIRFORMER_ENGINE_DIR"
BUILDER_FORCE_NUM_PROFILES="${BUILDER_FORCE_NUM_PROFILES:-2}" $PYTHON "$SCRIPT_DIR/build_engine.py" \
    --model openfold3 \
    --module pairformer \
    --checkpoint_dir "$PAIRFORMER_CKPT_DIR" \
    --output_dir "$PAIRFORMER_ENGINE_DIR" \
    --min_seqlen "$MIN_SEQLEN" \
    --max_seqlen "$MAX_SEQLEN" \
    --weakly_dtype bfloat16 \
    --norm_epsilon 0.001
log "✓ Pairformer engine built"

TT_ENGINE_DIR="$ENGINES_DIR/token_transformer_engine"
log "4/4 Building token transformer TRT engine → $TT_ENGINE_DIR"
BUILDER_FORCE_NUM_PROFILES="${BUILDER_FORCE_NUM_PROFILES:-2}" $PYTHON "$SCRIPT_DIR/build_engine.py" \
    --model openfold3 \
    --module token_transformer \
    --checkpoint_dir "$TT_CKPT_DIR" \
    --output_dir "$TT_ENGINE_DIR" \
    --min_seqlen "$MIN_SEQLEN" \
    --max_seqlen "$MAX_SEQLEN" \
    --weakly_dtype bfloat16 \
    --norm_epsilon 0.001
log "✓ Token transformer engine built"

# Patch engine configs: copy fields missing after trtbnm-build
# (support_batch, multiplicity etc. are not forwarded by the builder CLI)
# Also records GPU / TRT / CUDA metadata in each engine config.
log "Patching engine configs with missing fields from safetensors configs…"
$PYTHON - <<'PYEOF'
import json, sys, datetime
from pathlib import Path
import os

engines_dir = Path(os.environ["ENGINES_DIR"])

def _collect_build_metadata():
    """Collect GPU, CUDA, and TRT version info at build time."""
    meta = {"build_timestamp": datetime.datetime.utcnow().isoformat() + "Z"}
    try:
        import torch
        meta["cuda_version"] = torch.version.cuda or "unknown"
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            meta["gpu_name"] = props.name
            meta["compute_capability"] = f"{props.major}.{props.minor}"
        else:
            meta["gpu_name"] = "N/A"
            meta["compute_capability"] = "N/A"
    except Exception as e:
        meta["torch_error"] = str(e)
    try:
        import tensorrt as trt
        meta["trt_version"] = trt.__version__
    except Exception as e:
        meta["trt_version"] = "unknown"
    return meta

def patch(src_path, dst_path, fields):
    with open(src_path) as f:
        src = json.load(f)
    with open(dst_path) as f:
        dst = json.load(f)
    changed = {}
    for k in fields:
        if k in src and k not in dst:
            dst[k] = src[k]
            changed[k] = src[k]
    if changed:
        with open(dst_path, "w") as f:
            json.dump(dst, f, indent=4)
        print(f"  Patched {dst_path.name}: {changed}")
    else:
        print(f"  {dst_path.name}: nothing to patch")

def patch_dtype_from_pretrained(dst_path):
    """Promote dtype from pretrained_config to top-level (required by torch_dtype property)."""
    with open(dst_path) as f:
        dst = json.load(f)
    pc = dst.get("pretrained_config", {})
    if "dtype" not in dst and "dtype" in pc:
        dst["dtype"] = pc["dtype"]
        with open(dst_path, "w") as f:
            json.dump(dst, f, indent=4)
        print(f"  Promoted dtype={pc['dtype']} to top-level in {Path(dst_path).name}")

def patch_build_metadata(dst_path, metadata):
    """Write build-time GPU/TRT metadata into trt_build_metadata field."""
    with open(dst_path) as f:
        dst = json.load(f)
    dst["trt_build_metadata"] = metadata
    with open(dst_path, "w") as f:
        json.dump(dst, f, indent=4)
    print(f"  Wrote trt_build_metadata to {Path(dst_path).name}: {metadata}")

build_meta = _collect_build_metadata()
print(f"  Build metadata: {build_meta}")

patch(
    engines_dir / "pairformer_safetensors/trt/config.json",
    engines_dir / "pairformer_engine/trt/config.json",
    ["support_batch", "triangle_attention_backend"],
)
patch_dtype_from_pretrained(engines_dir / "pairformer_engine/trt/config.json")
patch_build_metadata(engines_dir / "pairformer_engine/trt/config.json", build_meta)

patch(
    engines_dir / "token_transformer_safetensors/trt/config.json",
    engines_dir / "token_transformer_engine/trt/config.json",
    ["support_batch", "multiplicity", "pairwise_attn_backend"],
)
patch_dtype_from_pretrained(engines_dir / "token_transformer_engine/trt/config.json")
patch_build_metadata(engines_dir / "token_transformer_engine/trt/config.json", build_meta)
PYEOF
log "✓ Engine configs patched"

log ""
log "Engine directory contents:"
find "$ENGINES_DIR" -name "*.engine" -o -name "config.json" | sort | sed 's|^|  |'

log "####################################"
log "# OpenFold3 TRT engine build DONE  #"
log "####################################"
