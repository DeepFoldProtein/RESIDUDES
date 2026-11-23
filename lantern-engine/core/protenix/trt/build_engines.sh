#!/bin/bash
# Build TRT engines for Protenix (pairformer + token transformer).
#
# Protenix IS OpenFold3 — uses the openfold3 model name in trtbnm-build.
#
# Usage:
#   export PROTENIX_CKPT=/path/to/protenix.pt
#   export ENGINES_DIR=/workspace/engines
#   bash build_engines.sh [--min-seqlen 4] [--max-seqlen 2048]
#
# Environment variables:
#   PROTENIX_CKPT     Path to Protenix checkpoint (.pt)
#   ENGINES_DIR       Output directory for engines (default: ./engines)
#   MIN_SEQLEN        Minimum sequence length (default: 4)
#   MAX_SEQLEN        Maximum sequence length (default: 2048)
#   MULTIPLICITY      Diffusion multiplicity (default: 5)

set -euo pipefail

log() { echo "$(date '+%Y-%m-%d %H:%M:%S') INFO $*"; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
THIRDPARTY="$REPO_ROOT/thirdparty"

# Defaults
MIN_SEQLEN="${MIN_SEQLEN:-4}"
MAX_SEQLEN="${MAX_SEQLEN:-2048}"
ENGINES_DIR="${ENGINES_DIR:-$REPO_ROOT/engines}"
MULTIPLICITY="${MULTIPLICITY:-5}"
TRI_CHUNK="${TRI_CHUNK:-32}"  # triangle_attn_node_chunk_size for pairformer
NO_TRIMUL_HP="${NO_TRIMUL_HP:-0}"  # 1 = force trimul_high_precision=False

# Parse flags
while [[ $# -gt 0 ]]; do
    case "$1" in
        --min-seqlen) MIN_SEQLEN="$2"; shift 2 ;;
        --max-seqlen) MAX_SEQLEN="$2"; shift 2 ;;
        --engines-dir) ENGINES_DIR="$2"; shift 2 ;;
        --checkpoint) PROTENIX_CKPT="$2"; shift 2 ;;
        --multiplicity) MULTIPLICITY="$2"; shift 2 ;;
        --model-name) MODEL_NAME="$2"; shift 2 ;;
        --tri-chunk) TRI_CHUNK="$2"; shift 2 ;;
        --no-trimul-hp) NO_TRIMUL_HP=1; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [[ -z "${PROTENIX_CKPT:-}" ]]; then
    echo "ERROR: Set PROTENIX_CKPT=/path/to/protenix.pt or use --checkpoint"
    exit 1
fi

export PYTHONPATH="$THIRDPARTY:${PYTHONPATH:-}"

# LD_LIBRARY_PATH: pip-installed native libs (.so) are in non-standard paths
_VENV_SITE="$(dirname "$(dirname "$REPO_ROOT/.venv/bin/python")")"/lib/python3.11/site-packages
if [[ -d "$_VENV_SITE" ]]; then
    _LD_EXTRA="$_VENV_SITE/cuequivariance_ops/lib:$_VENV_SITE/nvidia/cu13/lib:$_VENV_SITE/tensorrt_libs"
else
    _LD_EXTRA=""
fi

# C++ IdentitySZ plugin needs libstdc++ with GLIBCXX_3.4.32.
_STDCXX_DIR=""
for _STDCXX in \
    "$THIRDPARTY/tensorrt_bionemo/libs/libstdc++.so.6" \
    "/opt/conda/lib/libstdc++.so.6" \
    "/usr/lib/gcc/x86_64-linux-gnu/13/libstdc++.so" \
    "/usr/lib/gcc/x86_64-linux-gnu/12/libstdc++.so"; do
    if [[ -f "$_STDCXX" ]] && strings "$_STDCXX" 2>/dev/null | grep "GLIBCXX_3.4.32" >/dev/null 2>&1; then
        export LD_PRELOAD="$_STDCXX"
        _STDCXX_DIR="$(dirname "$_STDCXX")"
        log "Using LD_PRELOAD=$_STDCXX for GLIBCXX_3.4.32"
        break
    fi
done
if [[ -z "${LD_PRELOAD:-}" ]]; then
    log "WARNING: GLIBCXX_3.4.32 not found. C++ IdentitySZ plugin may fail."
    log "  Fix: apt install g++-13, or copy libstdc++.so.6 to $THIRDPARTY/tensorrt_bionemo/libs/"
fi
export LD_LIBRARY_PATH="${_LD_EXTRA}${_STDCXX_DIR:+:$_STDCXX_DIR}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

PYTHON="${PYTHON:-/opt/conda/bin/python3.11}"
PYTHON=python
TRIANGLE_ATTN_BACKEND="${TRIANGLE_ATTN_BACKEND:-VANILLA}"

# CUEQUIV: uses cuequivariance Python TRT plugins (cuequivariance-ops-torch)
#   - No pre-built .so needed; plugins register via tensorrt.plugin API

log "###################################"
log "# Protenix TRT engine build START #"
log "###################################"
log "Checkpoint:  $PROTENIX_CKPT"
log "Engines dir: $ENGINES_DIR"
log "Seq range:   [$MIN_SEQLEN, $MAX_SEQLEN]"
log "Multiplicity: $MULTIPLICITY"
log "Triangle attn node chunk: $TRI_CHUNK"
log "trimul_high_precision: $([ "$NO_TRIMUL_HP" = "1" ] && echo "FALSE (forced)" || echo "auto-by-model-name")"

mkdir -p "$ENGINES_DIR"

# ──────────────────────────────────────────────
# Step 1: Convert pairformer checkpoint
# ──────────────────────────────────────────────
PAIRFORMER_CKPT_DIR="$ENGINES_DIR/pairformer_safetensors"
log "1/4 Converting pairformer checkpoint → $PAIRFORMER_CKPT_DIR"
$PYTHON "$SCRIPT_DIR/convert_pairformer.py" \
    --local_checkpoint "$PROTENIX_CKPT" \
    --output_dir "$PAIRFORMER_CKPT_DIR" \
    --triangle_attention_backend "${TRIANGLE_ATTN_BACKEND:-VANILLA}" \
    --triangle_attn_node_chunk_size "$TRI_CHUNK" \
    $([ "$NO_TRIMUL_HP" = "1" ] && echo "--no_trimul_high_precision") \
    ${MODEL_NAME:+--model_name "$MODEL_NAME"}
log "✓ Pairformer checkpoint converted"

# ──────────────────────────────────────────────
# Step 2: Convert token transformer checkpoint
# ──────────────────────────────────────────────
TT_CKPT_DIR="$ENGINES_DIR/token_transformer_safetensors"
log "2/4 Converting token transformer checkpoint → $TT_CKPT_DIR"
$PYTHON "$SCRIPT_DIR/convert_token_transformer.py" \
    --local_checkpoint "$PROTENIX_CKPT" \
    --output_dir "$TT_CKPT_DIR" \
    --multiplicity "$MULTIPLICITY" \
    ${MODEL_NAME:+--model_name "$MODEL_NAME"}
log "✓ Token transformer checkpoint converted"

# ──────────────────────────────────────────────
# Step 3: Build pairformer TRT engine
# ──────────────────────────────────────────────
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

# ──────────────────────────────────────────────
# Step 4: Build token transformer TRT engine
# ──────────────────────────────────────────────
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

# Patch engine configs: copy fields missing after trtbnm-build and record build metadata
log "Patching engine configs…"
$PYTHON - <<'PYEOF'
import json, datetime
from pathlib import Path
import os

engines_dir = Path(os.environ["ENGINES_DIR"])

def _collect_build_metadata():
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
    except Exception:
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

def patch_dtype_from_pretrained(dst_path):
    with open(dst_path) as f:
        dst = json.load(f)
    pc = dst.get("pretrained_config", {})
    if "dtype" not in dst and "dtype" in pc:
        dst["dtype"] = pc["dtype"]
        with open(dst_path, "w") as f:
            json.dump(dst, f, indent=4)

def patch_build_metadata(dst_path, metadata):
    with open(dst_path) as f:
        dst = json.load(f)
    dst["trt_build_metadata"] = metadata
    with open(dst_path, "w") as f:
        json.dump(dst, f, indent=4)

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

log "###################################"
log "# Protenix TRT engine build DONE  #"
log "###################################"
