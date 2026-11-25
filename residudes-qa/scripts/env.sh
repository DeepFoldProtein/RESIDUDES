# scripts/env.sh — shared paths for the residudes-qa shell wrappers (sourced, not executed).
#
#   LQA_ROOT       the residudes-qa checkout (derived from this file, or exported by the caller —
#                  Slurm copies batch scripts to a spool dir, so sbatch'ed scripts get it via --export)
#   VENV_PY        python of the uv venv: $RESIDUDES_QA_PYTHON or $LQA_ROOT/.venv/bin/python
#   LQA_CDR_BIN    bin dir of the optional ANARCI/HMMER venv (antibody CDR masking, aADARS)
if [ -z "${LQA_ROOT:-}" ]; then
    LQA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
export LQA_ROOT
VENV_PY="${RESIDUDES_QA_PYTHON:-$LQA_ROOT/.venv/bin/python}"
export PYTHONPATH="$LQA_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
LQA_CDR_BIN="${RESIDUDES_QA_CDR_BIN:-$RESIDUDES_QA_CDR_BIN}"
if [ ! -x "$VENV_PY" ]; then
    echo "residudes-qa: no venv python at $VENV_PY — run 'uv sync --extra cu130' (or cu124) in $LQA_ROOT" >&2
    return 2 2>/dev/null || exit 2
fi
