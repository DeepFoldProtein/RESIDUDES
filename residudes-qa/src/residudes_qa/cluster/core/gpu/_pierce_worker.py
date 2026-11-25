"""Standalone Pierce-angle worker — runs inside `.venv-cdr/bin/python`.

Invoked as a subprocess by `core/gpu/tcr_angle.py` when angle CSV is
missing. Spawns its own ProcessPoolExecutor for parallel CIF processing.

The wrapper needs:
  - biopython 1.81 + ANARCI on PATH (this is what .venv-cdr provides)
  - the Pierce binary under $RESIDUDES_QA_TCR_TOOLS
                                 tcr_docking_angle/tcr_docking_angle
  - the tcr3d_pierce wrapper under $RESIDUDES_QA_TCR_TOOLS

Usage:
    .venv-cdr/bin/python _pierce_worker.py \\
        --cifs-list <txt one path per line> \\
        --output-csv <path> \\
        --chain-layout ours_5chain \\
        [--target-name <key for HLA offsets>] \\
        [--workers 16]
"""
from __future__ import annotations
import argparse
import csv
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Ensure both the Pierce wrapper and its ANARCI binary are reachable.
# residudes-qa: optional, needs the external Pierce binary; override with $RESIDUDES_QA_TCR_TOOLS.
_TCR_TOOLS = os.environ.get("RESIDUDES_QA_TCR_TOOLS", "")
sys.path.insert(0, _TCR_TOOLS)

# The wrapper looks for ANARCI on PATH; default it to .venv-cdr's bin.
_VENV_CDR_BIN = os.environ.get("RESIDUDES_QA_CDR_BIN", "")
if _VENV_CDR_BIN not in os.environ.get("PATH", ""):
    os.environ["PATH"] = f"{_VENV_CDR_BIN}:{os.environ.get('PATH', '')}"

# Same for the subprocess `python` referenced inside the wrapper's
# subprocess call to `renumber_tcr.py`. We can't override that easily
# without monkey-patching, so do it here so children inherit.

# Patch tcr3d_pierce so the renumber subprocess uses .venv-cdr python.
import tcr3d_pierce  # noqa: E402  (after the sys.path / availability block above)
import subprocess as _sp  # noqa: E402  (after the sys.path / availability block above)

_VENV_PY = os.path.join(_VENV_CDR_BIN, "python")
_orig_run = _sp.run


def _patched_run(cmd, *args, **kwargs):
    if cmd and isinstance(cmd, list) and cmd[0] == "python":
        cmd = [_VENV_PY] + cmd[1:]
    return _orig_run(cmd, *args, **kwargs)


tcr3d_pierce.subprocess.run = _patched_run
# Override the ANARCI bin dir module-global
tcr3d_pierce.ANARCI_BIN_DIR = _VENV_CDR_BIN
# Point at the patched renumber_tcr.py (two-pass id reassignment for
# biopython 1.81 sibling-uniqueness collisions). Original at
# $RESIDUDES_QA_TCR_TOOLS/.../renumber_tcr.py
# is read-only; the patched copy lives in playground.
_PATCHED_RENUMBER = Path(os.environ.get(
    "RESIDUDES_QA_TCR_RENUMBER",
    ""))                      # $RESIDUDES_QA_RENUMBER_TCR: a patched copy, when a site has one
if _PATCHED_RENUMBER.is_file():
    tcr3d_pierce.RENUMBER_SCRIPT = _PATCHED_RENUMBER


def _measure(args_tuple):
    """Worker entry — must be top-level for pickling under spawn."""
    cif_path, chain_layout, target_name = args_tuple
    try:
        c, i = tcr3d_pierce.compute_angles(
            cif_path,
            target=target_name,
            chain_layout=chain_layout,
        )
    except Exception as exc:
        return cif_path, float("nan"), float("nan"), repr(exc)
    return cif_path, c, i, None


def _parse_path_meta(p: Path):
    """Pull (group_dir, predictor, seed, sample) suitable for building the
    canonical pdb_name = ``<predictor>_seed<seed>_sample<sample>.pdb`` that
    the tcr-angle matcher expects (see core/gpu/tcr_angle.py:63-65).

    Supports two layouts:
      1. AF3:        .../<group>/<predictor>/seed_<N>/sample_<M>/model.cif
      2. MassiveFold (flat):   Model_<idx>_<params>_pred_<P>.{pdb,cif}
         → group='mf', predictor='mf', seed=idx, sample=P
         The resulting pdb_name `mf_seed<idx>_sample<P>.pdb` is matched
         against the canonical model_id built by the flat-dir indexer.
    Returns ('', '', 0, 0) on parse failure.
    """
    parts = p.parts
    name = p.name
    # AF3 layout
    if name == "model.cif" and len(parts) >= 5:
        try:
            sample = parts[-2].replace("sample_", "")
            seed = parts[-3].replace("seed_", "")
            predictor = parts[-4]
            group = parts[-5]
            return (group, predictor, int(seed), int(sample))
        except Exception:
            pass
    # MassiveFold flat layout
    import re
    m = re.match(r"^Model_(\d+)_(.*?)(?:_pred_(\d+))?\.(?:pdb|cif)$", name)
    if m:
        idx = int(m.group(1))
        sample = int(m.group(3)) if m.group(3) else 0
        return ("mf", "mf", idx, sample)
    return ("", "", 0, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cifs-list", required=True,
                    help="text file with one CIF/PDB path per line")
    ap.add_argument("--output-csv", required=True)
    ap.add_argument("--chain-layout", default="ours_5chain",
                    choices=["ours_5chain", "ours_4chain"])
    ap.add_argument("--target-name", default=None,
                    help="optional key for HLA_NUM_OFFSETS lookup (e.g. 9RU5)")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    cifs = []
    with open(args.cifs_list) as f:
        for line in f:
            p = line.strip()
            if p:
                cifs.append(Path(p))

    rows = []
    failures = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {
            ex.submit(_measure, (str(c), args.chain_layout, args.target_name)): c
            for c in cifs
        }
        for k, fut in enumerate(as_completed(futs)):
            cif = futs[fut]
            try:
                path, cross, inc, err = fut.result()
            except Exception as exc:
                path, cross, inc, err = str(cif), float("nan"), float("nan"), repr(exc)
            group, predictor, seed, sample = _parse_path_meta(Path(path))
            if err is not None or cross != cross or inc != inc:  # NaN
                failures += 1
            rows.append({
                "group_dir": group,
                "predictor": predictor,
                "seed": seed,
                "sample": sample,
                "crossing_deg": cross,
                "incident_deg": inc,
                "model_path": path,
                "error": err or "",
            })
            if (k + 1) % 200 == 0:
                print(f"[pierce-worker] {k+1}/{len(cifs)} done "
                      f"(failures so far: {failures})", flush=True)

    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[pierce-worker] wrote {len(rows)} rows → {out}  "
          f"(failures: {failures})", flush=True)


if __name__ == "__main__":
    main()
