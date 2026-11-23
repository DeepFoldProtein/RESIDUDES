#!/usr/bin/env python3
"""Parity check: lantern-qa interface scorer vs the CASP17 XGBDimerScorer on T1417 (A3).

Three sources, all on the same node in one job:
  ref    targets/T1417/xgb_trimer/  (written 2026-07-08 by XGBDimerScorer_tool, 2,460 models)
  orig   the original XGBDimerScorer_tool + analysis/.venv, run now on the same inputs
  new    lantern-qa (this package, its uv venv), run now — twice for the GPU stages

Comparisons (max |difference| per column, tolerance PARITY_ATOL):
  new  vs orig    is the port faithful?                      (must pass)
  new  vs new2    run-to-run noise of GPU stages (dfire/mpnn)
  orig vs ref     drift of the original tool since 2026-07-08 (models/code updated after the reference)

Stages: A = score the reference master.parquet (all models);
        B = features base/clash/cpu/dfire/mpnn on a stratified subset (PARITY_PER_METHOD per method).

    sbatch -p normal --gres=gpu:1 -c 8 --wrap ".venv/bin/python tests/parity/interface_t1417.py OUT_DIR"
Writes OUT_DIR/parity_summary.json; exit 1 if new-vs-orig fails.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CASP = Path("/gpfs/deepfold/casp/casp17")
REF = CASP / "targets/T1417/xgb_trimer"
ORIG_TOOL = CASP / "analysis/XGBDimerScorer_tool"
ORIG_PY = CASP / "analysis/.venv/bin/python"
STAGES = os.environ.get("PARITY_STAGES", "base,clash,cpu,dfire,mpnn").split(",")
GPU_STAGES = ["dfire", "mpnn"]
ATOL = float(os.environ.get("PARITY_ATOL", "1e-5"))
PER_METHOD = int(os.environ.get("PARITY_PER_METHOD", "6"))


def compare(a: pd.DataFrame, b: pd.DataFrame, key: str = "model_id") -> dict:
    m = a.merge(b, on=key, suffixes=("_a", "_b"))
    cols = [c[:-2] for c in m.columns if c.endswith("_a") and f"{c[:-2]}_b" in m.columns]
    worst, rel = {}, {}
    for c in cols:
        x, y = m[f"{c}_a"], m[f"{c}_b"]
        if not (pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y)):
            continue
        x, y = x.astype(float).to_numpy(), y.astype(float).to_numpy()
        d = np.where(np.isnan(x) & np.isnan(y), 0.0, np.abs(x - y))
        d = np.where(np.isnan(d), np.inf, d)
        worst[c] = float(d.max()) if len(d) else 0.0
        scale = float(np.nanmax(np.abs(y))) if len(y) and np.isfinite(np.nanmax(np.abs(y))) else 0.0
        rel[c] = worst[c] / scale if scale > 0 else worst[c]
    over = {c: {"abs": worst[c], "rel": rel[c]} for c in worst if worst[c] > ATOL}
    return dict(rows=len(m), columns=len(worst), max_abs=max(worst.values()) if worst else 0.0, over_tolerance=over)


def run_new(args: list[str]) -> None:
    from lantern_qa.interface.cli import main as interface
    interface(args)


def run_orig(args: list[str], log: Path) -> int:
    env = dict(os.environ, PYTHONPATH=str(ORIG_TOOL), EMA_PATH=str(CASP / "analysis"))
    for k in list(env):                                   # do not leak lantern-qa data paths into the original tool
        if k.startswith(("EMA_DFIRE", "EMA_BBDEP", "EMA_NDRD", "EMA_AADARS", "EMA_PROTEINMPNN", "LANTERN_QA_")):
            env.pop(k)
    with open(log, "a") as fh:
        return subprocess.call([str(ORIG_PY), "-m", "xgbdimer.cli", *args], cwd=CASP, env=env, stdout=fh, stderr=subprocess.STDOUT)


def stage_parquet(d: Path, st: str) -> pd.DataFrame | None:
    got = sorted(d.rglob(f"{st}_features.parquet"))
    return pd.read_parquet(got[0]) if got else None


def main(out: Path) -> int:
    out.mkdir(parents=True, exist_ok=True)
    os.chdir(CASP)                                        # index src_cif paths are relative to the casp17 root
    log = out / "orig_tool.log"
    S: dict = {"reference": str(REF), "atol": ATOL}

    # ---- A: score the reference master with both tools (PARITY_SKIP_SCORE=1 to skip)
    if os.environ.get("PARITY_SKIP_SCORE") != "1":
        run_new(["score", "--master", str(REF / "master.parquet"), "--mode", "ppi", "-o", str(out / "A_new")])
        rc = run_orig(["score", "--master", str(REF / "master.parquet"), "--mode", "ppi", "-o", str(out / "A_orig")], log)
        ref_s = pd.read_parquet(REF / "xgb_scores.parquet")[["model_id", "xgb_pred"]]
        new_s = pd.read_parquet(out / "A_new" / "xgb_scores.parquet")[["model_id", "xgb_pred"]]
        S["A_new_vs_ref"] = compare(new_s, ref_s)
        if rc == 0:
            orig_s = pd.read_parquet(out / "A_orig" / "xgb_scores.parquet")[["model_id", "xgb_pred"]]
            S["A_new_vs_orig"] = compare(new_s, orig_s)
            S["A_orig_vs_ref"] = compare(orig_s, ref_s)
        else:
            S["A_new_vs_orig"] = {"error": f"original tool exited {rc}, see {log}"}

    # ---- B: features on a stratified subset
    idx = pd.read_parquet(REF / "index.parquet")
    sub = (idx.sort_values(["method", "seed", "sample"]).groupby("method", group_keys=False)
              .apply(lambda g: g.iloc[np.linspace(0, len(g) - 1, min(PER_METHOD, len(g))).astype(int)]))
    sub_index = out / "B_index.parquet"
    sub.to_parquet(sub_index, index=False)
    S["B_models"] = len(sub)
    for st in STAGES:
        run_new(["features", "--index", str(sub_index), "--feature", st, "--workers", "4", "-o", str(out / "B_new")])
        rc = run_orig(["features", "--index", str(sub_index), "--feature", st, "--workers", "4", "-o", str(out / "B_orig")], log)
        new_f, orig_f = stage_parquet(out / "B_new", st), stage_parquet(out / "B_orig", st)
        ref_f = pd.read_parquet(REF / "stage_outputs" / f"{st}_features.parquet")
        S[f"B_{st}_new_vs_orig"] = compare(new_f, orig_f) if (new_f is not None and orig_f is not None) else {"error": f"missing output (orig rc={rc})"}
        S[f"B_{st}_orig_vs_ref"] = compare(orig_f, ref_f[ref_f.model_id.isin(sub.model_id)]) if orig_f is not None else {"error": "no orig"}
        if st in GPU_STAGES:
            run_new(["features", "--index", str(sub_index), "--feature", st, "--workers", "4", "-o", str(out / "B_new2")])
            S[f"B_{st}_new_vs_new2"] = compare(new_f, stage_parquet(out / "B_new2", st))

    def faithful(k: str) -> bool:
        r = S[k]
        if "error" in r:
            return False
        if not r["over_tolerance"]:
            return True
        noise = S.get(k.replace("new_vs_orig", "new_vs_new2"))          # GPU stages: allow run-to-run noise
        return bool(noise) and r["max_abs"] <= max(noise["max_abs"], ATOL) * 1.5
    S["pass"] = all(faithful(k) for k in S if k.endswith("new_vs_orig"))
    (out / "parity_summary.json").write_text(json.dumps(S, indent=1, default=str))
    print(json.dumps(S, indent=1, default=str))
    return 0 if S["pass"] else 1


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]).resolve()))
