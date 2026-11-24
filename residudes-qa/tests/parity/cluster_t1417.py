#!/usr/bin/env python3
"""Parity check: `residudes-qa cluster` vs the CASP17 MassiveClustering `run_clustering.py` on T1417 (A3).

Builds a test target that links seeds 1-PARITY_SEEDS of targets/T1417/results/A3/af3_msa/{af3,of3,pt2}
(the real target is never written to), then on the same GPU node runs:
  orig   analysis/.venv + MassiveClustering_tool/run_clustering.py
  new    residudes-qa (this package)
  new2   residudes-qa again (run-to-run noise)
each with --method tm and its own --output-base, and compares the TM matrix, the structure order
and the cluster labels (exact match and adjusted Rand index).

    sbatch -p normal --gres=gpu:1 -c 8 --wrap ".venv/bin/python tests/parity/cluster_t1417.py OUT_DIR"
Writes OUT_DIR/parity_summary.json; exit 1 if new and orig disagree beyond new-vs-new2 noise.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

CASP = Path("/gpfs/deepfold/casp/casp17")
SRC = CASP / "targets/T1417/results/A3/af3_msa"
ORIG_PY = CASP / "analysis/.venv/bin/python"
ORIG_TOOL = CASP / "analysis/MassiveClustering_tool"
LQA = Path(__file__).resolve().parents[2]
SEEDS = int(os.environ.get("PARITY_SEEDS", "20"))
METHOD = os.environ.get("PARITY_METHOD", "tm")


def build_target(root: Path) -> Path:
    tgt = root / "T1417"
    for model in ("af3", "of3", "pt2"):
        d = tgt / "results/A3/af3_msa" / model
        d.mkdir(parents=True, exist_ok=True)
        for s in range(1, SEEDS + 1):
            src, dst = SRC / model / f"seed_{s}", d / f"seed_{s}"
            if src.is_dir() and not dst.exists():
                dst.symlink_to(src)
    return tgt


def run(tag: str, cmd: list[str], env: dict, out: Path) -> int:
    with open(out / f"{tag}.log", "w") as fh:
        return subprocess.call(cmd, cwd=CASP, env=env, stdout=fh, stderr=subprocess.STDOUT)


def load(base: Path) -> dict:
    mats = sorted(base.rglob("matrices/*_matrix.npy"))
    names = sorted(base.rglob("matrices/structure_names.npy"))
    csvs = [p for p in sorted(base.rglob("results/*clustering.csv")) if "tm_tm" in p.name or p.name.endswith("_clustering.csv")]
    out = {"matrix": np.load(mats[0]) if mats else None,
           "names": [str(x) for x in np.load(names[0], allow_pickle=True)] if names else None,
           "labels": None, "csv": str(csvs[-1]) if csvs else None}
    if csvs:
        df = pd.read_csv(csvs[-1])
        name_col = next((c for c in df.columns if c.lower() in ("structure", "name", "model", "structure_name")), df.columns[0])
        lab_col = next((c for c in df.columns if c.lower() in ("cluster", "cluster_id", "label")), None)
        if lab_col:
            out["labels"] = dict(zip(df[name_col].astype(str), df[lab_col]))
    return out


def compare(a: dict, b: dict) -> dict:
    r: dict = {}
    if a["matrix"] is None or b["matrix"] is None:
        return {"error": "missing matrix"}
    r["n_structures"] = int(a["matrix"].shape[0])
    r["same_order"] = a["names"] == b["names"]
    r["matrix_max_abs"] = float(np.nanmax(np.abs(a["matrix"] - b["matrix"]))) if a["matrix"].shape == b["matrix"].shape else None
    if a["labels"] and b["labels"]:
        keys = sorted(set(a["labels"]) & set(b["labels"]))
        la, lb = [a["labels"][k] for k in keys], [b["labels"][k] for k in keys]
        r["labels_identical"] = la == lb
        r["adjusted_rand"] = float(adjusted_rand_score(la, lb))
        r["n_clusters"] = [len(set(la)), len(set(lb))]
    else:
        r["labels"] = "not found"
    return r


def main(out: Path) -> int:
    out.mkdir(parents=True, exist_ok=True)
    tgt = build_target(out / "target")
    args = [str(tgt), "--sto", "A3", "--method", METHOD, "--gpu", "0", "--no-medoid-pdbs"]
    env_orig = {k: v for k, v in os.environ.items() if not k.startswith(("EMA_", "RESIDUDES_QA_", "PYTHONPATH"))}
    env_orig["PYTHONPATH"] = str(ORIG_TOOL)
    rc = {
        "orig": run("orig", [str(ORIG_PY), str(ORIG_TOOL / "run_clustering.py"), *args, "--output-base", str(out / "orig")], env_orig, out),
        "new": run("new", [str(LQA / "bin/residudes-qa"), "cluster", *args, "--output-base", str(out / "new")], dict(os.environ), out),
        "new2": run("new2", [str(LQA / "bin/residudes-qa"), "cluster", *args, "--output-base", str(out / "new2")], dict(os.environ), out),
    }
    res = {k: load(out / k) for k in rc}
    S = {"target": str(tgt), "seeds": SEEDS, "method": METHOD, "returncodes": rc,
         "new_vs_orig": compare(res["new"], res["orig"]), "new_vs_new2": compare(res["new"], res["new2"]),
         "csv": {k: v["csv"] for k, v in res.items()}}
    a, n = S["new_vs_orig"], S["new_vs_new2"]
    S["pass"] = ("error" not in a and a.get("same_order") and a.get("matrix_max_abs") is not None
                 and a["matrix_max_abs"] <= max(n.get("matrix_max_abs") or 0.0, 1e-6)
                 and (a.get("labels_identical") or a.get("adjusted_rand", 0) >= n.get("adjusted_rand", 1)))
    (out / "parity_summary.json").write_text(json.dumps(S, indent=1, default=str))
    print(json.dumps(S, indent=1, default=str))
    return 0 if S["pass"] else 1


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]).resolve()))
