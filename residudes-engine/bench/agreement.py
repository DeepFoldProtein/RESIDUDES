#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["tmtools>=0.2", "gemmi>=0.6.5", "numpy>=1.26"]
# ///
"""Structural agreement between backends: TM-score / RMSD of each backend's
top-ranked model against the PyTorch backend's top model for the same input
and model (same weights, same seed).

Reads the benchmark run tree (runs/bench/<target>_<sto>/<model>-<backend>/seed_1
by default, or any --root with the same layout), aligns Cα (protein) / C1'
(nucleic) / first atom per residue otherwise, concatenated over chains in file
order, with TM-align (tmtools). TM-score is normalised by the reference length.

    uv run bench/agreement.py                 # → docs/benchmark/agreement.csv
    uv run bench/agreement.py --root runs/opt/tt --ref-root runs/bench
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import gemmi
import numpy as np
from tmtools import tm_align

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / "bench" / "targets.json").read_text())
MODELS = CFG["models"]
BACKENDS = ["trt", "torch", "jax", "jax-warm", "jax-mem85"]


def rep_atoms(cif: Path) -> tuple[np.ndarray, str]:
    """One representative atom per residue (Cα / C1' / first), plus a 1-letter sequence."""
    st = gemmi.read_structure(str(cif))
    st.setup_entities()
    model = st[0]
    coords, seq = [], []
    for chain in model:
        for res in chain:
            atom = res.find_atom("CA", "*") or res.find_atom("C1'", "*") or (res[0] if len(res) else None)
            if atom is None:
                continue
            coords.append([atom.pos.x, atom.pos.y, atom.pos.z])
            info = gemmi.find_tabulated_residue(res.name)
            seq.append(info.one_letter_code.upper() if info and info.one_letter_code.strip() else "X")
    return np.asarray(coords, dtype=float), "".join(seq)


def top_model(seed_dir: Path) -> Path | None:
    """The sample the writer ranked first (sample_0 in the unified layout)."""
    for cand in ("sample_0/model.cif", "sample_1/model.cif"):
        p = seed_dir / cand
        if p.is_file():
            return p
    return None


def compare(cif_a: Path, cif_b: Path) -> dict:
    xa, sa = rep_atoms(cif_a)
    xb, sb = rep_atoms(cif_b)
    if len(xa) != len(xb) or len(xa) == 0:
        return {"n_res": min(len(xa), len(xb)), "tm_score": None, "rmsd": None,
                "note": f"length mismatch {len(xa)} vs {len(xb)}"}
    res = tm_align(xa, xb, sa, sb)
    # same residues in the same order: RMSD after optimal superposition of all pairs
    ca = xa - xa.mean(0)
    cb = xb - xb.mean(0)
    u, _, vt = np.linalg.svd(ca.T @ cb)
    d = np.sign(np.linalg.det(u @ vt))
    r = u @ np.diag([1, 1, d]) @ vt
    rmsd = float(np.sqrt(((ca @ r - cb) ** 2).sum(1).mean()))
    return {"n_res": len(xa), "tm_score": float(res.tm_norm_chain2), "rmsd": rmsd, "note": ""}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT / "runs" / "bench")
    ap.add_argument("--ref-root", type=Path, default=None, help="where the torch reference runs live (default: --root)")
    ap.add_argument("--ref-backend", default="torch")
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "benchmark" / "agreement.csv")
    ns = ap.parse_args()
    ref_root = ns.ref_root or ns.root
    rows = []
    for t in CFG["targets"]:
        key = f"{t['name']}_{t['sto']}"
        for m in MODELS:
            # the reference is the first-series torch run; a model that joined later (v2 only) uses its v2 torch run
            ref_label = ns.ref_backend
            ref = top_model(ref_root / key / f"{m}-{ref_label}" / f"seed_{CFG['seed']}")
            if ref is None:
                ref_label = f"{ns.ref_backend}-v2"
                ref = top_model(ref_root / key / f"{m}-{ref_label}" / f"seed_{CFG['seed']}")
            if ref is None:
                continue
            # every run dir of this model (trt, jax, jax-warm, trt-v2, jax-v2-warm, …) except the reference
            labels = sorted(p.name[len(m) + 1:] for p in (ns.root / key).glob(f"{m}-*") if p.is_dir()) \
                if (ns.root / key).is_dir() else []
            for b in labels:
                if b == ref_label:
                    continue
                cif = top_model(ns.root / key / f"{m}-{b}" / f"seed_{CFG['seed']}")
                if cif is None:
                    continue
                r = compare(cif, ref)
                rows.append({"target": t["name"], "sto": t["sto"], "tokens": t["tokens"], "model": m,
                             "backend": b, "ref_backend": ref_label, **r})
                print(f"{t['name']:<6} {t['tokens']:>5} {m} {b:<9} vs {ns.ref_backend}: "
                      f"TM {r['tm_score'] if r['tm_score'] is None else round(r['tm_score'], 3)}  "
                      f"RMSD {r['rmsd'] if r['rmsd'] is None else round(r['rmsd'], 2)} Å  {r['note']}")
    ns.out.parent.mkdir(parents=True, exist_ok=True)
    with ns.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["target"])
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} comparisons → {ns.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
