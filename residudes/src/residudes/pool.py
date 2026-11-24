"""`residudes pool T`: read a residudes-engine layout prediction pool into the dossier.

Layout (residudes-engine / CASP17):
  <target_dir>/results/<sto>/<variant>/<runner>/seed_<N>/sample_<M>/summary.json        complexes
  <target_dir>/<chain>/results/<sto>/<variant>/<runner>/seed_<N>/sample_<M>/summary.json monomer chains

Per-chain runs get the stoichiometry label <sto>@<chain> (A1@B), so chains never share model names or z-scores.

Writes <dossier_dir>/pool.csv (one row per model) and returns a summary per stoichiometry x variant x runner:
model count, ranking_score / ipTM / pTM / pLDDT distribution, and each model's z-score of ranking_score
within its runner, robust to outliers and to runners whose scores barely spread (ranking_score is not comparable
across runners: bt2 runs high, ef2 low).
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

FIELDS = ["sto", "variant", "runner", "seed", "sample", "ranking_score", "iptm", "ptm", "plddt", "has_clash", "path"]


def _seed_dirs(root: Path):
    """(stoichiometry label, variant, runner, seed dir) for every seed of the pool; per-chain runs get A1@B."""
    bases = [("", root / "results")] + [(p.name, p / "results") for p in sorted(root.iterdir())
                                        if p.is_dir() and len(p.name) <= 2 and (p / "results").is_dir()]
    for chain, base in bases:
        if not base.is_dir():
            continue
        for sto in sorted(os.scandir(base), key=lambda e: e.name):
            if not sto.is_dir():
                continue
            for var in sorted(os.scandir(sto.path), key=lambda e: e.name):
                if not var.is_dir():
                    continue
                for run in sorted(os.scandir(var.path), key=lambda e: e.name):
                    if not run.is_dir():
                        continue
                    for seed in sorted(os.scandir(run.path), key=lambda e: e.name):
                        if seed.is_dir() and seed.name.startswith("seed_"):
                            yield (f"{sto.name}@{chain}" if chain else sto.name), var.name, run.name, seed.path


def _read_seed(item) -> tuple[list[dict], list[str]]:
    """Every sample of one seed: one directory listing and one summary.json read per sample. Also returns the samples
    whose summary.json exists but whose model.cif is missing or empty."""
    sto, var, run, seed_path = item
    rows, empty = [], []
    try:
        samples = [e for e in os.scandir(seed_path) if e.is_dir() and e.name.startswith("sample_")]
    except OSError:
        return rows, empty
    for smp in samples:
        s = Path(smp.path) / "summary.json"
        try:
            j = json.loads(s.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        try:                                           # a summary without a model (seen on CASP17: 0-byte model.cif)
            if os.stat(Path(smp.path) / "model.cif").st_size == 0:
                raise OSError("empty")
        except OSError:
            empty.append(f"{sto}/{var}/{run}/{Path(seed_path).name}/{smp.name}")
            continue
        rows.append({"sto": sto, "variant": var, "runner": run, "seed": Path(seed_path).name[5:], "sample": smp.name[7:],
                     "ranking_score": j.get("ranking_score"), "iptm": j.get("iptm"), "ptm": j.get("ptm"),
                     "plddt": j.get("plddt"), "has_clash": j.get("has_clash"), "path": str(s.parent / "model.cif")})
    return rows, empty


def _sample_key(r: dict):
    return (0, int(r["sample"]), "") if r["sample"].isdigit() else (1, 0, r["sample"])   # sample_best after sample_4


def scan(root: Path, threads: int = 8, empty: list[str] | None = None) -> list[dict]:
    """Read the pool. The reads are small files on a parallel file system, so they go through a few threads
    (tens of thousands of summary.json files: minutes one by one, about a minute with 8). Rows keep a stable order.
    `residudes pool --threads N` changes it; keep it modest on a shared login node. Models left out for a missing or
    empty model.cif are added, sorted, to `empty` when it is given."""
    from concurrent.futures import ThreadPoolExecutor
    seeds = list(_seed_dirs(root))
    with ThreadPoolExecutor(max_workers=max(1, threads)) as ex:
        parts = list(ex.map(_read_seed, seeds))
    rows = [r for part, _ in parts for r in sorted(part, key=_sample_key)]
    if empty is not None:
        empty.extend(sorted(e for _, es in parts for e in es))
    _zscore(rows)
    return rows


Z_FLOOR = 0.02          # ranking_score differences below a few hundredths are noise, not confidence


def _zscore(rows: list[dict]) -> None:
    """Each model's robust z of ranking_score within its stoichiometry and runner: (x - median) / max(1.4826 MAD,
    Z_FLOOR). Two failure modes of the plain z, both seen on CASP17 pools:
      - a runner whose scores barely spread (sd 0.007) turned a 0.015 difference into z 2.1 and led the slate with a
        uniformly confident but wrong model;
      - a few extreme scores (sd 30-40 from outliers) squashed every other model of that runner to z ~ 0, so it never led.
    The group is the runner, not the runner and input variant: a z per variant (putting templated and template-free inputs
    on equal footing) did worse on CASP17's released natives, model 1 DockQ 0.535 -> 0.503 over 53 complexes and lDDT
    0.783 -> 0.730 over 31 monomers (better on 4 and 1, worse on 10 and 7)."""
    groups: dict[tuple, list[float]] = {}
    for r in rows:
        if isinstance(r["ranking_score"], (int, float)):
            groups.setdefault((r["sto"], r["runner"]), []).append(float(r["ranking_score"]))
    stats = {}
    for k, v in groups.items():
        v = sorted(v)
        med = _median(v)
        mad = _median(sorted(abs(x - med) for x in v)) * 1.4826
        stats[k] = (med, max(mad, Z_FLOOR))
    for r in rows:
        k = (r["sto"], r["runner"])
        r["rs_z_runner"] = round((float(r["ranking_score"]) - stats[k][0]) / stats[k][1], 3) if k in stats and isinstance(r["ranking_score"], (int, float)) else None


def _median(v: list[float]) -> float:
    n = len(v)
    return v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2


def summarize(rows: list[dict], top: int = 3) -> dict:
    out: dict = {}
    for r in rows:
        key = f"{r['sto']}/{r['variant']}/{r['runner']}"
        out.setdefault(key, []).append(r)
    summary = {}
    for key, rs in sorted(out.items()):
        vals = lambda f: sorted(float(x[f]) for x in rs if isinstance(x[f], (int, float)))
        rk, ip = vals("ranking_score"), vals("iptm")
        best = sorted(rs, key=lambda x: -(x["ranking_score"] or -1))[:top]
        summary[key] = {"n": len(rs), "ranking_score_median": rk[len(rk) // 2] if rk else None, "ranking_score_max": rk[-1] if rk else None,
                        "iptm_median": ip[len(ip) // 2] if ip else None, "iptm_max": ip[-1] if ip else None,
                        "top": [f"seed {b['seed']} sample {b['sample']} rs {b['ranking_score']} ipTM {b['iptm']}" for b in best]}
    return summary


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS + ["rs_z_runner"])
        w.writeheader()
        w.writerows(rows)
