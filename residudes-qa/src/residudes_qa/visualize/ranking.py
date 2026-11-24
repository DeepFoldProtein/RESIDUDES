"""DataFrame aggregation for ranking and summary CSV export."""

from __future__ import annotations

from typing import Optional

import pandas as pd

from .reader import PredictionRun, SampleFiles, load_all_ranking_scores, load_summary_confidences
from .metrics import compute_plddt_stats, residue_plddt


def build_ranking_dataframe(run: PredictionRun) -> pd.DataFrame:
    """Per-sample ranking with quality metrics for model selection.

    Reads ranking_score, ptm, iptm, fraction_disordered, has_clash, gpde
    from each sample's summary.json.
    """
    rows = []
    for sf in run.samples:
        summary = load_summary_confidences(sf.summary_path)
        rows.append({
            "stoich":              run.stoich,
            "msa_variant":         run.msa_variant,
            "model":               run.model,
            "label":               run.label,
            "seed":                sf.seed,
            "sample":              sf.sample,
            "ranking_score":       summary.get("ranking_score"),
            "ptm":                 summary.get("ptm"),
            "iptm":                summary.get("iptm"),
            "fraction_disordered": summary.get("fraction_disordered"),
            "has_clash":           summary.get("has_clash"),
            "inter_pde_mean":      summary.get("gpde"),
        })

    df = pd.DataFrame(rows)
    for col in ["iptm", "inter_pde_mean"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def build_per_seed_summary(run: PredictionRun) -> pd.DataFrame:
    """One row per seed: average metrics across all samples in that seed."""
    rows = []
    for seed_dir in run.seed_dirs:
        seed_num = int(seed_dir.name.split("_")[1])
        summaries = []
        for sf in run.samples:
            if sf.seed == seed_num:
                try:
                    summaries.append(load_summary_confidences(sf.summary_path))
                except Exception:
                    pass
        if not summaries:
            continue
        # Use best sample's ranking_score for this seed
        best = max(summaries, key=lambda s: float(s.get("ranking_score", 0.0)))
        rows.append({
            "stoich":              run.stoich,
            "msa_variant":         run.msa_variant,
            "model":               run.model,
            "label":               run.label,
            "seed":                seed_num,
            "iptm":                best.get("iptm"),
            "ptm":                 best.get("ptm"),
            "ranking_score":       best.get("ranking_score"),
            "fraction_disordered": best.get("fraction_disordered"),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["iptm"] = pd.to_numeric(df["iptm"], errors="coerce")
    return df


def find_global_best(ranking_df: pd.DataFrame) -> tuple[int, int, float]:
    """Return (seed, sample, ranking_score) of the globally best sample."""
    idx = ranking_df["ranking_score"].idxmax()
    row = ranking_df.loc[idx]
    return int(row["seed"]), int(row["sample"]), float(row["ranking_score"])
