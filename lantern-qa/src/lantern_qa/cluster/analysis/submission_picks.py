"""Reference submission picker — RRF fusion + diversity-aware selection.

A **starting point**, not the final word: it fuses several quality scores by
Reciprocal Rank Fusion (scale-free, no per-metric normalization or weight
tuning) and selects N models balancing cluster diversity with quality. Inspect
the fusion table and adjust before trusting the picks.

RRF
---
Each model is ranked by every score column; the fused score is

    RRF(m) = Σ_metrics  1 / (k + rank_metric(m))          (k default 60)

Higher RRF = consistently good across metrics. Robust to scale and outliers,
which is why it suits heterogeneous scorers (xgb / ipTM / pDockQ / mpnn / dfire)
that disagree on calibration.

Selection strategies
--------------------
* ``hybrid`` (default) — round-robin: take the best-RRF unused model from each
  cluster (clusters ordered by their own best RRF), looping until N picked.
  Diversity first, quality within. This is what H1335's manual pick approximated.
* ``top_rrf`` — simply the top-N by RRF (may concentrate in one cluster).
* ``per_cluster`` — one per cluster, best RRF, no wrap (good when #clusters ≈ N).
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


def compute_rrf(
    df: pd.DataFrame,
    score_cols: Iterable[str],
    *,
    k: int = 60,
    lower_is_better: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Add per-metric rank columns + an ``rrf_score`` column.

    Parameters
    ----------
    score_cols
        Columns to fuse. Missing columns are skipped with a warning.
    k
        RRF damping constant (standard default 60).
    lower_is_better
        Subset of ``score_cols`` where a *lower* value is better (e.g. dfire,
        clash counts). Everything else is treated as higher-is-better.
    """
    lower = set(lower_is_better or [])
    out = df.copy()
    used = []
    rrf = np.zeros(len(out), dtype=np.float64)
    for col in score_cols:
        if col not in out.columns:
            print(f"  [rrf] skip missing column: {col}")
            continue
        vals = out[col]
        # rank 1 = best. ascending for lower-is-better, else descending.
        ascending = col in lower
        rank = vals.rank(method="min", ascending=ascending, na_option="bottom")
        out[f"rank_{col}"] = rank.astype(int)
        rrf += 1.0 / (k + rank.to_numpy())
        used.append(col)
    if not used:
        raise ValueError("no usable score columns for RRF")
    out["rrf_score"] = rrf
    out["rrf_n_metrics"] = len(used)
    print(f"  [rrf] fused {len(used)} metrics: {used}")
    return out


def select_picks(
    df: pd.DataFrame,
    *,
    n: int = 5,
    cluster_col: str = "cluster",
    rrf_col: str = "rrf_score",
    strategy: str = "hybrid",
) -> pd.DataFrame:
    """Return the N selected rows with a ``pick_rank`` + ``pick_reason`` column.

    ``df`` should already be filtered to the eligible (sane) pool and have an
    ``rrf_score`` column from :func:`compute_rrf`.
    """
    work = df.sort_values(rrf_col, ascending=False).reset_index(drop=True)

    if strategy == "top_rrf":
        picks = work.head(n).copy()
        picks["pick_reason"] = "global RRF top-" + (picks.index + 1).astype(str)

    elif strategy in ("hybrid", "per_cluster"):
        # order clusters by their best RRF
        cluster_best = (work.groupby(cluster_col)[rrf_col].max()
                        .sort_values(ascending=False))
        cluster_order = list(cluster_best.index)
        # per cluster, models sorted by RRF
        by_cluster = {c: work[work[cluster_col] == c].reset_index(drop=True)
                      for c in cluster_order}
        chosen_idx = []  # (cluster, local_row_position)
        round_i = 0
        while len(chosen_idx) < n:
            progressed = False
            for c in cluster_order:
                if round_i < len(by_cluster[c]):
                    chosen_idx.append((c, round_i))
                    progressed = True
                    if len(chosen_idx) >= n:
                        break
            if strategy == "per_cluster":
                break          # one round only
            if not progressed:
                break          # exhausted all clusters
            round_i += 1
        rows = []
        for rank, (c, pos) in enumerate(chosen_idx, 1):
            r = by_cluster[c].iloc[pos].copy()
            r["pick_reason"] = (f"cluster {c} #{pos+1} by RRF"
                                + (" (diversity)" if pos == 0 else " (fill)"))
            rows.append(r)
        picks = pd.DataFrame(rows)
    else:
        raise ValueError(f"unknown strategy: {strategy}")

    picks = picks.reset_index(drop=True)
    picks.insert(0, "pick_rank", picks.index + 1)
    return picks


def run_submission_picks(
    df: pd.DataFrame,
    output_dir: str | Path,
    *,
    score_cols: Iterable[str],
    n: int = 5,
    cluster_col: str = "cluster",
    model_col: str = "model_id",
    cif_col: Optional[str] = "src_cif",
    strategy: str = "hybrid",
    k: int = 60,
    lower_is_better: Optional[Iterable[str]] = None,
    sanity_cols: Optional[Iterable[str]] = None,
    copy_cif: bool = False,
) -> dict:
    """End-to-end: RRF table + picks + (optional) CIF copies.

    ``sanity_cols`` — boolean columns that must all be True/falsey-False to keep
    a model (e.g. ``has_atom_outlier`` should be False → pass it and we drop
    rows where it is True). Pass column names; we drop rows failing any.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pool = df.copy()
    n_before = len(pool)
    if sanity_cols:
        for col in sanity_cols:
            if col in pool.columns:
                pool = pool[~pool[col].fillna(False).astype(bool)]
        print(f"  [picks] sanity gate: {n_before} → {len(pool)} eligible")

    fused = compute_rrf(pool, score_cols, k=k, lower_is_better=lower_is_better)
    fused_sorted = fused.sort_values("rrf_score", ascending=False)
    fused_sorted.to_csv(output_dir / "fusion_table.csv", index=False)

    picks = select_picks(fused, n=n, cluster_col=cluster_col,
                         strategy=strategy)
    keep_cols = [c for c in (
        "pick_rank", model_col, cluster_col, "rrf_score", "pick_reason",
        *score_cols, *(f"rank_{c}" for c in score_cols),
    ) if c in picks.columns]
    picks[keep_cols].to_csv(output_dir / "picks.csv", index=False)

    copied = []
    if copy_cif and cif_col in picks.columns:
        import shutil
        for _, r in picks.iterrows():
            src = r[cif_col]
            if isinstance(src, str) and Path(src).exists():
                dst = output_dir / f"model_{int(r['pick_rank'])}.cif"
                shutil.copy(src, dst)
                copied.append(str(dst))

    print(f"  [picks] strategy={strategy}  n={len(picks)}")
    for _, r in picks.iterrows():
        print(f"    {int(r['pick_rank'])}. {r[model_col]}  "
              f"cluster={r[cluster_col]}  RRF={r['rrf_score']:.4f}  "
              f"({r['pick_reason']})")

    return {
        "fusion_table": str(output_dir / "fusion_table.csv"),
        "picks": str(output_dir / "picks.csv"),
        "n_eligible": int(len(pool)),
        "n_picks": int(len(picks)),
        "strategy": strategy,
        "score_cols": list(score_cols),
        "copied_cif": copied,
    }


__all__ = ["compute_rrf", "select_picks", "run_submission_picks"]
