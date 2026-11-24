"""Per-cluster pick selection — geometric medoid + xgb_top.

Generalized from H1335 ``44_strict_picks_2per.py``.  For each cluster we emit:

* exactly one **medoid** (model whose COM is closest to the cluster centroid),
* the top ``n_top_xgb`` models by ``xgb_pred``.

Each pick row is tagged with ``kind`` (``'medoid'`` or ``'xgb_top'``) and a
human-readable ``tag`` that combines cluster id, dominant epitope chain (if
present), and kind.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd


def _dist_to_center(g: pd.DataFrame, com_cols: list[str]) -> pd.Series:
    """Euclidean distance from each row's COM to the cluster centroid."""
    center = g[com_cols].mean()
    diff = g[com_cols].to_numpy() - center.to_numpy()
    return pd.Series(np.sqrt((diff ** 2).sum(axis=1)), index=g.index)


def pick_per_cluster(
    filtered_df: pd.DataFrame,
    n_top_xgb: int = 1,
    include_medoid: bool = True,
    com_cols: tuple[str, str, str] = ("com_x", "com_y", "com_z"),
    score_col: str = "xgb_pred",
    cluster_col: str = "cluster_id",
    ec_col: Optional[str] = "primary_epi_chain",
) -> pd.DataFrame:
    """Return per-cluster picks (medoid + ``n_top_xgb`` xgb-best).

    Parameters
    ----------
    filtered_df
        Output of one of the filter functions, must contain ``cluster_id``
        (or whatever ``cluster_col`` says) and the COM columns.
    n_top_xgb
        How many top-``score_col`` rows per cluster to include in addition
        to the medoid.
    include_medoid
        Set to False to skip the medoid pick.
    """
    if cluster_col not in filtered_df.columns:
        raise KeyError(f"filtered_df missing cluster column {cluster_col!r}")
    com_list = list(com_cols)
    missing_com = [c for c in com_list if c not in filtered_df.columns]
    if missing_com and include_medoid:
        raise KeyError(f"missing COM columns for medoid: {missing_com}")

    rows: list[dict[str, Any]] = []
    # Drop the noise / invalid cluster (-1) from picks.
    valid = filtered_df[filtered_df[cluster_col] >= 0]
    for c, g in valid.groupby(cluster_col):
        c = int(c)
        n_cluster = len(g)
        dom_epi: Optional[str] = None
        if ec_col and ec_col in g.columns and g[ec_col].notna().any():
            dom_epi = str(g[ec_col].mode().iloc[0])

        picks_this_cluster: list[tuple[str, pd.Series]] = []
        if include_medoid and all(c_ in g.columns for c_ in com_list):
            dists = _dist_to_center(g, com_list)
            medoid_idx = dists.idxmin()
            picks_this_cluster.append(("medoid", g.loc[medoid_idx]))

        if score_col in g.columns and n_top_xgb > 0:
            top = g.nlargest(n_top_xgb, score_col)
            for _, row in top.iterrows():
                picks_this_cluster.append(("xgb_top", row))

        for kind, row in picks_this_cluster:
            tag_parts = [f"c{c}"]
            if dom_epi:
                tag_parts.append(f"dom-{dom_epi}")
            tag_parts.append(kind)
            tag = "_".join(tag_parts)
            pick: dict[str, Any] = {
                "tag": tag,
                "kind": kind,
                "cluster_id": c,
                "cluster_n": int(n_cluster),
                "cluster_dom_epi": dom_epi,
            }
            # Carry the underlying row.
            for k, v in row.to_dict().items():
                if k not in pick:
                    pick[k] = v
            rows.append(pick)

    picks = pd.DataFrame(rows)
    # Stable de-duplication: same model can be both medoid and xgb_top.  We
    # keep the first occurrence; user can inspect ``kind`` to see.
    if "model_id" in picks.columns:
        picks = picks.drop_duplicates(subset=["model_id", "kind"], keep="first")
    print(
        f"  pick_per_cluster: produced {len(picks)} picks across {valid[cluster_col].nunique()} clusters",
        flush=True,
    )
    return picks.reset_index(drop=True)
