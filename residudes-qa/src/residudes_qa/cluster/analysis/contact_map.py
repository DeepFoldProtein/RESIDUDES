"""Per-cluster contact map analysis for Ab-Ag clustering.

Operates on a ``contacts_long.parquet``-style DataFrame with one row per
model × epitope-residue contact (or extended to model × paratope-epitope
pair if available). Produces:

* ``F_binary``  — fraction of cluster models where (CDR_i, epitope_j) is in contact
* ``F_soft``    — Gaussian-weighted (exp(-d² / 2σ²)) frequency
* ``D_min``     — mean min-distance Å
* ``F_all``     — global frequency map (across all models, used for enrichment)
* ``ΔF_c``      — enrichment vs global

Two main flows:

1. **Epitope-only** (works with H1335 stage-1 ``contacts_long.parquet``):
   columns ``model_id, abc_chain, abc_resnum, n_atom_contacts, min_dist``.
   ⇒ produces 1D epitope-residue maps per cluster (no paratope axis).

2. **CDR×epitope pair** (works with H1335 stage-13 ``pair_contacts.csv`` or
   user-provided ``contacts_pair`` DataFrame with extra ``ab_chain, ab_resnum``):
   ⇒ produces full 2D CDR×epitope map per cluster.

References
----------
- DockQ native-contact convention (Basu & Wallner 2016): heavy-atom pairs ≤ 5 Å.
- Soft contact weighting: Gaussian Φ(d) = exp(-d²/(2σ²)).
- Enrichment ΔF_c = F_c - F_global is the simplest specificity score; for
  zero-frequency baselines use log-odds with pseudocount.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def _resolve_iface_columns(df: pd.DataFrame) -> dict[str, str]:
    """Detect column naming: H1335 (abc_chain / abc_resnum) vs generic."""
    rename = {}
    if "abc_chain" in df.columns:
        rename["epitope_chain"] = "abc_chain"
        rename["epitope_res"] = "abc_resnum"
    elif "ag_chain" in df.columns:
        rename["epitope_chain"] = "ag_chain"
        rename["epitope_res"] = "ag_resnum"
    if "ab_chain" in df.columns:
        rename["paratope_chain"] = "ab_chain"
        rename["paratope_res"] = "ab_resnum"
    if "min_dist" in df.columns:
        rename["min_dist"] = "min_dist"
    elif "distance" in df.columns:
        rename["min_dist"] = "distance"
    return rename


def build_global_contact_map(
    contacts_long: pd.DataFrame,
    n_models: Optional[int] = None,
    cutoff: float = 5.0,
    sigma: float = 1.5,
) -> dict:
    """Aggregate global (pool-wide) epitope contact statistics.

    Parameters
    ----------
    contacts_long
        Long-format. Must have ``model_id``, and epitope chain/residue cols
        (auto-detected from common naming).
    n_models
        Total models in the pool. If None, derived from ``model_id.nunique()``.
    cutoff
        Distance cutoff (Å) for binary contact (used if rows are not already
        pre-filtered by stage-1).
    sigma
        Gaussian σ (Å) for soft contact weight.

    Returns
    -------
    dict
        ``{epitope_residues, F_binary, F_soft, D_min, n_models}``
        - ``epitope_residues``: list of (chain, resnum) tuples
        - ``F_binary``: 1D np.ndarray, fraction of models with this contact
        - ``F_soft``:   1D np.ndarray, mean exp(-d²/2σ²) across models (0 if no contact)
        - ``D_min``:    1D np.ndarray, mean min-distance among models that had it
    """
    cols = _resolve_iface_columns(contacts_long)
    df = contacts_long.copy()
    if "epitope_chain" not in cols:
        raise KeyError(f"contacts_long missing chain column; columns={contacts_long.columns.tolist()}")
    ch_col, res_col = cols["epitope_chain"], cols["epitope_res"]
    if "min_dist" in cols:
        dist_col = cols["min_dist"]
        df = df[df[dist_col] <= cutoff].copy()
    else:
        dist_col = None

    if n_models is None:
        n_models = df["model_id"].nunique()

    grouped = df.groupby([ch_col, res_col])
    epitope_keys = sorted(grouped.groups.keys())
    F_binary = np.zeros(len(epitope_keys), dtype=np.float32)
    F_soft = np.zeros(len(epitope_keys), dtype=np.float32)
    D_min = np.full(len(epitope_keys), np.nan, dtype=np.float32)
    for idx, key in enumerate(epitope_keys):
        sub = grouped.get_group(key)
        F_binary[idx] = sub["model_id"].nunique() / max(n_models, 1)
        if dist_col is not None:
            d = sub[dist_col].to_numpy(dtype=np.float32)
            F_soft[idx] = float(np.exp(-(d**2) / (2 * sigma**2)).mean())
            D_min[idx] = float(d.mean())
        else:
            F_soft[idx] = F_binary[idx]
    return dict(
        epitope_residues=epitope_keys,
        F_binary=F_binary,
        F_soft=F_soft,
        D_min=D_min,
        n_models=n_models,
    )


def build_cluster_contact_maps(
    contacts_long: pd.DataFrame,
    cluster_assignments: pd.DataFrame,
    *,
    cutoff: float = 5.0,
    sigma: float = 1.5,
    cluster_col: str = "cluster",
    model_col: str = "model_id",
) -> dict:
    """Build per-cluster contact maps + global baseline + per-cluster epitope union.

    Parameters
    ----------
    contacts_long
        Pool-wide long contacts. Auto-detects ``abc_chain/abc_resnum`` or
        ``ag_chain/ag_resnum`` for epitope side.
    cluster_assignments
        Two columns ``model_id``, ``cluster``.

    Returns
    -------
    dict
        ``{
            'global':    {epitope_residues, F_binary, F_soft, D_min, n_models},
            'clusters':  {cluster_id: same structure},
            'epitope_universe': sorted list of all (chain, res) found across all clusters,
            'F_binary_matrix': (n_clusters × n_epitope) aligned to epitope_universe,
        }``
    """
    cols = _resolve_iface_columns(contacts_long)
    if "epitope_chain" not in cols:
        raise KeyError(f"contacts_long missing chain column; columns={contacts_long.columns.tolist()}")

    merged = contacts_long.merge(
        cluster_assignments[[model_col, cluster_col]].rename(columns={model_col: "model_id"}),
        on="model_id",
        how="inner",
    )
    if len(merged) == 0:
        raise ValueError("no overlap between contacts_long.model_id and cluster_assignments.model_id")

    global_map = build_global_contact_map(merged, cutoff=cutoff, sigma=sigma)

    clusters_out: dict = {}
    epitope_union: set = set(global_map["epitope_residues"])
    for cid, group in merged.groupby(cluster_col):
        cmap = build_global_contact_map(group, cutoff=cutoff, sigma=sigma)
        clusters_out[int(cid)] = cmap
        epitope_union.update(cmap["epitope_residues"])

    universe = sorted(epitope_union)
    universe_index = {res: i for i, res in enumerate(universe)}

    # Build a (n_clusters × n_epitope) aligned matrix.
    cluster_ids = sorted(clusters_out.keys())
    n_clusters = len(cluster_ids)
    F_matrix = np.zeros((n_clusters, len(universe)), dtype=np.float32)
    for ci, cid in enumerate(cluster_ids):
        cmap = clusters_out[cid]
        for res, freq in zip(cmap["epitope_residues"], cmap["F_binary"]):
            F_matrix[ci, universe_index[res]] = freq

    return {
        "global": global_map,
        "clusters": clusters_out,
        "epitope_universe": universe,
        "F_binary_matrix": F_matrix,
        "cluster_ids": cluster_ids,
    }


def compute_enrichment(
    F_cluster: np.ndarray,
    F_global: np.ndarray,
    *,
    method: str = "delta",
    pseudocount: float = 0.01,
) -> np.ndarray:
    """Specificity score per residue.

    Parameters
    ----------
    method
        ``'delta'``    — ΔF = F_c - F_all (simplest, signed)
        ``'log_odds'`` — log((F_c + ε) / (F_all + ε)) (asymmetric, robust to F_all=0)
        ``'ratio'``    — (F_c + ε) / (F_all + ε)

    Returns
    -------
    np.ndarray of same length as F_cluster
    """
    if method == "delta":
        return F_cluster - F_global
    if method == "log_odds":
        return np.log((F_cluster + pseudocount) / (F_global + pseudocount))
    if method == "ratio":
        return (F_cluster + pseudocount) / (F_global + pseudocount)
    raise ValueError(f"unknown enrichment method: {method}")


def rank_specific_contacts(
    epitope_residues: list,
    enrichment: np.ndarray,
    top_n: int = 20,
    min_cluster_freq: float = 0.0,
    F_cluster: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Return top N (residue, enrichment) pairs.

    Optional ``min_cluster_freq`` filters out residues with low cluster
    frequency (avoids reporting cluster-specific but rare contacts).
    """
    rows = []
    for res, score in zip(epitope_residues, enrichment):
        if F_cluster is not None and F_cluster[epitope_residues.index(res)] < min_cluster_freq:
            continue
        rows.append({
            "chain": res[0],
            "resnum": int(res[1]),
            "label": f"{res[0]}{res[1]}",
            "enrichment": float(score),
            "cluster_freq": (
                float(F_cluster[epitope_residues.index(res)])
                if F_cluster is not None else None
            ),
        })
    df = pd.DataFrame(rows)
    df = df.sort_values("enrichment", ascending=False).head(top_n)
    return df.reset_index(drop=True)


def save_cluster_contact_maps(
    maps: dict,
    output_dir: str | Path,
) -> dict:
    """Save per-cluster contact maps to organised folders.

    Layout::

        output_dir/
        ├── manifest.json
        ├── global/
        │   ├── F_binary.csv          # res | freq
        │   ├── F_soft.csv
        │   └── D_min.csv
        ├── per_cluster/
        │   ├── cluster_0/
        │   │   ├── data/
        │   │   │   ├── F_binary.csv
        │   │   │   ├── F_soft.csv
        │   │   │   ├── D_min.csv
        │   │   │   └── enrichment.csv
        │   │   └── (figures/ filled by viz layer in Phase 3)
        │   ├── cluster_1/ ...
        ├── cross_cluster/
        │   └── F_binary_matrix.csv   # cluster × epitope wide table

    Returns
    -------
    dict
        Manifest of all written paths.
    """
    import json
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {"generated_at": pd.Timestamp.now().isoformat(),
                "global": {}, "per_cluster": {}, "cross_cluster": {}}

    # global
    gdir = output_dir / "global"
    gdir.mkdir(exist_ok=True)
    gmap = maps["global"]
    pd.DataFrame({
        "chain": [r[0] for r in gmap["epitope_residues"]],
        "resnum": [r[1] for r in gmap["epitope_residues"]],
        "F_binary": gmap["F_binary"],
        "F_soft": gmap["F_soft"],
        "D_min": gmap["D_min"],
    }).to_csv(gdir / "epitope_frequency.csv", index=False)
    manifest["global"]["epitope_frequency"] = str(gdir / "epitope_frequency.csv")
    manifest["global"]["n_models"] = int(gmap["n_models"])

    # per-cluster
    per_cluster_dir = output_dir / "per_cluster"
    per_cluster_dir.mkdir(exist_ok=True)
    F_global_by_res = dict(zip(gmap["epitope_residues"], gmap["F_binary"]))
    for cid, cmap in maps["clusters"].items():
        cdir = per_cluster_dir / f"cluster_{cid}"
        ddir = cdir / "data"
        ddir.mkdir(parents=True, exist_ok=True)
        (cdir / "figures").mkdir(exist_ok=True)
        # binary + soft + dist
        df = pd.DataFrame({
            "chain": [r[0] for r in cmap["epitope_residues"]],
            "resnum": [r[1] for r in cmap["epitope_residues"]],
            "F_binary": cmap["F_binary"],
            "F_soft": cmap["F_soft"],
            "D_min": cmap["D_min"],
        })
        df.to_csv(ddir / "epitope_frequency.csv", index=False)
        # enrichment
        F_global_aligned = np.array([F_global_by_res.get(r, 0.0)
                                     for r in cmap["epitope_residues"]], dtype=np.float32)
        delta = compute_enrichment(cmap["F_binary"], F_global_aligned, method="delta")
        log_odds = compute_enrichment(cmap["F_binary"], F_global_aligned, method="log_odds")
        df["enrichment_delta"] = delta
        df["enrichment_log_odds"] = log_odds
        df.sort_values("enrichment_delta", ascending=False).to_csv(
            ddir / "enrichment.csv", index=False
        )
        manifest["per_cluster"][f"cluster_{cid}"] = {
            "n_models": int(cmap["n_models"]),
            "data_dir": str(ddir),
            "epitope_frequency": str(ddir / "epitope_frequency.csv"),
            "enrichment": str(ddir / "enrichment.csv"),
            "n_epitope_residues": len(cmap["epitope_residues"]),
        }

    # cross-cluster matrix
    cc_dir = output_dir / "cross_cluster"
    cc_dir.mkdir(exist_ok=True)
    universe = maps["epitope_universe"]
    F_matrix = maps["F_binary_matrix"]
    matrix_df = pd.DataFrame(
        F_matrix,
        index=[f"cluster_{c}" for c in maps["cluster_ids"]],
        columns=[f"{r[0]}{r[1]}" for r in universe],
    )
    matrix_df.to_csv(cc_dir / "F_binary_matrix.csv")
    manifest["cross_cluster"]["F_binary_matrix"] = str(cc_dir / "F_binary_matrix.csv")
    manifest["cross_cluster"]["n_universe_residues"] = len(universe)

    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


__all__ = [
    "build_global_contact_map",
    "build_cluster_contact_maps",
    "compute_enrichment",
    "rank_specific_contacts",
    "save_cluster_contact_maps",
]
