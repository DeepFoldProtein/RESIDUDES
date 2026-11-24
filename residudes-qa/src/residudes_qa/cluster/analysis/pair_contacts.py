"""Paratope ↔ epitope pair-level contact extraction from CIF.

This is the **CDR × epitope contact map foundation** — fills the gap left
by epitope-only ``contacts_long`` tables. For each model, scan paratope
chain residues against antigen chain heavy atoms, record every pair within
``cutoff`` Å with min-distance and atom counts.

Output per cluster (aggregated): F_c(p, e) = fraction of cluster models in
which paratope residue p contacts epitope residue e.

Output schema (long)
--------------------
``model_id, paratope_chain, paratope_resnum, paratope_resname,
epitope_chain, epitope_resnum, epitope_resname, n_atom_contacts, min_dist``

Optional CDR loop annotation (requires ANARCI): adds ``cdr_loop`` column
with values ``H1, H2, H3, L1, L2, L3, framework``.

Pair-level data unlocks:
* CDR × epitope 2D heatmap (the "real" contact map)
* per-channel (physchem) contact decomposition  (uses paratope_resname)
* CDR loop dominance per cluster (with ANARCI annotation)
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import multiprocessing as mp
import numpy as np
import pandas as pd


AA_3 = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY",
    "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER",
    "THR", "TRP", "TYR", "VAL",
}


def extract_pair_contacts(
    cif_path: str | Path,
    paratope_chains: Iterable[str],
    epitope_chains: Iterable[str],
    *,
    cutoff: float = 5.0,
    sanitize_cutoff: Optional[float] = 10.0,
) -> pd.DataFrame:
    """Return long-format pair contacts for one CIF.

    Heavy-atom pairs only (H excluded). One row per (paratope_res, epitope_res)
    pair with at least one atom pair within ``cutoff`` Å.

    Parameters
    ----------
    cif_path
        Path to CIF.
    paratope_chains, epitope_chains
        Chain ID iterables for the two sides.
    cutoff
        Heavy-atom distance threshold (Å). DockQ/CAPRI convention = 5.0.
    sanitize_cutoff
        Atoms farther than this from their residue's CA are dropped (guards
        against pt2 terminal-O blow-ups). Set to ``None`` to disable.

    Returns
    -------
    pd.DataFrame with columns:
        paratope_chain, paratope_resnum, paratope_resname,
        epitope_chain, epitope_resnum, epitope_resname,
        n_atom_contacts, min_dist
    """
    import gemmi
    from scipy.spatial import cKDTree

    para_set = set(paratope_chains)
    epi_set = set(epitope_chains)
    st = gemmi.read_structure(str(cif_path))

    # Gather heavy atoms by (chain, res), with sanitize.
    para_atoms: dict[tuple, list] = {}
    para_names: dict[tuple, str] = {}
    epi_atoms: dict[tuple, list] = {}
    epi_names: dict[tuple, str] = {}

    for model in st:
        for ch in model:
            target = para_atoms if ch.name in para_set else (epi_atoms if ch.name in epi_set else None)
            target_names = para_names if ch.name in para_set else (epi_names if ch.name in epi_set else None)
            if target is None:
                continue
            for r in ch:
                if r.name not in AA_3:
                    continue
                ca = r.find_atom("CA", "*")
                if ca is None:
                    continue
                ca_p = np.array([ca.pos.x, ca.pos.y, ca.pos.z])
                heavy: list[np.ndarray] = []
                for a in r:
                    if a.element.name == "H":
                        continue
                    p = np.array([a.pos.x, a.pos.y, a.pos.z])
                    if sanitize_cutoff is not None:
                        if float(np.linalg.norm(p - ca_p)) > sanitize_cutoff:
                            continue
                    heavy.append(p)
                if heavy:
                    key = (ch.name, int(r.seqid.num))
                    target[key] = np.asarray(heavy, dtype=np.float32)
                    target_names[key] = r.name
        break

    if not para_atoms or not epi_atoms:
        return pd.DataFrame(columns=[
            "paratope_chain", "paratope_resnum", "paratope_resname",
            "epitope_chain", "epitope_resnum", "epitope_resname",
            "n_atom_contacts", "min_dist",
        ])

    # Flatten epitope atoms into one tree for fast queries
    epi_keys: list[tuple] = []
    epi_atom_idx: list[int] = []  # for each atom, which residue key
    epi_xyz_list: list[np.ndarray] = []
    for k, arr in epi_atoms.items():
        epi_keys.append(k)
        for atom in arr:
            epi_atom_idx.append(len(epi_keys) - 1)
            epi_xyz_list.append(atom)
    epi_xyz = np.asarray(epi_xyz_list, dtype=np.float32)
    epi_atom_idx_arr = np.asarray(epi_atom_idx, dtype=np.int32)
    tree = cKDTree(epi_xyz)

    rows = []
    for p_key, p_arr in para_atoms.items():
        # Query all atoms within cutoff
        # cKDTree.query_ball_point returns list-of-lists for multi-query
        nearby = tree.query_ball_point(p_arr, r=cutoff)
        # Aggregate by epitope residue
        epi_hit_atoms: dict[int, list[float]] = {}
        for atom_i, hits in enumerate(nearby):
            if not hits:
                continue
            d_atoms = np.linalg.norm(epi_xyz[hits] - p_arr[atom_i], axis=1)
            for hit_idx, d in zip(hits, d_atoms):
                e_key_idx = int(epi_atom_idx_arr[hit_idx])
                epi_hit_atoms.setdefault(e_key_idx, []).append(float(d))
        for e_idx, dists in epi_hit_atoms.items():
            e_key = epi_keys[e_idx]
            rows.append({
                "paratope_chain": p_key[0],
                "paratope_resnum": p_key[1],
                "paratope_resname": para_names[p_key],
                "epitope_chain": e_key[0],
                "epitope_resnum": e_key[1],
                "epitope_resname": epi_names[e_key],
                "n_atom_contacts": len(dists),
                "min_dist": float(min(dists)),
            })
    return pd.DataFrame(rows)


def _worker(args):
    cif, paratope_chains, epitope_chains, cutoff, sanitize_cutoff, model_id = args
    try:
        df = extract_pair_contacts(cif, paratope_chains, epitope_chains,
                                    cutoff=cutoff, sanitize_cutoff=sanitize_cutoff)
        df["model_id"] = model_id
        return df
    except Exception as e:
        return pd.DataFrame([{"model_id": model_id, "_err": f"{type(e).__name__}: {e}"}])


def extract_pair_contacts_batch(
    cif_records: pd.DataFrame,
    paratope_chains: Iterable[str],
    epitope_chains: Iterable[str],
    *,
    cif_col: str = "src_cif",
    model_id_col: str = "model_id",
    cutoff: float = 5.0,
    sanitize_cutoff: Optional[float] = 10.0,
    workers: int = 8,
) -> pd.DataFrame:
    """Compute pair contacts for many models in parallel.

    Returns one long table with every pair contact across all models, plus
    a ``model_id`` column.
    """
    args = [(row[cif_col], tuple(paratope_chains), tuple(epitope_chains),
              cutoff, sanitize_cutoff, row[model_id_col])
            for _, row in cif_records.iterrows()]
    if workers > 1 and len(args) > 1:
        with mp.Pool(workers) as pool:
            frames = pool.map(_worker, args)
    else:
        frames = [_worker(a) for a in args]
    return pd.concat(frames, ignore_index=True)


# ============================================================================
# Cluster aggregation
# ============================================================================


def build_cluster_pair_contact_map(
    pair_contacts: pd.DataFrame,
    cluster_assignments: pd.DataFrame,
    *,
    cluster_col: str = "cluster",
    model_col: str = "model_id",
) -> dict:
    """Aggregate per-model pair contacts to per-cluster matrices.

    Returns
    -------
    dict
        ``{
           cluster_ids: [...],
           paratope_universe: [(chain, res), ...],
           epitope_universe:  [(chain, res), ...],
           clusters: {
               cid: {
                   F_binary:       (n_para × n_epi) frequency in cluster,
                   D_mean:         (n_para × n_epi) mean min-dist (Å, NaN if absent),
                   D_std:          (n_para × n_epi) std of min-dist (Å),
                   n_models:       int,
                   top_pairs:      list of dicts sorted by freq, each with
                                   {paratope, epitope, freq, n_models,
                                    mean_min_dist, std_min_dist}
               }
           },
           global: { F_binary, n_models }
        }``

    Notes
    -----
    ``F_binary[i, j]`` = fraction of cluster models with at least one heavy-atom
    pair contact between paratope residue i and epitope residue j (cutoff used
    in :func:`extract_pair_contacts`).

    ``D_mean[i, j]`` / ``D_std[i, j]`` = mean and std of the per-model minimum
    heavy-atom distance for that pair, over the cluster models in which the
    contact occurs. Low mean + low std = a tight, reproducible contact (a
    geometric-stability signal, distinct from prediction frequency).
    """
    merged = pair_contacts.merge(
        cluster_assignments[[model_col, cluster_col]],
        on=model_col, how="inner",
    )

    para_keys = sorted(set(zip(merged["paratope_chain"], merged["paratope_resnum"].astype(int))))
    epi_keys = sorted(set(zip(merged["epitope_chain"], merged["epitope_resnum"].astype(int))))
    para_idx = {k: i for i, k in enumerate(para_keys)}
    epi_idx = {k: i for i, k in enumerate(epi_keys)}

    has_dist = "min_dist" in merged.columns
    pair_cols = ["paratope_chain", "paratope_resnum",
                 "epitope_chain", "epitope_resnum"]

    clusters_out = {}
    cluster_ids = sorted(merged[cluster_col].unique())
    for cid in cluster_ids:
        sub = merged[merged[cluster_col] == cid]
        n = sub[model_col].nunique()
        # For each (p, e) pair, count unique models (frequency).
        grouped = sub.groupby(pair_cols)[model_col].nunique()
        # Geometric consistency: mean / std of per-model min_dist for the pair.
        # Tells how *tight* and how *reproducible* the contact is across the
        # cluster ensemble — a contact at 3.0 ± 0.3 Å in 90% of models is far
        # more credible (and tighter) than one at 4.8 ± 1.5 Å in 30%.
        if has_dist:
            dstats = sub.groupby(pair_cols)["min_dist"].agg(["mean", "std"])
        F = np.zeros((len(para_keys), len(epi_keys)), dtype=np.float32)
        D_mean = np.full((len(para_keys), len(epi_keys)), np.nan, dtype=np.float32)
        D_std = np.full((len(para_keys), len(epi_keys)), np.nan, dtype=np.float32)
        top_pairs = []
        for (pc, pr, ec, er), nm in grouped.items():
            i = para_idx[(pc, int(pr))]
            j = epi_idx[(ec, int(er))]
            F[i, j] = nm / max(n, 1)
            d_mean = d_std = float("nan")
            if has_dist:
                d_mean = float(dstats.loc[(pc, pr, ec, er), "mean"])
                d_std = float(dstats.loc[(pc, pr, ec, er), "std"])
                if d_mean == d_mean:   # not NaN
                    D_mean[i, j] = d_mean
                if d_std == d_std:
                    D_std[i, j] = d_std
            top_pairs.append({
                "paratope": f"{pc}.{int(pr)}",
                "epitope": f"{ec}.{int(er)}",
                "freq": float(F[i, j]),
                "n_models": int(nm),
                "mean_min_dist": d_mean,
                "std_min_dist": d_std,
            })
        top_pairs.sort(key=lambda t: -t["freq"])
        clusters_out[int(cid)] = dict(
            F_binary=F, D_mean=D_mean, D_std=D_std,
            n_models=int(n), top_pairs=top_pairs[:50],
        )

    # global
    n_total = merged[model_col].nunique()
    grouped = merged.groupby([
        "paratope_chain", "paratope_resnum",
        "epitope_chain", "epitope_resnum",
    ])[model_col].nunique()
    F_g = np.zeros((len(para_keys), len(epi_keys)), dtype=np.float32)
    for (pc, pr, ec, er), nm in grouped.items():
        F_g[para_idx[(pc, int(pr))], epi_idx[(ec, int(er))]] = nm / max(n_total, 1)

    return {
        "cluster_ids": cluster_ids,
        "paratope_universe": para_keys,
        "epitope_universe": epi_keys,
        "clusters": clusters_out,
        "global": {"F_binary": F_g, "n_models": int(n_total)},
    }


def annotate_cdr_loops(
    pair_contacts: pd.DataFrame,
    *,
    heavy_chain: Optional[str] = None,
    light_chain: Optional[str] = None,
    scheme: str = "imgt",
) -> pd.DataFrame:
    """Add a ``cdr_loop`` column (H1/H2/H3/L1/L2/L3/framework) using IMGT ranges.

    IMGT default CDR boundaries:
        CDR-H1: 27-38
        CDR-H2: 56-65
        CDR-H3: 105-117
        CDR-L1: 27-38
        CDR-L2: 56-65
        CDR-L3: 105-117

    For non-IMGT-numbered models the residue numbers may not match these
    ranges — in that case run ANARCI first or pass explicit ranges.
    """
    if scheme.lower() != "imgt":
        raise NotImplementedError("only IMGT default ranges implemented")
    CDR_RANGES = {
        "H1": (27, 38), "H2": (56, 65), "H3": (105, 117),
        "L1": (27, 38), "L2": (56, 65), "L3": (105, 117),
    }

    def classify(row):
        chain = row["paratope_chain"]
        resnum = int(row["paratope_resnum"])
        if heavy_chain and chain == heavy_chain:
            for label in ("H1", "H2", "H3"):
                lo, hi = CDR_RANGES[label]
                if lo <= resnum <= hi:
                    return label
            return "framework_H"
        if light_chain and chain == light_chain:
            for label in ("L1", "L2", "L3"):
                lo, hi = CDR_RANGES[label]
                if lo <= resnum <= hi:
                    return label
            return "framework_L"
        return "other"

    df = pair_contacts.copy()
    df["cdr_loop"] = df.apply(classify, axis=1)
    return df


__all__ = [
    "extract_pair_contacts",
    "extract_pair_contacts_batch",
    "build_cluster_pair_contact_map",
    "annotate_cdr_loops",
]
