"""COM-based KMeans clustering with silhouette-driven k-sweep.

Adapted from H1335 ``29_com_clustering_extra.py``.  The general recipe is:

1. For each candidate CIF, compute the probe-chain centre of mass.  If an
   ``anchor_chain`` + ``ref_cif`` are supplied, the predicted anchor chain is
   first Kabsch-aligned onto the reference anchor so that all COMs live in
   one common reference frame (essential for cross-method comparison).
2. KMeans across ``k_range``, pick the best ``k`` by silhouette score.
3. Return labels + per-cluster summary.

For workflows that don't need the anchor alignment (e.g. all predictions
from the same backbone), pass ``ref_cif=None``.
"""
from __future__ import annotations

import time
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd


# Worker-state cache for the multiprocessing driver.
_WORKER_STATE: dict[str, Any] = {}


# ---------- chain CA loaders + Kabsch ----------------------------------------


def _load_ca_chains(cif_path: str | Path, chains: Iterable[str]) -> dict[str, list[np.ndarray]]:
    """Return ``{chain_id: [ca_xyz, ...]}`` for the requested chains."""
    import gemmi

    cset = set(chains)
    st = gemmi.read_structure(str(cif_path))
    out: dict[str, list[np.ndarray]] = {c: [] for c in cset}
    for model in st:
        for ch in model:
            if ch.name not in cset:
                continue
            for r in ch:
                ca = r.find_atom("CA", "*")
                if ca is not None:
                    out[ch.name].append(np.array([ca.pos.x, ca.pos.y, ca.pos.z]))
        break
    return out


def _kabsch(P: np.ndarray, Q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    Pc = P.mean(0)
    Qc = Q.mean(0)
    H = (P - Pc).T @ (Q - Qc)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.eye(3)
    D[2, 2] = d
    R = Vt.T @ D @ U.T
    return R, Qc - R @ Pc


def compute_probe_com(
    cif_path: str | Path,
    probe_chains: Iterable[str],
    anchor_chains: Optional[Iterable[str]] = None,
    ref_anchor_xyz: Optional[np.ndarray] = None,
) -> Optional[tuple[float, float, float]]:
    """Probe-chain COM, optionally in the reference frame after anchor Kabsch.

    Parameters
    ----------
    cif_path
        Predicted CIF.
    probe_chains
        Chain IDs of the probe (averaged across all listed chains).
    anchor_chains
        Chain IDs of the anchor to align onto.  ``None`` ⇒ raw frame.
    ref_anchor_xyz
        Concatenated reference CA coords for the anchor (shape ``(N, 3)``),
        in the same order as ``_load_ca_chains`` would yield them.  Required
        when ``anchor_chains`` is set.
    """
    all_chains = list(probe_chains) + (list(anchor_chains) if anchor_chains else [])
    try:
        chains = _load_ca_chains(cif_path, all_chains)
    except Exception:
        return None
    probe_pts = [pt for c in probe_chains for pt in chains.get(c, [])]
    if not probe_pts:
        return None
    probe_arr = np.array(probe_pts)

    if anchor_chains is not None and ref_anchor_xyz is not None:
        anchor_pts = [pt for c in anchor_chains for pt in chains.get(c, [])]
        if not anchor_pts:
            return None
        anc_arr = np.array(anchor_pts)
        n = min(len(anc_arr), len(ref_anchor_xyz))
        if n < 10:
            return None
        R, t = _kabsch(anc_arr[:n], ref_anchor_xyz[:n])
        probe_arr = (R @ probe_arr.T).T + t

    com = probe_arr.mean(axis=0)
    return float(com[0]), float(com[1]), float(com[2])


# ---------- KMeans + silhouette ----------------------------------------------


def cluster_com(
    com_xyz: np.ndarray,
    k_range: Iterable[int] = (2, 3, 4, 5, 6),
    random_state: int = 0,
) -> tuple[int, np.ndarray, dict[int, float]]:
    """KMeans sweep on COM coords; return ``(best_k, labels, silhouette_per_k)``."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    sil: dict[int, float] = {}
    best_lab: dict[int, np.ndarray] = {}
    for k in k_range:
        if len(com_xyz) <= k:
            continue
        lab = KMeans(k, n_init=10, random_state=random_state).fit_predict(com_xyz)
        try:
            s = float(silhouette_score(com_xyz, lab))
        except Exception:
            continue
        sil[int(k)] = round(s, 4)
        best_lab[int(k)] = lab
    if not sil:
        # fallback: single cluster
        return 1, np.zeros(len(com_xyz), dtype=int), {}
    best_k = max(sil, key=sil.get)
    return best_k, best_lab[best_k], sil


# ---------- driver -----------------------------------------------------------


def _init_worker(probe_chains: list[str], anchor_chains: Optional[list[str]],
                 ref_anchor_xyz: Optional[np.ndarray]) -> None:
    _WORKER_STATE["probe_chains"] = probe_chains
    _WORKER_STATE["anchor_chains"] = anchor_chains
    _WORKER_STATE["ref_anchor_xyz"] = ref_anchor_xyz


def _process_one(arg: tuple[str, str]) -> dict[str, Any]:
    model_id, cif_path = arg
    com = compute_probe_com(
        cif_path,
        _WORKER_STATE["probe_chains"],
        _WORKER_STATE["anchor_chains"],
        _WORKER_STATE["ref_anchor_xyz"],
    )
    if com is None:
        return {"model_id": model_id, "com_x": np.nan, "com_y": np.nan, "com_z": np.nan}
    return {"model_id": model_id, "com_x": com[0], "com_y": com[1], "com_z": com[2]}


def run_clustering(
    master_df: pd.DataFrame,
    probe_chains: Iterable[str],
    anchor_chains: Optional[Iterable[str]] = None,
    ref_cif: Optional[str | Path] = None,
    output_dir: Optional[str | Path] = None,
    k_range: Iterable[int] = (2, 3, 4, 5, 6),
    workers: int = 8,
    random_state: int = 0,
) -> pd.DataFrame:
    """Compute probe COM for each model and KMeans-cluster.

    Returns a DataFrame keyed on ``model_id`` with new columns:

    * ``com_x``, ``com_y``, ``com_z`` — probe COM (reference frame if applicable)
    * ``cluster_id`` — KMeans label, ``-1`` for COM-missing
    * a ``cluster_sil`` attribute mapped via ``df.attrs['silhouette']``.
    """
    probe_list = list(probe_chains)
    anchor_list = list(anchor_chains) if anchor_chains else None

    ref_anchor_xyz: Optional[np.ndarray] = None
    if anchor_list is not None and ref_cif is not None:
        ref_chains = _load_ca_chains(ref_cif, anchor_list)
        flat = [pt for c in anchor_list for pt in ref_chains.get(c, [])]
        if not flat:
            raise ValueError(f"Reference {ref_cif} missing anchor chains {anchor_list}")
        ref_anchor_xyz = np.array(flat)
        print(
            f"  reference anchor xyz: {ref_anchor_xyz.shape}  chains={anchor_list}",
            flush=True,
        )

    records = [(r["model_id"], r["src_cif"]) for _, r in master_df.iterrows()]
    n = len(records)
    print(
        f"  compute probe COM: {n} cifs  workers={workers}  "
        f"probe={probe_list}  anchor={anchor_list}",
        flush=True,
    )
    t0 = time.time()
    rows: list[dict[str, Any]] = []
    with Pool(
        workers,
        initializer=_init_worker,
        initargs=(probe_list, anchor_list, ref_anchor_xyz),
    ) as pool:
        for res in pool.imap_unordered(_process_one, records, chunksize=10):
            rows.append(res)
    com_df = pd.DataFrame(rows)
    print(f"  COM done: {(time.time()-t0)/60:.1f} min", flush=True)

    # Merge into master copy
    out = master_df.merge(com_df, on="model_id", how="left")
    com_xyz = out[["com_x", "com_y", "com_z"]].to_numpy()
    valid = ~np.isnan(com_xyz).any(axis=1)
    labels = np.full(len(out), -1, dtype=int)
    sil: dict[int, float] = {}
    if valid.sum() >= 4:
        best_k, lab_valid, sil = cluster_com(com_xyz[valid], k_range=k_range, random_state=random_state)
        labels[valid] = lab_valid
        print(f"  silhouette per k: {sil}", flush=True)
        print(f"  best k = {best_k}", flush=True)
    else:
        print(
            f"  too few valid COMs ({int(valid.sum())}) — skipping cluster, "
            "all rows assigned cluster_id=-1",
            flush=True,
        )
    out["cluster_id"] = labels
    out.attrs["silhouette"] = sil

    if output_dir is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        keep_cols = [c for c in (
            "model_id", "target", "variant", "method", "seed", "sample",
            "src_cif", "com_x", "com_y", "com_z", "cluster_id",
        ) if c in out.columns]
        out[keep_cols].to_csv(out_dir / "cluster_labels.csv", index=False)
        out.to_parquet(out_dir / "master_with_clusters.parquet", index=False)
        print(
            f"  saved: {out_dir/'cluster_labels.csv'} + master_with_clusters.parquet",
            flush=True,
        )

    return out
