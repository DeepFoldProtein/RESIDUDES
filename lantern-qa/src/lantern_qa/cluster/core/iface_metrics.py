"""Interface-level distance metrics for MassiveClustering.

Adds 4 new pairwise distance matrices to complement TM/iTM/dRMSD/COM/anchored
suite — all operate on the **interface** (paratope-epitope contacts) rather
than on the global fold.

Metrics
-------
``iface_jaccard_residue``     Distance = 1 − |Q_A ∩ Q_B| / |Q_A ∪ Q_B|
                              where Q_M = epitope residue set of model M.
                              Fast, robust to numbering. Range [0,1].

``iface_jaccard_contact``     Same shape but on the contact pair set
                              E_M = {(p_chain.res, q_chain.res) : d ≤ 5 Å}.
                              Sharper than residue version.

``iface_chamfer``             Symmetric Chamfer on paratope CA point clouds:
                                d(A,B) = (1/|A|)Σ_a min_b ||a-b||² +
                                         (1/|B|)Σ_b min_a ||a-b||²
                              Not a metric (triangle inequality breaks) but
                              fast and geometric.

``iface_wasserstein``         Sinkhorn-2 EMD on paratope CA:
                                W_2(A,B) = (min_γ Σ γ_ij ||a_i-b_j||²)^(1/2)
                              True metric. Requires ``pip install pot``.

All functions return an N×N float matrix indexed by ``model_id`` order.

References
----------
- Chamfer: Borgefors 1986 (image processing); 3D point clouds since
  Hausdorff distance era.
- Wasserstein-2: Villani 2008; Sinkhorn approx Cuturi 2013.

Reference H1335 case study showing these metrics resolve sub-modes:
``H1335/analysis/REPORT.md §13`` (gO core EC3 vs gO N-term EC5 separated
purely by Jaccard despite TM ≈ 0.55).
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


# ============================================================================
# 1) Jaccard on residue sets
# ============================================================================


def _jaccard_distance_from_sets(sets: dict, model_ids: list[str]) -> np.ndarray:
    """Vectorized 1 - Jaccard over per-model element sets.

    Replaces an O(N^2) pure-Python frozenset double-loop (which is ~99M
    set-intersection ops for N=14070 and takes hours) with a single
    incidence-matrix matmul:  I = B @ B.T gives |Q_i ∩ Q_j|, and
    |Q_i ∪ Q_j| = |Q_i| + |Q_j| - I. Identical values, seconds not hours.
    Uses a sparse incidence matrix so it scales to large element universes.
    """
    import numpy as np
    from scipy import sparse

    N = len(model_ids)
    universe: dict = {}
    rows: list[int] = []
    cols: list[int] = []
    for i, mid in enumerate(model_ids):
        for e in sets.get(mid, ()):
            k = universe.get(e)
            if k is None:
                k = len(universe)
                universe[e] = k
            rows.append(i)
            cols.append(k)
    M = len(universe)
    if M == 0:
        return np.zeros((N, N), dtype=np.float32)
    B = sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(N, M))
    inter = np.asarray((B @ B.T).todense(), dtype=np.float32)   # |Q_i ∩ Q_j|
    sizes = np.asarray(B.sum(axis=1)).ravel().astype(np.float32)
    union = sizes[:, None] + sizes[None, :] - inter
    with np.errstate(invalid="ignore", divide="ignore"):
        # empty union (both sets empty) => distance 0.0, matching the
        # reference loop's `if not union: 0.0` (NOT 1 - 0 = 1.0).
        D = np.where(union > 0, 1.0 - inter / union, 0.0).astype(np.float32)
    np.fill_diagonal(D, 0.0)
    return D


def iface_jaccard_residue_matrix(
    contacts_long: pd.DataFrame,
    model_ids: list[str],
    side: str = "epitope",
) -> np.ndarray:
    """N×N distance matrix on epitope (or paratope) residue sets.

    Parameters
    ----------
    contacts_long
        Long-format contacts table with columns
        ``model_id``, ``abc_chain``, ``abc_resnum`` (epitope side) — or for
        paratope use ``ab_chain`` / ``ab_resnum``.
    model_ids
        List of model_id strings; defines row/col order of returned matrix.
    side
        ``'epitope'`` (chain × resnum on Ag side) or
        ``'paratope'`` (Ab side). Column names probed accordingly.

    Returns
    -------
    np.ndarray
        N×N float matrix, D[i,j] = 1 − |Q_i ∩ Q_j| / |Q_i ∪ Q_j|.
        Diagonal = 0.0.

    Example (H1335, cluster 4 = gO core, cluster 5 = gO N-term)
    -----------------------------------------------------------
    Q_4 = {('A',405), ('A',423), ('A',425), ('A',427), ('A',432)}
    Q_5 = {('A',52),  ('A',54),  ('A',331), ('A',346)}
    intersect = ∅ → J = 0 → D = 1.0 (max distance, totally different
    epitope despite both being on chain A).
    """
    if side == "epitope":
        chain_col, res_col = "abc_chain", "abc_resnum"
    elif side == "paratope":
        chain_col, res_col = "ab_chain", "ab_resnum"
    else:
        raise ValueError("side must be 'epitope' or 'paratope'")

    for col in ("model_id", chain_col, res_col):
        if col not in contacts_long.columns:
            raise KeyError(f"contacts_long missing required column {col!r}")

    sets: dict[str, frozenset] = {}
    grouped = contacts_long.groupby("model_id")
    for mid in model_ids:
        if mid in grouped.groups:
            sub = grouped.get_group(mid)
            sets[mid] = frozenset(zip(sub[chain_col].tolist(), sub[res_col].astype(int).tolist()))
        else:
            sets[mid] = frozenset()

    return _jaccard_distance_from_sets(sets, model_ids)


# ============================================================================
# 2) Jaccard on contact pair sets
# ============================================================================


def iface_jaccard_contact_matrix(
    contacts_long: pd.DataFrame,
    model_ids: list[str],
) -> np.ndarray:
    """N×N distance matrix on contact-pair sets.

    Sharper than residue version: two models can share all paratope residues
    yet still differ on which epitope residue each one touches. Requires
    columns ``model_id``, ``ab_chain``, ``ab_resnum``, ``abc_chain``,
    ``abc_resnum``.

    D[i,j] = 1 - |E_i ∩ E_j| / |E_i ∪ E_j|
    """
    cols = ("model_id", "ab_chain", "ab_resnum", "abc_chain", "abc_resnum")
    for col in cols:
        if col not in contacts_long.columns:
            raise KeyError(f"contacts_long missing required column {col!r}")

    sets: dict[str, frozenset] = {}
    grouped = contacts_long.groupby("model_id")
    for mid in model_ids:
        if mid in grouped.groups:
            sub = grouped.get_group(mid)
            pairs = frozenset(
                zip(
                    sub["ab_chain"].tolist(),
                    sub["ab_resnum"].astype(int).tolist(),
                    sub["abc_chain"].tolist(),
                    sub["abc_resnum"].astype(int).tolist(),
                )
            )
            sets[mid] = pairs
        else:
            sets[mid] = frozenset()

    return _jaccard_distance_from_sets(sets, model_ids)


# ============================================================================
# 3) Symmetric Chamfer on paratope point clouds
# ============================================================================


def _extract_paratope_ca(
    cif_path: str | Path,
    ab_chains: Iterable[str],
    ag_chains: Iterable[str],
    cutoff: float = 5.0,
) -> np.ndarray:
    """Return paratope CA coordinates (Ab residues within cutoff of any Ag heavy atom).

    Lightweight: uses gemmi + a single CA→Ag scan. Returns (n, 3) array; empty
    array if no paratope residues found.
    """
    import gemmi
    from scipy.spatial import cKDTree

    st = gemmi.read_structure(str(cif_path))
    ab_set = set(ab_chains)
    ag_set = set(ag_chains)
    ab_ca: list[tuple[str, int, np.ndarray]] = []
    ag_heavy: list[np.ndarray] = []
    for model in st:
        for ch in model:
            if ch.name in ab_set:
                for r in ch:
                    ca = r.find_atom("CA", "*")
                    if ca is not None:
                        ab_ca.append((ch.name, r.seqid.num,
                                      np.array([ca.pos.x, ca.pos.y, ca.pos.z])))
            elif ch.name in ag_set:
                for r in ch:
                    for a in r:
                        if a.element.name == "H":
                            continue
                        ag_heavy.append(np.array([a.pos.x, a.pos.y, a.pos.z]))
        break

    if not ab_ca or not ag_heavy:
        return np.empty((0, 3), dtype=np.float32)

    ag_arr = np.asarray(ag_heavy)
    tree = cKDTree(ag_arr)
    paratope = []
    for _ch, _r, coord in ab_ca:
        # If any Ag heavy atom is within cutoff of this CA, mark as paratope.
        # CA→heavy distance < cutoff+~5 (residue radius slack).
        d, _ = tree.query(coord, k=1)
        if d <= cutoff + 5.0:
            paratope.append(coord)
    return np.asarray(paratope, dtype=np.float32)


def _symmetric_chamfer(A: np.ndarray, B: np.ndarray) -> float:
    """Symmetric Chamfer distance: mean(min ||a-b||^2)_A + mean(min ||a-b||^2)_B."""
    if len(A) == 0 or len(B) == 0:
        return float("nan")
    from scipy.spatial import cKDTree

    treeA, treeB = cKDTree(A), cKDTree(B)
    dAB, _ = treeB.query(A, k=1)
    dBA, _ = treeA.query(B, k=1)
    return float((dAB ** 2).mean() + (dBA ** 2).mean())


def _extract_worker(args):
    """Top-level wrapper for multiprocessing pickling."""
    p, ab_chains, ag_chains, cutoff = args
    return _extract_paratope_ca(p, ab_chains, ag_chains, cutoff)


# --- anchor-aware variant: align on the antigen, then compare paratope clouds ---


def _kabsch(P: np.ndarray, Q: np.ndarray):
    """Rigid transform (R, t) best-fitting P onto Q (corresponding (n,3) points)."""
    Pc, Qc = P - P.mean(0), Q - Q.mean(0)
    H = Pc.T @ Qc
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    t = Q.mean(0) - R @ P.mean(0)
    return R, t


def _extract_anchor_paratope(cif_path, ab_chains, ag_chains, anchor_chains, cutoff=5.0):
    """Return ``(anchor {(chain,resnum): CA xyz}, paratope (n,3) Ab CAs near Ag)``.

    The anchor CAs (one per residue, keyed by chain+resnum) let us Kabsch-align
    every model onto a common reference; the paratope cloud is then compared in
    that shared frame so the distance reflects *where the Ab sits on the Ag*.
    """
    import gemmi
    from scipy.spatial import cKDTree

    st = gemmi.read_structure(str(cif_path))
    ab_set, ag_set, anc_set = set(ab_chains), set(ag_chains), set(anchor_chains)
    ab_ca, ag_heavy = [], []
    anchor = {}
    for model in st:
        for ch in model:
            for r in ch:
                ca = r.find_atom("CA", "*")
                if ch.name in anc_set and ca is not None:
                    anchor[(ch.name, int(r.seqid.num))] = np.array(
                        [ca.pos.x, ca.pos.y, ca.pos.z], dtype=np.float64)
                if ch.name in ab_set and ca is not None:
                    ab_ca.append(np.array([ca.pos.x, ca.pos.y, ca.pos.z]))
                if ch.name in ag_set:
                    for a in r:
                        if a.element.name != "H":
                            ag_heavy.append(np.array([a.pos.x, a.pos.y, a.pos.z]))
        break
    if not ab_ca or not ag_heavy:
        return anchor, np.empty((0, 3), dtype=np.float32)
    tree = cKDTree(np.asarray(ag_heavy))
    para = [c for c in ab_ca if tree.query(c, k=1)[0] <= cutoff + 5.0]
    return anchor, np.asarray(para, dtype=np.float32)


def _extract_ap_worker(args):
    p, ab, ag, anchor, cutoff = args
    return _extract_anchor_paratope(p, ab, ag, anchor, cutoff)


def _align_paratope_clouds(extracted, ref_idx=0):
    """Kabsch-align each (anchor, paratope) onto the reference anchor frame.

    Returns a list of paratope clouds in the common frame (NaN-skipped where the
    anchor overlap with the reference is too small to fit).
    """
    ref_anchor, _ = extracted[ref_idx]
    aligned = []
    for anchor, para in extracted:
        if len(para) == 0 or not anchor:
            aligned.append(para)
            continue
        common = [k for k in anchor if k in ref_anchor]
        if len(common) < 3:
            aligned.append(para)  # can't align — leave as-is
            continue
        P = np.array([anchor[k] for k in common])
        Q = np.array([ref_anchor[k] for k in common])
        R, t = _kabsch(P, Q)
        aligned.append((R @ para.T).T + t)
    return aligned


def iface_chamfer_matrix(
    cif_paths: list[str | Path],
    ab_chains: Iterable[str],
    ag_chains: Iterable[str],
    cutoff: float = 5.0,
    workers: int = 8,
    anchor_chains: Optional[Iterable[str]] = None,
) -> np.ndarray:
    """N×N symmetric Chamfer distance on paratope CA point clouds.

    By default the structures are Kabsch-aligned on ``anchor_chains`` (the
    antigen, ``ag_chains``, if not specified) before the paratope clouds are
    compared — so the distance reflects *where the Ab binds on the Ag* (binding
    mode geometry), not the arbitrary global pose of each model. Pass
    ``anchor_chains=()`` to disable alignment (raw, frame-dependent — not
    recommended).

    Returns
    -------
    D : N×N float array
        D[i,j] in Å². D[i,i]=0.
    """
    import multiprocessing as mp

    ab_t, ag_t = tuple(ab_chains), tuple(ag_chains)
    anc_t = tuple(anchor_chains) if anchor_chains is not None else ag_t

    if anc_t:
        args = [(p, ab_t, ag_t, anc_t, cutoff) for p in cif_paths]
        if workers > 1 and len(cif_paths) > 1:
            with mp.Pool(workers) as pool:
                extracted = pool.map(_extract_ap_worker, args)
        else:
            extracted = [_extract_ap_worker(a) for a in args]
        # reference = first structure with both anchor + paratope
        ref_idx = next((i for i, (a, p) in enumerate(extracted) if a and len(p)), 0)
        clouds = _align_paratope_clouds(extracted, ref_idx=ref_idx)
    else:
        args = [(p, ab_t, ag_t, cutoff) for p in cif_paths]
        if workers > 1 and len(cif_paths) > 1:
            with mp.Pool(workers) as pool:
                clouds = pool.map(_extract_worker, args)
        else:
            clouds = [_extract_worker(a) for a in args]

    N = len(cif_paths)
    D = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        for j in range(i + 1, N):
            D[i, j] = D[j, i] = _symmetric_chamfer(clouds[i], clouds[j])
    return D


# ============================================================================
# 4) Wasserstein-2 (Sinkhorn) on paratope point clouds
# ============================================================================


def _sinkhorn_w2(A: np.ndarray, B: np.ndarray, reg: float = 0.1) -> float:
    """Sinkhorn approximation of W_2 distance between equal-weight point sets."""
    if len(A) == 0 or len(B) == 0:
        return float("nan")
    try:
        import ot
    except ImportError:
        warnings.warn(
            "POT library not installed — falling back to Chamfer. "
            "`pip install pot` to enable Wasserstein.",
            stacklevel=2,
        )
        return float(np.sqrt(_symmetric_chamfer(A, B)))

    M = ot.dist(A, B, metric="sqeuclidean")
    a = np.ones(len(A)) / len(A)
    b = np.ones(len(B)) / len(B)
    w2_sq = ot.sinkhorn2(a, b, M, reg=reg)
    return float(np.sqrt(max(w2_sq, 0.0)))


def iface_wasserstein_matrix(
    cif_paths: list[str | Path],
    ab_chains: Iterable[str],
    ag_chains: Iterable[str],
    cutoff: float = 5.0,
    reg: float = 0.1,
    workers: int = 8,
    anchor_chains: Optional[Iterable[str]] = None,
) -> np.ndarray:
    """N×N Sinkhorn-W_2 distance matrix on paratope CA point clouds.

    Anchor-aligned on ``anchor_chains`` (default ``ag_chains``) before comparison,
    like :func:`iface_chamfer_matrix`. Output unit: Å.
    """
    import multiprocessing as mp

    ab_t, ag_t = tuple(ab_chains), tuple(ag_chains)
    anc_t = tuple(anchor_chains) if anchor_chains is not None else ag_t

    if anc_t:
        args = [(p, ab_t, ag_t, anc_t, cutoff) for p in cif_paths]
        if workers > 1 and len(cif_paths) > 1:
            with mp.Pool(workers) as pool:
                extracted = pool.map(_extract_ap_worker, args)
        else:
            extracted = [_extract_ap_worker(a) for a in args]
        ref_idx = next((i for i, (a, p) in enumerate(extracted) if a and len(p)), 0)
        clouds = _align_paratope_clouds(extracted, ref_idx=ref_idx)
    else:
        args = [(p, ab_t, ag_t, cutoff) for p in cif_paths]
        if workers > 1 and len(cif_paths) > 1:
            with mp.Pool(workers) as pool:
                clouds = pool.map(_extract_worker, args)
        else:
            clouds = [_extract_worker(a) for a in args]

    N = len(cif_paths)
    D = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        for j in range(i + 1, N):
            D[i, j] = D[j, i] = _sinkhorn_w2(clouds[i], clouds[j], reg=reg)
    return D


# ============================================================================
# Convenience dispatcher
# ============================================================================


def compute_iface_matrix(
    metric: str,
    *,
    contacts_long: Optional[pd.DataFrame] = None,
    cif_paths: Optional[list[str | Path]] = None,
    model_ids: Optional[list[str]] = None,
    ab_chains: Optional[Iterable[str]] = None,
    ag_chains: Optional[Iterable[str]] = None,
    anchor_chains: Optional[Iterable[str]] = None,
    side: str = "epitope",
    cutoff: float = 5.0,
    reg: float = 0.1,
    workers: int = 8,
) -> np.ndarray:
    """Compute one of the four interface distance matrices.

    Parameters
    ----------
    metric : {'iface-jaccard-residue', 'iface-jaccard-contact',
              'iface-chamfer', 'iface-wasserstein'}
    contacts_long
        Required for Jaccard metrics.
    cif_paths
        Required for Chamfer / Wasserstein.
    model_ids
        Required for Jaccard metrics (row/col order).
    """
    if metric == "iface-jaccard-residue":
        if contacts_long is None or model_ids is None:
            raise ValueError("iface-jaccard-residue needs contacts_long + model_ids")
        return iface_jaccard_residue_matrix(contacts_long, model_ids, side=side)
    if metric == "iface-jaccard-contact":
        if contacts_long is None or model_ids is None:
            raise ValueError("iface-jaccard-contact needs contacts_long + model_ids")
        return iface_jaccard_contact_matrix(contacts_long, model_ids)
    if metric in ("iface-chamfer", "iface-wasserstein"):
        if cif_paths is None or ab_chains is None or ag_chains is None:
            raise ValueError(f"{metric} needs cif_paths + ab_chains + ag_chains")
        if metric == "iface-chamfer":
            return iface_chamfer_matrix(cif_paths, ab_chains, ag_chains, cutoff=cutoff,
                                        workers=workers, anchor_chains=anchor_chains)
        return iface_wasserstein_matrix(cif_paths, ab_chains, ag_chains, cutoff=cutoff,
                                        reg=reg, workers=workers, anchor_chains=anchor_chains)
    raise ValueError(f"unknown metric: {metric}")
