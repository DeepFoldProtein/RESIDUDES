"""gH-aligned regional RMSD generalised for any anchor/probe chain pair.

Generalized from H1335 ``41_filter_recluster_viz.py`` (the gO core RMSD check
that used 7LBE gH as a 100%-identical reference anchor).  The H1335 version
hard-coded chain ids (A=gH, C=gO) and search windows; this module accepts:

* arbitrary anchor / probe chain IDs in the predicted and reference CIFs,
* optional ``regions`` dict mapping ``label -> (lo_inclusive, hi_exclusive)``
  on the **predicted** probe chain numbering — the offset is searched so this
  maps onto the reference correctly.

Per-CIF returned dict::

    {
        'anchor_rmsd': float,          # gH-only Kabsch RMSD
        '<region>': float,             # regional probe RMSD after Kabsch
        '<region>_n': int,             # # residue pairs used for that region
        ...
    }

Errors return ``None`` for individual fields.
"""
from __future__ import annotations

import time
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd


# ---------- CA loaders --------------------------------------------------------


def _load_ca(cif_path: str | Path) -> dict[str, dict[int, np.ndarray]]:
    """Return ``{chain_id: {res_num: ndarray(3)}}`` for CA atoms only."""
    import gemmi

    st = gemmi.read_structure(str(cif_path))
    out: dict[str, dict[int, np.ndarray]] = {}
    for model in st:
        for ch in model:
            d: dict[int, np.ndarray] = {}
            for r in ch:
                ca = r.find_atom("CA", "*")
                if ca is not None:
                    d[r.seqid.num] = np.array([ca.pos.x, ca.pos.y, ca.pos.z])
            if d:
                out[ch.name] = d
        break
    return out


# ---------- math --------------------------------------------------------------


def kabsch(P: np.ndarray, Q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Standard Kabsch.  Aligns ``P`` onto ``Q``; returns rotation ``R`` and
    translation ``t`` such that ``Q ≈ R @ P + t``."""
    Pc = P.mean(0)
    Qc = Q.mean(0)
    H = (P - Pc).T @ (Q - Qc)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.eye(3)
    D[2, 2] = d
    R = Vt.T @ D @ U.T
    t = Qc - R @ Pc
    return R, t


def _best_offset(
    pred: dict[int, np.ndarray],
    ref: dict[int, np.ndarray],
    offsets: range,
    min_pairs: int,
    max_pairs: int = 500,
) -> tuple[Optional[float], Optional[np.ndarray], Optional[np.ndarray], Optional[int]]:
    """Search residue-number offset for max-overlap Kabsch alignment.

    Returns ``(rmsd, R, t, offset)`` for the best offset; ``(None, None, None, None)``
    if no offset has enough pairs.
    """
    best: tuple[float, Optional[np.ndarray], Optional[np.ndarray], Optional[int]] = (
        np.inf,
        None,
        None,
        None,
    )
    for off in offsets:
        pairs = [(pred[r], ref[r + off]) for r in pred if (r + off) in ref]
        if len(pairs) < min_pairs:
            continue
        P = np.array([p for p, _ in pairs[:max_pairs]])
        Q = np.array([q for _, q in pairs[:max_pairs]])
        R, t = kabsch(P, Q)
        diff = (R @ P.T).T + t - Q
        rmsd = float(np.sqrt((diff ** 2).sum(1).mean()))
        if rmsd < best[0]:
            best = (rmsd, R, t, off)
    if best[1] is None:
        return None, None, None, None
    return best[0], best[1], best[2], best[3]


# ---------- public API --------------------------------------------------------


def regional_rmsd(
    pred_cif: str | Path,
    ref_cif: str | Path,
    anchor_chain_pred: str,
    anchor_chain_ref: str,
    probe_chain_pred: str,
    probe_chain_ref: str,
    regions: Optional[dict[str, tuple[int, int]]] = None,
    anchor_offset_range: tuple[int, int] = (-50, 50),
    probe_offset_range: tuple[int, int] = (-60, 60),
    anchor_min_pairs: int = 100,
    probe_min_pairs: int = 50,
) -> dict[str, Any]:
    """Align predicted anchor onto reference anchor and measure probe RMSD.

    Parameters
    ----------
    pred_cif, ref_cif
        Predicted model + (single) reference structure.
    anchor_chain_pred, anchor_chain_ref
        Chain IDs of the anchor (e.g. ``B`` for gH on the prediction,
        ``A`` for gH on 7LBE).
    probe_chain_pred, probe_chain_ref
        Chain IDs of the probe being measured (e.g. ``A`` for gO on the
        prediction, ``C`` for gO on 7LBE).
    regions
        ``{name: (lo, hi)}`` half-open intervals on the predicted probe-chain
        residue numbering.  If ``None``, only the full-chain probe RMSD is
        returned under key ``'full'``.
    anchor_offset_range, probe_offset_range
        ``(lo, hi)`` for residue-number offset search.  Default: gH ±50, gO ±60.
    """
    try:
        pred = _load_ca(pred_cif)
        ref = _load_ca(ref_cif)
    except Exception as exc:
        return {"err": f"{type(exc).__name__}: {exc!s:.90}"}

    pred_anchor = pred.get(anchor_chain_pred, {})
    pred_probe = pred.get(probe_chain_pred, {})
    ref_anchor = ref.get(anchor_chain_ref, {})
    ref_probe = ref.get(probe_chain_ref, {})

    if not pred_anchor or not pred_probe or not ref_anchor or not ref_probe:
        return {"err": "missing chain"}

    a_rmsd, R, t, a_off = _best_offset(
        pred_anchor,
        ref_anchor,
        range(anchor_offset_range[0], anchor_offset_range[1]),
        min_pairs=min(anchor_min_pairs, max(10, int(0.3 * len(pred_anchor)))),
    )
    if R is None:
        return {"err": "anchor align failed"}

    # Build probe in reference frame (R applied to predicted probe).
    pred_probe_ref: dict[int, np.ndarray] = {
        r: (R @ xyz + t) for r, xyz in pred_probe.items()
    }

    # Search probe offset using full residue range.
    p_rmsd, _, _, p_off = _best_offset(
        pred_probe_ref,
        ref_probe,
        range(probe_offset_range[0], probe_offset_range[1]),
        min_pairs=probe_min_pairs,
    )
    if p_off is None:
        return {"anchor_rmsd": a_rmsd, "anchor_offset": a_off, "err": "probe align failed"}

    out: dict[str, Any] = {
        "anchor_rmsd": float(a_rmsd),
        "anchor_offset": int(a_off) if a_off is not None else None,
        "probe_offset": int(p_off),
    }

    if regions is None:
        regions_to_use: dict[str, tuple[int, int]] = {
            "full": (min(pred_probe.keys()), max(pred_probe.keys()) + 1),
        }
    else:
        regions_to_use = regions

    for label, (lo, hi) in regions_to_use.items():
        pairs = [
            (pred_probe_ref[r], ref_probe[r + p_off])
            for r in range(lo, hi)
            if r in pred_probe_ref and (r + p_off) in ref_probe
        ]
        if len(pairs) < 5:
            out[label] = None
            out[f"{label}_n"] = int(len(pairs))
            continue
        P = np.array([p for p, _ in pairs])
        Q = np.array([q for _, q in pairs])
        rmsd = float(np.sqrt(((P - Q) ** 2).sum(1).mean()))
        out[label] = rmsd
        out[f"{label}_n"] = int(len(pairs))

    return out


# ---------- worker-pool driver -----------------------------------------------


_REF_CA: Optional[dict[str, dict[int, np.ndarray]]] = None


def _init_worker(ref_cif: str) -> None:
    """Worker initialiser — cache reference CA dict once per child process."""
    global _REF_CA
    _REF_CA = _load_ca(ref_cif)


def _process_one(
    arg: tuple[str, str, str, str, str, str, Optional[dict[str, tuple[int, int]]]],
) -> dict[str, Any]:
    model_id, pred_cif, anchor_p, anchor_r, probe_p, probe_r, regions = arg
    out: dict[str, Any] = {"model_id": model_id}
    try:
        pred = _load_ca(pred_cif)
    except Exception as exc:
        out["err"] = f"{type(exc).__name__}: {exc!s:.90}"
        return out
    pred_anchor = pred.get(anchor_p, {})
    pred_probe = pred.get(probe_p, {})
    if _REF_CA is None or not pred_anchor or not pred_probe:
        out["err"] = "missing chain"
        return out
    ref_anchor = _REF_CA.get(anchor_r, {})
    ref_probe = _REF_CA.get(probe_r, {})
    if not ref_anchor or not ref_probe:
        out["err"] = "missing reference chain"
        return out

    a_rmsd, R, t, a_off = _best_offset(
        pred_anchor,
        ref_anchor,
        range(-50, 50),
        min_pairs=min(100, max(10, int(0.3 * len(pred_anchor)))),
    )
    if R is None:
        out["err"] = "anchor align failed"
        return out
    pred_probe_ref = {r: (R @ xyz + t) for r, xyz in pred_probe.items()}
    p_rmsd, _, _, p_off = _best_offset(
        pred_probe_ref,
        ref_probe,
        range(-60, 60),
        min_pairs=50,
    )
    out["anchor_rmsd"] = float(a_rmsd)
    if p_off is None:
        out["err"] = "probe align failed"
        return out

    if regions is None:
        regions_to_use = {
            "full": (min(pred_probe.keys()), max(pred_probe.keys()) + 1),
        }
    else:
        regions_to_use = regions

    for label, (lo, hi) in regions_to_use.items():
        pairs = [
            (pred_probe_ref[r], ref_probe[r + p_off])
            for r in range(lo, hi)
            if r in pred_probe_ref and (r + p_off) in ref_probe
        ]
        if len(pairs) < 5:
            out[label] = None
            out[f"{label}_n"] = int(len(pairs))
            continue
        P = np.array([p for p, _ in pairs])
        Q = np.array([q for _, q in pairs])
        out[label] = float(np.sqrt(((P - Q) ** 2).sum(1).mean()))
        out[f"{label}_n"] = int(len(pairs))
    return out


def run_regional_rmsd_parallel(
    index_df: pd.DataFrame,
    ref_cif: str | Path,
    anchor_chain_pred: str,
    anchor_chain_ref: str,
    probe_chain_pred: str,
    probe_chain_ref: str,
    regions: Optional[dict[str, tuple[int, int]]] = None,
    workers: int = 8,
    progress_every: int = 200,
) -> pd.DataFrame:
    """Pool-parallel regional RMSD across an index of predictions.

    Returns a DataFrame keyed on ``model_id`` with one column per region plus
    ``anchor_rmsd`` and ``anchor_offset`` / ``probe_offset``.
    """
    records: list[tuple[str, str, str, str, str, str, Optional[dict[str, tuple[int, int]]]]] = [
        (
            r["model_id"],
            r["src_cif"],
            anchor_chain_pred,
            anchor_chain_ref,
            probe_chain_pred,
            probe_chain_ref,
            regions,
        )
        for _, r in index_df.iterrows()
    ]
    n = len(records)
    print(
        f"  regional RMSD: {n} cifs  workers={workers}  "
        f"anchor={anchor_chain_pred}->{anchor_chain_ref}  "
        f"probe={probe_chain_pred}->{probe_chain_ref}  regions={list(regions or ['full'])}",
        flush=True,
    )
    t0 = time.time()
    rows: list[dict[str, Any]] = []
    n_err = 0
    with Pool(workers, initializer=_init_worker, initargs=(str(ref_cif),)) as pool:
        for i, res in enumerate(
            pool.imap_unordered(_process_one, records, chunksize=10), 1
        ):
            rows.append(res)
            if "err" in res:
                n_err += 1
            if i % progress_every == 0 or i == n:
                rate = i / (time.time() - t0 + 1e-6)
                eta = (n - i) / rate / 60 if rate else 0.0
                print(
                    f"    [{i}/{n}] {rate:.1f} cif/s  ETA {eta:.1f} min  err={n_err}",
                    flush=True,
                )
    df = pd.DataFrame(rows)
    print(
        f"  regional RMSD: shape={df.shape}  errors={n_err}  "
        f"({(time.time()-t0)/60:.1f} min)",
        flush=True,
    )
    return df
