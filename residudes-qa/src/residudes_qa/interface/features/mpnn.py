"""ProteinMPNN scoring on the full complex.

Adapted from H1335 ``22_h1335_mpnn.py``.  The scorer (``MPNNScorer``) is a
GPU-resident model so we run it **sequentially** in a single process — at
roughly 0.3 cif/s on a single GPU.  No multiprocessing here; H1335 used
slurm-array shards (``--shard A/B``) which the caller can still do by passing
a pre-split ``index_df``.

For each CIF we compute:

* ``mpnn_score``                — sequence-recovery / negative-log-likelihood
* ``overall_recovery``          — fraction of native AAs predicted correctly
* ``mpnn_iface_all_mean``       — per-residue mpnn score averaged over the
                                   ab–ag interface (Cβ within 8 Å)
* ``mpnn_iface_n_residues``     — interface residue count
* ``mpnn_iface_recovery_mean``  — recovery averaged over the interface

In **PPI mode** (``ab_chains=[]``) the interface aggregates are computed over
all chain pairs (matching :mod:`residudes_qa.interface.features.base`).
"""
from __future__ import annotations

import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd

EMA_PATH = str(__import__("pathlib").Path(__file__).resolve().parents[3])  # residudes-qa/src (package importable from here)

AA_3TO1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}
CONTACT_CB_CUTOFF = 8.0


# ----- helpers (chain ↔ flat residue index, Cβ interface mask) ---------------


def _collect_chain_geometry(structure) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Single-pass extraction of ``(chain_idx, chain_cb)`` from a gemmi
    ``Structure``.  Replaces what used to be two separate gemmi reads.

    Returns
    -------
    chain_idx
        ``{chain_id: flat_residue_indices}`` matching MPNN's flat residue
        order (CA-containing standard AAs only).
    chain_cb
        ``{chain_id: (n, 3) array of Cβ coords (or CA if Cβ missing)}``.
    """
    chain_idx: dict[str, np.ndarray] = {}
    chain_cb: dict[str, np.ndarray] = {}
    cur = 0
    for model in structure:
        for ch in model:
            idxs: list[int] = []
            cb: list[list[float]] = []
            for r in ch:
                if r.name not in AA_3TO1:
                    continue
                ca = r.find_atom("CA", "*")
                if ca is None:
                    continue
                cbat = r.find_atom("CB", "*") or ca
                idxs.append(cur)
                cur += 1
                cb.append([cbat.pos.x, cbat.pos.y, cbat.pos.z])
            if idxs:
                chain_idx[ch.name] = np.asarray(idxs, dtype=np.int32)
                chain_cb[ch.name] = np.asarray(cb, dtype=np.float32)
        break
    return chain_idx, chain_cb


def _compute_iface_mask(
    chain_cb: dict[str, np.ndarray],
    ab_chains: list[str],
    ag_chains: list[str],
    chain_idx: dict[str, np.ndarray],
    cutoff: float = CONTACT_CB_CUTOFF,
) -> np.ndarray:
    """Boolean mask over flat residues; True for residues at the ab×ag iface.

    Operates on pre-extracted ``chain_cb`` (no gemmi re-read).
    """
    total_n = sum(len(v) for v in chain_idx.values())
    mask = np.zeros(total_n, dtype=bool)
    cut2 = cutoff ** 2
    for ab in ab_chains:
        if ab not in chain_cb:
            continue
        for ag in ag_chains:
            if ag == ab or ag not in chain_cb:
                continue
            d2 = np.sum(
                (chain_cb[ab][:, None, :] - chain_cb[ag][None, :, :]) ** 2,
                axis=-1,
            )
            ab_iface = (d2 <= cut2).any(axis=1)
            ag_iface = (d2 <= cut2).any(axis=0)
            ab_idx = chain_idx.get(ab)
            ag_idx = chain_idx.get(ag)
            if ab_idx is not None:
                mask[ab_idx[ab_iface]] = True
            if ag_idx is not None:
                mask[ag_idx[ag_iface]] = True
    return mask


# ----- per-cif scoring -------------------------------------------------------


def run_mpnn(
    cif_path: str | Path,
    ab_chains: Iterable[str],
    ag_chains: Iterable[str],
    scorer: Any,
) -> dict:
    """Score one CIF with the given (already-loaded) MPNN scorer.

    Parameters
    ----------
    cif_path
        CIF to score.
    ab_chains, ag_chains
        Chain whitelists for the interface aggregates.  In PPI mode pass
        ``ab_chains=[]`` and ``ag_chains=`` the full chain list — the helper
        will form all-vs-all pairs (excluding self) when computing the mask.
    scorer
        Pre-initialized ``residudes_qa.ema.scorers.proteinmpnn_score.MPNNScorer`` instance.

    Returns dict with ``mpnn_score``, ``overall_recovery``,
    ``mpnn_iface_all_mean``, ``mpnn_iface_n_residues``,
    ``mpnn_iface_recovery_mean``.  On failure: ``{'mpnn_err': '...'}``.
    """
    import gemmi
    ab_list = list(ab_chains)
    ag_list = list(ag_chains)
    out: dict[str, Any] = {"cif_path": str(cif_path)}
    try:
        r = scorer.score_structure(str(cif_path))
        out["mpnn_score"] = float(r.mpnn_score)
        out["overall_recovery"] = float(r.overall_recovery)
        per_res = np.asarray(r.per_residue_score, dtype=np.float32)
        # Single gemmi parse for both chain_idx and chain_cb, replacing
        # what used to be 1 + N (=#chains in PPI mode) separate reads.
        st = gemmi.read_structure(str(cif_path))
        chain_idx, chain_cb = _collect_chain_geometry(st)
        # PPI mode: treat all chains as one "group" and form all pairs.
        if not ab_list:
            chains_present = [c for c in ag_list if c in chain_idx] or list(chain_idx)
            mask = np.zeros(len(per_res), dtype=bool)
            for i in range(len(chains_present)):
                m = _compute_iface_mask(
                    chain_cb,
                    [chains_present[i]],
                    chains_present[i + 1:],
                    chain_idx,
                )
                if len(m) == len(mask):
                    mask |= m
        else:
            mask = _compute_iface_mask(chain_cb, ab_list, ag_list, chain_idx)
        if mask.any() and len(mask) == len(per_res):
            out["mpnn_iface_all_mean"] = float(per_res[mask].mean())
            out["mpnn_iface_n_residues"] = int(mask.sum())
            out["mpnn_iface_recovery_mean"] = float(
                np.asarray(r.per_residue_recovery)[mask].mean()
            )
        else:
            out["mpnn_iface_all_mean"] = None
            out["mpnn_iface_n_residues"] = 0
            out["mpnn_iface_recovery_mean"] = None
    except Exception as e:  # noqa: BLE001
        out["mpnn_err"] = f"{type(e).__name__}: {str(e)[:80]}"
    return out


# ----- sequential driver -----------------------------------------------------


def _load_scorer(ema_path: str = EMA_PATH):
    """Lazy import + init of the GPU-resident MPNN scorer.

    Uses ``skip_sampling=True``: XGB only consumes forward()-derived NLL
    features (``mpnn_score``, ``mpnn_iface_all_mean``); recovery features
    contribute <0.3% of total gain and ablation across 4 targets +
    35k-row training master showed identical top-5 picks.  Skipping the
    AR sample loop removes ~80% of per-cif wall time.
    """
    if ema_path not in sys.path:
        sys.path.insert(0, ema_path)
    warnings.filterwarnings("ignore")
    from residudes_qa.ema.scorers.proteinmpnn_score import MPNNScorer
    return MPNNScorer(num_sequences=1, sampling_temp=0.1, skip_sampling=True)


def run_mpnn_sequential(
    index_df: pd.DataFrame,
    ab_chains: Iterable[str],
    ag_chains: Iterable[str],
    gpu: Optional[int] = 0,
    ema_path: str = EMA_PATH,
    progress_every: int = 100,
) -> pd.DataFrame:
    """Sequential MPNN over ``index_df``.  Slow (~0.3 cif/s on one GPU).

    GPU pinning is only done when not running under SLURM (which already sets
    ``CUDA_VISIBLE_DEVICES``).
    """
    if gpu is not None and "SLURM_JOB_ID" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    records = [
        {"model_id": r["model_id"], "src_cif": r["src_cif"]}
        for _, r in index_df.iterrows()
    ]
    n = len(records)
    t0 = time.time()
    print(f"  mpnn features: {n} cifs  gpu={gpu}", flush=True)
    print(f"  loading MPNN scorer...", flush=True)
    scorer = _load_scorer(ema_path)
    print(f"  MPNN initialized ({time.time()-t0:.1f}s)", flush=True)

    ab_list = list(ab_chains)
    ag_list = list(ag_chains)
    rows: list[dict] = []
    n_err = 0
    t1 = time.time()
    for i, rec in enumerate(records, 1):
        row = {"model_id": rec["model_id"]}
        row.update(run_mpnn(rec["src_cif"], ab_list, ag_list, scorer))
        rows.append(row)
        if "mpnn_err" in row:
            n_err += 1
        if i % progress_every == 0 or i == n:
            rate = i / (time.time() - t1 + 1e-6)
            eta = (n - i) / rate / 60 if rate else 0.0
            print(
                f"    [{i}/{n}] {rate:.1f} cif/s  ETA {eta:.1f} min  err={n_err}",
                flush=True,
            )

    df = pd.DataFrame(rows)
    print(
        f"  mpnn features: shape={df.shape}  errors={n_err}  "
        f"({(time.time()-t0)/60:.1f} min)",
        flush=True,
    )
    return df


def save_mpnn_features(df: pd.DataFrame, output_dir: str | Path) -> Path:
    """Write ``<output_dir>/stage_outputs/mpnn_features.parquet``."""
    out_dir = Path(output_dir) / "stage_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "mpnn_features.parquet"
    df.to_parquet(out, index=False)
    return out
