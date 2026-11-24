"""CPU feature extraction — hbond network, bbdep chi, NDRD.

Adapted from H1335 ``23_h1335_cpu.py`` (mode='cpu' branch).  The three CPU
scorers come from the EMA package:

* ``residudes_qa.ema.scorers.hbond_network.analyze_hbond_network``
* ``residudes_qa.ema.scorers.bbdep_chi_spline_v2.score_bbdep_chi_v2``
* ``residudes_qa.ema.scorers.ndrd_scorer.score_ndrd``

These are imported lazily inside :func:`_init_worker` so the parent process
need not be able to ``import EMA``.

Output columns:

* hbond  — ``hbond_n_total``, ``hbond_n_helix``, ``hbond_n_sheet``,
  ``hbond_largest_comp``, ``hbond_quality``, ``hbond_network_score``
* bbdep  — ``bbdep_chi_grid_per_res``, ``bbdep_chi_bilinear_per_res``
* ndrd   — ``ndrd_per_res``, ``ndrd_outlier_frac``, ``ndrd_n_outliers``

Plus per-scorer ``*_err`` strings when a single scorer crashes.
"""
from __future__ import annotations

import sys
import time
import warnings
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Path to the user's CASP17 EMA checkout; injected into workers via sys.path.
EMA_PATH = str(__import__("pathlib").Path(__file__).resolve().parents[3])  # residudes-qa/src (package importable from here)

_WORKER_STATE: dict[str, Any] = {}


def _init_worker(ema_path: str = EMA_PATH) -> None:
    """Worker initializer — import the three CPU scorers once per child."""
    if ema_path not in sys.path:
        sys.path.insert(0, ema_path)
    warnings.filterwarnings("ignore")
    from residudes_qa.ema.scorers.bbdep_chi_spline_v2 import score_bbdep_chi_v2
    from residudes_qa.ema.scorers.hbond_network import analyze_hbond_network
    from residudes_qa.ema.scorers.ndrd_scorer import score_ndrd
    _WORKER_STATE["hbond"] = analyze_hbond_network
    _WORKER_STATE["bbdep"] = score_bbdep_chi_v2
    _WORKER_STATE["ndrd"] = score_ndrd


def run_cpu_features(cif_path: str | Path) -> dict:
    """Score a single CIF with hbond + bbdep + ndrd.

    Must be called from a process where :func:`_init_worker` has populated
    ``_WORKER_STATE``.  Returns a dict with the columns described in the
    module docstring; any failing scorer populates a ``*_err`` field and the
    other scorers still run.
    """
    out: dict[str, Any] = {"cif_path": str(cif_path)}
    try:
        r = _WORKER_STATE["hbond"](str(cif_path))
        out["hbond_n_total"] = int(r.get("n_hbonds", 0))
        out["hbond_n_helix"] = int(r.get("n_helix_hbonds", 0))
        out["hbond_n_sheet"] = int(r.get("n_sheet_hbonds", 0))
        out["hbond_largest_comp"] = int(r.get("largest_component", 0))
        out["hbond_quality"] = float(r.get("hbond_quality", np.nan))
        out["hbond_network_score"] = float(r.get("network_score", np.nan))
    except Exception as e:  # noqa: BLE001
        out["hbond_err"] = f"{type(e).__name__}: {str(e)[:80]}"
    try:
        r = _WORKER_STATE["bbdep"](str(cif_path))
        out["bbdep_chi_grid_per_res"] = float(
            r.get("E_bbdep_chi_v2_grid_per_res", np.nan)
        )
        out["bbdep_chi_bilinear_per_res"] = float(
            r.get("E_bbdep_chi_v2_bilinear_per_res", np.nan)
        )
    except Exception as e:  # noqa: BLE001
        out["bbdep_chi_err"] = f"{type(e).__name__}: {str(e)[:80]}"
    try:
        r = _WORKER_STATE["ndrd"](str(cif_path))
        out["ndrd_per_res"] = float(r.get("E_ndrd_per_res", np.nan))
        out["ndrd_outlier_frac"] = float(r.get("ndrd_outlier_frac", np.nan))
        out["ndrd_n_outliers"] = int(r.get("n_ndrd_outliers", 0))
    except Exception as e:  # noqa: BLE001
        out["ndrd_err"] = f"{type(e).__name__}: {str(e)[:80]}"
    return out


def _process_one(arg: tuple[str, str]) -> dict:
    model_id, cif_path = arg
    out = {"model_id": model_id}
    out.update(run_cpu_features(cif_path))
    return out


def run_cpu_features_parallel(
    index_df: pd.DataFrame,
    workers: int = 8,
    ema_path: str = EMA_PATH,
    progress_every: int = 100,
) -> pd.DataFrame:
    """Pool-parallel driver over ``index_df`` (needs ``model_id``, ``src_cif``)."""
    records = [(r["model_id"], r["src_cif"]) for _, r in index_df.iterrows()]
    n = len(records)
    print(f"  cpu features: {n} cifs  workers={workers}", flush=True)
    t0 = time.time()
    rows: list[dict] = []
    n_err = 0
    with Pool(workers, initializer=_init_worker, initargs=(ema_path,)) as pool:
        for i, res in enumerate(
            pool.imap_unordered(_process_one, records, chunksize=10), 1
        ):
            rows.append(res)
            if any(k.endswith("_err") for k in res):
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
        f"  cpu features: shape={df.shape}  errors={n_err}  "
        f"({(time.time()-t0)/60:.1f} min)",
        flush=True,
    )
    return df


def save_cpu_features(df: pd.DataFrame, output_dir: str | Path) -> Path:
    """Write ``<output_dir>/stage_outputs/cpu_features.parquet``."""
    out_dir = Path(output_dir) / "stage_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "cpu_features.parquet"
    df.to_parquet(out, index=False)
    return out
