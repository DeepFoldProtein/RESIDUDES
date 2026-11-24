"""aADARS v3 scoring — CDR/framework energetics for Fab targets.

Adapted from H1335 ``25_h1335_aadars.py``.  ANARCI is required for CDR
annotation; this means the worker process must have HMMER on PATH.  In the
H1335 workflow this is achieved by running the wrapper script with
``PATH=/gpfs/deepfold/casp/casp17/analysis/.venv-cdr/bin:$PATH``.  No code
changes here — :func:`AADARSv3Scorer` will surface a helpful error if HMMER is
missing.

This module is **Ab mode only**.  Callers should skip it when
``ChainConfig.mode == 'ppi'``.

Output columns (subset; ``score`` may add more — every returned key is copied
through with an ``aadars_`` prefix if it doesn't already have one):

* per-CDR energies/counts/fracs
    ``aadars_e_cdr_{h1,h2,h3,l1,l2,l3}``
    ``aadars_n_cdr_{h1,h2,h3,l1,l2,l3}``
    ``aadars_frac_cdr_{h1,h2,h3,l1,l2,l3}``
* framework
    ``aadars_e_framework``, ``aadars_n_framework``, ``aadars_frac_framework``
* total
    ``aadars_e_total_v3``
"""
from __future__ import annotations

import sys
import time
import warnings
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Iterable, Optional

import pandas as pd

EMA_PATH = str(__import__("pathlib").Path(__file__).resolve().parents[3])  # residudes-qa/src (package importable from here)

_WORKER_STATE: dict[str, Any] = {}


def _init_worker(ema_path: str = EMA_PATH) -> None:
    if ema_path not in sys.path:
        sys.path.insert(0, ema_path)
    warnings.filterwarnings("ignore")
    from residudes_qa.ema.scorers.aadars_v3_scorer import AADARSv3Scorer
    _WORKER_STATE["scorer"] = AADARSv3Scorer()


def run_aadars(
    cif_path: str | Path,
    heavy_chain: str,
    light_chain: Optional[str],
    ag_chains: Iterable[str],
) -> dict:
    """Score a single CIF with aADARS v3.  Requires :func:`_init_worker`.

    Returns a dict with all keys from :meth:`AADARSv3Scorer.score`, prefixed
    with ``aadars_`` if not already present, all lowercased.  On failure
    returns ``{'aadars_err': '<exc>'}``.
    """
    out: dict[str, Any] = {"cif_path": str(cif_path)}
    try:
        r = _WORKER_STATE["scorer"].score(
            str(cif_path),
            heavy=heavy_chain,
            light=light_chain,
            ag_chains=list(ag_chains),
        )
        for k, v in r.items():
            col = k if k.startswith("aadars") else f"aadars_{k}"
            out[col.lower()] = v
    except Exception as e:  # noqa: BLE001
        out["aadars_err"] = f"{type(e).__name__}: {str(e)[:90]}"
    return out


# ----- worker-pool plumbing --------------------------------------------------


def _process_one(arg: tuple[str, str, str, Optional[str], list[str]]) -> dict:
    model_id, cif_path, heavy, light, ag_chains = arg
    out = {"model_id": model_id}
    out.update(run_aadars(cif_path, heavy, light, ag_chains))
    return out


def run_aadars_parallel(
    index_df: pd.DataFrame,
    heavy: str,
    light: Optional[str],
    ag_chains: Iterable[str],
    workers: int = 8,
    ema_path: str = EMA_PATH,
    progress_every: int = 100,
) -> pd.DataFrame:
    """Pool-parallel aADARS v3 over ``index_df``.

    Parameters
    ----------
    index_df
        DataFrame with at least ``model_id`` and ``src_cif`` columns.
    heavy, light, ag_chains
        Chain IDs.  ``light`` may be None for VHH / single-chain antibodies.
    """
    ag_list = list(ag_chains)
    records = [
        (r["model_id"], r["src_cif"], heavy, light, ag_list)
        for _, r in index_df.iterrows()
    ]
    n = len(records)
    print(
        f"  aadars features: {n} cifs  workers={workers}  "
        f"heavy={heavy} light={light} ag={ag_list}",
        flush=True,
    )
    t0 = time.time()
    rows: list[dict] = []
    n_err = 0
    with Pool(workers, initializer=_init_worker, initargs=(ema_path,)) as pool:
        for i, res in enumerate(
            pool.imap_unordered(_process_one, records, chunksize=10), 1
        ):
            rows.append(res)
            if "aadars_err" in res:
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
        f"  aadars features: shape={df.shape}  errors={n_err}  "
        f"({(time.time()-t0)/60:.1f} min)",
        flush=True,
    )
    return df


def save_aadars_features(df: pd.DataFrame, output_dir: str | Path) -> Path:
    """Write ``<output_dir>/stage_outputs/aadars_features.parquet``."""
    out_dir = Path(output_dir) / "stage_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "aadars_features.parquet"
    df.to_parquet(out, index=False)
    return out
