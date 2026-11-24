"""DFIRE v2 GPU scoring.

Adapted from the ``mode='dfire'`` branch of H1335 ``23_h1335_cpu.py``.
``DFIREv2Scorer`` is GPU-resident, so :func:`run_dfire_parallel` defaults to
``workers=2`` to avoid OOM on a single 24 GB GPU.  The H1335 stage-5 sbatch
also uses --workers 2 for the dfire-only re-run.

Output columns: ``dfire_v2_total``, ``dfire_v2_per_atom``, ``dfire_v2_n_atoms``
(plus ``dfire_err`` on failure).
"""
from __future__ import annotations

import os
import sys
import time
import warnings
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

EMA_PATH = str(__import__("pathlib").Path(__file__).resolve().parents[3])  # residudes-qa/src (package importable from here)

_WORKER_STATE: dict[str, Any] = {}


def _init_worker(ema_path: str = EMA_PATH, gpu: int | None = None) -> None:
    """Worker init: pin GPU, import + warm DFIRE scorer once per child."""
    if gpu is not None and "SLURM_JOB_ID" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    if ema_path not in sys.path:
        sys.path.insert(0, ema_path)
    warnings.filterwarnings("ignore")
    from residudes_qa.ema.scorers.dfire_v2_scorer import DFIREv2Scorer
    _WORKER_STATE["dfire"] = DFIREv2Scorer()


def run_dfire(cif_path: str | Path) -> dict:
    """Score one CIF with DFIRE v2.  Requires :func:`_init_worker` to have run."""
    out: dict[str, Any] = {"cif_path": str(cif_path)}
    try:
        r = _WORKER_STATE["dfire"].score(str(cif_path))
        out["dfire_v2_per_atom"] = float(r.get("E_dfire_v2_per_atom", np.nan))
        out["dfire_v2_total"] = float(r.get("E_dfire_v2", np.nan))
        out["dfire_v2_n_atoms"] = int(r.get("n_atoms", 0))
    except Exception as e:  # noqa: BLE001
        out["dfire_err"] = f"{type(e).__name__}: {str(e)[:80]}"
    return out


def _process_one(arg: tuple[str, str]) -> dict:
    model_id, cif_path = arg
    out = {"model_id": model_id}
    out.update(run_dfire(cif_path))
    return out


def run_dfire_parallel(
    index_df: pd.DataFrame,
    workers: int = 2,
    gpu: int | None = 0,
    ema_path: str = EMA_PATH,
    progress_every: int = 100,
) -> pd.DataFrame:
    """Run DFIRE v2 over the index with a small pool (default 2 workers).

    ``workers=2`` is the empirically safe default for a single 24 GB GPU
    (H1335 retry sbatch uses the same value).  Set ``gpu=None`` to defer
    GPU choice to ``CUDA_VISIBLE_DEVICES`` already in the environment.
    """
    records = [(r["model_id"], r["src_cif"]) for _, r in index_df.iterrows()]
    n = len(records)
    print(
        f"  dfire features: {n} cifs  workers={workers}  gpu={gpu}",
        flush=True,
    )
    t0 = time.time()
    rows: list[dict] = []
    n_err = 0
    with Pool(
        workers,
        initializer=_init_worker,
        initargs=(ema_path, gpu),
    ) as pool:
        for i, res in enumerate(
            pool.imap_unordered(_process_one, records, chunksize=4), 1
        ):
            rows.append(res)
            if "dfire_err" in res:
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
        f"  dfire features: shape={df.shape}  errors={n_err}  "
        f"({(time.time()-t0)/60:.1f} min)",
        flush=True,
    )
    return df


def save_dfire_features(df: pd.DataFrame, output_dir: str | Path) -> Path:
    """Write ``<output_dir>/stage_outputs/dfire_features.parquet``."""
    out_dir = Path(output_dir) / "stage_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "dfire_features.parquet"
    df.to_parquet(out, index=False)
    return out
