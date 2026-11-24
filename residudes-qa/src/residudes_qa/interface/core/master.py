"""Master feature join — consolidate per-stage parquets / CSVs into ``master.parquet``.

Adapted from H1335 ``27_join_features.py``.  Joins every stage output keyed on
``model_id`` and writes a single wide ``master.parquet`` for downstream
xgb / filter / cluster / pick stages.

Layout assumed::

    <stage_outputs_dir>/
        base_features.parquet
        cpu_features*.parquet
        dfire_features*.parquet
        aadars_features*.{parquet,csv}        (optional — Ab mode only)
        mpnn_features*.parquet                (optional)
        clash_features*.parquet               (optional, written by filter.clash)

Each stage may produce multiple shard files (e.g. mpnn array-job output); we
glob the wildcard names and concat with ``drop_duplicates(subset='model_id',
keep='last')`` so re-runs / patch files override originals.
"""
from __future__ import annotations

import glob
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

# Canonical method-master mapping used by xgb_v8 / xgb_ppi (matches H1335).
# Variant ID is kept as a meta column for filtering/cluster analysis.
METHOD_MASTER_MAP: dict[str, str] = {
    "af3": "af3",
    "bt2": "b2",
    "bt2_jax": "b2",
    "of3": "of3",
    "pt2": "v2",
    "ef2": "ef2",        # ESMFold2 single-seq (PPI v5 / AgAb v9)
    "esmfold2": "ef2",
    "single_seq": "ef2",
    # MassiveFold AF2-multimer models (1..5) — closest train category is af3
    # (both AlphaFold-family); xgb_v8 has no af2 one-hot so we coalesce.
    "af2m1": "af3", "af2m2": "af3", "af2m3": "af3",
    "af2m4": "af3", "af2m5": "af3",
    "afm": "af3",  # generic AF2-multimer fallback
}

# Columns to drop after merge — they duplicate index columns or are bulky.
_DROP_AFTER_MERGE: set[str] = {
    "cif_path",
    "summary_path",
    "unit_cif",
    "err",
    "dfire_v2_err",
    "hbond_err",
    "bbdep_chi_err",
    "ndrd_err",
    "aadars_err",
}


def _glob_concat(
    stage_dir: Path,
    pattern: str,
    reader: Callable[[str], pd.DataFrame],
) -> Optional[pd.DataFrame]:
    """Glob ``stage_dir/<pattern>`` and concat all matches.

    Returns ``None`` if no files match.  Deduplicates on ``model_id``
    keeping the **last** occurrence so user-supplied patch shards override
    base shards.
    """
    paths = sorted(glob.glob(str(stage_dir / pattern)))
    if not paths:
        return None
    frames = [reader(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    if "model_id" in df.columns:
        df = df.drop_duplicates(subset="model_id", keep="last")
    return df


def _load_stage(stage_dir: Path, name: str) -> Optional[pd.DataFrame]:
    """Load a stage output, trying both parquet and csv variants.

    Concrete patterns tried (in order)::

        <name>.parquet
        <name>_*.parquet
        <name>.csv
        <name>_*.csv
    """
    df = None
    # parquet first
    for pat in (f"{name}.parquet", f"{name}_*.parquet"):
        df = _glob_concat(stage_dir, pat, pd.read_parquet)
        if df is not None:
            break
    if df is None:
        for pat in (f"{name}.csv", f"{name}_*.csv"):
            df = _glob_concat(stage_dir, pat, pd.read_csv)
            if df is not None:
                break
    # 'device' is per-stage provenance (recorded in the stage parquet + run_meta
    # + stamped on xgb_scores); strip it here so it never pollutes master with a
    # single misleading value when stages ran on different hardware.
    if df is not None and "device" in df.columns:
        df = df.drop(columns=["device"])
    return df


def join_features(
    index_df: pd.DataFrame,
    stage_outputs_dir: str | Path,
    mode: str = "ab",
) -> pd.DataFrame:
    """Join all per-stage feature outputs onto ``index_df``.

    Parameters
    ----------
    index_df
        Index DataFrame from :func:`residudes_qa.interface.core.discover.discover_cifs`.  Must
        contain at minimum ``model_id`` plus the meta columns (``target``,
        ``variant``, ``method``, ``seed``, ``sample``, ``src_cif``).
    stage_outputs_dir
        Directory containing per-stage outputs (typically
        ``<output>/stage_outputs``).
    mode
        ``'ab'`` or ``'ppi'``.  In PPI mode, aADARS shards are not expected
        and a missing aADARS stage does not warn.

    Returns
    -------
    pd.DataFrame
        One row per ``model_id`` in ``index_df`` (left-merged), with all
        available feature columns plus ``method_master`` (xgb-canonical
        method name).
    """
    stage_dir = Path(stage_outputs_dir)
    if not stage_dir.is_dir():
        raise FileNotFoundError(f"stage_outputs_dir not found: {stage_dir}")

    # Anchor on the index — left-merge each block so the row set is preserved.
    m = index_df.copy()
    n_idx = len(m)
    print(f"  index: {n_idx} rows  mode={mode}", flush=True)

    stages = ["base", "cpu", "dfire", "aadars", "mpnn", "clash"]
    for stage in stages:
        # Try several naming conventions used across the feature modules.
        df_stage: Optional[pd.DataFrame] = None
        for prefix in (stage, f"{stage}_features"):
            df_stage = _load_stage(stage_dir, prefix)
            if df_stage is not None:
                break
        if df_stage is None:
            if stage == "aadars" and mode == "ppi":
                print(f"  {stage:6s}: skipped (ppi mode)", flush=True)
            else:
                print(f"  {stage:6s}: not found", flush=True)
            continue
        # Drop columns already in the index / bulky.
        drop = [c for c in df_stage.columns if c in _DROP_AFTER_MERGE]
        if "model_id" not in df_stage.columns:
            print(
                f"  {stage:6s}: WARN no model_id column, skipping ({df_stage.shape})",
                flush=True,
            )
            continue
        # Drop overlap with already-merged columns except model_id itself.
        already = set(m.columns) - {"model_id"}
        drop += [c for c in df_stage.columns if c in already]
        df_stage = df_stage.drop(columns=[c for c in set(drop) if c in df_stage.columns])
        m = m.merge(df_stage, on="model_id", how="left")
        cov = m[df_stage.columns.difference(["model_id"])[0]].notna().sum() if len(
            df_stage.columns
        ) > 1 else 0
        print(
            f"  {stage:6s}: {df_stage.shape}  merged → master {m.shape}  (probe nonnull {cov}/{n_idx})",
            flush=True,
        )

    # method_master one-hot key for xgb inference.
    if "method" in m.columns:
        m["method_master"] = m["method"].map(METHOD_MASTER_MAP)
        unmapped = sorted(m.loc[m["method_master"].isna(), "method"].dropna().unique())
        if unmapped:
            print(f"  WARN: unmapped methods (no method_master): {unmapped}", flush=True)

    return m


def save_master(df: pd.DataFrame, output_dir: str | Path) -> Path:
    """Write ``<output_dir>/master.parquet`` and return the path."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "master.parquet"
    df.to_parquet(out, index=False)
    print(f"  saved: {out}  shape={df.shape}", flush=True)
    return out


def join_and_save(
    index_df: pd.DataFrame,
    stage_outputs_dir: str | Path,
    output_dir: str | Path,
    mode: str = "ab",
) -> tuple[pd.DataFrame, Path]:
    """Convenience: :func:`join_features` then :func:`save_master`."""
    m = join_features(index_df, stage_outputs_dir, mode=mode)
    out = save_master(m, output_dir)
    return m, out
