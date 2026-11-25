"""Generic PPI xgb scoring (xgb_ppi_v6_full default (7 methods incl OpenFold3, FoldBench labels); v5 fallback).

v5 = v4 + ESMFold2 ``ef2`` (single-sequence) method (6 methods total).

Mirrors :mod:`residudes_qa.interface.score.xgb_ab` but the PPI meta does not ship a
``feat_cols_breakdown`` — instead we partition ``feat_cols`` by name pattern:

* ``method_*``         → one-hot on ``method_master`` (5 cols)
* ``*_z_method``       → per-method z-score from ``normalization.json``
* ``*_z_target``       → per-target z-score (computed across **this** target's
  own predictions, NOT from norm.json — the normalization file does not
  contain target-level stats).
* everything else      → base (raw) features

The xgb_ppi method-master mapping differs from xgb_v8::

    {af3 -> af3, bt2_jax -> b2, of3 -> b1 (closest), pt2 -> ptx}

For our targets, we use ``method_master`` as already filled in by
:mod:`residudes_qa.interface.core.master`; if a method does not appear in the meta
``method_oh`` list, the model receives an all-zero one-hot row.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd


# Models ship inside the tool (XGBDimerScorer_tool/models/) for portability.
# No runtime /store dependency: if a model is absent we fall back to the bundled
# previous version (also under models/), never to /store.
from residudes_qa.paths import xgb_models_dir
_BUNDLED_MODELS = xgb_models_dir()  # data root, see residudes_qa.paths


def _resolve_model(filename: str, legacy: str) -> str:
    bundled = _BUNDLED_MODELS / filename
    return str(bundled) if bundled.exists() else legacy


# v5 (v4 + ESMFold2 ef2 method added) is the new default; v4 is auto-fallback if v5 missing.
_DEFAULT_MODEL = _resolve_model(
    "xgb_ppi_v6_full.json",
    str(_BUNDLED_MODELS / "xgb_ppi_v5_full.json"))
_DEFAULT_META = _resolve_model(
    "xgb_ppi_v6_full.meta.json",
    str(_BUNDLED_MODELS / "xgb_ppi_v4_full.meta.json"))
_DEFAULT_NORM = _resolve_model(
    "xgb_ppi_v6_full.normalization.json",
    str(_BUNDLED_MODELS / "xgb_ppi_v4_full.normalization.json"))


# Multi-target scoring (v6): dockq is the default model above; ics/ips are the
# v6 companion models (same feature schema, different label). Column produced
# for each target — dockq keeps the historical ``xgb_pred`` name for back-compat.
PPI_TARGET_COL: dict[str, str] = {"dockq": "xgb_pred", "ics": "xgb_ics", "ips": "xgb_ips"}


def ppi_model_triple(target: str, version: str = "v6") -> tuple[str, str, str]:
    """Return ``(model, meta, norm)`` paths for ``target`` in {dockq, ics, ips}.

    ``version`` selects the model generation (default ``v6``). If the requested
    version isn't deployed, dockq falls back to the bundled default chain
    (v6→v5→v4); ics/ips raise (no fallback companion).
    """
    if target == "dockq":
        stem = _BUNDLED_MODELS / f"xgb_ppi_{version}_full"
        model = f"{stem}.json"
        if Path(model).exists():
            return model, f"{stem}.meta.json", f"{stem}.normalization.json"
        # requested version absent → bundled default (resolved at import).
        return _DEFAULT_MODEL, _DEFAULT_META, _DEFAULT_NORM
    if target not in ("ics", "ips"):
        raise ValueError(f"unknown PPI target {target!r} (use dockq|ics|ips)")
    stem = _BUNDLED_MODELS / f"xgb_ppi_{version}_{target}_full"
    model = f"{stem}.json"
    if not Path(model).exists():
        raise FileNotFoundError(
            f"PPI {target} model not deployed: {model} "
            f"(expected xgb_ppi_{version}_{{ics,ips}}_full.* in models/)")
    return model, f"{stem}.meta.json", f"{stem}.normalization.json"


# Optional remap for PPI mode (different method universe).  None entries mean
# "leave as-is" (which usually produces all-zero one-hot for unseen methods).
PPI_METHOD_REMAP: dict[str, str] = {
    "af3": "af3",
    "b2": "b2",      # bt2_jax → b2
    "v2": "ptx",     # pt2 → ptx
    "of3": "of3",    # not in PPI training set; will result in zero one-hot
    "ef2": "ef2",    # ESMFold2 single-seq (v5); method_ef2 one-hot + ef2 z-stats
}


def _partition_feat_cols(
    feat_cols: list[str],
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Return ``(oh, z_method, z_target, base)`` partitioned by name suffix."""
    oh = [c for c in feat_cols if c.startswith("method_")]
    z_method = [c for c in feat_cols if c.endswith("_z_method")]
    z_target = [c for c in feat_cols if c.endswith("_z_target")]
    used = set(oh) | set(z_method) | set(z_target)
    base = [c for c in feat_cols if c not in used]
    return oh, z_method, z_target, base


def prep_xgb_ppi_input(
    master_df: pd.DataFrame,
    xgb_meta_path: str | Path,
    xgb_norm_path: str | Path,
) -> tuple[np.ndarray, list[str]]:
    """Build feature matrix for xgb_ppi.  Returns ``(X, feat_cols)``."""
    meta = json.loads(Path(xgb_meta_path).read_text())
    norm = json.loads(Path(xgb_norm_path).read_text())

    feat_cols: list[str] = meta["feat_cols"]
    train_means: dict[str, float] = meta.get("train_feature_means", {})

    # PPI meta does not ship a breakdown — partition by suffix.
    oh_cols, z_method_cols, z_target_cols, base_cols = _partition_feat_cols(feat_cols)
    print(
        f"  prep_xgb_ppi_input: master={master_df.shape}  "
        f"feat_cols={len(feat_cols)} (base={len(base_cols)} oh={len(oh_cols)} "
        f"z_method={len(z_method_cols)} z_target={len(z_target_cols)})",
        flush=True,
    )

    df = master_df
    n = len(df)

    # ---- raw / base features ----
    raw_vals: dict[str, np.ndarray] = {}
    for c in base_cols:
        if c in df.columns:
            raw_vals[c] = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=np.float64)

    # ---- method one-hot ----
    if "method_master" in df.columns:
        mm_raw = df["method_master"].astype(str).to_numpy()
        # Apply optional remap (xgb_ab uses {b2,v2}; PPI uses {b2,ptx}).
        mm = np.array(
            [PPI_METHOD_REMAP.get(x, x) for x in mm_raw],
            dtype=object,
        )
    else:
        mm = np.array([""] * n, dtype=object)
    X_oh = np.zeros((n, len(oh_cols)), dtype=np.float64)
    for i, c in enumerate(oh_cols):
        name = c[len("method_"):]
        X_oh[:, i] = (mm == name).astype(np.float64)

    # ---- z_method (per-method z using norm.json) ----
    X_zm = np.full((n, len(z_method_cols)), np.nan, dtype=np.float64)
    for i, c_z in enumerate(z_method_cols):
        c_raw = c_z[: -len("_z_method")]
        # If master already has the precomputed _z_method, prefer that.
        if c_z in df.columns:
            X_zm[:, i] = pd.to_numeric(df[c_z], errors="coerce").to_numpy(dtype=np.float64)
            continue
        if c_raw not in df.columns:
            continue
        x = pd.to_numeric(df[c_raw], errors="coerce").to_numpy(dtype=np.float64)
        per_method = norm.get(c_raw, {})
        out = np.full(n, np.nan)
        for canon, s in per_method.items():
            sd = float(s.get("std", 0.0))
            mu = float(s.get("mean", 0.0))
            if sd <= 0:
                continue
            mask = mm == canon
            if mask.any():
                out[mask] = (x[mask] - mu) / sd
        X_zm[:, i] = out

    # ---- z_target (per-target z computed in-batch) ----
    X_zt = np.full((n, len(z_target_cols)), np.nan, dtype=np.float64)
    for i, c_z in enumerate(z_target_cols):
        c_raw = c_z[: -len("_z_target")]
        if c_z in df.columns:
            X_zt[:, i] = pd.to_numeric(df[c_z], errors="coerce").to_numpy(dtype=np.float64)
            continue
        if c_raw not in df.columns:
            continue
        x = pd.to_numeric(df[c_raw], errors="coerce").to_numpy(dtype=np.float64)
        finite = x[np.isfinite(x)]
        if len(finite) < 2:
            continue
        mu = float(finite.mean())
        sd = float(finite.std(ddof=1))
        if sd <= 0:
            continue
        X_zt[:, i] = (x - mu) / sd

    # ---- assemble in canonical column order ----
    n_feat = len(feat_cols)
    X = np.full((n, n_feat), np.nan, dtype=np.float64)
    for j, c in enumerate(feat_cols):
        if c in oh_cols:
            X[:, j] = X_oh[:, oh_cols.index(c)]
        elif c in z_method_cols:
            X[:, j] = X_zm[:, z_method_cols.index(c)]
        elif c in z_target_cols:
            X[:, j] = X_zt[:, z_target_cols.index(c)]
        elif c in raw_vals:
            X[:, j] = raw_vals[c]

    # impute NaN with train_means
    impute_count = 0
    for j, c in enumerate(feat_cols):
        mask = np.isnan(X[:, j])
        if mask.any():
            X[:, j] = np.where(mask, float(train_means.get(c, 0.0)), X[:, j])
            impute_count += int(mask.sum())
    print(f"  imputed {impute_count} NaN cells via train_feature_means", flush=True)

    return X.astype(np.float32), feat_cols


def predict_xgb_ppi(
    X: np.ndarray,
    feat_cols: list[str],
    model_path: str | Path,
) -> np.ndarray:
    """Load xgb booster and predict.  Returns 1D float32 array (n,)."""
    import xgboost as xgb

    booster = xgb.Booster()
    booster.load_model(str(model_path))
    dmat = xgb.DMatrix(X.astype(np.float64), feature_names=list(feat_cols))
    preds = booster.predict(dmat)
    print(
        f"  predict_xgb_ppi: n={len(preds)}  min={preds.min():.4f}  "
        f"mean={preds.mean():.4f}  max={preds.max():.4f}",
        flush=True,
    )
    return preds.astype(np.float32)


def score_ppi(
    master_df: pd.DataFrame,
    output_dir: str | Path,
    targets: tuple[str, ...] | list[str] = ("dockq",),
    model_path: str | Path | None = None,
    meta_path: str | Path | None = None,
    norm_path: str | Path | None = None,
) -> pd.DataFrame:
    """End-to-end PPI scoring pipeline.  See :func:`xgb_ab.score_ab` for I/O layout.

    ``targets`` selects which v6 models to run — any subset of
    ``{dockq, ics, ips}`` (default ``dockq`` only, producing the historical
    ``xgb_pred`` column). Each target adds one column (``xgb_pred`` / ``xgb_ics``
    / ``xgb_ips``). Passing ``model_path``/``meta_path``/``norm_path`` overrides
    the triple used for the ``dockq`` target (back-compat for direct callers).
    """
    t0 = time.time()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scores = pd.DataFrame({"model_id": master_df["model_id"].values})
    if "candidate_uid" in master_df.columns:
        scores["candidate_uid"] = master_df["candidate_uid"].values

    pred_cols: list[str] = []
    for tgt in targets:
        if tgt == "dockq" and model_path is not None:
            mp, mt, nm = model_path, meta_path or _DEFAULT_META, norm_path or _DEFAULT_NORM
        else:
            mp, mt, nm = ppi_model_triple(tgt)
        print(f"  --- PPI target={tgt} (model={Path(mp).name}) ---", flush=True)
        X, feat_cols = prep_xgb_ppi_input(master_df, mt, nm)
        col = PPI_TARGET_COL[tgt]
        scores[col] = predict_xgb_ppi(X, feat_cols, mp)
        pred_cols.append(col)

    scores.to_parquet(output_dir / "xgb_scores.parquet", index=False)
    # CSV companion for quick human inspection / handoff (parquet stays the source of truth).
    scores.sort_values(pred_cols[0], ascending=False).to_csv(
        output_dir / "xgb_scores.csv", index=False)

    merged = master_df.merge(scores[["model_id", *pred_cols]], on="model_id", how="left")
    merged.to_parquet(output_dir / "master_with_xgb.parquet", index=False)

    print(
        f"  score_ppi: scored {len(scores)} models x {len(pred_cols)} target(s) "
        f"{pred_cols}  ({(time.time()-t0)/60:.1f} min)",
        flush=True,
    )
    return merged
