"""Ab-Ag xgb scoring (xgb_v9_full; falls back to v8).

v9 = v8 + ESMFold2 ``ef2`` (single-sequence) method, per-interface (12 methods).

Replicates the H1335 ``31_xgb_input_table.py`` + ``32_xgb_v8_predict.py``
pipeline exactly:

1. Read meta JSON, get ``feat_cols_breakdown`` (base_clean + method_oh + method_z).
2. Build ``X_base`` from master, NaN → ``train_feature_means`` later.
3. Build ``X_oh`` from ``method_master`` one-hot, names listed in meta.
4. Build ``X_mz`` from per-method normalisation pooled across ``METHOD_GROUPS``.
5. Impute NaN cells with ``train_feature_means``.
6. Call ``xgb.Booster.predict``.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# Model files ship inside the tool (XGBDimerScorer_tool/models/) so the tool is
# portable across nodes — /gpfs/deepfold is mounted everywhere, /store may not.
# No runtime /store dependency: if a model is absent we fall back to the bundled
# previous version (also under models/), never to /store.
from residudes_qa.paths import xgb_models_dir
_BUNDLED_MODELS = xgb_models_dir()  # data root, see residudes_qa.paths


def _resolve_model(filename: str, legacy: str) -> str:
    bundled = _BUNDLED_MODELS / filename
    return str(bundled) if bundled.exists() else legacy


# v9 (v8 + ESMFold2 ef2 method) is the new default; v8 is auto-fallback if v9 missing.
_DEFAULT_MODEL = _resolve_model(
    "xgb_v9_full.json", str(_BUNDLED_MODELS / "xgb_v8_full.json"))
_DEFAULT_META = _resolve_model(
    "xgb_v9_full.meta.json", str(_BUNDLED_MODELS / "xgb_v8_full.meta.json"))
_DEFAULT_NORM = _resolve_model(
    "xgb_v9_full.normalization.json", str(_BUNDLED_MODELS / "xgb_v8_full.normalization.json"))


# Multi-target scoring: dockq uses the v9 default above (production model). ICS/IPS
# are only available from the v10.1 companion models (xgb_ab.py-compatible feature
# schema). Column produced per target — dockq keeps the historical ``xgb_pred``.
AB_TARGET_COL: dict[str, str] = {"dockq": "xgb_pred", "ics": "xgb_ics", "ips": "xgb_ips"}


def _v10_1_triple(target: str) -> tuple[str, str, str]:
    stem = _BUNDLED_MODELS / f"xgb_agab_v10_1_{target}_full"
    model = f"{stem}.json"
    if not Path(model).exists():
        raise FileNotFoundError(
            f"AgAb v10.1 {target} model not deployed: {model} "
            "(expected xgb_agab_v10_1_{dockq,ics,ips}_full.* in models/)")
    return model, f"{stem}.meta.json", f"{stem}.normalization.json"


def ab_model_triple(target: str, version: str = "v9") -> tuple[str, str, str]:
    """Return ``(model, meta, norm)`` paths for ``target`` in {dockq, ics, ips}.

    ``version`` selects the **dockq** model family: ``v9`` (default production,
    12 methods, LOPO 0.82) or ``v10_1`` (deepfold3 7-method, FoldBench labels,
    LOPO 0.766). ICS/IPS are only available from v10.1, so ``version`` is ignored
    for those (always v10.1).
    """
    if target not in ("dockq", "ics", "ips"):
        raise ValueError(f"unknown AgAb target {target!r} (use dockq|ics|ips)")
    if target == "dockq" and version == "v9":
        return _DEFAULT_MODEL, _DEFAULT_META, _DEFAULT_NORM
    if target == "dockq" and version not in ("v9", "v10_1"):
        raise ValueError(f"unknown AgAb version {version!r} (use v9|v10_1)")
    return _v10_1_triple(target)


# Method groups used to pool normalisation stats — replicates H1335 stage 31.
METHOD_GROUPS: dict[str, list[str]] = {
    "af3": ["af3", "af3_v2"],
    "b1":  ["b1", "b1_v2"],
    "b2":  ["b2", "b2_v2"],
    "v1":  ["v1", "v1_v2"],
    "v2":  ["v2", "v2_v2"],
    "of3": ["of3"],
    "ef2": ["ef2"],   # ESMFold2 single-seq (v9)
}


def _pool_stats(
    per_method_stats: dict[str, dict],
    members: list[str],
) -> Optional[tuple[float, float]]:
    """Pool ``(mean, std)`` across ``members`` weighted by ``n``.

    Mirrors H1335 stage 31 ``pool_stats``.
    """
    rows: list[tuple[int, float, float]] = []
    for m in members:
        s = per_method_stats.get(m)
        if s is None:
            continue
        n = int(s.get("n", 0))
        if n <= 1:
            continue
        rows.append((n, float(s["mean"]), float(s["std"])))
    if not rows:
        return None
    N = sum(n for n, _, _ in rows)
    mu = sum(n * mm for n, mm, _ in rows) / N
    if len(rows) == 1:
        return mu, rows[0][2]
    var = (
        sum((n - 1) * sd * sd for n, _, sd in rows)
        + sum(n * (mm - mu) ** 2 for n, mm, _ in rows)
    ) / (N - 1)
    return mu, float(np.sqrt(var))


def prep_xgb_v8_input(
    master_df: pd.DataFrame,
    xgb_meta_path: str | Path,
    xgb_norm_path: str | Path,
) -> tuple[np.ndarray, list[str]]:
    """Build the (n, 190) xgb_v8 feature matrix from ``master_df``.

    Returns
    -------
    X : np.ndarray (float32)
        Imputed matrix in canonical ``feat_cols`` order.
    feat_cols : list[str]
        Column order (same as ``meta['feat_cols']``).
    """
    meta = json.loads(Path(xgb_meta_path).read_text())
    norm = json.loads(Path(xgb_norm_path).read_text())

    feat_cols: list[str] = meta["feat_cols"]
    breakdown = meta["feat_cols_breakdown"]
    base_cols: list[str] = breakdown["base_clean"]
    oh_cols: list[str] = breakdown["method_oh"]
    mz_cols: list[str] = breakdown["method_z"]
    train_means: dict[str, float] = meta["train_feature_means"]

    df = master_df
    n = len(df)
    print(
        f"  prep_xgb_v8_input: master={df.shape}  "
        f"feat_cols={len(feat_cols)}  (base={len(base_cols)} oh={len(oh_cols)} mz={len(mz_cols)})",
        flush=True,
    )

    # ----- base -----
    X_base = np.full((n, len(base_cols)), np.nan, dtype=np.float64)
    for i, c in enumerate(base_cols):
        if c in df.columns:
            X_base[:, i] = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=np.float64)
    base_missing = [c for c in base_cols if c not in df.columns]
    print(
        f"  base: {len(base_cols) - len(base_missing)}/{len(base_cols)} "
        f"columns present, {len(base_missing)} fully imputed via train_means",
        flush=True,
    )

    # ----- method one-hot -----
    method_master = (
        df["method_master"].astype(str).to_numpy()
        if "method_master" in df.columns
        else np.array([""] * n, dtype=object)
    )
    X_oh = np.zeros((n, len(oh_cols)), dtype=np.float64)
    for i, c in enumerate(oh_cols):
        name = c[len("method_"):]
        X_oh[:, i] = (method_master == name).astype(np.float64)

    # ----- method-z -----
    X_mz = np.full((n, len(mz_cols)), np.nan, dtype=np.float64)
    for i, c_mz in enumerate(mz_cols):
        c_raw = c_mz[: -len("_mz")]
        if c_raw not in df.columns:
            continue
        x = pd.to_numeric(df[c_raw], errors="coerce").to_numpy(dtype=np.float64)
        # H1335 stage 31: ranking score sentinel
        if c_raw == "json_ranking_score":
            x = np.where(x > -10, x, np.nan)
        per_method = norm.get(c_raw, {})
        out = np.full(n, np.nan)
        for canon, members in METHOD_GROUPS.items():
            pooled = _pool_stats(per_method, members)
            if pooled is None:
                continue
            mu, sd = pooled
            if sd <= 0:
                continue
            mask = method_master == canon
            if mask.any():
                out[mask] = (x[mask] - mu) / sd
        X_mz[:, i] = out

    X = np.concatenate([X_base, X_oh, X_mz], axis=1)

    # impute NaN with train_means
    impute_count = 0
    for j, c in enumerate(feat_cols):
        mask = np.isnan(X[:, j])
        if mask.any():
            X[:, j] = np.where(mask, float(train_means.get(c, 0.0)), X[:, j])
            impute_count += int(mask.sum())
    print(f"  imputed {impute_count} NaN cells via train_feature_means", flush=True)

    return X.astype(np.float32), feat_cols


def predict_xgb_v8(
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
        f"  predict_xgb_v8: n={len(preds)}  min={preds.min():.4f}  "
        f"mean={preds.mean():.4f}  max={preds.max():.4f}",
        flush=True,
    )
    return preds.astype(np.float32)


def score_ab(
    master_df: pd.DataFrame,
    output_dir: str | Path,
    targets: tuple[str, ...] | list[str] = ("dockq",),
    ab_version: str = "v9",
    model_path: str | Path | None = None,
    meta_path: str | Path | None = None,
    norm_path: str | Path | None = None,
) -> pd.DataFrame:
    """End-to-end Ab-Ag scoring pipeline.

    ``targets`` selects which models to run — any subset of ``{dockq, ics, ips}``
    (default ``dockq`` only → historical ``xgb_pred`` column). ``ab_version``
    selects the **dockq** model family: ``v9`` (default production, 12-method,
    LOPO 0.82) or ``v10_1`` (deepfold3 7-method, FoldBench labels, LOPO 0.766).
    ICS/IPS always come from v10.1. Each target adds one column (``xgb_pred`` /
    ``xgb_ics`` / ``xgb_ips``). ``model_path``/``meta_path``/``norm_path`` override
    the ``dockq`` triple (back-compat for direct callers).

    Writes ``<output_dir>/{xgb_scores.parquet, xgb_scores.csv, master_with_xgb.parquet}``.
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
            mp, mt, nm = ab_model_triple(tgt, version=ab_version)
        print(f"  --- AgAb target={tgt} (model={Path(mp).name}) ---", flush=True)
        X, feat_cols = prep_xgb_v8_input(master_df, mt, nm)
        col = AB_TARGET_COL[tgt]
        scores[col] = predict_xgb_v8(X, feat_cols, mp)
        pred_cols.append(col)

    scores.to_parquet(output_dir / "xgb_scores.parquet", index=False)
    # CSV companion for quick human inspection / handoff (parquet stays the source of truth).
    scores.sort_values(pred_cols[0], ascending=False).to_csv(
        output_dir / "xgb_scores.csv", index=False)

    merged = master_df.merge(scores[["model_id", *pred_cols]], on="model_id", how="left")
    merged.to_parquet(output_dir / "master_with_xgb.parquet", index=False)

    print(
        f"  score_ab: scored {len(scores)} models x {len(pred_cols)} target(s) "
        f"{pred_cols}  ({(time.time()-t0)/60:.1f} min)",
        flush=True,
    )
    return merged
