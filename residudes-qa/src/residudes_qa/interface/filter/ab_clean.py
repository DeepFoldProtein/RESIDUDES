"""CDR-clean + strict filters for Ab-Ag mode.

Generalized from H1335 ``41_filter_recluster_viz.py`` (the cascade applied to
923 → 807 → 6-cluster STRICT pool).

Two entry points:

* :func:`apply_cdr_clean` — Ab-only sanity gate (frac framework / H3, iface
  contacts, MPNN coverage, no chain breaks).
* :func:`apply_strict_filter` — :func:`apply_cdr_clean` plus optional extras:
  ``no_of3`` (drop OpenFold3), ``no_clash_inter`` (require zero inter-chain
  1.7 Å clash), and ``gO_core_max`` (regional RMSD ceiling).

Both return ``(filtered_df, stats_dict)`` where stats records the per-step
cascade.
"""
from __future__ import annotations

from typing import Any, Optional

import pandas as pd


DEFAULT_CDR_THRESHOLDS: dict[str, float] = {
    "aadars_frac_framework_max": 0.30,
    "aadars_frac_cdr_h3_min": 0.20,
    "iface_n_contacts_total_min": 100,
    "mpnn_iface_n_residues_min": 20,
    "n_chain_breaks_max": 0,
}


def _check_col(df: pd.DataFrame, col: str) -> Optional[pd.Series]:
    """Return df[col] if present, else None (with a soft warning)."""
    if col in df.columns:
        return df[col]
    print(f"    WARN: column '{col}' missing — skipping that step", flush=True)
    return None


def apply_cdr_clean(
    master_df: pd.DataFrame,
    thresholds: Optional[dict[str, float]] = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """CDR-clean filter for Ab-Ag mode.

    Default thresholds (override individual keys via ``thresholds``):

    * ``aadars_frac_framework`` ≤ 0.30
    * ``aadars_frac_cdr_h3`` ≥ 0.20
    * ``iface_n_contacts_total`` ≥ 100
    * ``mpnn_iface_n_residues`` ≥ 20
    * ``n_chain_breaks`` == 0
    """
    th = {**DEFAULT_CDR_THRESHOLDS, **(thresholds or {})}
    n0 = len(master_df)
    by_step: list[dict[str, Any]] = []

    keep = pd.Series(True, index=master_df.index)
    steps = [
        ("aadars_frac_framework",   "aadars_frac_framework",     "<=", th["aadars_frac_framework_max"]),
        ("aadars_frac_cdr_h3",      "aadars_frac_cdr_h3",        ">=", th["aadars_frac_cdr_h3_min"]),
        ("iface_n_contacts_total",  "iface_n_contacts_total",    ">=", th["iface_n_contacts_total_min"]),
        ("mpnn_iface_n_residues",   "mpnn_iface_n_residues",     ">=", th["mpnn_iface_n_residues_min"]),
        ("n_chain_breaks",          "n_chain_breaks",            "==", th["n_chain_breaks_max"]),
    ]
    for name, col, op, val in steps:
        s = _check_col(master_df, col)
        if s is None:
            by_step.append({"step": name, "skipped": True, "n_after": int(keep.sum())})
            continue
        if op == "<=":
            mask = s <= val
        elif op == ">=":
            mask = s >= val
        elif op == "==":
            mask = s == val
        else:  # pragma: no cover
            raise ValueError(op)
        mask = mask & s.notna()
        keep &= mask
        by_step.append(
            {
                "step": name,
                "threshold": f"{col} {op} {val}",
                "n_after": int(keep.sum()),
            }
        )

    filt = master_df.loc[keep].copy()
    stats: dict[str, Any] = {
        "n_input": int(n0),
        "n_output": int(len(filt)),
        "fraction_passed": float(len(filt) / n0) if n0 else 0.0,
        "by_filter_step": by_step,
    }
    print(
        f"  apply_cdr_clean: {n0} → {len(filt)}  ({stats['fraction_passed']*100:.1f}%)",
        flush=True,
    )
    return filt, stats


DEFAULT_STRICT_EXTRA: dict[str, Any] = {
    "no_of3": True,
    "no_clash_inter": True,
    "gO_core_max": 6.0,
    "max_intra_clash": 2,
}


def apply_strict_filter(
    master_df: pd.DataFrame,
    mode: str = "ab",
    additional: Optional[dict[str, Any]] = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Full STRICT cascade: CDR-clean + optional method/clash/template-RMSD gates.

    ``additional`` keys (defaults in :data:`DEFAULT_STRICT_EXTRA`):

    * ``no_of3``         — drop ``method == 'of3'``
    * ``no_clash_inter`` — require ``n_inter_clash_17 == 0`` if column present
    * ``gO_core_max``    — keep ``gO_core <= value`` if column present (None = skip)
    * ``max_intra_clash``— keep ``n_clash_17 <= value`` if column present
    """
    add = {**DEFAULT_STRICT_EXTRA, **(additional or {})}
    if mode != "ab":
        print(
            f"  apply_strict_filter: mode={mode!r} — Ab-specific extras (gO RMSD) "
            "will be skipped unless columns are present",
            flush=True,
        )

    df, stats_cdr = apply_cdr_clean(master_df)
    by_step: list[dict[str, Any]] = list(stats_cdr["by_filter_step"])
    n0 = stats_cdr["n_input"]

    # extra: no_of3
    if add.get("no_of3") and "method" in df.columns:
        before = len(df)
        df = df[df["method"] != "of3"].copy()
        by_step.append(
            {"step": "no_of3", "threshold": "method != 'of3'", "n_after": int(len(df)), "removed": before - len(df)}
        )

    # extra: no_clash_inter
    if add.get("no_clash_inter") and "n_inter_clash_17" in df.columns:
        before = len(df)
        df = df[(df["n_inter_clash_17"].fillna(0) == 0)].copy()
        by_step.append(
            {"step": "no_clash_inter", "threshold": "n_inter_clash_17 == 0", "n_after": int(len(df)), "removed": before - len(df)}
        )

    # extra: max_intra_clash
    mic = add.get("max_intra_clash")
    if mic is not None and "n_clash_17" in df.columns:
        before = len(df)
        df = df[df["n_clash_17"].fillna(0) <= mic].copy()
        by_step.append(
            {"step": "max_intra_clash", "threshold": f"n_clash_17 <= {mic}", "n_after": int(len(df)), "removed": before - len(df)}
        )

    # extra: gO_core_max (or generic 'probe_core' RMSD threshold if column present)
    gO = add.get("gO_core_max")
    if gO is not None and "gO_core" in df.columns:
        before = len(df)
        df = df[df["gO_core"].notna() & (df["gO_core"] <= gO)].copy()
        by_step.append(
            {"step": "gO_core_max", "threshold": f"gO_core <= {gO}", "n_after": int(len(df)), "removed": before - len(df)}
        )

    stats: dict[str, Any] = {
        "n_input": int(n0),
        "n_output": int(len(df)),
        "fraction_passed": float(len(df) / n0) if n0 else 0.0,
        "by_filter_step": by_step,
    }
    print(
        f"  apply_strict_filter: {n0} → {len(df)}  ({stats['fraction_passed']*100:.1f}%)",
        flush=True,
    )
    return df, stats
