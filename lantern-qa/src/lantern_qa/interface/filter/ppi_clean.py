"""Generic PPI sanity filter (no Ab-specific features).

For protein-protein dimer / multimer targets where the aADARS CDR features
are unavailable.  Filters on interface size, chain integrity, MPNN coverage
and AF3 iptm.
"""
from __future__ import annotations

from typing import Any, Optional

import pandas as pd


DEFAULT_PPI_THRESHOLDS: dict[str, float] = {
    "iface_n_contacts_total_min": 80,
    "mpnn_iface_n_residues_min": 15,
    "json_iptm_min": 0.30,
    "n_chain_breaks_max": 0,
}


def _check_col(df: pd.DataFrame, col: str) -> Optional[pd.Series]:
    if col in df.columns:
        return df[col]
    print(f"    WARN: column '{col}' missing — skipping that step", flush=True)
    return None


def apply_ppi_clean(
    master_df: pd.DataFrame,
    thresholds: Optional[dict[str, float]] = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Generic PPI sanity filter.

    Default thresholds (override via ``thresholds``):

    * ``iface_n_contacts_total`` ≥ 80
    * ``mpnn_iface_n_residues`` ≥ 15
    * ``json_iptm`` ≥ 0.3
    * ``n_chain_breaks`` == 0
    """
    th = {**DEFAULT_PPI_THRESHOLDS, **(thresholds or {})}
    n0 = len(master_df)
    by_step: list[dict[str, Any]] = []

    keep = pd.Series(True, index=master_df.index)
    steps = [
        ("iface_n_contacts_total", "iface_n_contacts_total", ">=", th["iface_n_contacts_total_min"]),
        ("mpnn_iface_n_residues",  "mpnn_iface_n_residues",  ">=", th["mpnn_iface_n_residues_min"]),
        ("json_iptm",              "json_iptm",              ">=", th["json_iptm_min"]),
        ("n_chain_breaks",         "n_chain_breaks",         "==", th["n_chain_breaks_max"]),
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
        f"  apply_ppi_clean: {n0} → {len(filt)}  ({stats['fraction_passed']*100:.1f}%)",
        flush=True,
    )
    return filt, stats
