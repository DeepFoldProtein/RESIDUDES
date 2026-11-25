"""
Per-group confidence summaries (pLDDT / pDockQ / arbitrary scores).

Two grouping lenses are summarised the same way:
  * per **cluster**  (embedding-based; pair with silhouette)
  * per **community** (network/Leiden; pair with modularity + seed-ARI)

pLDDT comes straight from the CA B-factor column (CASP server PDBs and AF
outputs both store per-residue confidence there) — no PAE needed. pDockQ is a
PAE-free interface-confidence proxy (interface pLDDT × log(#contacts)); useful
when ipTM is unavailable (e.g. structure-only server submissions). Any extra
score (ipTM, ranking_score, ...) can be summarised by passing a value array.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# structure-derived confidence (no PAE required)
# --------------------------------------------------------------------------- #
def read_mean_plddt(path: Union[str, Path]) -> float:
    """Mean CA B-factor (= pLDDT for AF-style models). NaN if unreadable."""
    import gemmi
    try:
        st = gemmi.read_structure(str(path))
    except Exception:
        try:
            st = gemmi.read_pdb(str(path))     # PDB content in a .cif name
        except Exception:
            return float("nan")
    if len(st) == 0:
        return float("nan")
    bs = [a.b_iso for ch in st[0] for r in ch for a in r if a.name == "CA"]
    return float(np.mean(bs)) if bs else float("nan")


def interface_pdockq(path: Union[str, Path], cutoff: float = 8.0) -> float:
    """PAE-free interface-confidence proxy (Bryant et al. 2022), averaged over
    all inter-chain interfaces — appropriate for homo/hetero oligomers.

    pDockQ = 0.724 / (1 + e^{-0.052 (x - 152.611)}) + 0.018,
    with x = mean(interface CB pLDDT) · log(#interface CB-CB contacts < cutoff).
    Returns 0.0 if no interface contacts, NaN if unreadable.
    """
    import gemmi
    try:
        st = gemmi.read_structure(str(path))
    except Exception:
        try:
            st = gemmi.read_pdb(str(path))
        except Exception:
            return float("nan")
    if len(st) == 0:
        return float("nan")
    model = st[0]
    chains = []
    for ch in model:
        xs, bs = [], []
        for r in ch:
            cb = r.find_atom("CB", "*") or r.find_atom("CA", "*")
            if cb is not None:
                xs.append([cb.pos.x, cb.pos.y, cb.pos.z])
                bs.append(cb.b_iso)
        if xs:
            chains.append((np.array(xs), np.array(bs)))
    if len(chains) < 2:
        return float("nan")

    contacts = 0
    if_plddt: List[float] = []
    for i in range(len(chains)):
        for j in range(i + 1, len(chains)):
            (xi, bi), (xj, bj) = chains[i], chains[j]
            d = np.linalg.norm(xi[:, None, :] - xj[None, :, :], axis=2)
            mask = d < cutoff
            contacts += int(mask.sum())
            ii, jj = np.where(mask)
            if ii.size:
                if_plddt.extend(bi[np.unique(ii)].tolist())
                if_plddt.extend(bj[np.unique(jj)].tolist())
    if contacts == 0 or not if_plddt:
        return 0.0
    x = float(np.mean(if_plddt)) * np.log(contacts)
    return float(0.724 / (1.0 + np.exp(-0.052 * (x - 152.611))) + 0.018)


def confidence_from_paths(
    paths: Sequence[Union[str, Path]],
    *,
    pdockq: bool = False,
    pdockq_cutoff: float = 8.0,
) -> Dict[str, np.ndarray]:
    """Read per-structure confidence(s). Returns ``{"plddt": arr[, "pdockq": arr]}``."""
    plddt = np.array([read_mean_plddt(p) for p in paths], dtype=float)
    out = {"plddt": plddt}
    if pdockq:
        out["pdockq"] = np.array([interface_pdockq(p, pdockq_cutoff) for p in paths], dtype=float)
    return out


# --------------------------------------------------------------------------- #
# group summary
# --------------------------------------------------------------------------- #
def summarize_by_group(
    group_labels: Sequence[int],
    values: Dict[str, np.ndarray],
    *,
    group_name: str = "cluster",
    names: Optional[Sequence[str]] = None,
    highlight_prefix: Optional[str] = None,
) -> pd.DataFrame:
    """Per-group mean/median/min/max for each value array, sorted by group id.

    ``values`` maps a metric name (e.g. "plddt", "pdockq", "iptm") to a per-item
    array aligned with ``group_labels``. If ``names`` + ``highlight_prefix`` are
    given, a ``highlight`` column lists matching members per group (e.g. our
    submissions). Drops the -1 group (unassigned).
    """
    g = np.asarray(group_labels)
    rows = []
    for grp in sorted(set(int(x) for x in g) - {-1}):
        m = g == grp
        row = {group_name: grp, "n": int(m.sum())}
        for vname, arr in values.items():
            a = np.asarray(arr, float)[m]
            a = a[~np.isnan(a)]
            if a.size:
                row[f"{vname}_mean"] = round(float(a.mean()), 3)
                row[f"{vname}_median"] = round(float(np.median(a)), 3)
                row[f"{vname}_min"] = round(float(a.min()), 3)
                row[f"{vname}_max"] = round(float(a.max()), 3)
            else:
                for s in ("mean", "median", "min", "max"):
                    row[f"{vname}_{s}"] = float("nan")
        if names is not None and highlight_prefix is not None:
            hits = [str(names[i]).replace(highlight_prefix, "")
                    for i in np.where(m)[0] if str(names[i]).startswith(highlight_prefix)]
            row["highlight"] = ",".join(hits) if hits else ""
        rows.append(row)
    return pd.DataFrame(rows)


def format_group_table(
    df: pd.DataFrame,
    *,
    group_name: str = "cluster",
    value_cols: Optional[Sequence[str]] = None,
    footer: Optional[str] = None,
    max_rows: int = 20,
) -> str:
    """Compact monospace text of a group-summary DataFrame, for on-figure overlay.

    ``value_cols`` are the ``<metric>_mean`` columns to show (default: all means).
    ``footer`` appends a line (e.g. "modularity 0.65 | ARI 1.00").
    """
    if value_cols is None:
        value_cols = [c for c in df.columns if c.endswith("_mean")]
    has_hl = "highlight" in df.columns
    hdr = f"{group_name[:5]:<5}{'n':>4}" + "".join(f"{c.replace('_mean','')[:7]:>8}" for c in value_cols)
    if has_hl:
        hdr += "  ours"
    lines = [hdr, "-" * len(hdr)]
    show = df if len(df) <= max_rows else df.head(max_rows)
    for _, r in show.iterrows():
        line = f"{int(r[group_name]):<5}{int(r['n']):>4}" + "".join(
            f"{r[c]:>8.1f}" if pd.notna(r[c]) else f"{'-':>8}" for c in value_cols)
        if has_hl:
            line += f"  {r['highlight'] or '-'}"
        lines.append(line)
    if len(df) > max_rows:
        lines.append(f"... (+{len(df) - max_rows} more)")
    if footer:
        lines += ["", footer]
    return "\n".join(lines)


def plot_confidence_by_group(
    group_labels: Sequence[int],
    values: np.ndarray,
    *,
    value_name: str = "pLDDT",
    group_name: str = "community",
    names: Optional[Sequence[str]] = None,
    highlight_prefix: Optional[str] = None,
    title: Optional[str] = None,
    output_path: Optional[Union[str, Path]] = None,
    figsize=(9, 6),
):
    """Violin + strip of a confidence value per group (cluster/community).

    Useful on its own — highlighting specific members (e.g. our submissions via
    ``highlight_prefix``) is optional, so this is a standard QC plot even before
    any models are picked. Returns the matplotlib Figure.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    g = np.asarray(group_labels)
    v = np.asarray(values, float)
    groups = sorted(set(int(x) for x in g) - {-1})
    is_hl = (np.array([str(n).startswith(highlight_prefix) for n in names])
             if (names is not None and highlight_prefix) else np.zeros(len(g), bool))

    fig, ax = plt.subplots(figsize=figsize)
    data = [v[(g == c) & ~np.isnan(v)] for c in groups]
    if any(len(d) for d in data):
        ax.violinplot([d if len(d) else [np.nanmean(v)] for d in data],
                      positions=range(len(groups)), showmedians=True, widths=0.8)
    rng = np.random.default_rng(0)
    for i, c in enumerate(groups):
        m = (g == c) & ~np.isnan(v)
        base = m & ~is_hl
        x = rng.normal(i, 0.06, int(base.sum()))
        ax.scatter(x, v[base], s=16, alpha=0.5, color="#4c72b0")
        hm = m & is_hl
        if hm.any():
            hx = rng.normal(i, 0.06, int(hm.sum()))
            ax.scatter(hx, v[hm], s=240, marker="*", facecolor="white",
                       edgecolor="black", linewidth=1.4, zorder=5)
            for xi, idx in zip(hx, np.where(hm)[0]):
                ax.annotate(str(names[idx]).replace(highlight_prefix, ""), (xi, v[idx]),
                            fontsize=8, fontweight="bold", xytext=(5, 4),
                            textcoords="offset points")
    ax.set_xticks(range(len(groups)))
    many = len(groups) > 8
    ax.set_xticklabels([(f"{c}\n(n={int((g == c).sum())})" if many
                         else f"{group_name} {c}\n(n={int((g == c).sum())})") for c in groups],
                       rotation=(45 if many else 0), ha=("right" if many else "center"),
                       fontsize=(8 if many else 10))
    if many:
        ax.set_xlabel(group_name)
    ax.set_ylabel(f"mean {value_name}")
    ax.set_title(title or f"{value_name} by {group_name}"
                 + (" (★=highlighted)" if is_hl.any() else ""), fontweight="bold")
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    return fig


__all__ = [
    "read_mean_plddt",
    "interface_pdockq",
    "confidence_from_paths",
    "summarize_by_group",
    "format_group_table",
    "plot_confidence_by_group",
]
