"""1D contact map visualisations — epitope residue × cluster.

Uses the output of :mod:`contact_map.build_cluster_contact_maps` directly.

Three plot types:
* :func:`plot_cluster_epitope_bars`    — per-cluster horizontal bars of top
  enriched epitope residues (one PNG per cluster)
* :func:`plot_cluster_epitope_heatmap` — single matrix figure with clusters
  as rows and epitope residues as columns (frequency or enrichment)
* :func:`plot_enrichment_grid`         — small-multiples grid showing all
  clusters' enrichment maps side by side

For the full 2D CDR × epitope heatmap (the original "x=epitope, y=CDR" view
from the design discussion), see :mod:`contact_map_2d`, which needs pair-level
contact data (paratope residue × epitope residue per model).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _setup_axes(ax, title=None):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if title:
        ax.set_title(title, fontsize=10)


def plot_cluster_epitope_bars(
    cluster_id: int,
    epitope_residues: list,
    F_binary: np.ndarray,
    enrichment: Optional[np.ndarray] = None,
    output_path: str | Path = None,
    top_n: int = 25,
    title: Optional[str] = None,
) -> Path:
    """Horizontal bars of top epitope residues for one cluster.

    Bars are colored by enrichment if provided (red = positive specificity,
    blue = depleted). Bar length = frequency in cluster.
    """
    if enrichment is None:
        order = np.argsort(-F_binary)
    else:
        order = np.argsort(-enrichment)
    order = order[:top_n][::-1]   # reverse for bottom-to-top top

    labels = [f"{epitope_residues[i][0]}.{epitope_residues[i][1]}" for i in order]
    freqs = F_binary[order]
    if enrichment is not None:
        ev = enrichment[order]
        cmap = matplotlib.colormaps.get_cmap("coolwarm")
        norm = matplotlib.colors.TwoSlopeNorm(vmin=-max(0.3, abs(ev.min())),
                                               vcenter=0,
                                               vmax=max(0.3, ev.max()))
        colors = [cmap(norm(v)) for v in ev]
    else:
        colors = ["#1b9e77"] * len(order)

    h = max(3.5, 0.22 * len(order) + 1)
    fig, ax = plt.subplots(figsize=(8, h))
    bars = ax.barh(labels, freqs, color=colors, edgecolor="black", linewidth=0.5)
    if enrichment is not None:
        for bar, v, e in zip(bars, freqs, ev):
            ax.text(v + 0.005, bar.get_y() + bar.get_height()/2,
                    f"  Δ{e:+.2f}", va="center", fontsize=8)
    ax.set_xlim(0, max(1.0, freqs.max() * 1.18))
    ax.set_xlabel("Frequency in cluster")
    ax.set_ylabel("Epitope residue")
    _setup_axes(ax, title or f"Cluster {cluster_id} — top epitope residues")
    if enrichment is not None:
        sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
        cbar = plt.colorbar(sm, ax=ax, shrink=0.5, label="Enrichment ΔF")
    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_cluster_epitope_heatmap(
    F_binary_matrix: np.ndarray,
    cluster_ids: list,
    epitope_universe: list,
    output_path: str | Path,
    *,
    mode: str = "frequency",   # 'frequency' or 'enrichment'
    F_global: Optional[np.ndarray] = None,
    top_n_residues: Optional[int] = 50,
    title: Optional[str] = None,
) -> Path:
    """Heatmap: rows = clusters, columns = epitope residues, color = freq or ΔF.

    If ``mode='enrichment'`` and ``F_global`` provided, subtracts global
    frequency to show specificity per cluster.
    Optionally restrict to top N residues by global frequency or by total
    enrichment to keep the heatmap readable.
    """
    if mode == "enrichment":
        if F_global is None:
            raise ValueError("mode='enrichment' requires F_global aligned to epitope_universe")
        matrix = F_binary_matrix - F_global[np.newaxis, :]
        center = 0
        cmap = "coolwarm"
        cbar_label = "Enrichment ΔF (cluster - global)"
    else:
        matrix = F_binary_matrix
        center = None
        cmap = "viridis"
        cbar_label = "Frequency in cluster"

    if top_n_residues is not None and matrix.shape[1] > top_n_residues:
        # pick residues by max absolute value across clusters
        score = np.max(np.abs(matrix), axis=0)
        keep = np.argsort(-score)[:top_n_residues]
        keep.sort()
        matrix_show = matrix[:, keep]
        labels_show = [f"{epitope_universe[i][0]}.{epitope_universe[i][1]}" for i in keep]
    else:
        matrix_show = matrix
        labels_show = [f"{r[0]}.{r[1]}" for r in epitope_universe]

    w = max(8, 0.16 * len(labels_show) + 2)
    h = max(2.5, 0.45 * len(cluster_ids) + 1.5)
    fig, ax = plt.subplots(figsize=(w, h))

    if center is not None:
        v = max(abs(matrix_show.min()), abs(matrix_show.max()), 0.1)
        norm = matplotlib.colors.TwoSlopeNorm(vmin=-v, vcenter=center, vmax=v)
        im = ax.imshow(matrix_show, aspect="auto", cmap=cmap, norm=norm)
    else:
        im = ax.imshow(matrix_show, aspect="auto", cmap=cmap, vmin=0, vmax=1)

    ax.set_yticks(range(len(cluster_ids)))
    ax.set_yticklabels([f"C{c}" for c in cluster_ids])
    ax.set_xticks(range(len(labels_show)))
    ax.set_xticklabels(labels_show, rotation=90, fontsize=7)
    ax.set_xlabel("Epitope residue")
    ax.set_ylabel("Cluster")
    _setup_axes(ax, title or f"{mode.capitalize()} heatmap — {len(cluster_ids)} clusters × {len(labels_show)} residues")
    plt.colorbar(im, ax=ax, label=cbar_label, shrink=0.7)
    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_enrichment_grid(
    cluster_maps: dict,
    global_F: np.ndarray,
    global_residues: list,
    output_path: str | Path,
    top_per_cluster: int = 12,
) -> Path:
    """Small-multiples — one subplot per cluster, top N enriched residues.

    Compact overview: 6 clusters → 6 subplots in 2×3 grid.
    """
    cluster_ids = sorted(cluster_maps.keys())
    n = len(cluster_ids)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows),
                              squeeze=False)
    F_global_by_res = dict(zip(global_residues, global_F))

    for ax_i, cid in enumerate(cluster_ids):
        r, c = divmod(ax_i, ncols)
        ax = axes[r][c]
        cmap = cluster_maps[cid]
        F_g = np.array([F_global_by_res.get(rr, 0.0) for rr in cmap["epitope_residues"]],
                       dtype=np.float32)
        delta = cmap["F_binary"] - F_g
        order = np.argsort(-delta)[:top_per_cluster]
        labels = [f"{cmap['epitope_residues'][i][0]}.{cmap['epitope_residues'][i][1]}" for i in order]
        ev = delta[order]
        cmap_color = matplotlib.colormaps.get_cmap("coolwarm")
        norm = matplotlib.colors.TwoSlopeNorm(
            vmin=-max(0.3, abs(ev.min())), vcenter=0, vmax=max(0.3, ev.max())
        )
        ax.barh(labels[::-1], ev[::-1],
                color=[cmap_color(norm(v)) for v in ev[::-1]],
                edgecolor="black", linewidth=0.4)
        ax.axvline(0, color="black", linewidth=0.6)
        ax.set_title(f"Cluster {cid} (n={cmap['n_models']})", fontsize=9)
        ax.set_xlabel("ΔF (enrichment)", fontsize=8)
        ax.tick_params(axis="y", labelsize=7)
        ax.tick_params(axis="x", labelsize=8)
        _setup_axes(ax)

    # hide unused subplots
    for ax_i in range(n, nrows * ncols):
        r, c = divmod(ax_i, ncols)
        axes[r][c].set_visible(False)

    fig.suptitle("Per-cluster enrichment (top residues vs global)", y=1.01)
    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_physchem_bars(
    profile_df: pd.DataFrame,
    output_path: str | Path,
    *,
    channels: tuple = ("hydrophobic", "positive_charge", "negative_charge", "polar", "aromatic"),
) -> Path:
    """Stacked bar per cluster — residue type fractions on the epitope side.

    Uses output of :func:`physchem_channels.epitope_residue_type_profile`.
    """
    cluster_ids = profile_df.index.tolist()
    data = profile_df[list(channels)].to_numpy(dtype=np.float32)
    colors = {
        "hydrophobic": "#fdae61",
        "positive_charge": "#2b83ba",
        "negative_charge": "#d7191c",
        "polar": "#abd9e9",
        "aromatic": "#5e4fa2",
    }

    fig, ax = plt.subplots(figsize=(max(6, 1.4 * len(cluster_ids) + 2), 4))
    bottom = np.zeros(len(cluster_ids))
    for ch in channels:
        vals = profile_df[ch].to_numpy()
        ax.bar([f"C{c}" for c in cluster_ids], vals, bottom=bottom,
               label=ch, color=colors.get(ch, "#888"), edgecolor="black", linewidth=0.4)
        bottom += vals
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Fraction of epitope contacts")
    ax.set_xlabel("Cluster")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=5, fontsize=8)
    _setup_axes(ax, "Physicochemical composition of epitope contacts per cluster")
    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return output_path


from lantern_qa.cluster.visualization._colors import (
    METHOD_COLORS, normalize_method, get_method_color,
)


def plot_cluster_epitope_freq_by_method(
    cluster_id: int,
    contacts_long: pd.DataFrame,
    cluster_assignments: pd.DataFrame,
    master_df: pd.DataFrame,
    output_path: str | Path,
    *,
    top_n: int = 25,
    cluster_col: str = "cluster",
    model_col: str = "model_id",
    method_col: str = "method",
    epitope_chain_col: Optional[str] = None,
    epitope_res_col: Optional[str] = None,
    overall_panel: bool = True,
) -> Path:
    """Per-cluster epitope frequency plot with per-method breakdown.

    Layout (2 panels)::

        ┌──────────────────────────────────────────────────────────────┐
        │ Top   : aggregate frequency (Y) over top-N residues (X)      │
        │ Bottom: per-method line — fraction of method-M models that   │
        │         contact each residue (4 lines, color-coded by method)│
        └──────────────────────────────────────────────────────────────┘

    Color convention: af3=blue, pt2=red, bt2_jax=gold, of3=light-green.

    Parameters
    ----------
    cluster_id
        Cluster to plot.
    contacts_long
        Long contacts table (must have model_id + epitope chain/res columns).
    cluster_assignments
        DataFrame with ``model_id, cluster``.
    master_df
        DataFrame with ``model_id, method`` (for per-method breakdown).
    top_n
        Number of top residues (sorted by aggregate freq).
    """
    # auto-detect epitope columns
    if epitope_chain_col is None:
        epitope_chain_col = "abc_chain" if "abc_chain" in contacts_long.columns else "ag_chain"
    if epitope_res_col is None:
        epitope_res_col = "abc_resnum" if "abc_resnum" in contacts_long.columns else "ag_resnum"

    # cluster members
    members = cluster_assignments[cluster_assignments[cluster_col] == cluster_id][model_col].tolist()
    if not members:
        raise ValueError(f"no members in cluster {cluster_id}")
    member_method = master_df[master_df[model_col].isin(members)][[model_col, method_col]].drop_duplicates()

    # subset contacts to this cluster
    sub = contacts_long[contacts_long[model_col].isin(members)]

    # aggregate frequency per residue
    n_total = len(members)
    agg = sub.groupby([epitope_chain_col, epitope_res_col])[model_col].nunique() / max(n_total, 1)
    agg = agg.sort_values(ascending=False).head(top_n)
    residue_labels = [f"{c}.{int(r)}" for c, r in agg.index]
    residue_keys = list(agg.index)

    # per-method frequency: among method-M models in cluster, fraction with contact
    methods_in_cluster = member_method[method_col].value_counts()
    method_freqs: dict[str, np.ndarray] = {}
    for method, n_m in methods_in_cluster.items():
        if n_m == 0:
            continue
        m_models = member_method[member_method[method_col] == method][model_col].tolist()
        m_sub = sub[sub[model_col].isin(m_models)]
        # count unique model per residue
        m_per_res = m_sub.groupby([epitope_chain_col, epitope_res_col])[model_col].nunique()
        freqs = np.zeros(len(residue_keys), dtype=np.float32)
        for i, key in enumerate(residue_keys):
            freqs[i] = m_per_res.get(key, 0) / n_m
        method_freqs[method] = freqs

    # ---- plot ----
    n_panels = 2 if overall_panel else 1
    fig, axes = plt.subplots(n_panels, 1,
                              figsize=(max(8, 0.35 * top_n + 2), 6 if overall_panel else 3.5),
                              sharex=True)
    if n_panels == 1:
        axes = [axes]

    if overall_panel:
        ax = axes[0]
        ax.bar(range(len(residue_labels)), agg.to_numpy(),
               color="#444", edgecolor="black", linewidth=0.4)
        ax.set_ylabel("Aggregate frequency", fontsize=9)
        ax.set_ylim(0, max(1.0, agg.max() * 1.1))
        _setup_axes(ax, f"Cluster {cluster_id} — top {top_n} epitope residues (n={n_total})")

    ax = axes[-1]
    method_order = sorted(method_freqs.keys(),
                         key=lambda m: -methods_in_cluster.get(m, 0))
    for method in method_order:
        color = get_method_color(method)
        ax.plot(range(len(residue_labels)), method_freqs[method],
                marker="o", markersize=4, linewidth=1.4,
                color=color,
                label=f"{method} (n={int(methods_in_cluster.get(method, 0))})")
    ax.set_ylabel("Per-method frequency", fontsize=9)
    ax.set_xlabel("Epitope residue (sorted by aggregate freq)")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(range(len(residue_labels)))
    ax.set_xticklabels(residue_labels, rotation=70, fontsize=7)
    ax.legend(loc="upper right", fontsize=8, frameon=True)
    _setup_axes(ax, None if overall_panel else f"Cluster {cluster_id} per-method")

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_all_clusters_freq_by_method(
    contacts_long: pd.DataFrame,
    cluster_assignments: pd.DataFrame,
    master_df: pd.DataFrame,
    output_dir: str | Path,
    *,
    top_n: int = 25,
    **kwargs,
) -> dict[int, Path]:
    """Convenience: render the per-cluster method-breakdown plot for all clusters."""
    cluster_ids = sorted(cluster_assignments[
        kwargs.get("cluster_col", "cluster")
    ].unique())
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    for cid in cluster_ids:
        path = out_dir / f"cluster_{cid}_freq_by_method.png"
        plot_cluster_epitope_freq_by_method(
            int(cid), contacts_long, cluster_assignments, master_df,
            output_path=path, top_n=top_n, **kwargs,
        )
        written[int(cid)] = path
    return written


__all__ = [
    "plot_cluster_epitope_bars",
    "plot_cluster_epitope_heatmap",
    "plot_enrichment_grid",
    "plot_physchem_bars",
    "plot_cluster_epitope_freq_by_method",
    "plot_all_clusters_freq_by_method",
    "METHOD_COLORS",
]
