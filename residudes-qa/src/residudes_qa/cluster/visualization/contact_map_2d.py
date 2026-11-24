"""2D CDR × epitope contact map visualisation.

The "real" contact map view requested by users:
  x-axis: epitope residue (sorted by contact frequency)
  y-axis: CDR residue, grouped by CDR loop (H1, H2, H3, L1, L2, L3, framework)
  color: contact frequency in cluster

Two main functions:

* :func:`plot_cluster_2d_heatmap` — one cluster, full CDR × epitope matrix
* :func:`plot_all_clusters_2d`     — grid of all clusters

Uses CDR loop annotation from :mod:`anarci_cdr` for y-axis grouping. If
``cdr_loop`` column missing, residues are grouped by chain only.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# CDR loop ordering (visually grouped, H first then L)
CDR_LOOP_ORDER = ["H1", "H2", "H3", "framework_H",
                  "L1", "L2", "L3", "framework_L",
                  "framework", "other"]

CDR_LOOP_COLORS = {
    "H1": "#fdae6b", "H2": "#fd8d3c", "H3": "#d94801",
    "framework_H": "#fee6ce",
    "L1": "#9ecae1", "L2": "#6baed6", "L3": "#2171b5",
    "framework_L": "#deebf7",
    "framework": "#cccccc",
    "other": "#aaaaaa",
}

# Compact y-tick suffix so labels never truncate (framework_H -> fwH).
_SHORT_LOOP = {
    "H1": "H1", "H2": "H2", "H3": "H3",
    "L1": "L1", "L2": "L2", "L3": "L3",
    "framework_H": "fwH", "framework_L": "fwL",
    "framework": "fw", "other": "-",
}

# Readable text colors for y-tick labels (strip colors are too light on white).
_LOOP_TEXT_COLORS = {
    "H1": "#e6550d", "H2": "#d94801", "H3": "#a63603",
    "framework_H": "#969696",
    "L1": "#3182bd", "L2": "#2171b5", "L3": "#08519c",
    "framework_L": "#969696",
    "framework": "#777777", "other": "#777777",
}


def _sort_paratope_keys(paratope_keys, cdr_map):
    """Sort paratope (chain, resnum) keys by CDR loop order, then resnum."""
    def sort_key(pk):
        loop = cdr_map.get(pk, "other")
        try:
            order_i = CDR_LOOP_ORDER.index(loop)
        except ValueError:
            order_i = len(CDR_LOOP_ORDER)
        return (order_i, pk[0], int(pk[1]))
    return sorted(paratope_keys, key=sort_key)


def plot_cluster_2d_heatmap(
    cluster_id: int,
    F_binary: np.ndarray,
    paratope_universe: list[tuple[str, int]],
    epitope_universe: list[tuple[str, int]],
    output_path: str | Path,
    *,
    cdr_map: Optional[dict] = None,
    top_n_para: int = 30,
    top_n_epi: int = 30,
    title: Optional[str] = None,
) -> Path:
    """2D heatmap of CDR × epitope contact frequency for one cluster.

    Parameters
    ----------
    F_binary
        Matrix of shape ``(n_paratope, n_epitope)`` from
        :func:`pair_contacts.build_cluster_pair_contact_map`.
    paratope_universe, epitope_universe
        Ordered lists of ``(chain, resnum)`` tuples corresponding to F_binary
        rows / columns.
    cdr_map
        Optional ``(chain, resnum) → cdr_loop`` mapping. If provided,
        paratope rows are sorted into H1/H2/H3/L1/L2/L3/framework groups
        with colored sidebar.
    top_n_para, top_n_epi
        Show only N rows/cols (sorted by total frequency) for readability.
    """
    # Restrict to top residues by row/col sums
    row_sums = F_binary.sum(axis=1)
    col_sums = F_binary.sum(axis=0)
    top_para_idx = np.argsort(-row_sums)[:top_n_para]
    top_epi_idx = np.argsort(-col_sums)[:top_n_epi]

    para_keys_kept = [paratope_universe[i] for i in top_para_idx]
    epi_keys_kept = [epitope_universe[j] for j in top_epi_idx]

    # Sort paratope by CDR loop if cdr_map present
    if cdr_map is not None:
        order = _sort_paratope_keys(para_keys_kept, cdr_map)
        para_order_idx = [para_keys_kept.index(k) for k in order]
        para_keys_kept = order
    else:
        para_order_idx = list(range(len(para_keys_kept)))

    # Sort epitope by chain then resnum (natural sequence order)
    epi_sorted_idx = sorted(range(len(epi_keys_kept)),
                             key=lambda i: (epi_keys_kept[i][0], int(epi_keys_kept[i][1])))
    epi_keys_kept = [epi_keys_kept[i] for i in epi_sorted_idx]

    # Build display matrix
    M = np.zeros((len(para_keys_kept), len(epi_keys_kept)), dtype=np.float32)
    for ri, p_local in enumerate(para_order_idx):
        for ci, e_local in enumerate(epi_sorted_idx):
            M[ri, ci] = F_binary[top_para_idx[p_local], top_epi_idx[e_local]]

    # Labels — short CDR loop suffix to avoid truncation
    loop_labels = ([cdr_map.get(p, "other") for p in para_keys_kept]
                   if cdr_map is not None else None)
    if cdr_map is not None:
        para_labels = [f"{p[0]}.{int(p[1])}·{_SHORT_LOOP.get(l, l)}"
                       for p, l in zip(para_keys_kept, loop_labels)]
    else:
        para_labels = [f"{p[0]}.{int(p[1])}" for p in para_keys_kept]
    epi_labels = [f"{e[0]}.{int(e[1])}" for e in epi_keys_kept]

    # Figure sizing
    w = max(8, 0.32 * len(epi_labels) + 3)
    h = max(5, 0.32 * len(para_labels) + 2)
    fig = plt.figure(figsize=(w, h), constrained_layout=True)

    if cdr_map is not None:
        # GridSpec: [CDR strip | heatmap]. Sharing y keeps them aligned even
        # under constrained_layout (the old add_axes approach drifted after
        # tight_layout moved only the main axes).
        gs = fig.add_gridspec(1, 2, width_ratios=[0.05, 1.0], wspace=0.02)
        strip_ax = fig.add_subplot(gs[0, 0])
        ax = fig.add_subplot(gs[0, 1], sharey=strip_ax)
        loop_colors = [CDR_LOOP_COLORS.get(l, "#cccccc") for l in loop_labels]
        strip_ax.imshow(
            np.arange(len(para_keys_kept))[:, None],
            aspect="auto",
            cmap=matplotlib.colors.ListedColormap(loop_colors),
        )
        strip_ax.set_xticks([])
        # residue labels live on the strip (left), colored by loop
        strip_ax.set_yticks(range(len(para_labels)))
        strip_ax.set_yticklabels(para_labels, fontsize=7)
        for tick, l in zip(strip_ax.get_yticklabels(), loop_labels):
            tick.set_color(_LOOP_TEXT_COLORS.get(l, "#333333"))
        strip_ax.set_ylabel("Paratope residue · CDR loop")
    else:
        ax = fig.add_subplot(1, 1, 1)
        ax.set_yticks(range(len(para_labels)))
        ax.set_yticklabels(para_labels, fontsize=7)
        ax.set_ylabel("Paratope residue")

    im = ax.imshow(M, aspect="auto", cmap="viridis", vmin=0, vmax=max(0.3, M.max()))
    if cdr_map is not None:
        ax.tick_params(axis="y", labelleft=False)  # labels are on the strip

    ax.set_xticks(range(len(epi_labels)))
    ax.set_xticklabels(epi_labels, rotation=90, fontsize=7)
    ax.set_xlabel("Epitope residue")
    ax.set_title(title or f"Cluster {cluster_id} — CDR × epitope contact frequency",
                 fontsize=11)

    cbar = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02, label="Frequency in cluster")
    cbar.ax.tick_params(labelsize=8)

    if cdr_map is not None:
        from matplotlib.patches import Patch
        loops_present = sorted(
            set(loop_labels),
            key=lambda l: CDR_LOOP_ORDER.index(l) if l in CDR_LOOP_ORDER else len(CDR_LOOP_ORDER),
        )
        ax.legend(
            handles=[Patch(facecolor=CDR_LOOP_COLORS.get(l, "#ccc"), label=l)
                     for l in loops_present],
            loc="upper left", bbox_to_anchor=(1.12, 1.0), fontsize=8,
            title="CDR loop", title_fontsize=9, framealpha=0.9,
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_all_clusters_2d(
    cluster_pair_maps: dict,
    output_dir: str | Path,
    *,
    cdr_map: Optional[dict] = None,
    top_n_para: int = 25,
    top_n_epi: int = 25,
) -> dict[int, Path]:
    """Render 2D CDR × epitope heatmap for each cluster.

    Parameters
    ----------
    cluster_pair_maps
        Output from :func:`pair_contacts.build_cluster_pair_contact_map`.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for cid in cluster_pair_maps["cluster_ids"]:
        cm = cluster_pair_maps["clusters"][cid]
        p = out_dir / f"cluster_{cid}_cdr_epitope_2d.png"
        plot_cluster_2d_heatmap(
            cid,
            F_binary=cm["F_binary"],
            paratope_universe=cluster_pair_maps["paratope_universe"],
            epitope_universe=cluster_pair_maps["epitope_universe"],
            output_path=p,
            cdr_map=cdr_map,
            top_n_para=top_n_para,
            top_n_epi=top_n_epi,
            title=f"Cluster {cid} (n_models={cm['n_models']}) — CDR × epitope contact map",
        )
        paths[cid] = p
    return paths


def plot_cluster_channel_distribution(
    pair_contacts_annotated: pd.DataFrame,
    cluster_assignments: pd.DataFrame,
    output_path: str | Path,
    *,
    cluster_col: str = "cluster",
    model_col: str = "model_id",
    channels: tuple = ("hydrophobic", "salt_bridge", "polar_hbond", "aromatic", "cation_pi"),
) -> Path:
    """Stacked bar of physchem channel composition per cluster.

    Uses pair-level data (true interaction channels) — distinct from the
    epitope-only residue type profile in :mod:`physchem_channels`.
    """
    merged = pair_contacts_annotated.merge(
        cluster_assignments[[model_col, cluster_col]], on=model_col, how="inner",
    )
    rows = []
    for cid, grp in merged.groupby(cluster_col):
        n_pairs = len(grp)
        row = {"cluster": int(cid), "n_pairs": n_pairs}
        for ch in channels:
            col = f"ch_{ch}"
            if col in grp.columns:
                row[ch] = float(grp[col].sum() / max(n_pairs, 1))
            else:
                row[ch] = 0.0
        rows.append(row)
    df = pd.DataFrame(rows).set_index("cluster").sort_index()

    colors = {
        "hydrophobic": "#fdae61",
        "salt_bridge": "#d7191c",
        "polar_hbond": "#abd9e9",
        "aromatic":    "#5e4fa2",
        "cation_pi":   "#fee08b",
    }
    fig, ax = plt.subplots(figsize=(max(6, 1.4 * len(df) + 2), 4.5))
    bottom = np.zeros(len(df))
    for ch in channels:
        if ch not in df.columns:
            continue
        vals = df[ch].to_numpy()
        ax.bar([f"C{i}" for i in df.index], vals, bottom=bottom, label=ch,
               color=colors.get(ch, "#888"), edgecolor="black", linewidth=0.4)
        bottom += vals
    ax.set_ylim(0, max(1.1, float(bottom.max()) * 1.05))
    ax.set_ylabel("Channel fraction (pair contacts)")
    ax.set_xlabel("Cluster")
    ax.set_title("Physicochemical channel distribution per cluster (pair-level)",
                 fontsize=11)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=5, fontsize=9)
    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return output_path


__all__ = [
    "plot_cluster_2d_heatmap",
    "plot_all_clusters_2d",
    "plot_cluster_channel_distribution",
    "CDR_LOOP_ORDER",
    "CDR_LOOP_COLORS",
]
