"""
Cluster-level hierarchical dendrogram + clustermap.

Shows how the *clusters* (not individual structures) relate to one another:
which clusters are closest, and the order in which they merge / branch apart.

The tree is built on the K x K mean inter-cluster distance matrix — the same
quantity drawn by ``plot_inter_cluster_heatmap`` — using average linkage to
match the agglomerative clustering in ``core.clustering``. With K leaves the
plot stays readable no matter how many structures were clustered (tens of
thousands of models still collapse to K labelled leaves).

Two entry points:

* ``plot_cluster_dendrogram`` — standalone tree, leaves = clusters.
* ``plot_cluster_clustermap`` — tree on top + inter-cluster heatmap below,
  columns/rows reordered by the dendrogram leaf order (the "position +
  branching" combined view).
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt

from lantern_qa.cluster.visualization._colors import get_cluster_color

_NEUTRAL_LINK = "#555555"


def compute_inter_cluster_distance(
    distance_matrix: np.ndarray,
    labels: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean pairwise distance within / between clusters.

    Returns ``(inter_dist, unique_labels, sizes)`` where ``inter_dist`` is the
    K x K symmetric matrix (diagonal = mean intra-cluster distance) and
    ``sizes[i]`` is the membership count of cluster ``unique_labels[i]``.
    """
    unique_labels = np.unique(labels)
    k = len(unique_labels)
    inter = np.zeros((k, k), dtype=float)
    sizes = np.zeros(k, dtype=int)

    masks = [labels == lab for lab in unique_labels]
    for i in range(k):
        sizes[i] = int(masks[i].sum())
        for j in range(k):
            if i == j:
                sub = distance_matrix[masks[i]][:, masks[i]]
                triu = np.triu_indices_from(sub, k=1)
                inter[i, j] = sub[triu].mean() if triu[0].size else 0.0
            else:
                inter[i, j] = distance_matrix[masks[i]][:, masks[j]].mean()
    # Enforce exact symmetry (float means of A->B vs B->A can drift slightly).
    inter = (inter + inter.T) / 2.0
    return inter, unique_labels, sizes


def compute_cluster_linkage(
    distance_matrix: np.ndarray,
    labels: np.ndarray,
    linkage_method: str = "average",
) -> Tuple[Optional[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """Build a SciPy linkage matrix over clusters.

    Returns ``(Z, inter_dist, unique_labels, sizes)``. ``Z`` is ``None`` when
    there are fewer than two clusters (nothing to merge).
    """
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import squareform

    inter, unique_labels, sizes = compute_inter_cluster_distance(distance_matrix, labels)
    if len(unique_labels) < 2:
        return None, inter, unique_labels, sizes

    condensed = squareform(inter, checks=False)
    # optimal_ordering reorders leaves (without changing tree topology) to
    # minimise the distance between adjacent leaves — so the reordered
    # inter-cluster heatmap becomes block-diagonal and visually matches the
    # tree (same behaviour as seaborn clustermap).
    Z = linkage(condensed, method=linkage_method, optimal_ordering=True)
    return Z, inter, unique_labels, sizes


def _leaf_labels(
    unique_labels: np.ndarray,
    sizes: np.ndarray,
    medoid_indices: Optional[List[int]] = None,
    structure_names: Optional[List[str]] = None,
) -> List[str]:
    """``C{id} (n={size})`` per leaf, plus medoid name when available."""
    out = []
    for i, lab in enumerate(unique_labels):
        text = f"C{int(lab)} (n={int(sizes[i])})"
        if medoid_indices is not None and i < len(medoid_indices):
            mi = medoid_indices[i]
            if mi is not None and structure_names is not None and mi < len(structure_names):
                name = Path(str(structure_names[mi])).stem
                text += f"\n{name}"
        out.append(text)
    return out


def _draw_dendrogram(
    ax,
    Z: np.ndarray,
    unique_labels: np.ndarray,
    leaf_labels: List[str],
    orientation: str = "top",
):
    """Draw a cluster dendrogram on ``ax``; colour leaf labels by cluster.

    Returns the SciPy dendrogram dict (its ``leaves`` give the ordering).
    """
    from scipy.cluster.hierarchy import dendrogram

    dn = dendrogram(
        Z,
        ax=ax,
        orientation=orientation,
        labels=leaf_labels,
        color_threshold=0,                 # one neutral colour for all links;
        above_threshold_color=_NEUTRAL_LINK,  # leaves carry the cluster colours
        leaf_font_size=9,
    )
    # Colour each leaf tick label by its cluster's palette colour.
    leaf_order = dn["leaves"]
    if orientation in ("top", "bottom"):
        ticklabels = ax.get_xticklabels()
    else:
        ticklabels = ax.get_yticklabels()
    for pos, tl in enumerate(ticklabels):
        cid = int(unique_labels[leaf_order[pos]])
        tl.set_color(get_cluster_color(cid))
        tl.set_fontweight("bold")
    return dn


def plot_cluster_dendrogram(
    distance_matrix: np.ndarray,
    labels: np.ndarray,
    structure_names: Optional[List[str]] = None,
    medoid_indices: Optional[List[int]] = None,
    linkage_method: str = "average",
    title: str = "Cluster Hierarchy",
    output_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (10, 6),
) -> Optional[plt.Figure]:
    """Standalone cluster-level dendrogram (leaves = clusters).

    Returns the figure, or ``None`` if there are fewer than two clusters.
    """
    Z, _, unique_labels, sizes = compute_cluster_linkage(
        distance_matrix, labels, linkage_method
    )
    if Z is None:
        return None

    leaf_labels = _leaf_labels(unique_labels, sizes, medoid_indices, structure_names)
    fig, ax = plt.subplots(figsize=figsize)
    _draw_dendrogram(ax, Z, unique_labels, leaf_labels, orientation="top")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylabel(f"Mean inter-cluster distance ({linkage_method} linkage)", fontsize=10)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    return fig


def plot_cluster_clustermap(
    distance_matrix: np.ndarray,
    labels: np.ndarray,
    structure_names: Optional[List[str]] = None,
    medoid_indices: Optional[List[int]] = None,
    linkage_method: str = "average",
    title: str = "Cluster Hierarchy & Inter-Cluster Distance",
    cmap: str = "viridis",
    output_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (9, 9),
) -> Optional[plt.Figure]:
    """Dendrogram (top) + inter-cluster distance heatmap (below), co-ordered.

    The heatmap rows/columns are reordered to match the dendrogram leaf order,
    so adjacency in the tree lines up with adjacency in the map. Returns the
    figure, or ``None`` if there are fewer than two clusters.
    """
    Z, inter, unique_labels, sizes = compute_cluster_linkage(
        distance_matrix, labels, linkage_method
    )
    if Z is None:
        return None

    k = len(unique_labels)
    leaf_labels = _leaf_labels(unique_labels, sizes, medoid_indices, structure_names)

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(
        2, 2, height_ratios=[1, 3.2], width_ratios=[24, 1],
        hspace=0.04, wspace=0.04,
    )
    ax_dendro = fig.add_subplot(gs[0, 0])
    ax_heat = fig.add_subplot(gs[1, 0])
    ax_cbar = fig.add_subplot(gs[1, 1])

    # Top dendrogram defines the leaf ordering.
    dn = _draw_dendrogram(ax_dendro, Z, unique_labels, leaf_labels, orientation="top")
    order = dn["leaves"]
    ax_dendro.set_xticks([])
    ax_dendro.set_title(title, fontsize=13, fontweight="bold")
    for spine in ax_dendro.spines.values():
        spine.set_visible(False)

    # Heatmap reordered by leaf order.
    reordered = inter[np.ix_(order, order)]
    im = ax_heat.imshow(reordered, cmap=cmap, aspect="auto")
    ax_heat.set_xticks(range(k))
    ax_heat.set_yticks(range(k))
    ord_labels = [leaf_labels[o].split("\n")[0] for o in order]
    ax_heat.set_xticklabels(ord_labels, rotation=45, ha="right", fontsize=8)
    ax_heat.set_yticklabels(ord_labels, fontsize=8)
    for pos, o in enumerate(order):
        cid = int(unique_labels[o])
        col = get_cluster_color(cid)
        ax_heat.get_xticklabels()[pos].set_color(col)
        ax_heat.get_yticklabels()[pos].set_color(col)

    thr = reordered.mean()
    for i in range(k):
        for j in range(k):
            ax_heat.text(
                j, i, f"{reordered[i, j]:.2f}", ha="center", va="center",
                fontsize=7, color="white" if reordered[i, j] > thr else "black",
            )
    cbar = fig.colorbar(im, cax=ax_cbar)
    cbar.set_label("Mean distance", fontsize=9)

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    return fig


__all__ = [
    "compute_inter_cluster_distance",
    "compute_cluster_linkage",
    "plot_cluster_dendrogram",
    "plot_cluster_clustermap",
]
