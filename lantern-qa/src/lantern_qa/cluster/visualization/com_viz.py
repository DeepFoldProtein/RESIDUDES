"""
3D / 2D scatter of per-chain COM positions in the shared anchor frame.

Reads:
    com_matrix_com3d.npy   — shape (N_models, n_probe_chains, 3)
    <method>_clustering.csv — per-model 'cluster' column + 'structure'
    representatives/*_representatives.csv — per-cluster medoid (optional)

Produces:
    com_scatter_3d.png   — one 3D axis, all probe chains overlaid,
                           colored by cluster (medoid star)
    com_scatter_2d.png   — 4-panel (XY/XZ/YZ + size bar) per probe chain
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional, Sequence, Union
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)


def _make_palette(n_clusters: int):
    """Return list of colors of length n_clusters (tab20 then cycle)."""
    cmap = plt.get_cmap("tab20")
    return [cmap(i % 20) for i in range(n_clusters)]


def render_com_scatter(
    com3d_path: Union[str, Path],
    labels: np.ndarray,
    structure_names: Sequence[str],
    probe_chains: Sequence[str],
    medoid_indices: Optional[Sequence[int]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    title: str = "COM scatter — anchor-aligned",
    anchor_chains: Sequence[str] = (),
    chain_marker: dict | None = None,
) -> tuple[plt.Figure, plt.Figure]:
    """
    Build 3D + 2D scatter figures for per-chain COMs.

    Args:
      com3d_path      : npy file with array of shape (N, K, 3)
                        — (model, probe_chain, xyz) in shared anchor frame.
      labels          : (N,) cluster label per model.
      structure_names : (N,) names (for hover text in future plotly export).
      probe_chains    : list of K probe chain ids in the same order as com3d
                        last-but-one axis.
      medoid_indices  : optional medoid index per cluster. Highlighted as ★.
      output_dir      : if set, write PNGs there.
      title           : figure-level title.
      anchor_chains   : just for the label; not used in plotting.
      chain_marker    : optional {chain: matplotlib marker} per probe chain.

    Returns:
      (fig3d, fig2d)
    """
    com3d = np.load(com3d_path)
    if com3d.shape[1] != len(probe_chains):
        raise ValueError(
            f"probe_chains ({len(probe_chains)}) ≠ com3d K dim "
            f"({com3d.shape[1]}); check ordering"
        )
    N = com3d.shape[0]
    labels = np.asarray(labels)
    unique = sorted(set(labels[labels >= 0].tolist()))
    palette = _make_palette(len(unique))
    color_of = {c: palette[i] for i, c in enumerate(unique)}

    if chain_marker is None:
        markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
        chain_marker = {ch: markers[i % len(markers)] for i, ch in enumerate(probe_chains)}

    anchor_tag = "".join(anchor_chains) if anchor_chains else "none"
    probe_tag = "".join(probe_chains)

    # ----- 3D figure -----
    fig3d = plt.figure(figsize=(10, 8))
    ax = fig3d.add_subplot(111, projection="3d")
    for k, ch in enumerate(probe_chains):
        for cl in unique:
            mask = labels == cl
            xs, ys, zs = com3d[mask, k, 0], com3d[mask, k, 1], com3d[mask, k, 2]
            ax.scatter(xs, ys, zs,
                       c=[color_of[cl]] * mask.sum(),
                       marker=chain_marker[ch],
                       s=20, alpha=0.65,
                       edgecolors="none")
    # Medoid stars
    if medoid_indices is not None:
        for cl_idx, mi in enumerate(medoid_indices):
            if mi is None or mi < 0:
                continue
            cl = unique[cl_idx] if cl_idx < len(unique) else cl_idx
            for k, ch in enumerate(probe_chains):
                ax.scatter([com3d[mi, k, 0]], [com3d[mi, k, 1]], [com3d[mi, k, 2]],
                           marker="*", s=200,
                           c=[color_of.get(cl, "#000")],
                           edgecolors="black", linewidths=0.5, zorder=5)
    # Anchor center marker (origin region)
    ax.scatter([0], [0], [0], marker="x", c="black", s=60, label=f"anchor [{anchor_tag}]")
    ax.set_xlabel("X (Å)")
    ax.set_ylabel("Y (Å)")
    ax.set_zlabel("Z (Å)")
    ax.set_title(f"{title}   anchor={anchor_tag}  probe={probe_tag}")
    ax.legend(fontsize=8, loc="upper right")

    # Probe chain legend (markers)
    handles = []
    for ch in probe_chains:
        handles.append(
            ax.scatter([], [], [], marker=chain_marker[ch], c="gray", s=40, label=f"chain {ch}")
        )
    ax.legend(handles=handles, loc="lower left", fontsize=8, title="probe chains")

    fig3d.tight_layout()

    # ----- 2D figure: 3 views × K probe chains (max 4 chains to keep legible) -----
    K = min(len(probe_chains), 4)
    fig2d, axes = plt.subplots(K, 3, figsize=(12, 3.5 * K), squeeze=False)
    for k in range(K):
        ch = probe_chains[k]
        for view_idx, (lbl_xy, ix, iy) in enumerate([
            ("XY", 0, 1),
            ("XZ", 0, 2),
            ("YZ", 1, 2),
        ]):
            ax2 = axes[k, view_idx]
            for cl in unique:
                mask = labels == cl
                ax2.scatter(com3d[mask, k, ix], com3d[mask, k, iy],
                            c=[color_of[cl]] * mask.sum(),
                            s=15, alpha=0.6, edgecolors="none")
            if medoid_indices is not None:
                for cl_idx, mi in enumerate(medoid_indices):
                    if mi is None or mi < 0:
                        continue
                    cl = unique[cl_idx] if cl_idx < len(unique) else cl_idx
                    ax2.scatter(com3d[mi, k, ix], com3d[mi, k, iy],
                                marker="*", s=120,
                                c=[color_of.get(cl, "#000")],
                                edgecolors="black", linewidths=0.5, zorder=5)
            ax2.scatter([0], [0], marker="x", c="black", s=40)
            ax2.set_xlabel(f"{lbl_xy[0]} (Å)")
            ax2.set_ylabel(f"{lbl_xy[1]} (Å)")
            ax2.set_aspect("equal", adjustable="datalim")
            if view_idx == 0:
                ax2.set_title(f"chain {ch}  ({N} models)", fontweight="bold")
    fig2d.suptitle(f"{title}  —  anchor={anchor_tag}  probe={probe_tag}", fontsize=11)
    fig2d.tight_layout()

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        fig3d.savefig(output_dir / "com_scatter_3d.png",
                      dpi=140, bbox_inches="tight", facecolor="white")
        fig2d.savefig(output_dir / "com_scatter_2d.png",
                      dpi=140, bbox_inches="tight", facecolor="white")

    return fig3d, fig2d


__all__ = ["render_com_scatter"]
