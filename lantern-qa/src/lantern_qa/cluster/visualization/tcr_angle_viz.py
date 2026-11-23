"""
2D scatter of (crossing, incident) angles colored by cluster.

Companion to core/gpu/tcr_angle.py. Renders the user-facing plot for
TCR docking angle clustering.

Output:
    tcr_angle_scatter.png  — single scatter, crossing on x, incident on y
                              cluster color, medoid star, optional density
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional, Sequence, Union
import numpy as np
import matplotlib.pyplot as plt


def _palette(n):
    cmap = plt.get_cmap("tab20")
    return [cmap(i % 20) for i in range(n)]


def render_tcr_angle_scatter(
    features_deg: np.ndarray,         # (N, 2)
    labels: np.ndarray,                # (N,)
    structure_names: Sequence[str],
    medoid_indices: Optional[Sequence[int]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    title: str = "TCR docking angle clusters",
) -> plt.Figure:
    """
    Args:
      features_deg : (N, 2) [crossing_deg, incident_deg]
      labels       : (N,) cluster id per model
      structure_names : for tooltip-style annotation (future plotly).
      medoid_indices  : per-cluster medoid global index → star marker.
      output_dir   : if set, save tcr_angle_scatter.png there.
    """
    features_deg = np.asarray(features_deg)
    labels = np.asarray(labels)
    unique = sorted(set(labels[labels >= 0].tolist()))
    pal = _palette(len(unique))
    color_of = {c: pal[i] for i, c in enumerate(unique)}

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- Left: scatter colored by cluster ---
    ax = axes[0]
    for cl in unique:
        mask = labels == cl
        ax.scatter(features_deg[mask, 0], features_deg[mask, 1],
                   c=[color_of[cl]] * mask.sum(),
                   s=14, alpha=0.55, edgecolors="none",
                   label=f"L{cl}  n={mask.sum()}")
    if medoid_indices is not None:
        for cl_idx, mi in enumerate(medoid_indices):
            if mi is None or mi < 0:
                continue
            cl = unique[cl_idx] if cl_idx < len(unique) else cl_idx
            ax.scatter(features_deg[mi, 0], features_deg[mi, 1],
                       marker="*", s=200,
                       c=[color_of.get(cl, "#000")],
                       edgecolors="black", linewidths=0.5, zorder=5)
    ax.set_xlabel("crossing angle (deg)")
    ax.set_ylabel("incident angle (deg)")
    ax.set_title(f"{title}  (cluster colors)")
    if len(unique) <= 12:
        ax.legend(fontsize=7, loc="best", ncol=2)
    ax.grid(alpha=0.2)

    # --- Right: 2D density + canonical/atypical labels ---
    ax2 = axes[1]
    ax2.hexbin(features_deg[:, 0], features_deg[:, 1],
               gridsize=40, cmap="Blues", mincnt=1)
    # canonical TCR-pMHC docking range (literature): crossing 30-70, incident 5-25
    from matplotlib.patches import Rectangle
    canonical = Rectangle((30, 5), 40, 20,
                          linewidth=1.5, edgecolor="green",
                          facecolor="none", linestyle="--",
                          label="canonical (Rudolph/Wilson)")
    ax2.add_patch(canonical)
    ax2.set_xlabel("crossing angle (deg)")
    ax2.set_ylabel("incident angle (deg)")
    ax2.set_title("Density + canonical pose box")
    ax2.legend(fontsize=8, loc="upper right")
    ax2.grid(alpha=0.2)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir / "tcr_angle_scatter.png",
                    dpi=140, bbox_inches="tight", facecolor="white")
    return fig


__all__ = ["render_tcr_angle_scatter"]
