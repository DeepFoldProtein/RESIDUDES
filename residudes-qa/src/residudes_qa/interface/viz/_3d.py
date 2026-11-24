"""3D visualisations (matplotlib only).

Generalized from H1335 ``36_3d_viz.py`` + ``41_filter_recluster_viz.py``.
Three public functions:

* :func:`plot_3d_clusters`     — colour by cluster id, 3-view subplot
* :func:`plot_3d_xgb_gradient` — colour by xgb_pred (viridis), 3-view
* :func:`plot_filter_funnel`   — horizontal bar chart of filter cascade

All functions write PNG to ``output_png`` and ``plt.close()`` the figure
before returning.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd


# Default rotation angles for the 3-view subplot.
_VIEWS: list[tuple[int, int, str]] = [
    (20, -60, "top"),
    (10, 30, "side"),
    (-5, 120, "front"),
]


def _setup_mpl():
    """Headless matplotlib import — deferred to call time."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _scatter_chains(ax, ref_chain_ca: Optional[dict[str, np.ndarray]]) -> None:
    """Optional grey scatter of reference chain CA + chain-name labels."""
    if not ref_chain_ca:
        return
    for cname, arr in ref_chain_ca.items():
        if arr is None or len(arr) == 0:
            continue
        ax.scatter(*arr.T, s=1, c="lightgray", alpha=0.25)
        ax.text(*arr.mean(0), cname, fontsize=11, weight="bold")


def plot_3d_clusters(
    master_df: pd.DataFrame,
    com_xyz: np.ndarray,
    labels: np.ndarray,
    output_png: str | Path,
    title: Optional[str] = None,
    picks: Optional[pd.DataFrame] = None,
    ref_chain_ca: Optional[dict[str, np.ndarray]] = None,
    pick_score_col: str = "xgb_pred",
) -> Path:
    """3-view 3D scatter coloured by cluster label.

    Parameters
    ----------
    master_df
        Used only for length sanity; coordinates come from ``com_xyz``.
    com_xyz
        ``(n, 3)`` array of COM coordinates (one row per model in master).
    labels
        Cluster id per row in ``com_xyz``.  ``-1`` = unassigned (greyed out).
    picks
        Optional picks-DataFrame to over-plot as red stars.  Must contain
        ``com_x`` / ``com_y`` / ``com_z``.
    ref_chain_ca
        Optional ``{label: (N,3) CA coords}`` for the reference structure to
        scatter as a faint grey backdrop.
    """
    plt = _setup_mpl()
    com_xyz = np.asarray(com_xyz)
    labels = np.asarray(labels)
    n_clusters = int(labels.max()) + 1 if (labels >= 0).any() else 0

    fig = plt.figure(figsize=(18, 6))
    cmap = plt.get_cmap("tab10", max(n_clusters, 1))
    for view_i, (elev, azim, name) in enumerate(_VIEWS, 1):
        ax = fig.add_subplot(1, 3, view_i, projection="3d")
        _scatter_chains(ax, ref_chain_ca)
        # noise / unassigned points
        noise = labels < 0
        if noise.any():
            ax.scatter(
                com_xyz[noise, 0], com_xyz[noise, 1], com_xyz[noise, 2],
                s=6, c="lightgray", alpha=0.3, label="unassigned",
            )
        for k in range(n_clusters):
            mask = labels == k
            if not mask.any():
                continue
            ax.scatter(
                com_xyz[mask, 0], com_xyz[mask, 1], com_xyz[mask, 2],
                s=20, c=[cmap(k)], alpha=0.7,
                label=f"C{k} (n={int(mask.sum())})",
            )
            # cluster centroid
            cx = com_xyz[mask].mean(0)
            ax.scatter(
                [cx[0]], [cx[1]], [cx[2]],
                s=180, c=[cmap(k)], edgecolors="black", linewidths=1.5,
                marker="o", zorder=10,
            )
        # picks: red stars
        if picks is not None and len(picks):
            req = {"com_x", "com_y", "com_z"}
            if req.issubset(picks.columns):
                ax.scatter(
                    picks["com_x"], picks["com_y"], picks["com_z"],
                    s=250, c="red", marker="*", edgecolors="black",
                    linewidths=1.5, zorder=11,
                )
        ax.view_init(elev=elev, azim=azim)
        sub_title = f"{name} view"
        if title and view_i == 1:
            sub_title = f"{title} — {name} view"
        ax.set_title(sub_title, fontsize=10)
        if view_i == 1:
            ax.legend(loc="upper left", fontsize=8)

    if title:
        fig.suptitle(title, fontsize=12, weight="bold")
    fig.tight_layout()
    out = Path(output_png)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out}", flush=True)
    return out


def plot_3d_xgb_gradient(
    master_df: pd.DataFrame,
    com_xyz: np.ndarray,
    xgb_pred: np.ndarray,
    output_png: str | Path,
    title: Optional[str] = None,
    ref_chain_ca: Optional[dict[str, np.ndarray]] = None,
) -> Path:
    """3-view scatter coloured by xgb_pred (viridis colormap)."""
    plt = _setup_mpl()
    com_xyz = np.asarray(com_xyz)
    xgb_pred = np.asarray(xgb_pred, dtype=float)
    valid = np.isfinite(com_xyz).all(axis=1) & np.isfinite(xgb_pred)

    fig = plt.figure(figsize=(18, 6))
    for view_i, (elev, azim, name) in enumerate(_VIEWS, 1):
        ax = fig.add_subplot(1, 3, view_i, projection="3d")
        _scatter_chains(ax, ref_chain_ca)
        sc = ax.scatter(
            com_xyz[valid, 0], com_xyz[valid, 1], com_xyz[valid, 2],
            c=xgb_pred[valid], cmap="viridis", s=14, alpha=0.75,
        )
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(f"{name} view", fontsize=10)
        if view_i == 3:
            fig.colorbar(sc, ax=ax, shrink=0.6, label="xgb_pred")
    if title:
        fig.suptitle(title, fontsize=12, weight="bold")
    fig.tight_layout()
    out = Path(output_png)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out}", flush=True)
    return out


def plot_filter_funnel(
    stats_history: Iterable[dict[str, Any]] | dict[str, Any],
    output_png: str | Path,
    title: str = "Filter cascade",
) -> Path:
    """Horizontal bar chart of filter cascade survival.

    Accepts a sequence of filter-stats dicts (as returned by the
    :mod:`residudes_qa.interface.filter` functions) **or** a single dict (its
    ``by_filter_step`` list is used).
    """
    plt = _setup_mpl()
    # Normalize into a single list of (label, n_after, threshold) tuples.
    rows: list[tuple[str, int, str]] = []
    if isinstance(stats_history, dict):
        stats_history = [stats_history]
    for stats in stats_history:
        if "n_input" in stats:
            rows.append((f"input", int(stats["n_input"]), ""))
        for step in stats.get("by_filter_step", []):
            rows.append(
                (
                    str(step.get("step", "")),
                    int(step.get("n_after", 0)),
                    str(step.get("threshold", "")),
                )
            )
        if "n_output" in stats and (not rows or rows[-1][1] != stats["n_output"]):
            rows.append(("output", int(stats["n_output"]), ""))

    labels = [r[0] for r in rows]
    counts = [r[1] for r in rows]
    thresholds = [r[2] for r in rows]

    fig, ax = plt.subplots(figsize=(10, max(3, 0.4 * len(rows) + 1)))
    ypos = np.arange(len(rows))
    ax.barh(ypos, counts, color="steelblue")
    ax.set_yticks(ypos)
    ax.set_yticklabels(
        [f"{lbl}  {thr}".rstrip() for lbl, thr in zip(labels, thresholds)],
        fontsize=9,
    )
    ax.invert_yaxis()
    ax.set_xlabel("# models surviving")
    ax.set_title(title, fontsize=11)
    for i, c in enumerate(counts):
        ax.text(c, i, f"  {c}", va="center", fontsize=8)
    fig.tight_layout()
    out = Path(output_png)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out}", flush=True)
    return out
