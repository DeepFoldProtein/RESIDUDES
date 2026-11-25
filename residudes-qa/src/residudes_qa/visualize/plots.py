"""Matplotlib-based visualization for AF3 analysis results."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import numpy as np

from .metrics import (
    chain_boundaries, residue_plddt,
    pae_matrix, contact_prob_matrix, mean_pae_per_chain_pair,
    pde_matrix, mean_pde_per_chain_pair,
)

# Chain color palette (up to 8 chains)
CHAIN_COLORS = [
    "#2196F3", "#F44336", "#4CAF50", "#FF9800",
    "#9C27B0", "#00BCD4", "#795548", "#607D8B",
]

_PLDDT_THRESHOLDS = [50, 70, 90]
_PLDDT_THRESHOLD_COLORS = ["#d62728", "#ff7f0e", "#2ca02c"]
_PLDDT_THRESHOLD_LABELS = ["Very low (50)", "Low (70)", "Confident (90)"]


def _chain_color_map(chain_labels: list[str]) -> dict[str, str]:
    return {c: CHAIN_COLORS[i % len(CHAIN_COLORS)] for i, c in enumerate(chain_labels)}


def _add_chain_boundary_lines(ax: plt.Axes, boundaries: list[tuple[str, int, int]], orientation: str = "both") -> None:
    """Draw dashed boundary lines and chain label annotations."""
    for chain_label, start, end in boundaries:
        mid = (start + end) / 2
        if orientation in ("v", "both"):
            if start > 0:
                ax.axvline(start - 0.5, color="white", linestyle="--", linewidth=0.8, alpha=0.8)
            ax.text(mid, ax.get_ylim()[1] * 1.01, chain_label,
                    ha="center", va="bottom", fontsize=8, color="black", fontweight="bold",
                    transform=ax.transData, clip_on=False)
        if orientation in ("h", "both"):
            if start > 0:
                ax.axhline(start - 0.5, color="white", linestyle="--", linewidth=0.8, alpha=0.8)


# ---------------------------------------------------------------------------
# pLDDT
# ---------------------------------------------------------------------------

def plot_plddt(
    samples: list[tuple[int, dict]],
    title: str = "Per-residue pLDDT",
    figsize: tuple[int, int] = (12, 4),
) -> plt.Figure:
    """Overlay pLDDT for multiple samples on a single plot.

    Args:
        samples: [(sample_num, confidences_dict), ...] — all samples within one seed.
    """
    if not samples:
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title(title)
        return fig

    n = len(samples)
    sample_palette = plt.cm.tab10

    fig, ax = plt.subplots(figsize=figsize)

    bounds = chain_boundaries(samples[0][1])
    first_res_idx, _, _ = residue_plddt(samples[0][1])

    for idx, (sample_num, conf) in enumerate(samples):
        res_idx, plddt, _ = residue_plddt(conf)
        color = sample_palette(idx / max(n - 1, 1))
        ax.plot(res_idx, plddt, color=color, label=f"Sample {sample_num}",
                linewidth=1.0, alpha=0.8)

    for thresh, ref_color, lbl in zip(_PLDDT_THRESHOLDS, _PLDDT_THRESHOLD_COLORS, _PLDDT_THRESHOLD_LABELS):
        ax.axhline(thresh, color=ref_color, linestyle="--", linewidth=0.8, alpha=0.6, label=lbl)

    for _, start, _ in bounds:
        if start > 0:
            ax.axvline(first_res_idx[start] - 0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)

    ax.set_xlim(first_res_idx[0], first_res_idx[-1])
    ax.set_ylim(0, 100)
    ax.set_xlabel("Residue index")
    ax.set_ylabel("pLDDT")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8, ncol=3)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# PAE heatmap
# ---------------------------------------------------------------------------

def plot_pae_heatmap(
    samples: list[tuple[int, dict]],
    title: str = "Predicted Aligned Error (PAE)",
    vmax: float = 30.0,
) -> plt.Figure:
    """1 row × N_samples columns of PAE heatmaps.

    Args:
        samples: [(sample_num, confidences_dict), ...] — all samples within one seed.
    """
    n = len(samples)
    if n == 0:
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.set_title(title)
        return fig

    ps = 3.5
    fig, axes = plt.subplots(1, n, figsize=(n * ps, ps + 0.8), squeeze=False)

    for col, (sample_num, conf) in enumerate(samples):
        pae = pae_matrix(conf)
        bounds = chain_boundaries(conf)
        pair_means = mean_pae_per_chain_pair(pae, bounds)
        pair_str = " ".join(f"{ci}→{cj}:{v:.1f}" for (ci, cj), v in pair_means.items())

        ax = axes[0, col]
        im = ax.imshow(pae, cmap="RdYlGn_r", vmin=0, vmax=vmax, aspect="auto", origin="upper")
        if col == n - 1:
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("PAE (Å)", fontsize=8)

        for _, start, _ in bounds[1:]:
            ax.axvline(start - 0.5, color="white", linestyle="--", linewidth=0.8, alpha=0.8)
            ax.axhline(start - 0.5, color="white", linestyle="--", linewidth=0.8, alpha=0.8)

        tick_pos = [(s + e) // 2 for _, s, e in bounds]
        tick_lbl = [c for c, _, _ in bounds]
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_lbl, fontsize=7)
        ax.set_yticks(tick_pos)
        ax.set_yticklabels(tick_lbl, fontsize=7)
        ax.set_title(f"Sample {sample_num}\n{pair_str}", fontsize=8)
        ax.tick_params(labelsize=6)

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Contact probability heatmap
# ---------------------------------------------------------------------------

def plot_contact_heatmap(
    confidences: dict,
    title: str = "Contact Probability",
    figsize: tuple[int, int] = (7, 6),
) -> plt.Figure:
    contact = contact_prob_matrix(confidences)
    bounds = chain_boundaries(confidences)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(contact, cmap="Blues", vmin=0, vmax=1, aspect="auto", origin="upper")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("P(contact)", fontsize=9)

    _add_chain_boundary_lines(ax, bounds, orientation="both")

    tick_positions = [(s + e) // 2 for _, s, e in bounds]
    tick_labels = [c for c, _, _ in bounds]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=9)
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels, fontsize=9)
    ax.set_xlabel("Residue i", fontsize=9)
    ax.set_ylabel("Residue j", fontsize=9)
    ax.set_title(title)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# PDE heatmap
# ---------------------------------------------------------------------------

def plot_pde_heatmap(
    samples: list[tuple[int, dict]],
    title: str = "Predicted Distance Error (PDE)",
    vmax: float = 30.0,
) -> plt.Figure:
    """1 row × N_samples columns of PDE heatmaps (symmetric, viridis colormap).

    Args:
        samples: [(sample_num, confidences_dict), ...] — all samples within one seed.
    """
    n = len(samples)
    if n == 0:
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.set_title(title)
        return fig

    ps = 3.5
    fig, axes = plt.subplots(1, n, figsize=(n * ps, ps + 0.8), squeeze=False)

    for col, (sample_num, conf) in enumerate(samples):
        pde = pde_matrix(conf)
        bounds = chain_boundaries(conf)
        pair_means = mean_pde_per_chain_pair(pde, bounds)
        pair_str = " ".join(
            f"{ci}↔{cj}:{v:.1f}" for (ci, cj), v in pair_means.items() if ci <= cj
        )

        ax = axes[0, col]
        im = ax.imshow(pde, cmap="viridis", vmin=0, vmax=vmax, aspect="auto", origin="upper")
        if col == n - 1:
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("PDE (Å)", fontsize=8)

        for _, start, _ in bounds[1:]:
            ax.axvline(start - 0.5, color="white", linestyle="--", linewidth=0.8, alpha=0.8)
            ax.axhline(start - 0.5, color="white", linestyle="--", linewidth=0.8, alpha=0.8)

        tick_pos = [(s + e) // 2 for _, s, e in bounds]
        tick_lbl = [c for c, _, _ in bounds]
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_lbl, fontsize=7)
        ax.set_yticks(tick_pos)
        ax.set_yticklabels(tick_lbl, fontsize=7)
        ax.set_title(f"Sample {sample_num}\n{pair_str}", fontsize=8)
        ax.tick_params(labelsize=6)

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Ranking bar chart
# ---------------------------------------------------------------------------

def plot_ranking_bars(
    ranking_df,
    title: str = "Ranking Scores by (Seed, Sample)",
    figsize: tuple[int, int] = (14, 5),
) -> plt.Figure:
    import matplotlib.cm as cm

    df = ranking_df.sort_values(["seed", "sample"]).reset_index(drop=True)
    labels = [f"S{row.seed}-{row.sample}" for row in df.itertuples()]
    scores = df["ranking_score"].values

    norm = plt.Normalize(vmin=scores.min() - 0.01, vmax=1.0)
    cmap = cm.viridis
    colors = [cmap(norm(s)) for s in scores]

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(range(len(scores)), scores, color=colors, edgecolor="none", width=0.7)

    # Add seed group separators
    seeds = df["seed"].values
    for i in range(1, len(seeds)):
        if seeds[i] != seeds[i - 1]:
            ax.axvline(i - 0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

    ax.axhline(0.8, color="#ff7f0e", linestyle="--", linewidth=1.0, label="Good (0.8)", alpha=0.8)
    ax.axhline(0.9, color="#2ca02c", linestyle="--", linewidth=1.0, label="Excellent (0.9)", alpha=0.8)

    ax.set_xticks(range(len(scores)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Ranking Score")
    ax.set_title(title)
    ax.legend(fontsize=8)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.01)
    cbar.set_label("Ranking Score", fontsize=8)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Metrics comparison
# ---------------------------------------------------------------------------

def plot_metrics_comparison(
    summary_df,
    metrics: list[str] | None = None,
    title: str = "Summary Metrics Comparison",
    figsize: tuple[int, int] = (10, 5),
) -> plt.Figure:
    if metrics is None:
        metrics = ["ptm", "iptm", "ranking_score", "fraction_disordered"]

    metric_labels = {
        "ptm": "pTM",
        "iptm": "iPTM",
        "ranking_score": "Ranking Score",
        "fraction_disordered": "Frac. Disordered",
    }

    labels = summary_df["label"].tolist()
    x = np.arange(len(metrics))
    width = 0.8 / len(labels)
    offsets = np.linspace(-(len(labels) - 1) * width / 2, (len(labels) - 1) * width / 2, len(labels))

    fig, ax = plt.subplots(figsize=figsize)
    colors = [CHAIN_COLORS[i % len(CHAIN_COLORS)] for i in range(len(labels))]

    for i, (label, color) in enumerate(zip(labels, colors)):
        row = summary_df[summary_df["label"] == label].iloc[0]
        vals: list[float] = []
        for m in metrics:
            v = row.get(m)
            vals.append(float(v) if v is not None and not (isinstance(v, float) and np.isnan(v)) else 0.0)
        hatches = [
            "//" if (row.get(m) is None or (isinstance(row.get(m), float) and np.isnan(row.get(m)))) else ""
            for m in metrics
        ]
        bars = ax.bar(x + offsets[i], vals, width, label=label, color=color, edgecolor="white", linewidth=0.5)
        for bar, hatch in zip(bars, hatches):
            bar.set_hatch(hatch)

    ax.set_xticks(x)
    ax.set_xticklabels([metric_labels.get(m, m) for m in metrics], fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score (0–1)")
    ax.set_title(title)
    ax.legend(title="Prediction type", fontsize=9)
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Distogram
# ---------------------------------------------------------------------------

def plot_distogram(
    disto_list: list[tuple[int, np.ndarray, np.ndarray]],
    confidences: dict | None = None,
    title: str = "Distogram",
) -> plt.Figure:
    """2 rows × N_seeds columns. Row 0 = Expected, Row 1 = Argmax.

    Distogram is saved once per seed (not per sample) by AF3 — the trunk representation
    computed before structure sampling. One PNG per prediction type.

    Args:
        disto_list: [(seed_num, dist_expected, dist_argmax), ...] — one entry per seed.
    """
    n = len(disto_list)
    if n == 0:
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.set_title(title)
        return fig

    bounds = chain_boundaries(confidences) if confidences else []
    ps = 3.5
    fig, axes = plt.subplots(2, n, figsize=(n * ps, 2 * ps), squeeze=False)

    def _chain_lines(ax: plt.Axes) -> None:
        for _, start, _ in bounds[1:]:
            ax.axvline(start - 0.5, color="white", linestyle="--", linewidth=0.8, alpha=0.8)
            ax.axhline(start - 0.5, color="white", linestyle="--", linewidth=0.8, alpha=0.8)

    for col, (seed_num, dist_exp, dist_arg) in enumerate(disto_list):
        ax0 = axes[0, col]
        im0 = ax0.imshow(dist_exp, cmap="jet", vmin=2, vmax=22, aspect="equal", origin="upper")
        _chain_lines(ax0)
        ax0.set_title(f"Seed {seed_num}", fontsize=9)
        ax0.set_xlabel("j", fontsize=7)
        ax0.set_ylabel("Expected (Å)" if col == 0 else "", fontsize=7)
        ax0.tick_params(labelsize=6)
        if col == n - 1:
            fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04, label="Å")

        ax1 = axes[1, col]
        im1 = ax1.imshow(dist_arg, cmap="jet", vmin=2, vmax=22, aspect="equal", origin="upper")
        _chain_lines(ax1)
        ax1.set_xlabel("j", fontsize=7)
        ax1.set_ylabel("Argmax (Å)" if col == 0 else "", fontsize=7)
        ax1.tick_params(labelsize=6)
        if col == n - 1:
            fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label="Å")

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Ramachandran
# ---------------------------------------------------------------------------

def plot_ramachandran(
    samples: list[tuple[int, np.ndarray, np.ndarray, list[str]]],
    title: str = "Ramachandran Plot",
) -> plt.Figure:
    """Grid of Ramachandran plots, one panel per sample within a seed.

    Args:
        samples: [(sample_num, phi_deg, psi_deg, chain_ids), ...] — all samples in one seed.
        Call once per seed and save as {label}_ramachandran_seed{N}.png.
    """
    n = len(samples)
    if n == 0:
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.set_title(title)
        return fig

    panel_size = 3.2
    fig, axes = plt.subplots(1, n, figsize=(n * panel_size, panel_size + 0.4), squeeze=False)

    unique_chains = list(dict.fromkeys(samples[0][3])) if samples[0][3] else []
    cmap = _chain_color_map(unique_chains)

    for col, (sample_num, phi, psi, chain_ids) in enumerate(samples):
        ax = axes[0, col]

        ax.add_patch(mpatches.Rectangle((-170, 90), 110, 80, alpha=0.08, color="#2ca02c"))
        ax.add_patch(mpatches.Rectangle((-170, -50), 110, 20, alpha=0.08, color="#d62728"))
        ax.add_patch(mpatches.Rectangle((60, 20), 20, 70, alpha=0.08, color="#1f77b4"))

        for chain in list(dict.fromkeys(chain_ids)):
            mask = np.array([c == chain for c in chain_ids])
            ax.scatter(phi[mask], psi[mask], s=5, alpha=0.5,
                       color=cmap.get(chain, "#333333"), edgecolors="none",
                       label=f"Chain {chain}" if col == 0 else "_")

        ax.axhline(0, color="gray", linewidth=0.4, alpha=0.4)
        ax.axvline(0, color="gray", linewidth=0.4, alpha=0.4)
        ax.set_xlim(-180, 180)
        ax.set_ylim(-180, 180)
        ax.set_xlabel("φ (°)", fontsize=7)
        ax.set_ylabel("ψ (°)", fontsize=7)
        ax.set_title(f"Sample {sample_num}", fontsize=9)
        ax.tick_params(labelsize=6)
        ax.set_xticks(range(-180, 181, 90))
        ax.set_yticks(range(-180, 181, 90))
        ax.grid(alpha=0.15, linestyle="--")

    if unique_chains:
        axes[0, 0].legend(fontsize=7, markerscale=2, loc="lower left")

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# MSA
# ---------------------------------------------------------------------------

def plot_msa(
    msa_data: dict,
    title: str = "Multiple Sequence Alignment",
    figsize: tuple[int, int] = (12, 6),
) -> plt.Figure:
    """MSA coverage heatmap matching CASP-Visualization-demo / bfd_uniclust_hits_coverage style.

    - Each row = one sequence, colored uniformly by its identity to the query sequence.
    - White cells = gaps (NaN).
    - Sequences sorted ascending by identity within each coverage-pattern cluster
      (low identity at bottom, high identity at top).
    - Black line = per-position coverage count (number of non-gap sequences).
    - Vertical black lines = chain boundaries (for multimers).
    """
    lines_raw = msa_data.get("lines", [])
    ln = msa_data.get("ln", [])
    nn = msa_data.get("nn", [])
    neff = msa_data.get("neff", 0.0)
    coverage = msa_data.get("coverage")

    if not lines_raw:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No MSA data available", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title(title)
        return fig

    mat = np.array(lines_raw, dtype=np.float32)
    n_seqs, seq_len = mat.shape

    fig, ax = plt.subplots(figsize=figsize)

    # Heatmap: origin="lower" puts row 0 (lowest identity) at the bottom
    im = ax.imshow(
        mat,
        cmap="rainbow",
        vmin=0, vmax=1,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Sequence identity to query", fontsize=9)

    # Coverage line: number of non-gap sequences per position
    # Y-axis shares scale with imshow (0 to n_seqs), so this overlays naturally
    if coverage is not None:
        cov_arr = np.array(coverage, dtype=float)
    else:
        cov_arr = np.sum(~np.isnan(mat), axis=0).astype(float)
    ax.plot(np.arange(seq_len), cov_arr, color="black", linewidth=0.8)

    # Chain boundary vertical lines (for complex)
    for pos in ln[1:-1]:
        ax.axvline(pos - 0.5, color="black", linewidth=1.5)

    # Cluster group horizontal lines (thin, sampled to avoid clutter)
    max_h_lines = 40
    nn_interior = nn[1:-1] if len(nn) > 2 else []
    step = max(1, len(nn_interior) // max_h_lines)
    for pos in nn_interior[::step]:
        ax.axhline(pos - 0.5, color="black", linewidth=0.3, alpha=0.4)

    ax.set_xlim(-0.5, seq_len - 0.5)
    ax.set_ylim(-0.5, n_seqs - 0.5)
    ax.set_xlabel("Positions", fontsize=10)
    ax.set_ylabel("Sequences", fontsize=10)
    ax.set_title(f"Sequence coverage (Neff: {neff})", fontsize=10)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Save utility
# ---------------------------------------------------------------------------

def save_figure(fig: plt.Figure, path: Path, dpi: int = 150) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
