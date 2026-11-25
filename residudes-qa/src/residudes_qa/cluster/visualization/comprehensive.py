"""
Comprehensive Clustering Visualization Module

Creates multi-panel figures combining:
- t-SNE / UMAP scatter plots with cluster ellipses
- Assessment score colormaps (TM-score, lDDT, RMSD, DockQ)
- Distance matrix heatmaps sorted by cluster
- Native/Best structure markers
- Cluster medoid annotations

Usage:
    from residudes_qa.cluster.visualization.comprehensive import create_comprehensive_figure

    fig = create_comprehensive_figure(
        distance_matrices={'tm': tm_dist, 'drmsd': drmsd_dist},
        labels=cluster_labels,
        scores_df=assessment_df,
        target_name="T1212"
    )
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from typing import Optional, Dict, List, Union
from scipy import linalg
import warnings
warnings.filterwarnings('ignore')

# Red-Yellow-Green colormap for scores
RYG_CMAP = LinearSegmentedColormap.from_list('RYG', ['#d73027', '#fee08b', '#1a9850'])


def create_comprehensive_figure(
    distance_matrices: Dict[str, np.ndarray],
    labels: np.ndarray,
    embedding: Optional[np.ndarray] = None,
    scores_df: Optional[pd.DataFrame] = None,
    structure_names: Optional[List[str]] = None,
    medoid_indices: Optional[List[int]] = None,
    native_idx: Optional[int] = None,
    best_idx: Optional[int] = None,
    gmm_means: Optional[np.ndarray] = None,
    gmm_covariances: Optional[np.ndarray] = None,
    target_name: str = "Target",
    output_path: Optional[Union[str, Path]] = None,
    methods: List[str] = ['tsne', 'umap'],
    score_columns: Optional[List[str]] = None,
    membership_confidence: Optional[np.ndarray] = None,
) -> plt.Figure:
    """
    Create comprehensive multi-panel visualization figure.

    Args:
        distance_matrices: Dict of {metric_name: distance_matrix}
            e.g., {'tm': 1-tm_matrix, 'drmsd': drmsd_matrix, 'itm': 1-itm_matrix}
        labels: Cluster labels
        embedding: Pre-computed 2D embedding (optional, will compute if None)
        scores_df: DataFrame with assessment scores
        structure_names: List of structure names
        medoid_indices: Indices of cluster medoids
        native_idx: Index of native/best structure
        gmm_means: GMM cluster centers for ellipse drawing
        gmm_covariances: GMM covariances for ellipse drawing
        target_name: Name of target for title
        output_path: Path to save figure
        methods: Embedding methods to use ['tsne', 'umap']
        score_columns: Score columns to visualize

    Returns:
        Matplotlib Figure object

    Example:
        >>> fig = create_comprehensive_figure(
        ...     distance_matrices={'1-TM': 1-tm, 'dRMSD': drmsd},
        ...     labels=result.labels,
        ...     scores_df=df,
        ...     target_name="T1212"
        ... )
    """
    from residudes_qa.cluster.visualization.embeddings import compute_embedding

    n_metrics = len(distance_matrices)
    n_methods = len(methods)

    # Auto-detect score columns
    if score_columns is None and scores_df is not None:
        candidates = ['tm_score', 'native_tm', 'lddt', 'native_lddt',
                     'rmsd', 'native_rmsd', 'dockq_ave', 'native_dockq', 'plddt']
        score_columns = [c for c in candidates if c in scores_df.columns][:4]

    n_scores = len(score_columns) if score_columns else 0

    # Figure layout: rows = metrics × methods, cols = [cluster + membership + scores]
    # No heatmap — replaced by membership probability panel
    n_cols = 1 + 1 + n_scores  # cluster plot + membership + score plots
    n_rows = n_metrics * n_methods

    figsize = (5 * n_cols, 4 * n_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, constrained_layout=True)

    if n_rows == 1:
        axes = axes.reshape(1, -1)

    # Get unique cluster colors — use tab20 for more clusters
    n_clusters = len(np.unique(labels))
    if n_clusters <= 10:
        cluster_colors = plt.cm.tab10(np.linspace(0, 1, 10))
    elif n_clusters <= 20:
        cluster_colors = plt.cm.tab20(np.linspace(0, 1, 20))
    else:
        cluster_colors = plt.cm.gist_ncar(np.linspace(0.05, 0.95, n_clusters))

    row = 0
    for metric_name, dist_matrix in distance_matrices.items():
        for method in methods:
            # Compute embedding
            emb = compute_embedding(dist_matrix, method=method)

            col = 0

            # 1. Cluster scatter plot with ellipses + prominent medoids
            ax = axes[row, col]
            _plot_cluster_scatter(
                ax, emb, labels, cluster_colors,
                gmm_means=gmm_means,
                gmm_covariances=gmm_covariances,
                medoid_indices=medoid_indices,
                native_idx=native_idx,
                best_idx=best_idx,
                title=f"{method.upper()} Clusters",
                metric_name=metric_name.upper(),
            )
            col += 1

            # 2. Membership confidence panel
            ax = axes[row, col]
            _plot_membership(
                ax, emb, labels, dist_matrix, cluster_colors,
                medoid_indices=medoid_indices,
                native_idx=native_idx,
                best_idx=best_idx,
                title="Membership Confidence",
                metric_name=metric_name.upper(),
                method_name=method.upper(),
                membership_confidence=membership_confidence,
            )
            col += 1

            # 3. Score-colored scatter plots
            if scores_df is not None and score_columns:
                score_arrays = _match_scores_to_structures(
                    scores_df, structure_names, score_columns
                )

                for score_name in score_columns:
                    ax = axes[row, col]
                    if score_name in score_arrays:
                        _plot_score_scatter(
                            ax, emb, score_arrays[score_name], score_name,
                            method_name=method.upper(),
                            native_idx=native_idx,
                            best_idx=best_idx,
                            medoid_indices=medoid_indices,
                            metric_name=metric_name.upper(),
                            labels=labels,
                        )
                    else:
                        ax.set_title(f"{method.upper()} - {score_name}\n(N/A)")
                        ax.axis('off')
                    col += 1

            row += 1

    # Overall title — let constrained_layout position it (y=1.01 + tight
    # bbox was clipping the second line on long target names).
    n_structures = len(labels)
    fig.suptitle(
        f'{target_name} - Comprehensive Clustering Analysis\n'
        f'{n_structures} structures | {n_clusters} clusters',
        fontsize=14, fontweight='bold',
    )

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight',
                    pad_inches=0.4, facecolor='white')

    return fig


def _plot_cluster_scatter(
    ax, embedding, labels, colors,
    gmm_means=None, gmm_covariances=None,
    medoid_indices=None, native_idx=None, best_idx=None,
    title="Clusters", metric_name='',
):
    """Plot cluster scatter with ellipses, prominent medoids and labels."""
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    for i, label in enumerate(unique_labels):
        mask = labels == label
        color = colors[i % len(colors)]

        # Hide native from scatter (plot separately)
        if native_idx is not None and native_idx < len(mask):
            plot_mask = mask.copy()
            plot_mask[native_idx] = False
        else:
            plot_mask = mask

        ax.scatter(
            embedding[plot_mask, 0], embedding[plot_mask, 1],
            c=[color], s=15, alpha=0.6, edgecolors='none',
            label=f'C{label} ({mask.sum()})' if n_clusters <= 20 else None,
        )

        # Draw ellipse
        if gmm_means is not None and gmm_covariances is not None:
            if label < len(gmm_means):
                _draw_ellipse(ax, gmm_means[label], gmm_covariances[label],
                             color, alpha=0.2, n_std=2.0)
        else:
            cluster_points = embedding[mask]
            if len(cluster_points) >= 3:
                mean = cluster_points.mean(axis=0)
                cov = np.cov(cluster_points.T)
                if cov.shape == (2, 2):
                    _draw_ellipse(ax, mean, cov, color, alpha=0.15, n_std=2.0)

    # Mark medoids — small stars with cluster label (smaller to reduce
    # overlap with surrounding scatter points; plot canvas itself is now
    # bigger so the stars don't need to be huge to be visible).
    if medoid_indices:
        for idx in medoid_indices:
            if idx < 0 or idx >= len(embedding):
                continue
            cid = labels[idx]
            c = colors[cid % len(colors)]
            ax.scatter(
                embedding[idx, 0], embedding[idx, 1],
                marker='*', s=90, c=[c], edgecolors='black',
                linewidths=0.8, zorder=95
            )
            # Only label medoids for ≤25 clusters to avoid clutter
            if n_clusters <= 25:
                ax.annotate(
                    f'C{cid}', xy=(embedding[idx, 0], embedding[idx, 1]),
                    xytext=(3, 3), textcoords='offset points',
                    fontsize=4.5, fontweight='bold', color='black',
                    bbox=dict(boxstyle='round,pad=0.1', facecolor='white',
                              edgecolor='gray', alpha=0.6, linewidth=0.5),
                    zorder=96,
                )
        ax.scatter([], [], marker='*', s=60, c='gray', edgecolors='black',
                   linewidths=0.6, label='Representative')

    # Mark native (red) or best-predicted (blue)
    if native_idx is not None and native_idx < len(embedding):
        ax.scatter(
            embedding[native_idx, 0], embedding[native_idx, 1],
            marker='o', s=300, facecolors='none', edgecolors='red',
            linewidths=2.5, label='Native/Best', zorder=100
        )
    elif best_idx is not None and best_idx < len(embedding):
        ax.scatter(
            embedding[best_idx, 0], embedding[best_idx, 1],
            marker='o', s=300, facecolors='none', edgecolors='blue',
            linewidths=2.5, label='Best Predicted', zorder=100
        )

    ax.set_title(title, fontsize=10)
    # Axis labels with metric info
    ax.set_xlabel(f'Dim 1 (from {metric_name} distance)' if metric_name else 'Dim 1',
                  fontsize=8)
    ax.set_ylabel(f'Dim 2 (from {metric_name} distance)' if metric_name else 'Dim 2',
                  fontsize=8)
    ax.legend(loc='best', fontsize=6, ncol=2 if n_clusters <= 20 else 1,
              framealpha=0.8, markerscale=0.8)
    ax.grid(alpha=0.2)


def _plot_score_scatter(
    ax, embedding, scores, score_name,
    method_name='', native_idx=None, best_idx=None, medoid_indices=None,
    metric_name='', labels=None,
):
    """Plot scatter colored by assessment score."""
    valid_mask = ~np.isnan(scores)

    if native_idx is not None and native_idx < len(scores):
        valid_mask[native_idx] = False

    if not valid_mask.any():
        ax.set_title(f"{method_name} - {score_name}\n(No valid scores)")
        return

    min_score = np.nanmin(scores)
    max_score = np.nanmax(scores)

    # RMSD: invert colormap (lower is better = green)
    if 'rmsd' in score_name.lower():
        vmin, vmax = 0, 20
        cmap = RYG_CMAP.reversed()
    else:
        vmin, vmax = 0.0, 1.0
        cmap = RYG_CMAP

    scatter = ax.scatter(
        embedding[valid_mask, 0], embedding[valid_mask, 1],
        c=scores[valid_mask], cmap=cmap,
        s=15, alpha=0.7, edgecolors='none',
        vmin=vmin, vmax=vmax
    )

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.7)
    cbar.set_label(score_name, fontsize=8)
    if 'rmsd' not in score_name.lower():
        cbar.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    # Mark native (red) or best-predicted (blue)
    if native_idx is not None and native_idx < len(embedding):
        ax.scatter(
            embedding[native_idx, 0], embedding[native_idx, 1],
            marker='o', s=200, facecolors='none', edgecolors='red',
            linewidths=2, zorder=100
        )
    elif best_idx is not None and best_idx < len(embedding):
        ax.scatter(
            embedding[best_idx, 0], embedding[best_idx, 1],
            marker='o', s=200, facecolors='none', edgecolors='blue',
            linewidths=2, zorder=100
        )

    # Mark medoids
    if medoid_indices:
        for idx in medoid_indices:
            if idx >= 0 and idx < len(embedding):
                ax.scatter(
                    embedding[idx, 0], embedding[idx, 1],
                    marker='*', s=80, c='black', edgecolors='white',
                    linewidths=0.5, zorder=90
                )

    ax.set_title(f"{score_name}  (range: {min_score:.2f}–{max_score:.2f})",
                fontsize=9)
    ax.set_xlabel(f'{method_name} Dim 1 ({metric_name})' if metric_name else f'{method_name} Dim 1',
                  fontsize=7)
    ax.set_ylabel(f'{method_name} Dim 2 ({metric_name})' if metric_name else f'{method_name} Dim 2',
                  fontsize=7)
    ax.tick_params(axis='both', labelsize=7)


def _plot_membership(
    ax, embedding, labels, dist_matrix, colors,
    medoid_indices=None, native_idx=None, best_idx=None,
    title="Membership Confidence", metric_name='', method_name='',
    membership_confidence=None,
):
    """
    Plot membership confidence: each point colored by its GMM posterior probability
    (max_membership) or distance-based confidence to its medoid.
    Green = strong member, red = borderline.
    """
    n = len(labels)
    unique_labels = np.unique(labels)

    # Use pre-computed GMM membership if available
    confidence = np.full(n, 0.5)  # default mid-confidence
    if membership_confidence is not None and len(membership_confidence) == n:
        confidence = np.clip(membership_confidence, 0, 1)
    elif medoid_indices is not None and len(medoid_indices) > 0:
        for cid in unique_labels:
            mask = labels == cid
            if cid < len(medoid_indices) and medoid_indices[cid] >= 0:
                mid = medoid_indices[cid]
                dists = dist_matrix[mask, mid]
                d_max = dists.max() if dists.max() > 0 else 1.0
                confidence[mask] = 1.0 - (dists / d_max)  # 1=at medoid, 0=farthest
            else:
                # Fallback: use distance to cluster centroid in embedding space
                cluster_pts = embedding[mask]
                centroid = cluster_pts.mean(axis=0)
                dists = np.linalg.norm(cluster_pts - centroid, axis=1)
                d_max = dists.max() if dists.max() > 0 else 1.0
                confidence[mask] = 1.0 - (dists / d_max)

    # Map confidence to size and alpha
    sizes = 3 + 30 * confidence ** 2  # emphasize high-confidence points
    0.15 + 0.7 * confidence  # range 0.15-0.85

    # Plot points colored by confidence (RdYlGn)
    sc = ax.scatter(
        embedding[:, 0], embedding[:, 1],
        c=confidence, cmap='RdYlGn', s=sizes, alpha=0.7,
        edgecolors='none', vmin=0, vmax=1,
    )
    cbar = plt.colorbar(sc, ax=ax, shrink=0.7)
    cbar.set_label('Membership confidence', fontsize=7)
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])

    # Mark medoids (small marker, plot canvas is large)
    if medoid_indices:
        for idx in medoid_indices:
            if idx < 0 or idx >= len(embedding):
                continue
            cid = labels[idx]
            c = colors[cid % len(colors)]
            ax.scatter(
                embedding[idx, 0], embedding[idx, 1],
                marker='*', s=70, c=[c], edgecolors='black',
                linewidths=0.7, zorder=95,
            )

    # Mark native (red) or best-predicted (blue)
    if native_idx is not None and native_idx < len(embedding):
        ax.scatter(
            embedding[native_idx, 0], embedding[native_idx, 1],
            marker='o', s=250, facecolors='none', edgecolors='red',
            linewidths=2, zorder=100,
        )
    elif best_idx is not None and best_idx < len(embedding):
        ax.scatter(
            embedding[best_idx, 0], embedding[best_idx, 1],
            marker='o', s=250, facecolors='none', edgecolors='blue',
            linewidths=2, zorder=100,
        )

    ax.set_title(title, fontsize=10)
    ax.set_xlabel(f'{method_name} Dim 1 ({metric_name})' if metric_name else 'Dim 1',
                  fontsize=8)
    ax.set_ylabel(f'{method_name} Dim 2 ({metric_name})' if metric_name else 'Dim 2',
                  fontsize=8)
    ax.grid(alpha=0.2)

    # Add text: strong/weak member counts
    strong = (confidence >= 0.7).sum()
    weak = (confidence < 0.3).sum()
    ax.text(0.02, 0.98, f'Strong: {strong} | Weak: {weak}',
            transform=ax.transAxes, fontsize=6.5, va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))


def _plot_heatmap(ax, distance_matrix, labels, title="Distance Matrix"):
    """Plot distance matrix heatmap sorted by cluster."""
    from matplotlib.patches import Rectangle

    # Sort by cluster
    sort_idx = np.argsort(labels)
    sorted_matrix = distance_matrix[sort_idx][:, sort_idx]
    sorted_labels = labels[sort_idx]

    im = ax.imshow(sorted_matrix, cmap='viridis', aspect='auto')
    plt.colorbar(im, ax=ax, shrink=0.8)

    # Draw cluster borders
    unique_labels = np.unique(sorted_labels)
    n_clusters = len(unique_labels)
    colors = plt.cm.tab10(np.linspace(0, 1, min(n_clusters, 10)))

    for i, label in enumerate(unique_labels):
        idx = np.where(sorted_labels == label)[0]
        if len(idx) == 0:
            continue

        start = idx[0]
        end = idx[-1] + 1
        size = end - start

        rect = Rectangle(
            (start - 0.5, start - 0.5), size, size,
            linewidth=2, edgecolor=colors[i % len(colors)],
            facecolor='none', linestyle='-', alpha=0.9
        )
        ax.add_patch(rect)

    # White separator lines
    boundaries = []
    for label in unique_labels:
        idx = np.where(sorted_labels == label)[0]
        if len(idx) > 0:
            boundaries.append(idx[0])

    for b in boundaries[1:]:
        ax.axhline(y=b-0.5, color='white', linewidth=1, alpha=0.8)
        ax.axvline(x=b-0.5, color='white', linewidth=1, alpha=0.8)

    ax.set_title(title, fontsize=10)
    ax.set_xlabel('Index', fontsize=9)
    ax.set_ylabel('Index', fontsize=9)


def _draw_ellipse(ax, mean, cov, color, alpha=0.3, n_std=2.0):
    """Draw confidence ellipse."""
    if cov.shape != (2, 2):
        return

    try:
        eigenvalues, eigenvectors = linalg.eigh(cov)
        order = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

        angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
        width = 2 * n_std * np.sqrt(eigenvalues[0])
        height = 2 * n_std * np.sqrt(eigenvalues[1])

        ellipse = Ellipse(
            xy=mean, width=width, height=height, angle=angle,
            facecolor=color, alpha=alpha, edgecolor=color, linewidth=2
        )
        ax.add_patch(ellipse)
    except Exception:
        pass


def _match_scores_to_structures(scores_df, structure_names, score_columns):
    """Match assessment scores to structure indices."""
    score_arrays = {}

    if structure_names is None or scores_df is None:
        return score_arrays

    # Build mapping
    if 'file' in scores_df.columns:
        scores_df = scores_df.copy()
        scores_df['model_key'] = scores_df['file'].str.replace('.pdb', '', regex=False)
    elif 'model_name' in scores_df.columns:
        scores_df = scores_df.copy()
        scores_df['model_key'] = scores_df['model_name'].str.replace('.pdb', '', regex=False)
    elif 'structure' in scores_df.columns:
        scores_df = scores_df.copy()
        scores_df['model_key'] = scores_df['structure'].str.replace('.pdb', '', regex=False)
    else:
        return score_arrays

    for col in score_columns:
        if col in scores_df.columns:
            score_map = dict(zip(scores_df['model_key'], scores_df[col]))
            score_arrays[col] = np.array([
                score_map.get(s.replace('.pdb', ''), np.nan)
                for s in structure_names
            ])

    return score_arrays


def plot_k_selection_diagnostic(
    ax, k_scores: Dict[str, List],
    selected_k: int,
    metric_name: str = '',
    method_name: str = '',
):
    """
    Plot K selection diagnostic showing why a particular K was chosen.

    Displays normalized individual metrics (Silhouette, Penalized Sil, Delta Sil, BIC)
    on left axis and combined score on right axis.  Includes boundary markers,
    ascending-at-boundary warnings, and plateau shading when available.

    Args:
        ax: Matplotlib axis
        k_scores: Dict from select_optimal_k() with k_values, silhouette_scores, etc.
        selected_k: The chosen K value
        metric_name: Distance metric name (e.g., 'TM')
        method_name: Embedding method name (e.g., 't-SNE')
    """
    if not k_scores or 'k_values' not in k_scores:
        ax.text(0.5, 0.5, f'K Selection\n({metric_name} {method_name})\nNo data',
                ha='center', va='center', fontsize=10, transform=ax.transAxes)
        ax.set_axis_off()
        return

    ks = k_scores['k_values']
    sils = k_scores['silhouette_scores']
    pen_scores = k_scores.get('pen_scores', [])
    delta_sils = k_scores.get('delta_sil_scores', [])
    bic_scores = k_scores.get('bic_scores', [])
    balance_scores = k_scores.get('balance_scores', [])
    combined = k_scores['combined_scores']

    def _to_01(arr, reverse=False):
        a = np.array(arr, dtype=float)
        if a.max() == a.min():
            return np.ones_like(a) * 0.5
        n = (a - a.min()) / (a.max() - a.min())
        return 1.0 - n if reverse else n

    ax2 = ax.twinx()

    # Individual metrics (left axis, normalized)
    ax.plot(ks, _to_01(sils), 'o-', color='#2196F3', lw=1.5, ms=4,
            label='Silhouette (20%)', alpha=0.8)
    if pen_scores:
        ax.plot(ks, _to_01(pen_scores), 's-', color='#FF9800', lw=1.5, ms=4,
                label='Penalized Sil (30%)', alpha=0.8)
    if delta_sils:
        ax.plot(ks, _to_01(delta_sils), 'D-', color='#9C27B0', lw=1.5, ms=4,
                label='Delta Sil (10%)', alpha=0.8)
    if bic_scores:
        ax.plot(ks, _to_01(bic_scores, reverse=True), '^-', color='#4CAF50', lw=1.5,
                ms=4, label='BIC (25%)', alpha=0.8)
    if balance_scores:
        ax.plot(ks, _to_01(balance_scores), 'v-', color='#E91E63', lw=1.5, ms=4,
                label='Balance (15%)', alpha=0.8)

    # Combined (right axis, thick black)
    ax2.plot(ks, combined, 'k-', linewidth=3, label='Combined', alpha=0.9)
    ax2.fill_between(ks, combined, alpha=0.08, color='black')

    # Mark selected K
    sel_idx = next((i for i, k in enumerate(ks) if k == selected_k), None)
    if sel_idx is not None:
        ax.axvline(x=selected_k, color='red', linestyle='--', linewidth=2, alpha=0.6)
        ax2.scatter([selected_k], [combined[sel_idx]], c='red', s=200, zorder=10,
                    edgecolors='white', linewidths=2, marker='*')

        raw_sil = sils[sel_idx]
        ax.annotate(
            f'K={selected_k}\nSil={raw_sil:.3f}',
            xy=(selected_k, _to_01(sils)[sel_idx]),
            xytext=(8, -15), textcoords='offset points',
            fontsize=6.5, color='red', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='red', alpha=0.85)
        )

    # k_max boundary marker
    k_max_eff = k_scores.get('k_max_effective', ks[-1] if ks else None)
    k_max_req = k_scores.get('k_max_requested', k_max_eff)
    if k_max_eff is not None and k_max_eff == ks[-1]:
        ax.axvline(x=k_max_eff, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
        cap_text = f'k_max={k_max_eff}'
        if k_max_req is not None and k_max_req != k_max_eff:
            cap_text += f'\n(requested {k_max_req})'
        ax.annotate(
            cap_text, xy=(k_max_eff, 1.0), xycoords=('data', 'axes fraction'),
            xytext=(-5, -5), textcoords='offset points',
            fontsize=5.5, color='gray', ha='right', va='top',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='#f5f5f5',
                      edgecolor='gray', alpha=0.8)
        )

    # Plateau region shading
    plateau_k = k_scores.get('plateau_k')
    if plateau_k is not None and k_max_eff is not None:
        ax.axvspan(plateau_k, k_max_eff, alpha=0.06, color='gray',
                   label='Plateau region')

    # Ascending-at-boundary warning
    if k_scores.get('is_ascending_at_boundary', False):
        ax.annotate(
            'Sil still ascending at k_max!\nConsider increasing --k-max',
            xy=(ks[-1], _to_01(sils)[-1]),
            xytext=(-90, 25), textcoords='offset points',
            fontsize=6, color='darkorange', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#fff3e0',
                      edgecolor='orange', alpha=0.9),
            arrowprops=dict(arrowstyle='->', color='orange', lw=1.5)
        )

    # Warning when optimal_k == k_max
    if selected_k == ks[-1] and not k_scores.get('is_ascending_at_boundary', False):
        ax.annotate(
            'optimal_k = k_max\n(may need wider range)',
            xy=(selected_k, 0.0), xycoords=('data', 'axes fraction'),
            xytext=(-60, 15), textcoords='offset points',
            fontsize=5.5, color='#d32f2f',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffebee',
                      edgecolor='#d32f2f', alpha=0.85)
        )

    # BIC optimal k marker
    bic_optimal_k = k_scores.get('bic_optimal_k')
    if bic_optimal_k is not None and bic_optimal_k != selected_k:
        bic_idx = next((i for i, k in enumerate(ks) if k == bic_optimal_k), None)
        if bic_idx is not None:
            ax.axvline(x=bic_optimal_k, color='#4CAF50', linestyle='--',
                       linewidth=1, alpha=0.4)
            ax.annotate(
                f'BIC opt={bic_optimal_k}', xy=(bic_optimal_k, 0.02),
                xycoords=('data', 'axes fraction'),
                fontsize=5, color='#4CAF50', ha='center',
            )

    ax.set_xlabel('K (clusters)', fontsize=8)
    ax.set_ylabel('Normalized Score (0-1)', fontsize=8)
    ax2.set_ylabel('Combined Score', fontsize=8)
    ax.set_title(f"K Selection ({metric_name} {method_name})\n"
                 f"Selected K={selected_k}", fontsize=9, fontweight='bold')
    ax.set_xticks(ks)
    ax.tick_params(axis='both', labelsize=7)
    ax2.tick_params(axis='both', labelsize=7)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2,
              loc='lower left', fontsize=5.5, framealpha=0.85, ncol=1)
    ax.grid(alpha=0.2)


def create_k_diagnostic_figure(
    k_scores: Dict[str, List],
    target_name: str = '',
    metric_name: str = '',
) -> Optional[plt.Figure]:
    """
    Standalone wrapper: create a figure with the K selection diagnostic plot.

    Args:
        k_scores: Dict from select_optimal_k() with k_values, silhouette_scores, etc.
        target_name: Target label for title
        metric_name: Distance metric name (e.g., 'TM')

    Returns:
        matplotlib Figure or None
    """
    if not k_scores or 'k_values' not in k_scores:
        return None

    selected_k = k_scores.get('optimal_k', k_scores['k_values'][0])
    fig, ax = plt.subplots(figsize=(7, 4))
    plot_k_selection_diagnostic(ax, k_scores, selected_k,
                                metric_name=metric_name, method_name=target_name)
    fig.tight_layout()
    return fig


def create_simple_comprehensive(
    distance_matrix: np.ndarray,
    labels: np.ndarray,
    embedding: np.ndarray,
    scores: Optional[np.ndarray] = None,
    score_name: str = "Score",
    medoid_indices: Optional[List[int]] = None,
    native_idx: Optional[int] = None,
    gmm_means: Optional[np.ndarray] = None,
    gmm_covariances: Optional[np.ndarray] = None,
    title: str = "Clustering Analysis",
    output_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Create a simple 3-panel comprehensive figure.

    Panels: [Cluster scatter] [Score scatter] [Distance heatmap]

    Args:
        distance_matrix: Distance matrix
        labels: Cluster labels
        embedding: 2D embedding
        scores: Score values for coloring
        score_name: Name of score
        medoid_indices: Medoid indices
        native_idx: Native structure index
        gmm_means: GMM means for ellipses
        gmm_covariances: GMM covariances
        title: Figure title
        output_path: Output path

    Returns:
        Matplotlib Figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    n_clusters = len(np.unique(labels))
    if n_clusters <= 10:
        colors = plt.cm.tab10(np.linspace(0, 1, 10))
    elif n_clusters <= 20:
        colors = plt.cm.tab20(np.linspace(0, 1, 20))
    else:
        colors = plt.cm.gist_ncar(np.linspace(0.05, 0.95, n_clusters))

    # 1. Cluster scatter
    _plot_cluster_scatter(
        axes[0], embedding, labels, colors,
        gmm_means=gmm_means, gmm_covariances=gmm_covariances,
        medoid_indices=medoid_indices, native_idx=native_idx,
        title="Clusters"
    )

    # 2. Score scatter
    if scores is not None:
        _plot_score_scatter(
            axes[1], embedding, scores, score_name,
            native_idx=native_idx, medoid_indices=medoid_indices,
            labels=labels,
        )
    else:
        axes[1].set_title("No scores available")
        axes[1].axis('off')

    # 3. Membership confidence
    _plot_membership(
        axes[2], embedding, labels, distance_matrix, colors,
        medoid_indices=medoid_indices, native_idx=native_idx,
        title="Membership Confidence"
    )

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')

    return fig


def create_comprehensive_report(
    distance_matrix: np.ndarray,
    labels: np.ndarray,
    structure_names: Optional[List[str]] = None,
    assessment_df: Optional[pd.DataFrame] = None,
    target_name: str = "Target",
    output_dir: Optional[str] = None,
    embedding: Optional[np.ndarray] = None,
    score_columns: Optional[List[str]] = None,
    medoid_indices: Optional[List[int]] = None,
    membership_confidence: Optional[np.ndarray] = None,
    native_idx: Optional[int] = None,
    best_idx: Optional[int] = None,
) -> plt.Figure:
    """
    CLI-friendly wrapper for create_comprehensive_figure.

    Accepts a single distance matrix and generates the comprehensive
    visualization, saving to output_dir/{target_name}_comprehensive.png.
    """
    output_path = None
    if output_dir:
        output_path = Path(output_dir) / f"{target_name}_comprehensive.png"

    return create_comprehensive_figure(
        distance_matrices={'tm': distance_matrix},
        labels=labels,
        embedding=embedding,
        scores_df=assessment_df,
        structure_names=structure_names,
        medoid_indices=medoid_indices,
        native_idx=native_idx,
        best_idx=best_idx,
        membership_confidence=membership_confidence,
        target_name=target_name,
        output_path=output_path,
        score_columns=score_columns,
    )
