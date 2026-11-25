"""
Distance Matrix Heatmap Visualization

Creates heatmaps of distance matrices sorted by cluster assignment,
showing the block structure of clustered data.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap
from typing import Optional, Union, Tuple
from pathlib import Path


def plot_distance_heatmap(
    distance_matrix: np.ndarray,
    labels: np.ndarray,
    title: str = "Distance Matrix",
    output_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = 'viridis',
    show_cluster_borders: bool = True,
    annotate_clusters: bool = True,
) -> plt.Figure:
    """
    Plot distance matrix heatmap sorted by cluster labels.

    Structures are reordered so that members of the same cluster are adjacent,
    revealing the block diagonal structure of well-clustered data.

    Args:
        distance_matrix: Pairwise distance matrix (N x N)
        labels: Cluster labels (N,)
        title: Plot title
        output_path: Path to save figure (optional)
        figsize: Figure size
        cmap: Colormap ('viridis', 'viridis_r', 'RdYlGn_r', etc.)
        show_cluster_borders: Whether to draw rectangles around cluster blocks
        annotate_clusters: Whether to add cluster labels on axes

    Returns:
        Matplotlib Figure object

    Example:
        >>> fig = plot_distance_heatmap(
        ...     dist_matrix, labels,
        ...     title="TM-score Distance Matrix",
        ...     show_cluster_borders=True
        ... )
    """
    # Sort by cluster labels
    sort_idx = np.argsort(labels)
    sorted_matrix = distance_matrix[sort_idx][:, sort_idx]
    sorted_labels = labels[sort_idx]

    fig, ax = plt.subplots(figsize=figsize)

    # Plot heatmap
    im = ax.imshow(sorted_matrix, cmap=cmap, aspect='auto')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Distance', fontsize=11)

    # Draw cluster block borders
    if show_cluster_borders:
        _draw_cluster_borders(ax, sorted_labels)

    # Add cluster annotations
    if annotate_clusters:
        _annotate_cluster_regions(ax, sorted_labels)

    ax.set_xlabel('Structure Index (sorted by cluster)', fontsize=11)
    ax.set_ylabel('Structure Index (sorted by cluster)', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')

    return fig


def _draw_cluster_borders(ax, sorted_labels: np.ndarray):
    """Draw colored rectangles around cluster diagonal blocks."""
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

        # Draw rectangle around diagonal block
        rect = Rectangle(
            (start - 0.5, start - 0.5), size, size,
            linewidth=2.5,
            edgecolor=colors[i % len(colors)],
            facecolor='none',
            linestyle='-',
            alpha=0.9
        )
        ax.add_patch(rect)

    # Add white separator lines between clusters
    boundaries = []
    for label in unique_labels:
        idx = np.where(sorted_labels == label)[0]
        if len(idx) > 0:
            boundaries.append(idx[0])

    for b in boundaries[1:]:
        ax.axhline(y=b - 0.5, color='white', linewidth=1.5, alpha=0.8)
        ax.axvline(x=b - 0.5, color='white', linewidth=1.5, alpha=0.8)


def _annotate_cluster_regions(ax, sorted_labels: np.ndarray):
    """Add cluster labels along the diagonal."""
    unique_labels = np.unique(sorted_labels)

    for label in unique_labels:
        idx = np.where(sorted_labels == label)[0]
        if len(idx) == 0:
            continue

        center = (idx[0] + idx[-1]) / 2

        # Add small annotation inside the block
        ax.annotate(
            f'C{label}',
            xy=(center, center),
            fontsize=8,
            color='white',
            ha='center',
            va='center',
            fontweight='bold',
            alpha=0.7,
        )


def plot_multi_metric_heatmaps(
    distance_matrices: dict,
    labels: np.ndarray,
    output_path: Optional[Union[str, Path]] = None,
    figsize: Optional[Tuple[int, int]] = None,
) -> plt.Figure:
    """
    Plot multiple distance matrix heatmaps side by side.

    Args:
        distance_matrices: Dict of {metric_name: distance_matrix}
            e.g., {'TM': tm_dist, 'dRMSD': drmsd_dist, 'iTM': itm_dist}
        labels: Cluster labels
        output_path: Path to save figure
        figsize: Figure size (auto-calculated if None)

    Returns:
        Matplotlib Figure object

    Example:
        >>> matrices = {'1-TM': 1-tm_matrix, 'dRMSD': drmsd_matrix}
        >>> fig = plot_multi_metric_heatmaps(matrices, labels)
    """
    n_metrics = len(distance_matrices)

    if figsize is None:
        figsize = (5 * n_metrics, 4)

    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)

    if n_metrics == 1:
        axes = [axes]

    # Sort indices once
    sort_idx = np.argsort(labels)
    sorted_labels = labels[sort_idx]

    for ax, (metric_name, dist_matrix) in zip(axes, distance_matrices.items()):
        sorted_matrix = dist_matrix[sort_idx][:, sort_idx]

        # Choose colormap based on metric
        if 'rmsd' in metric_name.lower():
            cmap = 'viridis'  # Lower is better for RMSD
        else:
            cmap = 'viridis_r'  # Lower distance = more similar

        im = ax.imshow(sorted_matrix, cmap=cmap, aspect='auto')
        plt.colorbar(im, ax=ax, shrink=0.8)

        _draw_cluster_borders(ax, sorted_labels)

        ax.set_title(f'{metric_name}', fontsize=11, fontweight='bold')
        ax.set_xlabel('Index')
        ax.set_ylabel('Index')

    plt.suptitle('Distance Matrices by Metric (sorted by cluster)',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')

    return fig


def plot_similarity_heatmap(
    similarity_matrix: np.ndarray,
    labels: np.ndarray,
    title: str = "Similarity Matrix",
    output_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (10, 8),
    vmin: float = 0.0,
    vmax: float = 1.0,
) -> plt.Figure:
    """
    Plot similarity matrix (e.g., TM-score) as heatmap.

    Uses Red-Yellow-Green colormap where green = high similarity.

    Args:
        similarity_matrix: Pairwise similarity matrix (N x N), values in [0, 1]
        labels: Cluster labels
        title: Plot title
        output_path: Path to save figure
        figsize: Figure size
        vmin: Minimum value for colormap
        vmax: Maximum value for colormap

    Returns:
        Matplotlib Figure object
    """
    # Red-Yellow-Green colormap
    RYG_CMAP = LinearSegmentedColormap.from_list(
        'RYG', ['#d73027', '#fee08b', '#1a9850']
    )

    # Sort by cluster
    sort_idx = np.argsort(labels)
    sorted_matrix = similarity_matrix[sort_idx][:, sort_idx]
    sorted_labels = labels[sort_idx]

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(sorted_matrix, cmap=RYG_CMAP, aspect='auto',
                   vmin=vmin, vmax=vmax)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Similarity', fontsize=11)

    _draw_cluster_borders(ax, sorted_labels)

    ax.set_xlabel('Structure Index', fontsize=11)
    ax.set_ylabel('Structure Index', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')

    return fig


def plot_inter_cluster_heatmap(
    distance_matrix: np.ndarray,
    labels: np.ndarray,
    title: str = "Inter-Cluster Distance",
    output_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (8, 6),
) -> plt.Figure:
    """
    Plot heatmap of mean distances between clusters.

    Shows the relationship between clusters at a glance.

    Args:
        distance_matrix: Pairwise distance matrix (N x N)
        labels: Cluster labels
        title: Plot title
        output_path: Path to save figure
        figsize: Figure size

    Returns:
        Matplotlib Figure object
    """
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    # Compute inter-cluster distance matrix
    inter_dist = np.zeros((n_clusters, n_clusters))

    for i, label_i in enumerate(unique_labels):
        for j, label_j in enumerate(unique_labels):
            mask_i = labels == label_i
            mask_j = labels == label_j

            if i == j:
                # Intra-cluster distance
                sub_dist = distance_matrix[mask_i][:, mask_i]
                # Mean of upper triangle (excluding diagonal)
                triu = np.triu_indices_from(sub_dist, k=1)
                if len(triu[0]) > 0:
                    inter_dist[i, j] = sub_dist[triu].mean()
                else:
                    inter_dist[i, j] = 0
            else:
                # Inter-cluster distance
                inter_dist[i, j] = distance_matrix[mask_i][:, mask_j].mean()

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(inter_dist, cmap='viridis', aspect='auto')

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Mean Distance', fontsize=10)

    # Add value annotations
    for i in range(n_clusters):
        for j in range(n_clusters):
            color = 'white' if inter_dist[i, j] > inter_dist.mean() else 'black'
            ax.text(j, i, f'{inter_dist[i, j]:.3f}',
                    ha='center', va='center', fontsize=9, color=color)

    ax.set_xticks(range(n_clusters))
    ax.set_yticks(range(n_clusters))
    ax.set_xticklabels([f'C{lab}' for lab in unique_labels])
    ax.set_yticklabels([f'C{lab}' for lab in unique_labels])

    ax.set_xlabel('Cluster', fontsize=11)
    ax.set_ylabel('Cluster', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')

    return fig
