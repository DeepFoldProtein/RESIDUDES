"""
Cluster visualization module

Various plotting functions for cluster analysis results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from typing import Optional, Union, List
from pathlib import Path


def plot_clusters(
    embedding: np.ndarray,
    labels: np.ndarray,
    medoid_indices: Optional[List[int]] = None,
    best_idx: Optional[int] = None,
    best_label: str = "Best",
    title: str = "Cluster Visualization",
    output_path: Optional[Union[str, Path]] = None,
    figsize: tuple = (12, 10),
) -> plt.Figure:
    """
    Plot cluster visualization with medoids and best structure markers.

    Args:
        embedding: 2D coordinates (N x 2)
        labels: Cluster labels (N,)
        medoid_indices: Indices of cluster medoids
        best_idx: Index of best structure (e.g., highest DockQ)
        best_label: Label for best structure marker
        title: Plot title
        output_path: Path to save figure
        figsize: Figure size

    Returns:
        Matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    n_clusters = len(np.unique(labels))
    colors = plt.cm.tab20(np.linspace(0, 1, max(n_clusters, 20)))

    # Plot each cluster
    for cluster_id in range(n_clusters):
        mask = labels == cluster_id
        ax.scatter(
            embedding[mask, 0], embedding[mask, 1],
            c=[colors[cluster_id]], alpha=0.4, s=15,
            label=f'C{cluster_id} (n={mask.sum()})'
        )

    # Mark medoids with stars
    if medoid_indices:
        for i, medoid_idx in enumerate(medoid_indices):
            if medoid_idx >= 0 and medoid_idx < len(embedding):
                ax.scatter(
                    embedding[medoid_idx, 0], embedding[medoid_idx, 1],
                    marker='*', s=400, c=[colors[i % len(colors)]],
                    edgecolors='black', linewidths=2, zorder=10
                )

    # Mark best structure with diamond
    if best_idx is not None and best_idx < len(embedding):
        ax.scatter(
            embedding[best_idx, 0], embedding[best_idx, 1],
            marker='D', s=250, c='red',
            edgecolors='white', linewidths=2, zorder=11,
            label=best_label
        )

    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.set_title(f'{title}\n☆=Medoid, ◆={best_label}')
    ax.legend(loc='upper right', fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')

    return fig


def plot_cluster_distribution(
    df: pd.DataFrame,
    cluster_column: str = "cluster",
    value_column: str = "native_dockq",
    title: Optional[str] = None,
    output_path: Optional[Union[str, Path]] = None,
    figsize: tuple = (12, 6),
) -> plt.Figure:
    """
    Plot value distribution per cluster (boxplot).

    Args:
        df: DataFrame with clustering data
        cluster_column: Column containing cluster labels
        value_column: Column containing values to plot
        title: Plot title
        output_path: Path to save figure
        figsize: Figure size

    Returns:
        Matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    n_clusters = df[cluster_column].nunique()
    colors = plt.cm.tab20(np.linspace(0, 1, max(n_clusters, 20)))

    # Prepare data for boxplot
    cluster_data = []
    cluster_labels = []

    for cluster_id in sorted(df[cluster_column].unique()):
        values = df[df[cluster_column] == cluster_id][value_column].dropna()
        if len(values) > 0:
            cluster_data.append(values.values)
            cluster_labels.append(f'C{cluster_id}\n(n={len(values)})')

    if cluster_data:
        bp = ax.boxplot(cluster_data, labels=cluster_labels, patch_artist=True)

        for i, box in enumerate(bp['boxes']):
            box.set_facecolor(colors[i % len(colors)])
            box.set_alpha(0.6)

        # Add reference lines for DockQ
        if 'dockq' in value_column.lower():
            ax.axhline(y=0.23, color='orange', linestyle='--', alpha=0.7, label='Acceptable (0.23)')
            ax.axhline(y=0.49, color='green', linestyle='--', alpha=0.7, label='Medium (0.49)')
            ax.axhline(y=0.80, color='blue', linestyle='--', alpha=0.7, label='High (0.80)')
            ax.legend(loc='upper right')

        # Add reference line for TM-score
        elif 'tm' in value_column.lower():
            ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='TM=0.5 (fold similarity)')
            ax.legend(loc='upper right')

    ax.set_ylabel(value_column)
    ax.set_xlabel('Cluster')
    ax.set_title(title or f'{value_column} Distribution by Cluster')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')

    return fig


def plot_cluster_sizes(
    labels: np.ndarray,
    title: str = "Cluster Sizes",
    output_path: Optional[Union[str, Path]] = None,
    figsize: tuple = (10, 6),
) -> plt.Figure:
    """
    Plot cluster size distribution.

    Args:
        labels: Cluster labels
        title: Plot title
        output_path: Path to save figure
        figsize: Figure size

    Returns:
        Matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    unique_labels, counts = np.unique(labels, return_counts=True)
    colors = plt.cm.tab20(np.linspace(0, 1, max(len(unique_labels), 20)))

    bars = ax.bar(range(len(unique_labels)), counts, color=colors)

    ax.set_xticks(range(len(unique_labels)))
    ax.set_xticklabels([f'C{lab}' for lab in unique_labels])
    ax.set_ylabel('Size')
    ax.set_xlabel('Cluster')
    ax.set_title(title)

    # Add value labels
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
            str(count), ha='center', va='bottom', fontsize=9
        )

    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')

    return fig


def plot_clusters_with_ellipse(
    embedding: np.ndarray,
    labels: np.ndarray,
    medoid_indices: Optional[List[int]] = None,
    title: str = "Cluster Visualization",
    output_path: Optional[Union[str, Path]] = None,
    figsize: tuple = (12, 10),
    n_std: float = 2.0,
) -> plt.Figure:
    """
    Plot cluster visualization with confidence ellipses.

    Each cluster is shown with a 2-std confidence ellipse.
    Medoids are marked with stars.

    Args:
        embedding: 2D coordinates (N x 2)
        labels: Cluster labels (N,)
        medoid_indices: Indices of cluster medoids
        title: Plot title
        output_path: Path to save figure
        figsize: Figure size
        n_std: Number of standard deviations for ellipse

    Returns:
        Matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    n_clusters = len(np.unique(labels))
    colors = plt.cm.tab20(np.linspace(0, 1, max(n_clusters, 20)))

    for cluster_id in range(n_clusters):
        mask = labels == cluster_id
        cluster_points = embedding[mask]

        if len(cluster_points) < 3:
            # Too few points for ellipse
            ax.scatter(
                cluster_points[:, 0], cluster_points[:, 1],
                c=[colors[cluster_id]], alpha=0.6, s=30,
                label=f'C{cluster_id} (n={mask.sum()})'
            )
            continue

        # Plot points
        ax.scatter(
            cluster_points[:, 0], cluster_points[:, 1],
            c=[colors[cluster_id]], alpha=0.4, s=20,
            label=f'C{cluster_id} (n={mask.sum()})'
        )

        # Draw confidence ellipse
        mean = cluster_points.mean(axis=0)
        cov = np.cov(cluster_points.T)

        if cov.shape == (2, 2):
            try:
                eigenvalues, eigenvectors = np.linalg.eigh(cov)
                angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
                width = 2 * n_std * np.sqrt(eigenvalues[0])
                height = 2 * n_std * np.sqrt(eigenvalues[1])

                ellipse = Ellipse(
                    mean, width, height, angle=angle,
                    facecolor=colors[cluster_id], alpha=0.15,
                    edgecolor=colors[cluster_id], linewidth=2
                )
                ax.add_patch(ellipse)
            except Exception:
                # Degenerate covariance (singular eigvecs) — skip ellipse.
                pass

    # Mark medoids with stars
    if medoid_indices:
        for i, medoid_idx in enumerate(medoid_indices):
            if medoid_idx >= 0 and medoid_idx < len(embedding):
                ax.scatter(
                    embedding[medoid_idx, 0], embedding[medoid_idx, 1],
                    marker='*', s=500, c=[colors[i % len(colors)]],
                    edgecolors='black', linewidths=2, zorder=10
                )

    ax.set_xlabel('Dimension 1', fontsize=12)
    ax.set_ylabel('Dimension 2', fontsize=12)
    ax.set_title(f'{title}\n☆ = Medoid (평균 거리 최소)', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')

    return fig


def plot_metric_comparison(
    df: pd.DataFrame,
    cluster_column: str = "cluster",
    metrics: List[str] = None,
    output_path: Optional[Union[str, Path]] = None,
    figsize: tuple = (14, 6),
) -> plt.Figure:
    """
    Plot comparison of multiple metrics across clusters.

    Args:
        df: DataFrame with metrics
        cluster_column: Column containing cluster labels
        metrics: List of metric columns to compare
        output_path: Path to save figure
        figsize: Figure size

    Returns:
        Matplotlib Figure object
    """
    if metrics is None:
        metrics = ['native_tm', 'native_dockq', 'native_lddt', 'plddt']
        metrics = [m for m in metrics if m in df.columns]

    if not metrics:
        raise ValueError("No valid metrics found")

    fig, ax = plt.subplots(figsize=figsize)

    n_clusters = df[cluster_column].nunique()
    x = np.arange(n_clusters)
    width = 0.8 / len(metrics)

    for i, metric in enumerate(metrics):
        means = df.groupby(cluster_column)[metric].mean().values
        offset = (i - len(metrics)/2 + 0.5) * width
        ax.bar(x + offset, means, width, label=metric, alpha=0.8)

    ax.set_xlabel('Cluster')
    ax.set_ylabel('Mean Score')
    ax.set_title('Metric Comparison Across Clusters')
    ax.set_xticks(x)
    ax.set_xticklabels([f'C{i}' for i in range(n_clusters)])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')

    return fig
