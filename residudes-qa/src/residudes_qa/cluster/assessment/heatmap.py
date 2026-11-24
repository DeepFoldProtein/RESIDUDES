"""
Heatmap visualization module

Generate comprehensive heatmaps showing multiple metrics:
- Confidence metrics: pLDDT, pTM, piTM, pDockQ
- Native comparison: native_TM, native_iTM, native_DockQ
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, List, Union


def generate_heatmap(
    df: pd.DataFrame,
    metrics: Optional[List[str]] = None,
    row_index: str = "structure",
    cluster_column: Optional[str] = None,
    output_path: Optional[Union[str, Path]] = None,
    figsize: tuple = (14, 10),
    cmap: str = "RdYlGn",
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Generate a multi-metric heatmap for clustering assessment.

    Args:
        df: DataFrame with metrics (structures as rows)
        metrics: List of metric columns to include (auto-detect if None)
        row_index: Column to use as row labels
        cluster_column: Optional column for cluster grouping
        output_path: Path to save figure
        figsize: Figure size
        cmap: Colormap
        title: Figure title

    Returns:
        Matplotlib Figure object

    Example:
        >>> fig = generate_heatmap(
        ...     df=results_df,
        ...     metrics=["plddt", "native_tm", "native_dockq"],
        ...     cluster_column="cluster"
        ... )
    """
    # Auto-detect metric columns
    if metrics is None:
        metric_candidates = [
            'plddt', 'ptm', 'pitm', 'pdockq', 'iptm',
            'native_tm', 'native_itm', 'native_dockq',
            'native_rmsd', 'native_lddt', 'native_ilddt',
            'native_ics', 'native_ips',
            'tm_score', 'dockq_ave', 'dockq_wave', 'lddt', 'ilddt', 'ics', 'ips',
        ]
        metrics = [c for c in metric_candidates if c in df.columns]

    if not metrics:
        raise ValueError("No metric columns found in DataFrame")

    # Prepare data
    plot_df = df.copy()

    # Sort by cluster if specified
    if cluster_column and cluster_column in plot_df.columns:
        plot_df = plot_df.sort_values(cluster_column)

    # Set index
    if row_index in plot_df.columns:
        plot_df = plot_df.set_index(row_index)

    # Select only metric columns
    heatmap_data = plot_df[metrics].astype(float)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Generate heatmap
    sns.heatmap(
        heatmap_data,
        ax=ax,
        cmap=cmap,
        annot=len(heatmap_data) <= 50,  # Only annotate small heatmaps
        fmt='.2f',
        vmin=0,
        vmax=1,
        cbar_kws={'label': 'Score'},
        linewidths=0.5 if len(heatmap_data) <= 50 else 0,
    )

    # Add cluster separators if applicable
    if cluster_column and cluster_column in df.columns:
        _add_cluster_separators(ax, df, cluster_column)

    # Title
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    else:
        ax.set_title('Multi-Metric Assessment Heatmap', fontsize=14, fontweight='bold')

    ax.set_ylabel('Structure')
    ax.set_xlabel('Metric')

    plt.tight_layout()

    # Save if path specified
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches='tight')

    return fig


def generate_cluster_summary_heatmap(
    df: pd.DataFrame,
    cluster_column: str = "cluster",
    metrics: Optional[List[str]] = None,
    output_path: Optional[Union[str, Path]] = None,
    figsize: tuple = (12, 8),
) -> plt.Figure:
    """
    Generate heatmap summarizing metrics per cluster (mean values).

    Args:
        df: DataFrame with clustering and metric data
        cluster_column: Column containing cluster labels
        metrics: List of metric columns
        output_path: Path to save figure
        figsize: Figure size

    Returns:
        Matplotlib Figure object
    """
    # Auto-detect metrics
    if metrics is None:
        metric_candidates = [
            'plddt', 'ptm', 'pitm', 'native_tm', 'native_dockq',
            'native_lddt', 'native_ilddt', 'dockq_ave', 'ilddt',
        ]
        metrics = [c for c in metric_candidates if c in df.columns]

    if not metrics:
        raise ValueError("No metric columns found")

    # Compute cluster means
    cluster_means = df.groupby(cluster_column)[metrics].mean()

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Heatmap of cluster means
    ax1 = axes[0]
    sns.heatmap(
        cluster_means,
        ax=ax1,
        cmap='RdYlGn',
        annot=True,
        fmt='.3f',
        vmin=0,
        vmax=1,
        cbar_kws={'label': 'Mean Score'},
    )
    ax1.set_title('Cluster Mean Metrics')
    ax1.set_ylabel('Cluster')

    # Cluster sizes bar chart
    ax2 = axes[1]
    cluster_sizes = df.groupby(cluster_column).size()
    colors = plt.cm.tab20(np.linspace(0, 1, len(cluster_sizes)))
    bars = ax2.bar(range(len(cluster_sizes)), cluster_sizes.values, color=colors)
    ax2.set_xticks(range(len(cluster_sizes)))
    ax2.set_xticklabels([f'C{i}' for i in cluster_sizes.index])
    ax2.set_ylabel('Size')
    ax2.set_xlabel('Cluster')
    ax2.set_title('Cluster Sizes')

    # Add value labels on bars
    for bar, size in zip(bars, cluster_sizes.values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                 str(size), ha='center', va='bottom', fontsize=9)

    plt.suptitle('Cluster Summary', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')

    return fig


def _add_cluster_separators(ax, df, cluster_column):
    """Add horizontal lines between clusters in heatmap."""
    cluster_counts = df[cluster_column].value_counts().sort_index()
    cumsum = 0
    for i, count in enumerate(cluster_counts.values[:-1]):
        cumsum += count
        ax.axhline(y=cumsum, color='black', linewidth=2)


def generate_medoid_heatmap(
    medoid_df: pd.DataFrame,
    metrics: Optional[List[str]] = None,
    output_path: Optional[Union[str, Path]] = None,
    figsize: tuple = (10, 6),
) -> plt.Figure:
    """
    Generate heatmap for cluster medoids/representatives only.

    Args:
        medoid_df: DataFrame with medoid metrics (one row per cluster)
        metrics: List of metric columns
        output_path: Path to save figure
        figsize: Figure size

    Returns:
        Matplotlib Figure object
    """
    if metrics is None:
        metric_candidates = [
            'plddt', 'ptm', 'native_tm', 'native_dockq', 'native_lddt',
        ]
        metrics = [c for c in metric_candidates if c in medoid_df.columns]

    if not metrics:
        raise ValueError("No metric columns found")

    fig, ax = plt.subplots(figsize=figsize)

    # Use cluster or medoid name as index
    if 'cluster' in medoid_df.columns:
        index_col = 'cluster'
    elif 'medoid_name' in medoid_df.columns:
        index_col = 'medoid_name'
    else:
        medoid_df = medoid_df.reset_index()
        index_col = 'index'

    plot_df = medoid_df.set_index(index_col)[metrics]

    sns.heatmap(
        plot_df,
        ax=ax,
        cmap='RdYlGn',
        annot=True,
        fmt='.3f',
        vmin=0,
        vmax=1,
        linewidths=1,
        cbar_kws={'label': 'Score'},
    )

    ax.set_title('Cluster Representative Metrics', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cluster / Medoid')

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')

    return fig
