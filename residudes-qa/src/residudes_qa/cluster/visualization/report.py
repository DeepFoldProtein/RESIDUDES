"""
Report generation module

Generates comprehensive analysis reports combining multiple visualizations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Union, Dict, Any
from datetime import datetime

from residudes_qa.cluster.visualization.embeddings import compute_embedding
from residudes_qa.cluster.visualization.cluster_plots import (
    plot_clusters,
    plot_cluster_distribution,
    plot_cluster_sizes,
)
from residudes_qa.cluster.assessment.heatmap import generate_cluster_summary_heatmap


def generate_report(
    distance_matrix: np.ndarray,
    labels: np.ndarray,
    structure_names: Optional[list] = None,
    assessment_df: Optional[pd.DataFrame] = None,
    medoid_indices: Optional[list] = None,
    target_name: str = "Unknown",
    metric_type: str = "TM",
    output_dir: Union[str, Path] = "report",
    k_scores: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Generate a comprehensive clustering analysis report.

    Creates multiple plots and saves them to the output directory:
    - UMAP/t-SNE embeddings
    - Cluster distribution plots
    - Metric heatmaps
    - K selection diagnostic (if k_scores provided)
    - Summary statistics

    Args:
        distance_matrix: Precomputed distance matrix
        labels: Cluster labels
        structure_names: Optional list of structure names
        assessment_df: Optional DataFrame with assessment scores
        medoid_indices: Indices of cluster medoids
        target_name: Name of the target
        metric_type: Type of distance metric used
        output_dir: Directory to save report files
        k_scores: Optional K selection scores from select_optimal_k()

    Returns:
        Dictionary with report metadata and file paths

    Example:
        >>> report = generate_report(
        ...     distance_matrix=dist_matrix,
        ...     labels=clustering_result.labels,
        ...     assessment_df=scores_df,
        ...     target_name="T1234",
        ...     output_dir="results/T1234"
        ... )
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report_files = []
    n_clusters = len(np.unique(labels))
    n_structures = len(labels)

    print(f"\n{'='*60}")
    print(f"  Generating Report: {target_name}")
    print(f"{'='*60}")
    print(f"  Structures: {n_structures}")
    print(f"  Clusters: {n_clusters}")
    print(f"  Metric: {metric_type}")

    # 1. Compute embedding
    print("\n  [1/5] Computing UMAP embedding...")
    embedding = compute_embedding(distance_matrix, method="umap")

    # 2. Main cluster plot
    print("  [2/5] Generating cluster plot...")
    best_idx = None
    if assessment_df is not None and 'native_dockq' in assessment_df.columns:
        best_idx = assessment_df['native_dockq'].idxmax()
    elif assessment_df is not None and 'native_tm' in assessment_df.columns:
        best_idx = assessment_df['native_tm'].idxmax()

    fig = plot_clusters(
        embedding=embedding,
        labels=labels,
        medoid_indices=medoid_indices,
        best_idx=best_idx,
        best_label="Best Native",
        title=f"{target_name} - {metric_type} Clustering",
        output_path=output_dir / f"{target_name}_clusters.png",
    )
    plt.close(fig)
    report_files.append(f"{target_name}_clusters.png")

    # 2b. K selection diagnostic (if k_scores available)
    if k_scores and 'combined_scores' in k_scores:
        print("  [2b/5] Generating K selection diagnostic...")
        try:
            from residudes_qa.cluster.visualization.comprehensive import create_k_diagnostic_figure
            k_scores.get('optimal_k', n_clusters)
            fig_k = create_k_diagnostic_figure(
                k_scores, target_name=target_name, metric_name=metric_type
            )
            if fig_k is not None:
                fig_k.savefig(
                    output_dir / f"{target_name}_k_diagnostic.png",
                    dpi=150, bbox_inches='tight'
                )
                plt.close(fig_k)
                report_files.append(f"{target_name}_k_diagnostic.png")
        except Exception as e:
            print(f"    Warning: K diagnostic failed: {e}")

    # 3. Cluster size distribution
    print("  [3/5] Generating cluster size plot...")
    fig = plot_cluster_sizes(
        labels=labels,
        title=f"{target_name} - Cluster Sizes",
        output_path=output_dir / f"{target_name}_sizes.png",
    )
    plt.close(fig)
    report_files.append(f"{target_name}_sizes.png")

    # 4. Assessment metric distributions
    if assessment_df is not None:
        print("  [4/5] Generating metric distribution plots...")

        # Add cluster labels to assessment_df if not present
        if 'cluster' not in assessment_df.columns:
            assessment_df = assessment_df.copy()
            assessment_df['cluster'] = labels

        for metric in ['native_dockq', 'native_tm', 'native_lddt']:
            if metric in assessment_df.columns:
                fig = plot_cluster_distribution(
                    df=assessment_df,
                    cluster_column='cluster',
                    value_column=metric,
                    title=f"{target_name} - {metric} by Cluster",
                    output_path=output_dir / f"{target_name}_{metric}_dist.png",
                )
                plt.close(fig)
                report_files.append(f"{target_name}_{metric}_dist.png")

        # 5. Summary heatmap
        print("  [5/5] Generating summary heatmap...")
        try:
            fig = generate_cluster_summary_heatmap(
                df=assessment_df,
                cluster_column='cluster',
                output_path=output_dir / f"{target_name}_summary_heatmap.png",
            )
            plt.close(fig)
            report_files.append(f"{target_name}_summary_heatmap.png")
        except Exception as e:
            print(f"    Warning: Could not generate heatmap: {e}")
    else:
        print("  [4/5] Skipping metric plots (no assessment data)")
        print("  [5/5] Skipping heatmap (no assessment data)")

    # Generate summary statistics
    summary_stats = _compute_summary_stats(labels, assessment_df, medoid_indices, k_scores)

    # Save summary CSV
    summary_file = output_dir / f"{target_name}_summary.csv"
    pd.DataFrame([summary_stats]).to_csv(summary_file, index=False)
    report_files.append(f"{target_name}_summary.csv")

    # Generate text report
    report_text = _generate_text_report(
        target_name, metric_type, n_structures, n_clusters,
        summary_stats, report_files
    )

    report_file = output_dir / f"{target_name}_report.txt"
    with open(report_file, 'w') as f:
        f.write(report_text)
    report_files.append(f"{target_name}_report.txt")

    print(f"\n  Report saved to: {output_dir}")
    print(f"  Files: {len(report_files)}")

    return {
        'target': target_name,
        'output_dir': str(output_dir),
        'files': report_files,
        'summary': summary_stats,
        'embedding': embedding,
    }


def _compute_summary_stats(
    labels: np.ndarray,
    assessment_df: Optional[pd.DataFrame],
    medoid_indices: Optional[list],
    k_scores: Optional[Dict] = None,
) -> Dict[str, Any]:
    """Compute summary statistics."""
    unique_labels, counts = np.unique(labels, return_counts=True)

    stats = {
        'n_structures': len(labels),
        'n_clusters': len(unique_labels),
        'largest_cluster_size': int(counts.max()),
        'smallest_cluster_size': int(counts.min()),
        'mean_cluster_size': float(counts.mean()),
        'max_cluster_fraction': float(counts.max() / counts.sum()),
    }

    # K selection info
    if k_scores:
        stats['selected_k'] = k_scores.get('optimal_k')
        stats['bic_optimal_k'] = k_scores.get('bic_optimal_k')
        stats['best_silhouette'] = k_scores.get('best_silhouette')
        stats['plateau_k'] = k_scores.get('plateau_k')
        stats['ascending_at_boundary'] = k_scores.get('is_ascending_at_boundary')
        stats['k_max_requested'] = k_scores.get('k_max_requested')
        stats['k_max_effective'] = k_scores.get('k_max_effective')

    if assessment_df is not None:
        for metric in ['native_tm', 'native_dockq', 'native_lddt']:
            if metric in assessment_df.columns:
                values = assessment_df[metric].dropna()
                if len(values) > 0:
                    stats[f'{metric}_mean'] = float(values.mean())
                    stats[f'{metric}_max'] = float(values.max())
                    stats[f'{metric}_min'] = float(values.min())

    return stats


def _generate_text_report(
    target_name: str,
    metric_type: str,
    n_structures: int,
    n_clusters: int,
    summary_stats: Dict,
    report_files: list,
) -> str:
    """Generate text report."""
    lines = [
        "=" * 60,
        "  MassiveClustering Analysis Report",
        "=" * 60,
        "",
        f"Target: {target_name}",
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Metric: {metric_type}",
        "",
        "-" * 60,
        "  Summary Statistics",
        "-" * 60,
        "",
        f"  Total Structures: {n_structures}",
        f"  Number of Clusters: {n_clusters}",
        f"  Mean Cluster Size: {summary_stats.get('mean_cluster_size', 0):.1f}",
        f"  Largest Cluster: {summary_stats.get('largest_cluster_size', 0)}",
        f"  Smallest Cluster: {summary_stats.get('smallest_cluster_size', 0)}",
        "",
    ]

    # Add K selection info if available
    if 'selected_k' in summary_stats:
        lines.extend([
            "-" * 60,
            "  K Selection",
            "-" * 60,
            "",
            f"  Selected K: {summary_stats.get('selected_k', '?')}",
            f"  BIC Optimal K: {summary_stats.get('bic_optimal_k', '?')}",
            f"  Plateau K: {summary_stats.get('plateau_k', 'None')}",
            f"  Silhouette: {summary_stats.get('best_silhouette', 0):.4f}",
            f"  Max Cluster Fraction: {summary_stats.get('max_cluster_fraction', 0):.1%}",
            f"  Ascending at boundary: {summary_stats.get('ascending_at_boundary', False)}",
            f"  K range: {summary_stats.get('k_max_requested', '?')} (effective: {summary_stats.get('k_max_effective', '?')})",
            "",
        ])

    # Add native comparison stats if available
    if 'native_tm_mean' in summary_stats:
        lines.extend([
            "-" * 60,
            "  Native Comparison",
            "-" * 60,
            "",
            f"  Mean Native TM: {summary_stats.get('native_tm_mean', 0):.3f}",
            f"  Best Native TM: {summary_stats.get('native_tm_max', 0):.3f}",
            "",
        ])

    if 'native_dockq_mean' in summary_stats:
        lines.extend([
            f"  Mean DockQ: {summary_stats.get('native_dockq_mean', 0):.3f}",
            f"  Best DockQ: {summary_stats.get('native_dockq_max', 0):.3f}",
            "",
        ])

    lines.extend([
        "-" * 60,
        "  Generated Files",
        "-" * 60,
        "",
    ])
    for f in report_files:
        lines.append(f"  - {f}")

    lines.extend([
        "",
        "=" * 60,
        "  Generated by MassiveClustering",
        "=" * 60,
    ])

    return "\n".join(lines)
