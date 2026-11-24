"""
MassiveClustering - Clustering analysis tool for MassiveFold predictions

A comprehensive toolkit for analyzing and clustering protein structure predictions
from MassiveFold, AlphaFold-Multimer, and ColabFold.

Features:
    - GPU-accelerated distance matrix computation (TM, iTM, dRMSD)
    - UMAP + Fuzzy C-means clustering with automatic K selection
    - Multi-metric K estimation (Silhouette, Davies-Bouldin, Calinski-Harabasz, FPC)
    - Medoid extraction from original distance matrix
    - Rich visualizations with confidence ellipses
"""

__version__ = "0.1.0"
__author__ = "Gunwoong Park"

from residudes_qa.cluster.core import (
    compute_distance_matrix,
    similarity_to_distance,
    run_clustering,
    run_fuzzy_clustering,
    select_optimal_k,
    ClusteringResult,
    get_ambiguous_samples,
    GPU_AVAILABLE,
)
from residudes_qa.cluster.assessment import (
    load_assessment_scores,
    merge_with_clustering,
    compute_cluster_stats,
    classify_dockq_quality,
    generate_heatmap,
)
from residudes_qa.cluster.visualization import (
    plot_umap,
    plot_tsne,
    plot_clusters,
    generate_report,
)
from residudes_qa.cluster.visualization.cluster_plots import plot_clusters_with_ellipse

# GPU functions (optional)
if GPU_AVAILABLE:
    from residudes_qa.cluster.core.gpu import (
        compute_tm_matrix_gpu,
        compute_itm_matrix_gpu,
        compute_drmsd_matrix_gpu,
    )

__all__ = [
    # Core
    "compute_distance_matrix",
    "similarity_to_distance",
    "run_clustering",
    "run_fuzzy_clustering",
    "select_optimal_k",
    "ClusteringResult",
    "get_ambiguous_samples",
    "GPU_AVAILABLE",
    # Assessment
    "load_assessment_scores",
    "merge_with_clustering",
    "compute_cluster_stats",
    "classify_dockq_quality",
    "generate_heatmap",
    # Visualization
    "plot_umap",
    "plot_tsne",
    "plot_clusters",
    "plot_clusters_with_ellipse",
    "generate_report",
]

if GPU_AVAILABLE:
    __all__.extend([
        "compute_tm_matrix_gpu",
        "compute_itm_matrix_gpu",
        "compute_drmsd_matrix_gpu",
    ])
