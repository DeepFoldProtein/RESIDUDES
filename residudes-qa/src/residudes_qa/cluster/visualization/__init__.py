"""
Visualization modules

- embeddings: UMAP and t-SNE dimensionality reduction
- cluster_plots: Cluster visualization with various annotations
- distance_heatmap: Distance matrix heatmaps sorted by cluster
- comprehensive: Multi-panel comprehensive figures
- summary_table: Cluster summary tables (PNG + CSV)
- report: Comprehensive report generation
"""

from residudes_qa.cluster.visualization.embeddings import plot_umap, plot_tsne, compute_embedding
from residudes_qa.cluster.visualization.cluster_plots import plot_clusters, plot_cluster_distribution
from residudes_qa.cluster.visualization.distance_heatmap import (
    plot_distance_heatmap,
    plot_similarity_heatmap,
    plot_multi_metric_heatmaps,
    plot_inter_cluster_heatmap,
)
from residudes_qa.cluster.visualization.comprehensive import (
    create_comprehensive_figure,
    create_comprehensive_report,
    create_simple_comprehensive,
    plot_k_selection_diagnostic,
    create_k_diagnostic_figure,
)
from residudes_qa.cluster.visualization.summary_table import (
    create_cluster_summary_table,
    create_medoid_summary_table,
    print_cluster_summary,
)
from residudes_qa.cluster.visualization.report import generate_report
from residudes_qa.cluster.visualization.network_viz import (
    plot_network_graph,
    plot_community_comparison,
    plot_transition_heatmap,
    plot_resolution_scan,
    create_network_report,
    create_pretty_community_graph,
)
from residudes_qa.cluster.visualization.medoid_faces import render_medoid_faces
from residudes_qa.cluster.visualization.dendrogram import (
    plot_cluster_dendrogram,
    plot_cluster_clustermap,
    compute_cluster_linkage,
    compute_inter_cluster_distance,
)

__all__ = [
    # Embeddings
    "plot_umap",
    "plot_tsne",
    "compute_embedding",
    # Cluster plots
    "plot_clusters",
    "plot_cluster_distribution",
    # Distance heatmaps
    "plot_distance_heatmap",
    "plot_similarity_heatmap",
    "plot_multi_metric_heatmaps",
    "plot_inter_cluster_heatmap",
    # Comprehensive
    "create_comprehensive_figure",
    "create_simple_comprehensive",
    "plot_k_selection_diagnostic",
    "create_k_diagnostic_figure",
    # Summary tables
    "create_cluster_summary_table",
    "create_medoid_summary_table",
    "print_cluster_summary",
    # Report
    "generate_report",
    # Network visualization
    "plot_network_graph",
    "plot_community_comparison",
    "plot_transition_heatmap",
    "plot_resolution_scan",
    "create_network_report",
    "create_pretty_community_graph",
    # Medoid face overlay
    "render_medoid_faces",
    # Cluster dendrogram
    "plot_cluster_dendrogram",
    "plot_cluster_clustermap",
    "compute_cluster_linkage",
    "compute_inter_cluster_distance",
]
