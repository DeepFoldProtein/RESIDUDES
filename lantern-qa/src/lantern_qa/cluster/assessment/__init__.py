"""
Assessment module - Score loading and merging utilities

Provides functions for:
- Loading pre-computed assessment scores (DockQ, TM-score, lDDT, etc.)
- Merging scores with clustering results
- Computing per-cluster statistics
"""

from lantern_qa.cluster.assessment.scores import (
    load_assessment_scores,
    merge_with_clustering,
    compute_cluster_stats,
    classify_dockq_quality,
    add_dockq_categories,
)

# Keep heatmap for visualization
from lantern_qa.cluster.assessment.heatmap import generate_heatmap

__all__ = [
    "load_assessment_scores",
    "merge_with_clustering",
    "compute_cluster_stats",
    "classify_dockq_quality",
    "add_dockq_categories",
    "generate_heatmap",
]
