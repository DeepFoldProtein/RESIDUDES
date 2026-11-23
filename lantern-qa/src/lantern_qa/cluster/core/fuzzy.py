"""
Fuzzy C-means clustering utilities

Provides helper functions for fuzzy clustering analysis:
- Membership analysis
- Ambiguous sample detection
- High-membership sample extraction

Main clustering is now in clustering.py (run_clustering with method="fuzzy")
"""

import numpy as np
from typing import Optional, List
from dataclasses import dataclass

# Re-export from clustering for backwards compatibility
from lantern_qa.cluster.core.clustering import ClusteringResult, run_clustering


def get_high_membership_samples(
    membership: np.ndarray,
    cluster_id: int,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Get samples with membership above threshold for a cluster.

    Args:
        membership: Membership matrix (n_samples, n_clusters)
        cluster_id: Cluster index
        threshold: Membership threshold

    Returns:
        Array of sample indices
    """
    if membership is None:
        return np.array([])
    return np.where(membership[:, cluster_id] >= threshold)[0]


def get_ambiguous_samples(
    membership: np.ndarray,
    max_membership_threshold: float = 0.6,
) -> np.ndarray:
    """
    Find samples with ambiguous cluster assignment.

    Args:
        membership: Membership matrix (n_samples, n_clusters)
        max_membership_threshold: Samples with max membership below this are ambiguous

    Returns:
        Array of ambiguous sample indices
    """
    if membership is None:
        return np.array([])
    max_membership = membership.max(axis=1)
    return np.where(max_membership < max_membership_threshold)[0]


def get_membership_entropy(membership: np.ndarray) -> np.ndarray:
    """
    Compute entropy of membership distribution for each sample.

    Higher entropy = more ambiguous cluster assignment.

    Args:
        membership: Membership matrix (n_samples, n_clusters)

    Returns:
        Entropy values for each sample
    """
    if membership is None:
        return np.array([])

    # Avoid log(0)
    m = np.clip(membership, 1e-10, 1.0)
    entropy = -np.sum(m * np.log2(m), axis=1)
    return entropy


def get_cluster_membership_stats(
    membership: np.ndarray,
    labels: np.ndarray,
) -> dict:
    """
    Compute membership statistics per cluster.

    Args:
        membership: Membership matrix (n_samples, n_clusters)
        labels: Hard cluster labels

    Returns:
        Dict with per-cluster membership statistics
    """
    if membership is None:
        return {}

    n_clusters = membership.shape[1]
    stats = {}

    for k in range(n_clusters):
        cluster_mask = labels == k
        if cluster_mask.sum() == 0:
            continue

        cluster_membership = membership[cluster_mask, k]
        stats[k] = {
            'mean_membership': cluster_membership.mean(),
            'min_membership': cluster_membership.min(),
            'max_membership': cluster_membership.max(),
            'std_membership': cluster_membership.std(),
            'n_high_confidence': (cluster_membership >= 0.8).sum(),
            'n_low_confidence': (cluster_membership < 0.5).sum(),
        }

    return stats


# Backwards compatibility alias
def run_fuzzy_clustering(
    distance_matrix: np.ndarray,
    n_clusters: int,
    fuzziness: float = 2.0,
    structure_names: Optional[List[str]] = None,
) -> ClusteringResult:
    """
    Run Fuzzy C-means clustering on a distance matrix.

    This is a convenience wrapper around run_clustering(method="fuzzy").

    Args:
        distance_matrix: Precomputed distance matrix (N x N)
        n_clusters: Number of clusters
        fuzziness: Fuzziness parameter (m > 1, typically 2)
        structure_names: Optional list of structure names

    Returns:
        ClusteringResult object with membership probabilities
    """
    return run_clustering(
        distance_matrix=distance_matrix,
        n_clusters=n_clusters,
        method="fuzzy",
        structure_names=structure_names,
        fuzziness=fuzziness,
    )
