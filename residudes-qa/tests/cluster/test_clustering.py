"""Tests for clustering module."""

import numpy as np
import pytest

from residudes_qa.cluster.core.clustering import run_clustering, select_optimal_k


class TestRunClustering:
    @pytest.mark.parametrize("method", ["fuzzy", "kmeans", "hierarchical", "gmm"])
    def test_all_methods_run(self, medium_distance_matrix, method):
        """All clustering methods should produce valid labels."""
        result = run_clustering(medium_distance_matrix, n_clusters=3, method=method)
        assert len(result.labels) == 50
        assert len(np.unique(result.labels)) <= 3
        assert result.medoid_indices is not None
        assert len(result.medoid_indices) == 3

    def test_labels_are_contiguous(self, medium_distance_matrix):
        """Labels should be 0..k-1."""
        result = run_clustering(medium_distance_matrix, n_clusters=4, method="hierarchical")
        unique = np.unique(result.labels)
        assert unique[0] == 0
        assert unique[-1] <= 3

    def test_medoids_within_bounds(self, medium_distance_matrix):
        """Medoid indices must be valid indices."""
        result = run_clustering(medium_distance_matrix, n_clusters=3, method="fuzzy")
        for idx in result.medoid_indices:
            assert 0 <= idx < 50

    def test_embedding_exists(self, medium_distance_matrix):
        """Clustering should produce a 2D embedding."""
        result = run_clustering(medium_distance_matrix, n_clusters=3, method="fuzzy")
        assert result.embedding is not None
        assert result.embedding.shape == (50, 2)


class TestSelectOptimalK:
    def test_returns_valid_k(self, medium_distance_matrix):
        """Optimal K should be within requested range."""
        k, scores = select_optimal_k(medium_distance_matrix, k_range=(2, 10))
        assert 2 <= k <= 10

    def test_scores_dict_keys(self, medium_distance_matrix):
        """Scores dict should contain expected keys."""
        k, scores = select_optimal_k(medium_distance_matrix, k_range=(2, 10))
        expected_keys = [
            "k_values", "silhouette_scores", "bic_scores",
            "combined_scores", "optimal_k", "best_silhouette",
            "bic_optimal_k", "k_max_requested", "k_max_effective",
            "is_ascending_at_boundary", "plateau_k",
        ]
        for key in expected_keys:
            assert key in scores, f"Missing key: {key}"

    def test_clear_clusters_found(self, clustered_distance_matrix):
        """With 3 clear clusters, K should be near 3."""
        k, scores = select_optimal_k(clustered_distance_matrix, k_range=(2, 10))
        assert 2 <= k <= 5  # should find ~3

    def test_k_range_single(self, medium_distance_matrix):
        """k_range=(5,5) should return k=5."""
        k, scores = select_optimal_k(medium_distance_matrix, k_range=(5, 5))
        assert k == 5

    def test_adaptive_detection(self, medium_distance_matrix):
        """Adaptive mode should set plateau_k and is_ascending."""
        k, scores = select_optimal_k(
            medium_distance_matrix, k_range=(2, 15), adaptive=True
        )
        assert "plateau_k" in scores
        assert "is_ascending_at_boundary" in scores
        assert scores["is_ascending_at_boundary"] in (True, False)

    def test_no_artificial_cap(self, medium_distance_matrix):
        """k_max should not be capped by n//20 anymore."""
        k, scores = select_optimal_k(medium_distance_matrix, k_range=(2, 30))
        assert scores["k_max_effective"] == 30  # not capped at 50//20=2
