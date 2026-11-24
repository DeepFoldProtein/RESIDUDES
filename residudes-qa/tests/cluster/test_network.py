"""Tests for network analysis module."""

import numpy as np
import pytest

from residudes_qa.cluster.core.network import (
    build_structure_graph,
    analyze_cluster_network,
    compute_cluster_transition_matrix,
    resolution_scan,
    NetworkAnalysisResult,
)


class TestBuildStructureGraph:
    def test_auto_knn(self, medium_distance_matrix, sample_names):
        G = build_structure_graph(medium_distance_matrix, structure_names=sample_names)
        assert G.number_of_nodes() == 50
        assert G.number_of_edges() > 0

    def test_threshold_mode(self, medium_distance_matrix):
        G = build_structure_graph(medium_distance_matrix, threshold=0.3)
        assert G.number_of_nodes() == 50
        for u, v, d in G.edges(data=True):
            assert d["distance"] < 0.3

    def test_knn_mode(self, medium_distance_matrix):
        G = build_structure_graph(medium_distance_matrix, k_neighbors=3)
        assert G.number_of_nodes() == 50
        # Each node has at least 3 edges (KNN is symmetric)
        for node in G.nodes():
            assert G.degree(node) >= 3

    def test_edge_attributes(self, medium_distance_matrix):
        G = build_structure_graph(medium_distance_matrix)
        u, v, d = list(G.edges(data=True))[0]
        assert "weight" in d
        assert "distance" in d
        assert d["weight"] >= 0


class TestAnalyzeClusterNetwork:
    @pytest.mark.parametrize("method", ["auto", "leiden", "louvain"])
    def test_community_methods(self, medium_distance_matrix, sample_labels, sample_names, method):
        result = analyze_cluster_network(
            medium_distance_matrix, sample_labels,
            structure_names=sample_names, community_method=method,
        )
        assert isinstance(result, NetworkAnalysisResult)
        assert result.n_nodes == 50
        assert result.n_edges > 0
        assert 0 < result.density <= 1
        assert result.community_method in ("leiden", "louvain")

    def test_inter_cluster_matrix(self, medium_distance_matrix, sample_labels):
        result = analyze_cluster_network(medium_distance_matrix, sample_labels)
        n_clusters = len(np.unique(sample_labels))
        assert result.inter_cluster_matrix.shape == (n_clusters, n_clusters)
        assert np.allclose(result.inter_cluster_matrix, result.inter_cluster_matrix.T, atol=0.01)

    def test_degree_centrality(self, medium_distance_matrix, sample_labels):
        result = analyze_cluster_network(medium_distance_matrix, sample_labels)
        assert result.degree_centrality.shape == (50,)
        assert result.degree_centrality.min() >= 0
        assert result.degree_centrality.max() <= 1

    def test_betweenness_off_by_default(self, medium_distance_matrix, sample_labels):
        result = analyze_cluster_network(medium_distance_matrix, sample_labels)
        assert result.betweenness_centrality is None

    def test_betweenness_on(self, medium_distance_matrix, sample_labels):
        result = analyze_cluster_network(
            medium_distance_matrix, sample_labels, compute_betweenness=True,
        )
        assert result.betweenness_centrality is not None
        assert result.betweenness_centrality.shape == (50,)

    def test_hub_indices(self, medium_distance_matrix, sample_labels):
        result = analyze_cluster_network(
            medium_distance_matrix, sample_labels, n_hubs=5,
        )
        assert len(result.hub_indices) == 5
        for idx in result.hub_indices:
            assert 0 <= idx < 50

    def test_leiden_resolution(self, medium_distance_matrix, sample_labels):
        r1 = analyze_cluster_network(
            medium_distance_matrix, sample_labels, community_method="leiden", resolution=0.01,
        )
        r2 = analyze_cluster_network(
            medium_distance_matrix, sample_labels, community_method="leiden", resolution=2.0,
        )
        n1 = len(np.unique(r1.communities[r1.communities >= 0]))
        n2 = len(np.unique(r2.communities[r2.communities >= 0]))
        # Higher resolution -> more communities (or equal)
        assert n2 >= n1

    def test_metadata(self, medium_distance_matrix, sample_labels):
        result = analyze_cluster_network(medium_distance_matrix, sample_labels)
        assert "n_clusters" in result.metadata
        assert "graph_components" in result.metadata
        assert result.metadata["graph_components"] >= 1


class TestEdgeCases:
    def test_n2(self):
        """Smallest possible input."""
        d = np.array([[0, 0.5], [0.5, 0]])
        labels = np.array([0, 1])
        result = analyze_cluster_network(d, labels)
        assert result.n_nodes == 2

    def test_n3(self):
        d = np.array([[0, 0.2, 0.8], [0.2, 0, 0.5], [0.8, 0.5, 0]])
        labels = np.array([0, 0, 1])
        result = analyze_cluster_network(d, labels)
        assert result.n_nodes == 3
        assert result.inter_cluster_matrix.shape == (2, 2)

    def test_single_cluster(self, small_distance_matrix):
        labels = np.zeros(5, dtype=int)
        result = analyze_cluster_network(small_distance_matrix, labels)
        assert result.inter_cluster_matrix.shape == (1, 1)

    def test_all_zero_distance(self):
        d = np.zeros((5, 5))
        labels = np.array([0, 0, 1, 1, 2])
        result = analyze_cluster_network(d, labels, community_method="louvain")
        assert result.n_nodes == 5

    def test_no_structure_names(self, medium_distance_matrix, sample_labels):
        result = analyze_cluster_network(medium_distance_matrix, sample_labels)
        assert result.n_nodes == 50


class TestTransitionMatrix:
    def test_shape(self, medium_distance_matrix, sample_labels):
        T = compute_cluster_transition_matrix(medium_distance_matrix, sample_labels)
        n_clusters = len(np.unique(sample_labels))
        assert T.shape == (n_clusters, n_clusters)

    def test_row_sums(self, medium_distance_matrix, sample_labels):
        T = compute_cluster_transition_matrix(medium_distance_matrix, sample_labels)
        assert np.allclose(T.sum(axis=1), 1.0, atol=1e-6)

    def test_nonnegative(self, medium_distance_matrix, sample_labels):
        T = compute_cluster_transition_matrix(medium_distance_matrix, sample_labels)
        assert (T >= 0).all()

    def test_zero_diagonal(self, medium_distance_matrix, sample_labels):
        T = compute_cluster_transition_matrix(medium_distance_matrix, sample_labels)
        assert np.allclose(np.diag(T), 0)


class TestResolutionScan:
    def test_auto_resolutions(self, medium_distance_matrix, sample_names):
        G = build_structure_graph(medium_distance_matrix, structure_names=sample_names)
        scan = resolution_scan(G, n_points=10)
        assert len(scan["resolutions"]) == 10
        assert len(scan["n_communities"]) == 10
        assert len(scan["modularity"]) == 10
        assert "suggested_resolution" in scan

    def test_custom_resolutions(self, medium_distance_matrix, sample_names):
        G = build_structure_graph(medium_distance_matrix, structure_names=sample_names)
        scan = resolution_scan(G, resolutions=[0.01, 0.1, 1.0])
        assert len(scan["resolutions"]) == 3
