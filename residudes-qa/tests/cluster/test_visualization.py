"""Tests for visualization modules."""

import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from residudes_qa.cluster.core.clustering import run_clustering
from residudes_qa.cluster.core.network import (
    build_structure_graph, analyze_cluster_network,
    resolution_scan, compute_cluster_transition_matrix,
)
from residudes_qa.cluster.visualization.comprehensive import (
    create_comprehensive_figure,
    create_simple_comprehensive,
    create_comprehensive_report,
    plot_k_selection_diagnostic,
    create_k_diagnostic_figure,
)
from residudes_qa.cluster.visualization.network_viz import (
    plot_network_graph,
    plot_community_comparison,
    plot_transition_heatmap,
    plot_resolution_scan,
    create_network_report,
)
from residudes_qa.cluster.visualization.summary_table import (
    create_cluster_summary_table,
    create_medoid_summary_table,
)
from residudes_qa.cluster.visualization.distance_heatmap import (
    plot_distance_heatmap,
    plot_multi_metric_heatmaps,
    plot_inter_cluster_heatmap,
)
from residudes_qa.cluster.visualization.embeddings import compute_embedding


@pytest.fixture(autouse=True)
def close_plots():
    """Close all matplotlib figures after each test."""
    yield
    plt.close("all")


class TestComprehensiveFigure:
    def test_basic(self, medium_distance_matrix, sample_labels, tmp_output):
        fig = create_comprehensive_figure(
            distance_matrices={"tm": medium_distance_matrix},
            labels=sample_labels,
            target_name="Test",
            output_path=tmp_output / "comp.png",
            methods=["tsne"],
        )
        assert fig is not None
        assert (tmp_output / "comp.png").exists()

    def test_with_native_idx(self, medium_distance_matrix, sample_labels, tmp_output):
        result = run_clustering(medium_distance_matrix, n_clusters=3, method="fuzzy")
        fig = create_comprehensive_figure(
            distance_matrices={"tm": medium_distance_matrix},
            labels=result.labels,
            medoid_indices=result.medoid_indices,
            native_idx=5,
            target_name="NativeTest",
            output_path=tmp_output / "native.png",
            methods=["tsne"],
        )
        assert fig is not None

    def test_with_best_idx(self, medium_distance_matrix, sample_labels, tmp_output):
        result = run_clustering(medium_distance_matrix, n_clusters=3, method="fuzzy")
        fig = create_comprehensive_figure(
            distance_matrices={"tm": medium_distance_matrix},
            labels=result.labels,
            medoid_indices=result.medoid_indices,
            native_idx=None,
            best_idx=10,
            target_name="BestTest",
            output_path=tmp_output / "best.png",
            methods=["tsne"],
        )
        assert fig is not None

    def test_multi_metric(self, medium_distance_matrix, sample_labels, tmp_output):
        fig = create_comprehensive_figure(
            distance_matrices={
                "tm": medium_distance_matrix,
                "drmsd": medium_distance_matrix * 2,
            },
            labels=sample_labels,
            target_name="MultiMetric",
            output_path=tmp_output / "multi.png",
            methods=["tsne"],
        )
        assert fig is not None


class TestSimpleComprehensive:
    def test_basic(self, medium_distance_matrix, sample_labels, tmp_output):
        emb = compute_embedding(medium_distance_matrix, method="tsne")
        fig = create_simple_comprehensive(
            medium_distance_matrix, sample_labels, embedding=emb,
            output_path=tmp_output / "simple.png",
        )
        assert fig is not None

    def test_with_markers(self, medium_distance_matrix, tmp_output):
        result = run_clustering(medium_distance_matrix, n_clusters=3, method="fuzzy")
        fig = create_simple_comprehensive(
            medium_distance_matrix, result.labels,
            embedding=result.embedding,
            native_idx=0,
            medoid_indices=result.medoid_indices,
            output_path=tmp_output / "markers.png",
        )
        assert fig is not None


class TestComprehensiveReport:
    def test_basic(self, medium_distance_matrix, sample_labels, sample_names, tmp_output):
        fig = create_comprehensive_report(
            distance_matrix=medium_distance_matrix,
            labels=sample_labels,
            structure_names=sample_names,
            target_name="Report",
            output_dir=str(tmp_output),
        )
        assert fig is not None


class TestKDiagnostic:
    def test_plot_on_ax(self, medium_distance_matrix):
        from residudes_qa.cluster.core.clustering import select_optimal_k
        k, scores = select_optimal_k(medium_distance_matrix, k_range=(2, 10))
        fig, ax = plt.subplots()
        plot_k_selection_diagnostic(ax, scores, selected_k=k)
        assert fig is not None

    def test_create_figure(self, medium_distance_matrix):
        from residudes_qa.cluster.core.clustering import select_optimal_k
        k, scores = select_optimal_k(medium_distance_matrix, k_range=(2, 10))
        fig = create_k_diagnostic_figure(scores, target_name="KTest")
        assert fig is not None


class TestNetworkVisualization:
    @pytest.fixture
    def net_fixtures(self, medium_distance_matrix, sample_labels, sample_names):
        result = analyze_cluster_network(
            medium_distance_matrix, sample_labels,
            structure_names=sample_names, community_method="auto",
        )
        G = build_structure_graph(medium_distance_matrix, structure_names=sample_names)
        scan = resolution_scan(G, n_points=5)
        trans = compute_cluster_transition_matrix(medium_distance_matrix, sample_labels)
        return {
            "dist": medium_distance_matrix, "labels": sample_labels,
            "names": sample_names, "result": result,
            "G": G, "scan": scan, "trans": trans,
        }

    def test_network_graph(self, net_fixtures, tmp_output):
        nf = net_fixtures
        fig = plot_network_graph(
            nf["dist"], nf["labels"],
            communities=nf["result"].communities,
            hub_indices=nf["result"].hub_indices,
            degree_centrality=nf["result"].degree_centrality,
            output_path=tmp_output / "graph.png",
        )
        assert fig is not None
        assert (tmp_output / "graph.png").exists()

    def test_community_comparison(self, net_fixtures, tmp_output):
        nf = net_fixtures
        fig = plot_community_comparison(
            nf["labels"], nf["result"].communities,
            output_path=tmp_output / "comm.png",
        )
        assert fig is not None

    def test_community_comparison_none(self, net_fixtures, tmp_output):
        """communities=None should not crash."""
        nf = net_fixtures
        fig = plot_community_comparison(
            nf["labels"], None,
            output_path=tmp_output / "comm_none.png",
        )
        assert fig is not None

    def test_transition_heatmap(self, net_fixtures, tmp_output):
        fig = plot_transition_heatmap(
            net_fixtures["trans"],
            output_path=tmp_output / "trans.png",
        )
        assert fig is not None

    def test_resolution_scan_plot(self, net_fixtures, tmp_output):
        fig = plot_resolution_scan(
            net_fixtures["scan"],
            output_path=tmp_output / "scan.png",
        )
        assert fig is not None

    def test_network_report(self, net_fixtures, tmp_output):
        nf = net_fixtures
        fig = create_network_report(
            nf["dist"], nf["labels"], nf["result"],
            transition_matrix=nf["trans"],
            scan_results=nf["scan"],
            target_name="ReportTest",
            output_dir=str(tmp_output),
        )
        assert fig is not None

    def test_network_report_minimal(self, net_fixtures, tmp_output):
        """Report with no transition/scan data."""
        nf = net_fixtures
        fig = create_network_report(
            nf["dist"], nf["labels"], nf["result"],
            target_name="Minimal",
            output_dir=str(tmp_output),
        )
        assert fig is not None


class TestSummaryTable:
    def test_cluster_summary(self, medium_distance_matrix, sample_labels, sample_names, tmp_output):
        fig, df = create_cluster_summary_table(
            sample_labels,
            similarity_matrices={"tm": 1 - medium_distance_matrix},
            structure_names=sample_names,
            output_dir=str(tmp_output),
            target_name="SumTest",
        )
        assert df is not None
        assert len(df) == len(np.unique(sample_labels))

    def test_medoid_summary(self, sample_labels, sample_names, tmp_output):
        fig, df = create_medoid_summary_table(
            medoid_indices=[0, 10, 20],
            structure_names=sample_names,
            labels=sample_labels,
            output_path=tmp_output / "med.png",
        )
        assert df is not None


class TestHeatmaps:
    def test_distance_heatmap(self, medium_distance_matrix, sample_labels, tmp_output):
        fig = plot_distance_heatmap(
            medium_distance_matrix, labels=sample_labels,
            output_path=tmp_output / "heat.png",
        )
        assert fig is not None

    def test_multi_metric(self, medium_distance_matrix, sample_labels, tmp_output):
        fig = plot_multi_metric_heatmaps(
            {"tm": medium_distance_matrix, "drmsd": medium_distance_matrix * 2},
            labels=sample_labels,
            output_path=tmp_output / "multi.png",
        )
        assert fig is not None

    def test_inter_cluster(self, medium_distance_matrix, sample_labels, tmp_output):
        fig = plot_inter_cluster_heatmap(
            medium_distance_matrix, sample_labels,
            output_path=tmp_output / "inter.png",
        )
        assert fig is not None


class TestEmbeddings:
    def test_tsne(self, medium_distance_matrix):
        emb = compute_embedding(medium_distance_matrix, method="tsne")
        assert emb.shape == (50, 2)

    def test_umap(self, medium_distance_matrix):
        emb = compute_embedding(medium_distance_matrix, method="umap")
        assert emb.shape == (50, 2)
