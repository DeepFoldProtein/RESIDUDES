"""Tests for cluster dendrogram / clustermap (residudes_qa.cluster.visualization.dendrogram)."""
import tempfile
from pathlib import Path

import numpy as np

from residudes_qa.cluster.visualization.dendrogram import (
    compute_inter_cluster_distance, compute_cluster_linkage,
    plot_cluster_dendrogram, plot_cluster_clustermap,
)


def _clustered_matrix(n_per=10, k=4, seed=0):
    rng = np.random.default_rng(seed)
    n = n_per * k
    d = np.ones((n, n)) * 0.8
    for c in range(k):
        s = c * n_per
        d[s:s + n_per, s:s + n_per] = rng.random((n_per, n_per)) * 0.1
    d = (d + d.T) / 2
    np.fill_diagonal(d, 0.0)
    labels = np.repeat(np.arange(k), n_per)
    return d, labels


def test_inter_cluster_distance_shape_and_symmetry():
    d, labels = _clustered_matrix()
    inter, uniq, sizes = compute_inter_cluster_distance(d, labels)
    assert inter.shape == (4, 4)
    assert np.allclose(inter, inter.T)                       # symmetric
    assert (np.diag(inter) < inter[np.triu_indices(4, 1)].mean()).all()  # intra < inter
    assert list(sizes) == [10, 10, 10, 10]


def test_cluster_linkage_k_ge_2():
    d, labels = _clustered_matrix()
    Z, inter, uniq, sizes = compute_cluster_linkage(d, labels)
    assert Z is not None and Z.shape == (3, 4)               # K-1 merges


def test_cluster_linkage_single_cluster_returns_none():
    d, _ = _clustered_matrix()
    labels = np.zeros(len(d), dtype=int)                     # one cluster
    Z, inter, uniq, sizes = compute_cluster_linkage(d, labels)
    assert Z is None


def test_plots_render():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d, labels = _clustered_matrix()
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        f1 = plot_cluster_dendrogram(d, labels, output_path=t / "dn.png")
        f2 = plot_cluster_clustermap(d, labels, output_path=t / "cm.png")
        assert (t / "dn.png").exists() and (t / "cm.png").exists()
        plt.close(f1)
        plt.close(f2)
        # < 2 clusters -> None
        assert plot_cluster_clustermap(d, np.zeros(len(d), int)) is None


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
