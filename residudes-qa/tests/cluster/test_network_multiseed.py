"""Tests for multi-seed Leiden (residudes_qa.cluster.core.network.multiseed_communities)."""
import numpy as np

from residudes_qa.cluster.core.network import build_structure_graph, multiseed_communities


def _clustered_matrix(n_per=15, k=4, seed=0):
    rng = np.random.default_rng(seed)
    n = n_per * k
    d = np.ones((n, n)) * 0.85
    for c in range(k):
        s = c * n_per
        d[s:s + n_per, s:s + n_per] = rng.random((n_per, n_per)) * 0.2
    d = (d + d.T) / 2
    np.fill_diagonal(d, 0.0)
    return d


def test_multiseed_keys_and_shapes():
    d = _clustered_matrix()
    G = build_structure_graph(d, structure_names=[f"m{i}" for i in range(len(d))])
    ms = multiseed_communities(G, n_seeds=8)
    for key in ("best_labels", "best_modularity", "best_seed", "modularities",
                "n_communities_per_seed", "mean_ari", "std_ari", "coassoc",
                "node_order", "n_seeds"):
        assert key in ms
    assert ms["n_seeds"] <= 8 and ms["n_seeds"] > 0
    assert len(ms["best_labels"]) == len(d)
    assert ms["coassoc"].shape == (len(d), len(d))


def test_multiseed_recovers_planted_clusters():
    d = _clustered_matrix(n_per=15, k=4)
    G = build_structure_graph(d, structure_names=[f"m{i}" for i in range(len(d))])
    ms = multiseed_communities(G, n_seeds=10)
    # clean planted structure -> ~4 communities, seed-robust (mean ARI high)
    assert 3 <= len(np.unique(ms["best_labels"])) <= 6
    assert ms["mean_ari"] > 0.8
    assert ms["best_modularity"] > 0.4


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
