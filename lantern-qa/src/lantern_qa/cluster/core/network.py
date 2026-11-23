"""
Network Analysis Module for Protein Structure Clustering

Builds a graph from distance matrices and performs:
- Cluster-cluster connectivity analysis
- Community detection (Louvain/Leiden)
- Network centrality metrics
- Inter-cluster transition analysis

Usage:
    from lantern_qa.cluster.core.network import build_structure_graph, analyze_cluster_network

    G = build_structure_graph(distance_matrix, threshold=0.3)
    stats = analyze_cluster_network(G, labels)
"""

import numpy as np
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field


@dataclass
class NetworkAnalysisResult:
    """Container for network analysis results."""
    n_nodes: int
    n_edges: int
    density: float
    # Cluster-level
    inter_cluster_matrix: np.ndarray  # mean distance between clusters
    cluster_connectivity: Dict[int, Dict[int, float]]  # edges between clusters
    # Node-level
    degree_centrality: np.ndarray
    betweenness_centrality: Optional[np.ndarray]
    hub_indices: List[int]  # high-centrality nodes (bridge structures)
    # Community detection
    communities: Optional[np.ndarray]
    modularity: Optional[float]
    community_method: str = "unknown"
    resolution: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


def build_structure_graph(
    distance_matrix: np.ndarray,
    threshold: Optional[float] = None,
    k_neighbors: Optional[int] = None,
    structure_names: Optional[List[str]] = None,
) -> Any:
    """
    Build a graph from a distance matrix.

    Three modes (in priority order):
    - k_neighbors: connect each node to its k nearest neighbors (mutual KNN)
    - threshold: connect nodes with distance < threshold
    - auto (neither specified): use KNN with adaptive k = max(5, ceil(2*log2(n)))
      This ensures a connected graph for typical protein structure datasets.

    Args:
        distance_matrix: N×N symmetric distance matrix
        threshold: Distance cutoff for edges
        k_neighbors: Number of nearest neighbors
        structure_names: Optional node labels

    Returns:
        networkx.Graph with edge weights = similarity (1 - distance)
    """
    import networkx as nx

    n = len(distance_matrix)
    G = nx.Graph()

    # Add nodes
    for i in range(n):
        name = structure_names[i] if structure_names else f"s{i}"
        G.add_node(i, name=name)

    if threshold is None and k_neighbors is None:
        # Auto: adaptive KNN that scales with dataset size
        # k = max(5, ceil(2*log2(n))) ensures ~connected graph
        k_neighbors = max(5, int(np.ceil(2 * np.log2(max(n, 2)))))
        k_neighbors = min(k_neighbors, n - 1)

    if k_neighbors is not None:
        # KNN graph (symmetric: if i→j or j→i, add edge)
        for i in range(n):
            dists = distance_matrix[i].copy()
            dists[i] = np.inf
            neighbors = np.argsort(dists)[:k_neighbors]
            for j in neighbors:
                similarity = max(0, 1.0 - distance_matrix[i, j])
                G.add_edge(i, j, weight=similarity, distance=distance_matrix[i, j])
    else:
        # Threshold graph (vectorized edge discovery)
        rows, cols = np.where(np.triu(distance_matrix < threshold, k=1))
        for i, j in zip(rows, cols):
            similarity = max(0.0, 1.0 - distance_matrix[i, j])
            G.add_edge(int(i), int(j), weight=similarity, distance=float(distance_matrix[i, j]))

    return G


def analyze_cluster_network(
    distance_matrix: np.ndarray,
    labels: np.ndarray,
    structure_names: Optional[List[str]] = None,
    compute_betweenness: bool = False,
    n_hubs: int = 10,
    community_method: str = "auto",
    resolution: Optional[float] = None,
) -> NetworkAnalysisResult:
    """
    Analyze network properties of clustered structures.

    Args:
        distance_matrix: N*N distance matrix
        labels: Cluster labels for each structure
        structure_names: Optional structure names
        compute_betweenness: Whether to compute betweenness centrality (slow for large N)
        n_hubs: Number of hub structures to identify
        community_method: "auto" (Leiden with Louvain fallback), "leiden", or "louvain"
        resolution: Leiden CPM resolution parameter (None = modularity objective)

    Returns:
        NetworkAnalysisResult with inter-cluster metrics and node centralities
    """
    import networkx as nx

    n = len(distance_matrix)
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    # Build graph
    G = build_structure_graph(distance_matrix, structure_names=structure_names)

    # --- Inter-cluster distance matrix ---
    inter_cluster = np.zeros((n_clusters, n_clusters))
    for i, li in enumerate(unique_labels):
        for j, lj in enumerate(unique_labels):
            if i == j:
                # Intra-cluster mean distance
                idx = np.where(labels == li)[0]
                if len(idx) > 1:
                    sub = distance_matrix[np.ix_(idx, idx)]
                    inter_cluster[i, j] = np.mean(sub[np.triu_indices(len(idx), k=1)])
            else:
                idx_i = np.where(labels == li)[0]
                idx_j = np.where(labels == lj)[0]
                inter_cluster[i, j] = np.mean(distance_matrix[np.ix_(idx_i, idx_j)])

    # --- Cluster connectivity (edge count between clusters) ---
    cluster_connectivity = {int(l): {} for l in unique_labels}
    for u, v in G.edges():
        lu, lv = labels[u], labels[v]
        if lu != lv:
            key_u, key_v = int(lu), int(lv)
            cluster_connectivity[key_u][key_v] = cluster_connectivity[key_u].get(key_v, 0) + 1
            cluster_connectivity[key_v][key_u] = cluster_connectivity[key_v].get(key_u, 0) + 1

    # --- Degree centrality ---
    degree_cent = nx.degree_centrality(G)
    degree_array = np.array([degree_cent.get(i, 0) for i in range(n)])

    # --- Betweenness centrality (optional, expensive) ---
    betweenness_array = None
    if compute_betweenness and n < 5000:
        between_cent = nx.betweenness_centrality(G, weight='distance')
        betweenness_array = np.array([between_cent.get(i, 0) for i in range(n)])

    # --- Hub structures (high degree, connecting different clusters) ---
    # Score: degree * number of distinct cluster neighbors
    hub_scores = np.zeros(n)
    for i in range(n):
        neighbors = list(G.neighbors(i))
        if not neighbors:
            continue
        neighbor_clusters = set(labels[j] for j in neighbors)
        hub_scores[i] = degree_array[i] * len(neighbor_clusters)

    hub_indices = np.argsort(hub_scores)[::-1][:n_hubs].tolist()

    # --- Community detection (Leiden default, Louvain fallback) ---
    communities, modularity_val, method_used = _detect_communities(
        G, method=community_method, resolution=resolution
    )

    return NetworkAnalysisResult(
        n_nodes=G.number_of_nodes(),
        n_edges=G.number_of_edges(),
        density=nx.density(G),
        inter_cluster_matrix=inter_cluster,
        cluster_connectivity=cluster_connectivity,
        degree_centrality=degree_array,
        betweenness_centrality=betweenness_array,
        hub_indices=hub_indices,
        communities=communities,
        modularity=modularity_val,
        community_method=method_used,
        resolution=resolution,
        metadata={
            "n_clusters": n_clusters,
            "n_hubs": n_hubs,
            "graph_components": nx.number_connected_components(G),
        }
    )


def compute_cluster_transition_matrix(
    distance_matrix: np.ndarray,
    labels: np.ndarray,
    sigma: Optional[float] = None,
) -> np.ndarray:
    """
    Compute transition probability matrix between clusters.

    Uses Gaussian kernel on inter-cluster distances:
        P(i→j) = exp(-d²/2σ²) / Σ_k exp(-d²/2σ²)

    Args:
        distance_matrix: N×N distance matrix
        labels: Cluster labels
        sigma: Gaussian kernel width (default: auto from data)

    Returns:
        K×K transition probability matrix
    """
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    # Inter-cluster mean distances
    inter_dist = np.zeros((n_clusters, n_clusters))
    for i, li in enumerate(unique_labels):
        for j, lj in enumerate(unique_labels):
            idx_i = np.where(labels == li)[0]
            idx_j = np.where(labels == lj)[0]
            if i == j:
                if len(idx_i) > 1:
                    sub = distance_matrix[np.ix_(idx_i, idx_i)]
                    inter_dist[i, j] = np.mean(sub[np.triu_indices(len(idx_i), k=1)])
            else:
                inter_dist[i, j] = np.mean(distance_matrix[np.ix_(idx_i, idx_j)])

    if sigma is None:
        # Auto: median of off-diagonal distances
        off_diag = inter_dist[np.triu_indices(n_clusters, k=1)]
        sigma = np.median(off_diag) if len(off_diag) > 0 else 1.0

    # Gaussian kernel
    kernel = np.exp(-inter_dist ** 2 / (2 * sigma ** 2))
    np.fill_diagonal(kernel, 0)

    # Normalize rows to get transition probabilities
    row_sums = kernel.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    transition = kernel / row_sums

    return transition


def _detect_communities(
    G: Any,
    method: str = "auto",
    resolution: Optional[float] = None,
) -> Tuple[Optional[np.ndarray], Optional[float], str]:
    """
    Detect communities using Leiden (preferred) or Louvain (fallback).

    Args:
        G: networkx Graph
        method: "leiden", "louvain", or "auto" (tries Leiden first)
        resolution: Resolution parameter for Leiden CPM quality function.
                    If None, uses modularity objective.

    Returns:
        Tuple of (community_labels, modularity_score, method_used)
    """
    if method == "auto":
        try:
            comms, mod = _leiden_communities(G, resolution)
            return comms, mod, "leiden"
        except ImportError:
            comms, mod = _louvain_communities(G)
            return comms, mod, "louvain"
    elif method == "leiden":
        comms, mod = _leiden_communities(G, resolution)
        return comms, mod, "leiden"
    elif method == "louvain":
        comms, mod = _louvain_communities(G)
        return comms, mod, "louvain"
    else:
        raise ValueError(f"Unknown community method: {method}")


def _leiden_communities(
    G: Any,
    resolution: Optional[float] = None,
    seed: int = 42,
) -> Tuple[Optional[np.ndarray], Optional[float]]:
    """
    Leiden community detection via leidenalg + igraph.

    Guarantees well-connected communities (Traag et al. 2019).
    Handles non-contiguous node IDs by mapping to 0..n-1.

    ``seed`` fixes the RNG so a single call is reproducible. Leiden only finds
    a *local* optimum of modularity, and the optimum reached depends on the
    seed — vary it (see ``multiseed_communities``) to probe robustness.
    """
    import igraph as ig
    import leidenalg

    nodes = sorted(G.nodes())
    n = len(nodes)
    if n == 0:
        return None, None

    # Map networkx node IDs → contiguous igraph indices
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    edges = [(node_to_idx[u], node_to_idx[v]) for u, v in G.edges()]
    weights = [G[u][v].get('weight', 1.0) for u, v in G.edges()]

    ig_graph = ig.Graph(n=n, edges=edges, directed=False)
    ig_graph.es['weight'] = weights

    # A fixed seed makes one Leiden partition reproducible. Without an explicit
    # seed leidenalg uses non-deterministic node ordering, which can change
    # n_communities by ±1 across runs on the same data.
    if resolution is not None:
        partition = leidenalg.find_partition(
            ig_graph,
            leidenalg.CPMVertexPartition,
            weights='weight',
            resolution_parameter=resolution,
            seed=seed,
        )
    else:
        partition = leidenalg.find_partition(
            ig_graph,
            leidenalg.ModularityVertexPartition,
            weights='weight',
            seed=seed,
        )

    # Map back to original node IDs
    membership = np.array(partition.membership)
    # Build full-size array (assumes max node ID < some reasonable bound)
    max_node = max(nodes) if nodes else 0
    communities = np.full(max_node + 1, -1, dtype=int)
    for node, idx in node_to_idx.items():
        communities[node] = membership[idx]

    modularity = partition.modularity
    return communities, modularity


def _louvain_communities(
    G: Any,
) -> Tuple[Optional[np.ndarray], Optional[float]]:
    """Louvain fallback using networkx."""
    import networkx as nx

    n = G.number_of_nodes()
    try:
        community_map = nx.community.louvain_communities(G, weight='weight', seed=42)
        communities = np.zeros(n, dtype=int)
        for comm_id, members in enumerate(community_map):
            for m in members:
                communities[m] = comm_id
        modularity = nx.community.modularity(G, community_map, weight='weight')
        return communities, modularity
    except Exception:
        return None, None


def resolution_scan(
    G: Any,
    resolutions: Optional[List[float]] = None,
    n_points: int = 20,
) -> Dict[str, Any]:
    """
    Scan Leiden resolutions to find optimal community granularity.

    Uses CPM (Constant Potts Model) quality function at varying resolutions.

    Args:
        G: networkx Graph
        resolutions: List of resolutions to try.
            If None, auto-generates log-spaced from 0.01 to 2.0
        n_points: Number of resolution points if auto-generating

    Returns:
        Dict with resolution values, n_communities, modularity scores, and
        suggested_resolution (resolution with highest modularity)
    """
    if resolutions is None:
        resolutions = np.logspace(-2, np.log10(2.0), n_points).tolist()

    results = []
    for res in resolutions:
        try:
            communities, modularity = _leiden_communities(G, resolution=res)
            if communities is not None:
                n_comm = len(np.unique(communities))
                results.append({
                    'resolution': res,
                    'n_communities': n_comm,
                    'modularity': modularity,
                })
        except Exception:
            continue

    if not results:
        return {'resolutions': [], 'n_communities': [], 'modularity': [], 'results': []}

    # Suggest resolution: highest resolution that still has modularity >= 0.5
    # This gives meaningful communities without over-splitting
    valid = [r for r in results if r['modularity'] is not None and r['modularity'] >= 0.5]
    if valid:
        suggested = valid[-1]['resolution']  # highest res with mod >= 0.5
    else:
        best = max(results, key=lambda r: r['modularity'] if r['modularity'] is not None else -1)
        suggested = best['resolution']

    return {
        'resolutions': [r['resolution'] for r in results],
        'n_communities': [r['n_communities'] for r in results],
        'modularity': [r['modularity'] for r in results],
        'suggested_resolution': suggested,
        'results': results,
    }


# ---------------------------------------------------------------------------
# Robustness: bootstrap stability + consensus communities
# (a single Leiden run is not enough — these quantify how reproducible the
#  community structure is, and fuse multiple partitions into a consensus.)
# ---------------------------------------------------------------------------


def _partition_on_nodes(G: Any, nodes: List[int], resolution: Optional[float]):
    """Detect communities on the induced subgraph over ``nodes``.

    Returns a dict ``{node: community_label}`` for the given nodes (-1 dropped).
    """
    sub = G.subgraph(nodes)
    comms, _ = _leiden_communities(sub, resolution=resolution)
    if comms is None:
        return {}
    return {n: int(comms[n]) for n in nodes if n < len(comms) and comms[n] >= 0}


def bootstrap_community_stability(
    G: Any,
    *,
    n_boot: int = 30,
    sample_frac: float = 0.8,
    resolution: Optional[float] = None,
    seed: int = 0,
) -> Dict[str, Any]:
    """Quantify how reproducible the community structure is.

    Repeatedly subsamples ``sample_frac`` of the nodes, re-detects communities
    on the induced subgraph, and measures agreement with the full-graph
    partition via the Adjusted Rand Index (ARI). Also accumulates a
    co-association matrix (how often each pair of nodes lands in the same
    community, among bootstraps that sampled both).

    Returns ``{mean_ari, std_ari, ari_values, coassoc, node_order,
    per_node_stability}``. High mean ARI (→1) + tight co-association blocks =
    robust communities; low/variable ARI = the single-run partition is fragile.
    """
    from sklearn.metrics import adjusted_rand_score

    rng = np.random.default_rng(seed)
    nodes = sorted(G.nodes())
    n = len(nodes)
    if n < 3:
        return {"mean_ari": float("nan"), "std_ari": float("nan"),
                "ari_values": [], "coassoc": np.zeros((n, n)), "node_order": nodes}

    full = _partition_on_nodes(G, nodes, resolution)

    idx_of = {node: i for i, node in enumerate(nodes)}
    co_same = np.zeros((n, n), dtype=np.float64)
    co_seen = np.zeros((n, n), dtype=np.float64)
    ari_values = []

    k = max(3, int(round(sample_frac * n)))
    for _ in range(n_boot):
        samp = sorted(rng.choice(nodes, size=k, replace=False).tolist())
        part = _partition_on_nodes(G, samp, resolution)
        common = [s for s in samp if s in part and s in full]
        if len(common) >= 3:
            ari_values.append(float(adjusted_rand_score(
                [full[s] for s in common], [part[s] for s in common])))
        # co-association over sampled pairs
        labelled = [s for s in samp if s in part]
        for a_i in range(len(labelled)):
            for b_i in range(a_i + 1, len(labelled)):
                u, v = labelled[a_i], labelled[b_i]
                iu, iv = idx_of[u], idx_of[v]
                co_seen[iu, iv] += 1
                co_seen[iv, iu] += 1
                if part[u] == part[v]:
                    co_same[iu, iv] += 1
                    co_same[iv, iu] += 1

    with np.errstate(invalid="ignore", divide="ignore"):
        coassoc = np.where(co_seen > 0, co_same / co_seen, 0.0)
    np.fill_diagonal(coassoc, 1.0)
    # per-node stability: mean co-association with its full-partition community-mates
    per_node = np.full(n, np.nan)
    for node in nodes:
        if node not in full:
            continue
        mates = [m for m in full if m != node and full[m] == full[node]]
        if mates:
            per_node[idx_of[node]] = float(np.mean([coassoc[idx_of[node], idx_of[m]] for m in mates]))

    return {
        "mean_ari": float(np.mean(ari_values)) if ari_values else float("nan"),
        "std_ari": float(np.std(ari_values)) if ari_values else float("nan"),
        "ari_values": ari_values,
        "coassoc": coassoc,
        "node_order": nodes,
        "per_node_stability": per_node,
    }


def consensus_communities(
    G: Any,
    *,
    resolutions: Optional[List[float]] = None,
    n_points: int = 12,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Fuse Leiden partitions across resolutions into one consensus partition.

    Runs Leiden at several resolutions, builds a co-association matrix (how
    often each node pair shares a community across resolutions), then derives a
    consensus partition by single-linkage agglomeration: pairs co-clustered in
    ≥ ``threshold`` of resolutions stay together.

    Returns ``{consensus_labels, coassoc, node_order, n_consensus, resolutions}``.
    """
    if resolutions is None:
        resolutions = np.logspace(-1.5, np.log10(2.0), n_points).tolist()
    nodes = sorted(G.nodes())
    n = len(nodes)
    idx_of = {node: i for i, node in enumerate(nodes)}
    co = np.zeros((n, n), dtype=np.float64)
    n_used = 0
    for res in resolutions:
        part = _partition_on_nodes(G, nodes, res)
        if not part:
            continue
        n_used += 1
        labelled = [s for s in nodes if s in part]
        for a_i in range(len(labelled)):
            for b_i in range(a_i + 1, len(labelled)):
                u, v = labelled[a_i], labelled[b_i]
                if part[u] == part[v]:
                    co[idx_of[u], idx_of[v]] += 1
                    co[idx_of[v], idx_of[u]] += 1
    if n_used == 0:
        return {"consensus_labels": np.zeros(n, dtype=int), "coassoc": co,
                "node_order": nodes, "n_consensus": 1, "resolutions": resolutions}
    coassoc = co / n_used
    np.fill_diagonal(coassoc, 1.0)

    # Consensus partition: cut the co-association at `threshold`.
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    dist = 1.0 - coassoc
    np.fill_diagonal(dist, 0.0)
    dist = (dist + dist.T) / 2.0
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="average")
    labels = fcluster(Z, t=1.0 - threshold, criterion="distance")

    return {
        "consensus_labels": labels,
        "coassoc": coassoc,
        "node_order": nodes,
        "n_consensus": int(len(np.unique(labels))),
        "resolutions": resolutions,
    }


def multiseed_communities(
    G: Any,
    *,
    n_seeds: int = 20,
    resolution: Optional[float] = None,
    base_seed: int = 0,
) -> Dict[str, Any]:
    """Run Leiden under ``n_seeds`` different RNG seeds on the *full* graph.

    Leiden is a greedy modularity heuristic, so it only ever reaches a *local*
    optimum, and which one depends on the seed (node-visit order). Running many
    seeds lets us (a) keep the highest-modularity partition found, and
    (b) measure how seed-stable the community structure is — pairwise ARI
    between seed partitions and a co-association matrix (how often each node
    pair lands together across seeds).

    Unlike ``bootstrap_community_stability`` (which subsamples nodes), every
    seed sees all nodes, so this isolates the optimiser's seed sensitivity.

    Returns ``{best_labels, best_modularity, best_seed, modularities,
    n_communities_per_seed, mean_ari, std_ari, ari_pairs, coassoc, node_order,
    n_seeds}``. mean ARI → 1.0 = the partition is seed-robust; lower/variable =
    the single-run community split is an artifact of the seed.
    """
    from sklearn.metrics import adjusted_rand_score

    nodes = sorted(G.nodes())
    n = len(nodes)
    idx_of = {node: i for i, node in enumerate(nodes)}
    if n < 3:
        return {"best_labels": np.zeros(n, dtype=int), "best_modularity": float("nan"),
                "best_seed": base_seed, "modularities": [], "n_communities_per_seed": [],
                "mean_ari": float("nan"), "std_ari": float("nan"), "ari_pairs": [],
                "coassoc": np.zeros((n, n)), "node_order": nodes, "n_seeds": 0}

    partitions = []           # list of {node: label}
    modularities = []
    n_comms = []
    best_mod, best_seed, best_full = -np.inf, base_seed, None

    for s in range(base_seed, base_seed + n_seeds):
        comms, mod = _leiden_communities(G, resolution=resolution, seed=s)
        if comms is None:
            continue
        part = {node: int(comms[node]) for node in nodes
                if node < len(comms) and comms[node] >= 0}
        partitions.append(part)
        modularities.append(float(mod) if mod is not None else float("nan"))
        n_comms.append(int(len(set(part.values()))))
        if mod is not None and mod > best_mod:
            best_mod, best_seed = float(mod), s
            best_full = comms

    # Pairwise ARI between every pair of seed partitions (seed-agreement).
    ari_pairs = []
    common_nodes = nodes
    for i in range(len(partitions)):
        for j in range(i + 1, len(partitions)):
            a = [partitions[i].get(x, -1) for x in common_nodes]
            b = [partitions[j].get(x, -1) for x in common_nodes]
            ari_pairs.append(float(adjusted_rand_score(a, b)))

    # Co-association across seeds.
    co = np.zeros((n, n), dtype=np.float64)
    for part in partitions:
        labelled = [x for x in nodes if x in part]
        for a_i in range(len(labelled)):
            for b_i in range(a_i + 1, len(labelled)):
                u, v = labelled[a_i], labelled[b_i]
                if part[u] == part[v]:
                    co[idx_of[u], idx_of[v]] += 1
                    co[idx_of[v], idx_of[u]] += 1
    if partitions:
        co /= len(partitions)
    np.fill_diagonal(co, 1.0)

    # best_labels as a full-size array over node_order.
    best_labels = np.zeros(n, dtype=int)
    if best_full is not None:
        for node in nodes:
            best_labels[idx_of[node]] = int(best_full[node]) if node < len(best_full) else -1

    return {
        "best_labels": best_labels,
        "best_modularity": best_mod if np.isfinite(best_mod) else float("nan"),
        "best_seed": best_seed,
        "modularities": modularities,
        "n_communities_per_seed": n_comms,
        "mean_ari": float(np.mean(ari_pairs)) if ari_pairs else float("nan"),
        "std_ari": float(np.std(ari_pairs)) if ari_pairs else float("nan"),
        "ari_pairs": ari_pairs,
        "coassoc": co,
        "node_order": nodes,
        "n_seeds": len(partitions),
    }
