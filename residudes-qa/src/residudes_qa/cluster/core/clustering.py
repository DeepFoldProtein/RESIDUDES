"""
Clustering algorithms module

Supports:
- Fuzzy C-means clustering (primary, via skfuzzy)
- GMM (Gaussian Mixture Model) clustering with covariance ellipses
- K-means clustering (fallback)
- Hierarchical/Agglomerative clustering
- Automatic K selection (Silhouette, Davies-Bouldin, Calinski-Harabasz, FPC)
"""

import numpy as np
from typing import Optional, Literal, Tuple, Dict, Any, List
from dataclasses import dataclass

ClusterMethod = Literal["fuzzy", "kmeans", "hierarchical", "gmm", "leiden"]


@dataclass
class ClusteringResult:
    """Container for clustering results."""
    labels: np.ndarray
    n_clusters: int
    method: str
    metric: str
    medoid_indices: List[int]
    medoid_names: Optional[List[str]]
    cluster_sizes: List[int]
    intra_cluster_distances: List[float]
    silhouette_score: Optional[float]
    membership: Optional[np.ndarray]  # Fuzzy membership matrix
    fpc: Optional[float]  # Fuzzy Partition Coefficient
    embedding: Optional[np.ndarray]  # UMAP embedding used
    metadata: Dict[str, Any]
    # GMM specific fields
    gmm_means: Optional[np.ndarray] = None  # GMM cluster centers
    gmm_covariances: Optional[np.ndarray] = None  # GMM covariance matrices for ellipses


def run_clustering(
    distance_matrix: np.ndarray,
    n_clusters: Optional[int] = None,
    method: ClusterMethod = "fuzzy",
    k_range: Tuple[int, int] = (3, 20),
    structure_names: Optional[List[str]] = None,
    fuzziness: float = 2.0,

) -> ClusteringResult:
    """
    Run clustering on a distance matrix.

    Args:
        distance_matrix: Precomputed distance matrix (N x N)
        n_clusters: Number of clusters (auto-select if None)
        method: Clustering method ('fuzzy', 'kmeans', or 'hierarchical')
        k_range: Range for automatic K selection (min, max)
        structure_names: Optional list of structure names
        fuzziness: Fuzziness parameter for fuzzy clustering (m > 1)

    Returns:
        ClusteringResult object with labels, medoids, and statistics

    Example:
        >>> result = run_clustering(
        ...     distance_matrix=dist_matrix,
        ...     n_clusters=5,
        ...     method="fuzzy"
        ... )
        >>> print(result.labels)
        >>> print(result.medoid_names)
    """
    n = len(distance_matrix)

    # Compute UMAP embedding for clustering
    embedding = _compute_umap_embedding(distance_matrix)

    # Auto-select K if not provided
    if n_clusters is None:
        n_clusters, _ = select_optimal_k(distance_matrix, k_range, embedding=embedding)

    # Run clustering based on method
    gmm_means = None
    gmm_covariances = None

    if method == "fuzzy":
        labels, membership, fpc = _fuzzy_cmeans_clustering(
            embedding, n_clusters, fuzziness
        )
    elif method == "kmeans":
        labels, membership, fpc = _kmeans_clustering(embedding, n_clusters)
    elif method == "hierarchical":
        labels, membership, fpc = _hierarchical_clustering(distance_matrix, n_clusters)
    elif method == "gmm":
        labels, membership, fpc, gmm_means, gmm_covariances = _gmm_clustering(
            embedding, n_clusters
        )
    elif method == "leiden":
        labels, membership, fpc, n_clusters = _leiden_clustering(
            distance_matrix, n_clusters
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    # Find medoids using original distance matrix
    medoid_indices = []
    cluster_sizes = []
    intra_distances = []

    for cluster_id in range(n_clusters):
        mask = labels == cluster_id
        cluster_indices = np.where(mask)[0]
        cluster_sizes.append(len(cluster_indices))

        if len(cluster_indices) > 0:
            # Find medoid (point with minimum average distance to others in cluster)
            medoid_idx = _find_medoid(distance_matrix, cluster_indices)
            medoid_indices.append(medoid_idx)

            # Compute intra-cluster distance
            if len(cluster_indices) > 1:
                cluster_dists = distance_matrix[np.ix_(cluster_indices, cluster_indices)]
                triu_idx = np.triu_indices(len(cluster_indices), k=1)
                intra_dist = cluster_dists[triu_idx].mean()
            else:
                intra_dist = 0.0
            intra_distances.append(intra_dist)
        else:
            medoid_indices.append(-1)
            intra_distances.append(0.0)

    # Compute silhouette score
    try:
        from sklearn.metrics import silhouette_score
        sil_score = silhouette_score(distance_matrix, labels, metric='precomputed')
    except Exception:
        sil_score = None

    # Get medoid names
    medoid_names = None
    if structure_names:
        medoid_names = [
            structure_names[i] if i >= 0 else None
            for i in medoid_indices
        ]

    return ClusteringResult(
        labels=labels,
        n_clusters=n_clusters,
        method=method,
        metric="precomputed",
        medoid_indices=medoid_indices,
        medoid_names=medoid_names,
        cluster_sizes=cluster_sizes,
        intra_cluster_distances=intra_distances,
        silhouette_score=sil_score,
        membership=membership,
        fpc=fpc,
        embedding=embedding,
        metadata={
            "k_range": k_range,
            "n_samples": n,
            "fuzziness": fuzziness if method == "fuzzy" else None,
        },
        gmm_means=gmm_means,
        gmm_covariances=gmm_covariances,
    )


def select_optimal_k(
    distance_matrix: np.ndarray,
    k_range: Tuple[int, int] = (3, 20),
    embedding: Optional[np.ndarray] = None,
    cluster_method: ClusterMethod = "hierarchical",
    adaptive: bool = True,
    plateau_window: int = 3,
    plateau_threshold: float = 0.01,
) -> Tuple[int, Dict[str, Any]]:
    """
    Automatically select optimal number of clusters.

    Uses hierarchical clustering (average linkage) directly on the precomputed
    distance matrix, evaluated by Silhouette on the same distance matrix.
    Embedding-based methods (GMM/fuzzy on 2D) are avoided because 2D t-SNE/UMAP
    projections create artificial small clusters -> Silhouette keeps rising to k_max.

    Combines five signals with a balance gate:
      - Penalized Silhouette (35%): elbow detection via cumulative drop penalty
      - Raw Silhouette (20%): cluster quality
      - Delta Silhouette (10%): rate of change / elbow
      - BIC (25%): Bayesian Information Criterion from WCSS
      - Balance (10%): (1 - max_cluster_fraction)^2

    Silhouette-derived signals are gated by a sigmoid balance function that
    sharply penalizes solutions where one cluster dominates (>60% of data).
    This prevents high-silhouette but trivially imbalanced K from winning.

    Args:
        distance_matrix: Precomputed distance matrix (N*N, symmetric, zero diagonal)
        k_range: Range of K values to try (min, max)
        embedding: Ignored (kept for API compatibility)
        cluster_method: Ignored (always hierarchical); kept for API compatibility
        adaptive: Enable adaptive convergence detection (plateau + ascending warning)
        plateau_window: Window size for plateau detection
        plateau_threshold: Threshold for silhouette change to count as plateau

    Returns:
        Tuple of (optimal_k, scores_dict)
    """
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score

    n = len(distance_matrix)
    k_min, k_max_requested = k_range
    # If k_max not specified (None), search up to n-1 (no artificial cap)
    if k_max_requested is None:
        k_max_requested = n - 1
    # Only enforce sklearn hard limit (need at least 2 clusters, at most n-1)
    k_max = min(k_max_requested, n - 1)

    # Early stopping: if silhouette keeps declining for this many consecutive K,
    # further search is unlikely to find a better solution.
    EARLY_STOP_PATIENCE = 30  # stop after 30 consecutive declining silhouette
    declining_count = 0
    best_sil_so_far = -1.0

    results = []
    for k in range(k_min, k_max + 1):
        try:
            hc = AgglomerativeClustering(
                n_clusters=k, metric='precomputed', linkage='average'
            )
            labels = hc.fit_predict(distance_matrix)
            if len(set(labels)) < 2:
                continue
            sil = silhouette_score(distance_matrix, labels, metric='precomputed')
            bic = _compute_hierarchical_bic(distance_matrix, labels, k)
            # Cluster balance: (1 - max_fraction)^2
            _, counts = np.unique(labels, return_counts=True)
            max_frac = counts.max() / counts.sum()
            balance = (1.0 - max_frac) ** 2
            results.append({
                'k': k, 'silhouette': sil, 'bic': bic,
                'balance': balance, 'max_frac': max_frac, 'labels': labels,
            })

            # Early stopping check
            if sil > best_sil_so_far:
                best_sil_so_far = sil
                declining_count = 0
            else:
                declining_count += 1
            if declining_count >= EARLY_STOP_PATIENCE and len(results) >= 20:
                k_max = k  # record where we actually stopped
                break
        except Exception:
            continue

    if not results:
        return k_min, {}

    sil_arr = np.array([r['silhouette'] for r in results])
    bic_arr = np.array([r['bic'] for r in results])
    bal_arr = np.array([r['balance'] for r in results])
    sil_range = sil_arr.max() - sil_arr.min() + 1e-8
    lambda_pen = 0.5
    peak_sil = sil_arr[0]
    cumulative_drop = 0.0
    pen_scores = []
    delta_sil = np.zeros(len(results))
    for i, s in enumerate(sil_arr):
        if i > 0:
            delta_sil[i] = sil_arr[i] - sil_arr[i - 1]
        if s < peak_sil:
            cumulative_drop += peak_sil - s
        else:
            peak_sil = s
            cumulative_drop = 0.0
        pen_scores.append(s - lambda_pen * cumulative_drop / sil_range)

    def _normalize(arr, reverse=False):
        a = np.array(arr, dtype=float)
        if a.max() == a.min():
            return np.ones_like(a) * 0.5
        norm = (a - a.min()) / (a.max() - a.min())
        return 1.0 - norm if reverse else norm

    pen_n  = _normalize(np.array(pen_scores))
    sil_n  = _normalize(sil_arr)
    dsil_n = _normalize(delta_sil)
    bic_n  = _normalize(bic_arr, reverse=True)  # lower BIC = better
    bal_n  = _normalize(bal_arr)  # higher balance = better

    # Combined score: additive base (60%) + BIC (25%) + balance gate (15%)
    # Balance acts as a multiplicative gate on silhouette-derived signals:
    # if max cluster has >70% of data, combined score is scaled down
    for i, r in enumerate(results):
        r['delta_sil']      = delta_sil[i]
        r['pen_score']      = pen_scores[i]
        r['balance']        = bal_arr[i]
        # Soft balance gate: sigmoid around max_frac=0.5
        max_frac = r['max_frac']
        balance_gate = 1.0 / (1.0 + np.exp(10 * (max_frac - 0.6)))  # drops sharply above 60%
        sil_component = 0.35 * pen_n[i] + 0.20 * sil_n[i] + 0.10 * dsil_n[i]
        r['combined_score'] = (
            sil_component * balance_gate + 0.25 * bic_n[i] + 0.10 * bal_n[i]
        )
        r['sil_n']          = float(sil_n[i])
        r['delta_sil_n']    = float(dsil_n[i])
        r['pen_n']          = float(pen_n[i])
        r['bic_n']          = float(bic_n[i])
        r['bal_n']          = float(bal_n[i])
        r['balance_gate']   = float(balance_gate)

    best = max(results, key=lambda x: x['combined_score'])
    optimal_k = best['k']

    # Adaptive convergence detection
    plateau_k = None
    is_ascending = False
    if adaptive and len(results) >= plateau_window + 1:
        plateau_k, is_ascending = _detect_silhouette_plateau(
            sil_arr, [r['k'] for r in results], plateau_window, plateau_threshold
        )

    # BIC optimal k (minimum BIC)
    bic_optimal_k = results[int(np.argmin(bic_arr))]['k']

    scores = {
        "k_values":          [r['k'] for r in results],
        "silhouette_scores": [r['silhouette'] for r in results],
        "delta_sil_scores":  [r['delta_sil'] for r in results],
        "pen_scores":        [r['pen_score'] for r in results],
        "bic_scores":        [r['bic'] for r in results],
        "balance_scores":    [r['balance'] for r in results],
        "combined_scores":   [r['combined_score'] for r in results],
        "optimal_k":         optimal_k,
        "best_silhouette":   best['silhouette'],
        "bic_optimal_k":     bic_optimal_k,
        "plateau_k":         plateau_k,
        "is_ascending_at_boundary": is_ascending,
        "k_max_requested":   k_max_requested,
        "k_max_effective":   k_max,
    }

    return optimal_k, scores


def _compute_hierarchical_bic(
    distance_matrix: np.ndarray,
    labels: np.ndarray,
    k: int,
) -> float:
    """
    BIC for hierarchical clustering using precomputed distances.

    Approximates within-cluster sum of squares (WCSS) from pairwise distances:
        WCSS_c = sum(d_ij^2) / (2 * n_c)  for each cluster c
        BIC = n * log(WCSS_total / n) + k * log(n)

    This is an approximation since we use pairwise distances rather than
    distances to a centroid, but sufficient for detecting the BIC elbow.
    """
    n = len(distance_matrix)
    wcss = 0.0
    for c in range(k):
        mask = labels == c
        n_c = mask.sum()
        if n_c > 1:
            sub = distance_matrix[np.ix_(mask, mask)]
            wcss += np.sum(sub ** 2) / (2 * n_c)
        elif n_c == 1:
            pass  # single-point cluster contributes 0
    wcss = max(wcss, 1e-10)
    return n * np.log(wcss / n) + k * np.log(n)


def _detect_silhouette_plateau(
    sil_arr: np.ndarray,
    k_values: List[int],
    window: int = 3,
    threshold: float = 0.01,
) -> Tuple[Optional[int], bool]:
    """
    Detect if silhouette score has plateaued or is still ascending.

    Args:
        sil_arr: Array of silhouette scores
        k_values: Corresponding K values
        window: Number of consecutive deltas to check
        threshold: Max absolute delta to consider as plateau

    Returns:
        (plateau_k, is_ascending_at_boundary)
        plateau_k: K at which plateau begins (None if not detected)
        is_ascending_at_boundary: True if silhouette still rising at k_max
    """
    if len(sil_arr) < window + 1:
        return None, False

    deltas = np.diff(sil_arr)

    # Scan for plateau: `window` consecutive deltas all below threshold
    for i in range(len(deltas) - window + 1):
        segment = deltas[i:i + window]
        if np.all(np.abs(segment) < threshold):
            return k_values[i], False

    # Check if still ascending at boundary
    is_ascending = deltas[-1] > threshold
    return None, is_ascending


def _compute_umap_embedding(
    distance_matrix: np.ndarray,
    n_components: int = 2,
    n_neighbors: int = 30,
    min_dist: float = 0.1,
    random_state: int = 42,
) -> Optional[np.ndarray]:
    """Compute UMAP embedding from distance matrix."""
    try:
        import umap
    except ImportError:
        return None

    n = len(distance_matrix)
    n_neighbors = min(n_neighbors, n - 1)

    import warnings
    try:
        reducer = umap.UMAP(
            n_components=n_components,
            metric='precomputed',
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=random_state,
        )
        with warnings.catch_warnings():
            # a fixed random_state (reproducible layouts) makes UMAP warn that it runs single-threaded, and the
            # precomputed metric that inverse_transform is unavailable: both are intended
            warnings.filterwarnings("ignore", message=".*(precomputed metric|overridden to 1 by setting random_state).*")
            return reducer.fit_transform(distance_matrix)
    except Exception:
        return None


def _fuzzy_cmeans_clustering(
    embedding: np.ndarray,
    n_clusters: int,
    m: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Fuzzy C-means clustering using skfuzzy.

    Returns:
        Tuple of (labels, membership_matrix, fpc)
    """
    try:
        import skfuzzy as fuzz
    except ImportError:
        # Fallback to kmeans
        labels, _, _ = _kmeans_clustering(embedding, n_clusters)
        return labels, None, 0.5

    # skfuzzy expects data as (features, samples)
    data = embedding.T

    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        data, n_clusters, m, error=0.005, maxiter=1000, seed=42
    )

    # u is (clusters, samples), get hard labels
    labels = np.argmax(u, axis=0)
    membership = u.T  # Convert to (samples, clusters)

    return labels, membership, fpc


def _kmeans_clustering(
    embedding: np.ndarray,
    n_clusters: int,
) -> Tuple[np.ndarray, Optional[np.ndarray], float]:
    """K-means clustering fallback."""
    from sklearn.cluster import KMeans

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embedding)

    return labels, None, 0.5


def _hierarchical_clustering(
    distance_matrix: np.ndarray,
    n_clusters: int,
    linkage_method: str = 'average',
) -> Tuple[np.ndarray, Optional[np.ndarray], float]:
    """Hierarchical clustering using average linkage."""
    from scipy.cluster.hierarchy import linkage, fcluster

    n = len(distance_matrix)
    condensed = distance_matrix[np.triu_indices(n, k=1)]
    Z = linkage(condensed, method=linkage_method)
    labels = fcluster(Z, n_clusters, criterion='maxclust') - 1  # 0-indexed

    return labels, None, 0.5


def _gmm_clustering(
    embedding: np.ndarray,
    n_clusters: int,
    covariance_type: str = 'full',
    n_init: int = 5,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
    """
    Gaussian Mixture Model clustering.

    Returns cluster means and covariances for ellipse visualization.

    Args:
        embedding: 2D embedding coordinates (N x 2)
        n_clusters: Number of clusters
        covariance_type: 'full', 'tied', 'diag', 'spherical'
        n_init: Number of initializations
        random_state: Random seed

    Returns:
        Tuple of (labels, membership/probabilities, silhouette_score,
                  gmm_means, gmm_covariances)
    """
    from sklearn.mixture import GaussianMixture
    from sklearn.metrics import silhouette_score as sil_score

    gmm = GaussianMixture(
        n_components=n_clusters,
        covariance_type=covariance_type,
        n_init=n_init,
        random_state=random_state,
    )

    labels = gmm.fit_predict(embedding)

    # Get soft membership (posterior probabilities)
    membership = gmm.predict_proba(embedding)

    # Compute silhouette score. Falls back to 0.5 (neutral) on degenerate
    # cases like single-cluster or NaN embedding rows.
    try:
        sil = sil_score(embedding, labels)
    except Exception:
        sil = 0.5

    # Extract means and covariances for ellipse drawing
    gmm_means = gmm.means_
    gmm_covariances = gmm.covariances_

    return labels, membership, sil, gmm_means, gmm_covariances


def _find_medoid(distance_matrix: np.ndarray, cluster_indices: np.ndarray) -> int:
    """
    Find medoid: the point with minimum average distance to others in cluster.
    Uses original distance matrix for accuracy.
    """
    if len(cluster_indices) == 0:
        return -1

    if len(cluster_indices) == 1:
        return cluster_indices[0]

    # Extract sub-distance matrix for cluster
    sub_dist = distance_matrix[np.ix_(cluster_indices, cluster_indices)]

    # Find point with minimum average distance
    mean_dists = sub_dist.mean(axis=1)
    medoid_local_idx = np.argmin(mean_dists)

    return cluster_indices[medoid_local_idx]


def _select_k_silhouette_only(
    distance_matrix: np.ndarray,
    k_range: Tuple[int, int],
) -> Tuple[int, Dict[str, Any]]:
    """Fallback K selection using only silhouette score."""
    from sklearn.cluster import KMeans
    from sklearn.manifold import MDS
    from sklearn.metrics import silhouette_score

    n = len(distance_matrix)
    k_min, k_max = k_range
    k_max = min(k_max, n - 1)

    # MDS for coordinates
    mds = MDS(n_components=min(10, n - 1), dissimilarity='precomputed', random_state=42)
    coords = mds.fit_transform(distance_matrix)

    best_k = k_min
    best_sil = -1
    scores = []

    for k in range(k_min, k_max + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(coords)

        try:
            sil = silhouette_score(distance_matrix, labels, metric='precomputed')
            if sil > best_sil:
                best_sil = sil
                best_k = k
            scores.append({'k': k, 'silhouette': sil})
        except Exception:
            continue

    return best_k, {"silhouette_scores": scores, "optimal_k": best_k}


def _leiden_clustering(
    distance_matrix: np.ndarray,
    n_clusters_hint: Optional[int] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], float, int]:
    """
    Leiden community detection as a clustering method.

    Builds a KNN graph from the distance matrix and runs Leiden with
    modularity objective. The number of clusters is determined by the
    algorithm (n_clusters_hint is ignored — Leiden finds natural communities).

    Args:
        distance_matrix: Precomputed distance matrix (N x N)
        n_clusters_hint: Ignored (Leiden determines K automatically)

    Returns:
        Tuple of (labels, membership, fpc_equivalent, actual_n_clusters)
    """
    from residudes_qa.cluster.core.network import build_structure_graph

    n = len(distance_matrix)

    # Build KNN graph
    G = build_structure_graph(distance_matrix)

    # Try Leiden, fallback to Louvain
    try:
        import igraph as ig
        import leidenalg

        nodes = sorted(G.nodes())
        node_to_idx = {node: idx for idx, node in enumerate(nodes)}
        edges = [(node_to_idx[u], node_to_idx[v]) for u, v in G.edges()]
        weights = [G[u][v].get('weight', 1.0) for u, v in G.edges()]

        ig_graph = ig.Graph(n=n, edges=edges, directed=False)
        ig_graph.es['weight'] = weights

        partition = leidenalg.find_partition(
            ig_graph,
            leidenalg.ModularityVertexPartition,
            weights='weight',
        )

        membership = np.array(partition.membership)
        # Map back from igraph index to original node order
        labels = np.zeros(n, dtype=int)
        for node, idx in node_to_idx.items():
            labels[node] = membership[idx]

        modularity = partition.modularity

    except ImportError:
        import networkx as nx
        community_map = nx.community.louvain_communities(G, weight='weight', seed=42)
        labels = np.zeros(n, dtype=int)
        for comm_id, members in enumerate(community_map):
            for m in members:
                labels[m] = comm_id
        modularity = nx.community.modularity(G, community_map, weight='weight')

    actual_k = len(np.unique(labels))

    # No soft membership for graph-based methods
    return labels, None, modularity if modularity else 0.5, actual_k
