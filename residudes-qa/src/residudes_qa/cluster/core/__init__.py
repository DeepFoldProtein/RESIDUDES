"""
Core clustering modules

- distance: Distance matrix loading and conversion
- clustering: UMAP + Fuzzy C-means/GMM/K-means/Hierarchical clustering
- fuzzy: Fuzzy clustering utilities (membership analysis)
- network: Network analysis (graph, community detection, hub identification)
- consensus: Consensus structure extraction (RMSF, conserved core)
- gpu: GPU-accelerated distance matrix computation (TM, iTM, dRMSD)
"""

from residudes_qa.cluster.core.distance import compute_distance_matrix, similarity_to_distance
from residudes_qa.cluster.core.clustering import run_clustering, select_optimal_k, ClusteringResult
from residudes_qa.cluster.core.fuzzy import (
    run_fuzzy_clustering,
    get_ambiguous_samples,
    get_high_membership_samples,
    get_membership_entropy,
)
from residudes_qa.cluster.core.network import (
    build_structure_graph,
    analyze_cluster_network,
    compute_cluster_transition_matrix,
    resolution_scan,
    NetworkAnalysisResult,
)
from residudes_qa.cluster.core.consensus import (
    extract_cluster_consensus,
    extract_all_cluster_consensus,
    write_consensus_pdb,
    write_consensus_cif,
    ConsensusResult,
    find_conserved_segments,
    find_domain_boundaries,
    compare_segments_across_clusters,
    extract_local_substructures,
    write_segment_pdb,
    write_segment_cif,
    LocalSegment,
    SegmentComparisonResult,
)

# GPU functions (optional, requires PyTorch)
try:
    from residudes_qa.cluster.core.gpu import (
        compute_tm_matrix_gpu,
        compute_itm_matrix_gpu,
        compute_drmsd_matrix_gpu,
    )
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

__all__ = [
    # Distance
    "compute_distance_matrix",
    "similarity_to_distance",
    # Clustering
    "run_clustering",
    "select_optimal_k",
    "ClusteringResult",
    # Fuzzy utilities
    "run_fuzzy_clustering",
    "get_ambiguous_samples",
    "get_high_membership_samples",
    "get_membership_entropy",
    # Network
    "build_structure_graph",
    "analyze_cluster_network",
    "compute_cluster_transition_matrix",
    "resolution_scan",
    "NetworkAnalysisResult",
    # Consensus & Local Substructures
    "extract_cluster_consensus",
    "extract_all_cluster_consensus",
    "write_consensus_pdb",
    "ConsensusResult",
    "find_conserved_segments",
    "find_domain_boundaries",
    "compare_segments_across_clusters",
    "extract_local_substructures",
    "LocalSegment",
    "SegmentComparisonResult",
    # GPU (conditional)
    "GPU_AVAILABLE",
]

if GPU_AVAILABLE:
    __all__.extend([
        "compute_tm_matrix_gpu",
        "compute_itm_matrix_gpu",
        "compute_drmsd_matrix_gpu",
    ])
