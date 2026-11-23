"""
File writers module

Save clustering results, matrices, and export medoids.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Union, List
import shutil


def save_clustering_results(
    labels: np.ndarray,
    output_path: Union[str, Path],
    structure_names: Optional[List[str]] = None,
    membership: Optional[np.ndarray] = None,
    embedding: Optional[np.ndarray] = None,
    additional_data: Optional[dict] = None,
) -> Path:
    """
    Save clustering results to CSV file.

    Args:
        labels: Cluster labels (N,)
        output_path: Output file path
        structure_names: Optional structure names
        membership: Optional fuzzy membership matrix (N x K)
        embedding: Optional UMAP/t-SNE coordinates (N x 2)
        additional_data: Optional dict of additional columns

    Returns:
        Path to saved file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n = len(labels)

    data = {'cluster': labels}

    if structure_names:
        data['structure'] = structure_names
    else:
        data['structure'] = [f'struct_{i}' for i in range(n)]

    if membership is not None:
        data['max_membership'] = membership.max(axis=1)
        for k in range(membership.shape[1]):
            data[f'membership_c{k}'] = membership[:, k]

    if embedding is not None:
        data['umap1'] = embedding[:, 0]
        data['umap2'] = embedding[:, 1]

    if additional_data:
        data.update(additional_data)

    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)

    # Also drop a parquet sibling so the XGBDimerScorer bridge (`join-xgb`,
    # `filter-*`) can consume the clustering output directly — its loader reads
    # parquet only. Non-fatal if pyarrow/fastparquet is unavailable.
    try:
        df.to_parquet(output_path.with_suffix('.parquet'), index=False)
    except Exception:
        pass

    return output_path


def save_distance_matrix(
    matrix: np.ndarray,
    output_path: Union[str, Path],
    structure_names: Optional[List[str]] = None,
    format: str = "npy",
) -> Path:
    """
    Save distance/similarity matrix.

    Args:
        matrix: Distance matrix (N x N)
        output_path: Output file path
        structure_names: Optional structure names
        format: Output format ('npy', 'npz', 'csv')

    Returns:
        Path to saved file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == "npy":
        np.save(output_path.with_suffix('.npy'), matrix)
        if structure_names:
            np.save(
                output_path.with_name(output_path.stem + '_names.npy'),
                np.array(structure_names)
            )
    elif format == "npz":
        if structure_names:
            np.savez(output_path.with_suffix('.npz'),
                     matrix=matrix, names=np.array(structure_names))
        else:
            np.savez(output_path.with_suffix('.npz'), matrix=matrix)
    elif format == "csv":
        if structure_names:
            df = pd.DataFrame(matrix, index=structure_names, columns=structure_names)
        else:
            df = pd.DataFrame(matrix)
        df.to_csv(output_path.with_suffix('.csv'))
    else:
        raise ValueError(f"Unknown format: {format}")

    return output_path


def save_cluster_stats(
    labels: np.ndarray,
    output_path: Union[str, Path],
    medoid_indices: Optional[List[int]] = None,
    medoid_names: Optional[List[str]] = None,
    intra_distances: Optional[List[float]] = None,
    native_scores: Optional[dict] = None,
) -> Path:
    """
    Save per-cluster statistics.

    Args:
        labels: Cluster labels
        output_path: Output file path
        medoid_indices: Indices of cluster medoids
        medoid_names: Names of cluster medoids
        intra_distances: Intra-cluster distances
        native_scores: Dict of native scores per medoid

    Returns:
        Path to saved file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    unique_labels, counts = np.unique(labels, return_counts=True)

    data = {
        'cluster': unique_labels,
        'size': counts,
    }

    if medoid_indices:
        data['medoid_idx'] = medoid_indices

    if medoid_names:
        data['medoid_name'] = medoid_names

    if intra_distances:
        data['intra_dist_mean'] = intra_distances

    if native_scores:
        for metric, scores in native_scores.items():
            data[f'medoid_{metric}'] = scores

    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)

    return output_path


def save_k_selection_scores(
    k_scores: dict,
    output_path: Union[str, Path],
    target_name: str = "Unknown",
    metric_name: str = "",
) -> Path:
    """
    Save K selection scores (all metrics per K) to CSV.

    Includes: silhouette, penalized silhouette, delta silhouette,
    BIC, balance, and combined score for each K value.

    Args:
        k_scores: Dict from select_optimal_k()
        output_path: Output file path
        target_name: Target name for header
        metric_name: Metric name for header

    Returns:
        Path to saved file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not k_scores or 'k_values' not in k_scores:
        return output_path

    df = pd.DataFrame({
        'k': k_scores['k_values'],
        'silhouette': k_scores['silhouette_scores'],
        'penalized_sil': k_scores.get('pen_scores', [None] * len(k_scores['k_values'])),
        'delta_sil': k_scores.get('delta_sil_scores', [None] * len(k_scores['k_values'])),
        'bic': k_scores.get('bic_scores', [None] * len(k_scores['k_values'])),
        'balance': k_scores.get('balance_scores', [None] * len(k_scores['k_values'])),
        'combined_score': k_scores['combined_scores'],
    })

    with open(output_path, 'w') as f:
        f.write(f"# Target: {target_name}\n")
        f.write(f"# Metric: {metric_name}\n")
        f.write(f"# Selected K: {k_scores.get('optimal_k', '?')}\n")
        f.write(f"# BIC optimal K: {k_scores.get('bic_optimal_k', '?')}\n")
        f.write(f"# Plateau K: {k_scores.get('plateau_k', 'None')}\n")
        f.write(f"# Ascending at boundary: {k_scores.get('is_ascending_at_boundary', False)}\n")
        f.write(f"# k_max requested: {k_scores.get('k_max_requested', '?')}\n")
        f.write(f"# k_max effective: {k_scores.get('k_max_effective', '?')}\n")
        df.to_csv(f, index=False)

    return output_path


def export_medoids(
    medoid_indices: List[int],
    source_paths: List[Union[str, Path]],
    output_dir: Union[str, Path],
    labels: Optional[np.ndarray] = None,
) -> List[Path]:
    """
    Export medoid structures to a separate directory.

    Args:
        medoid_indices: Indices of medoid structures
        source_paths: List of all structure paths
        output_dir: Output directory for medoids
        labels: Optional cluster labels for naming

    Returns:
        List of paths to exported medoid files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exported = []

    for i, medoid_idx in enumerate(medoid_indices):
        if medoid_idx < 0 or medoid_idx >= len(source_paths):
            continue

        src = Path(source_paths[medoid_idx])
        if not src.exists():
            continue

        if labels is not None:
            cluster_id = labels[medoid_idx]
            dst_name = f"cluster_{cluster_id}_medoid_{src.name}"
        else:
            dst_name = f"medoid_{i}_{src.name}"

        dst = output_dir / dst_name
        shutil.copy2(src, dst)
        exported.append(dst)

    return exported


def export_cluster_members(
    cluster_id: int,
    labels: np.ndarray,
    source_paths: List[Union[str, Path]],
    output_dir: Union[str, Path],
    max_members: Optional[int] = None,
) -> List[Path]:
    """
    Export all members of a specific cluster.

    Args:
        cluster_id: Cluster ID to export
        labels: Cluster labels
        source_paths: List of all structure paths
        output_dir: Output directory
        max_members: Maximum number of members to export

    Returns:
        List of paths to exported files
    """
    output_dir = Path(output_dir) / f"cluster_{cluster_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    indices = np.where(labels == cluster_id)[0]

    if max_members:
        indices = indices[:max_members]

    exported = []

    for idx in indices:
        src = Path(source_paths[idx])
        if src.exists():
            dst = output_dir / src.name
            shutil.copy2(src, dst)
            exported.append(dst)

    return exported
