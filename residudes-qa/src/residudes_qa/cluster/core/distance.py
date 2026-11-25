"""
Distance matrix computation module

Supports multiple distance/similarity metrics:
- TM-score (Template Modeling score)
- iTM (interface TM-score for multimers)
- dRMSD (distance-based RMSD)
"""

import numpy as np
from pathlib import Path
from typing import Union, Literal, Optional, Tuple

MetricType = Literal["tm", "itm", "drmsd"]


def compute_distance_matrix(
    structures: list,
    metric: MetricType = "tm",
    output_dir: Optional[Path] = None,
    use_gpu: bool = False,
    n_jobs: int = -1,
) -> Tuple[np.ndarray, list]:
    """
    Compute pairwise distance matrix for protein structures.

    Args:
        structures: List of PDB file paths or structure objects
        metric: Distance metric ('tm', 'itm', 'drmsd')
        output_dir: Directory to save computed matrices
        use_gpu: Use GPU acceleration if available
        n_jobs: Number of parallel jobs (-1 for all CPUs)

    Returns:
        Tuple of (distance_matrix, structure_names)

    Example:
        >>> matrix, names = compute_distance_matrix(
        ...     structures=["pred1.pdb", "pred2.pdb"],
        ...     metric="tm"
        ... )
    """
    len(structures)
    structure_names = [Path(s).name if isinstance(s, (str, Path)) else f"struct_{i}"
                       for i, s in enumerate(structures)]

    if metric == "tm":
        matrix = _compute_tm_matrix(structures, use_gpu, n_jobs)
    elif metric == "itm":
        matrix = _compute_itm_matrix(structures, use_gpu, n_jobs)
    elif metric == "drmsd":
        matrix = _compute_drmsd_matrix(structures, use_gpu, n_jobs)
    else:
        raise ValueError(f"Unknown metric: {metric}. Use 'tm', 'itm', or 'drmsd'")

    # Save if output_dir specified
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(output_dir / f"{metric}_matrix.npy", matrix)
        np.save(output_dir / "structure_names.npy", np.array(structure_names))

    return matrix, structure_names


def _compute_tm_matrix(structures: list, use_gpu: bool, n_jobs: int) -> np.ndarray:
    """Compute TM-score matrix using US-align or similar tool."""
    n = len(structures)
    np.zeros((n, n))

    # TODO: Integrate with US-align or TMalign
    # For now, placeholder implementation
    raise NotImplementedError(
        "TM-score computation requires US-align. "
        "Use load_precomputed_matrix() for pre-computed matrices."
    )


def _compute_itm_matrix(structures: list, use_gpu: bool, n_jobs: int) -> np.ndarray:
    """Compute interface TM-score matrix for multimers."""
    raise NotImplementedError(
        "iTM computation requires specialized multimer alignment. "
        "Use load_precomputed_matrix() for pre-computed matrices."
    )


def _compute_drmsd_matrix(structures: list, use_gpu: bool, n_jobs: int) -> np.ndarray:
    """Compute distance-based RMSD matrix."""
    raise NotImplementedError(
        "dRMSD computation not yet implemented. "
        "Use load_precomputed_matrix() for pre-computed matrices."
    )


def load_precomputed_matrix(
    matrix_path: Union[str, Path],
    names_path: Optional[Union[str, Path]] = None,
) -> Tuple[np.ndarray, Optional[list]]:
    """
    Load a precomputed distance/similarity matrix.

    Args:
        matrix_path: Path to .npy matrix file
        names_path: Optional path to structure names file

    Returns:
        Tuple of (matrix, structure_names or None)

    Example:
        >>> matrix, names = load_precomputed_matrix(
        ...     "tm_matrix_gpu.npy",
        ...     "structure_names.npy"
        ... )
    """
    matrix = np.load(matrix_path)

    names = None
    if names_path and Path(names_path).exists():
        names = np.load(names_path, allow_pickle=True).tolist()

    return matrix, names


def similarity_to_distance(matrix: np.ndarray, metric: MetricType = "tm") -> np.ndarray:
    """
    Convert similarity matrix to distance matrix.

    For TM-score and iTM: distance = 1 - similarity
    For dRMSD: already a distance metric (normalize if needed)

    Args:
        matrix: Similarity or distance matrix
        metric: Metric type for proper conversion

    Returns:
        Distance matrix suitable for clustering
    """
    if metric in ["tm", "itm"]:
        dist = 1 - matrix
    else:  # drmsd
        # Normalize to 0-1 range
        if matrix.max() > 0:
            dist = matrix / matrix.max()
        else:
            dist = matrix.copy()

    # Ensure diagonal is zero
    np.fill_diagonal(dist, 0)

    return dist
