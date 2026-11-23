#!/usr/bin/env python3
"""
GPU-accelerated TM-align superposition core extraction.

Instead of RMSF (indirect, mean-based), uses TM-align's iterative Kabsch
weights to directly identify structurally aligned residues.

Key concept:
  w_i = 1 / (1 + (d_i / d0)^2)
  - w_i > 0.5 → residue i is within d0 after optimal superposition ("core")
  - w_i < 0.5 → residue i deviates significantly ("variable")

For a cluster of N structures:
  1. Align all to medoid via iterative weighted Kabsch
  2. Collect per-residue weights for each pair
  3. Core = residues with mean weight > threshold across all pairs
"""

import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Union, Dict
from dataclasses import dataclass, field

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class SuperpositionCore:
    """Result of TM-align superposition core extraction."""
    n_structures: int
    n_residues: int
    # Per-residue metrics
    mean_weight: np.ndarray        # (n_residues,) mean TM-weight across pairs
    core_mask: np.ndarray          # (n_residues,) bool — True if core residue
    n_core: int
    core_fraction: float
    # Per-residue distance stats
    mean_distance: np.ndarray      # (n_residues,) mean CA distance after superposition
    # Params
    weight_threshold: float
    d0: float
    # Comparison with RMSF
    rmsf: Optional[np.ndarray] = None   # if computed simultaneously


@dataclass
class CoreSegment:
    """Contiguous core region from superposition."""
    segment_id: int
    start: int
    end: int          # exclusive
    length: int
    mean_weight: float
    min_weight: float
    mean_distance: float


def compute_d0(L: int) -> float:
    """TM-score d0 normalization."""
    return max(0.5, 1.24 * ((L - 15) ** (1.0/3.0)) - 1.8)


def _parse_pdb_ca(pdb_path: Union[str, Path]) -> np.ndarray:
    """Extract CA atom coordinates from PDB file."""
    coords = []
    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith('ATOM') and line[12:16].strip() == 'CA':
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                coords.append([x, y, z])
    return np.array(coords, dtype=np.float32)


def extract_superposition_core_gpu(
    coords_list: List[np.ndarray],
    reference_idx: int = 0,
    weight_threshold: float = 0.5,
    n_iter: int = 5,
    batch_size: int = 512,
    gpu_id: int = 0,
    max_structures: int = 200,
) -> SuperpositionCore:
    """
    Extract per-residue superposition core using GPU TM-align weights.

    Aligns all structures to reference using iterative weighted Kabsch,
    then collects per-residue TM-weights. Core = residues consistently
    well-aligned (weight > threshold) across pairs.

    Args:
        coords_list: List of CA coordinate arrays, each (L, 3)
        reference_idx: Index of reference structure (medoid)
        weight_threshold: Core threshold on TM-weight (default 0.5 ≈ d_i < d0)
        n_iter: Iterative Kabsch refinement steps
        batch_size: GPU batch size
        gpu_id: CUDA device
        max_structures: Subsample if more than this

    Returns:
        SuperpositionCore with per-residue weights and core mask
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required for GPU computation")

    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')

    n_total = len(coords_list)
    n_residues = len(coords_list[reference_idx])
    d0 = compute_d0(n_residues)

    # Subsample if needed (keep reference)
    if n_total > max_structures:
        rng = np.random.RandomState(42)
        indices = list(rng.choice(
            [i for i in range(n_total) if i != reference_idx],
            max_structures - 1, replace=False
        ))
        indices = [reference_idx] + sorted(indices)
        coords_list = [coords_list[i] for i in indices]
        n_total = len(coords_list)
        reference_idx = 0

    # Filter to same length
    valid = []
    for i, c in enumerate(coords_list):
        if len(c) == n_residues:
            valid.append(i)
    if reference_idx not in valid:
        raise ValueError("Reference structure has different length")
    coords_list = [coords_list[i] for i in valid]
    reference_idx = valid.index(reference_idx) if reference_idx in valid else 0
    n_total = len(coords_list)

    ref_coords = coords_list[reference_idx]

    # All structures to align against reference
    others = [i for i in range(n_total) if i != reference_idx]
    n_pairs = len(others)

    # Accumulate per-residue weights and distances
    all_weights = np.zeros((n_pairs, n_residues), dtype=np.float32)
    all_distances = np.zeros((n_pairs, n_residues), dtype=np.float32)

    ref_tensor = torch.tensor(ref_coords, device=device, dtype=torch.float32)

    for batch_start in range(0, n_pairs, batch_size):
        batch_end = min(batch_start + batch_size, n_pairs)
        batch_indices = others[batch_start:batch_end]
        bs = len(batch_indices)

        # Stack batch
        coords_batch = np.stack([coords_list[i] for i in batch_indices])
        mobile = torch.tensor(coords_batch, device=device, dtype=torch.float32)
        target = ref_tensor.unsqueeze(0).expand(bs, -1, -1)

        # Iterative weighted Kabsch
        weights = torch.ones(bs, n_residues, device=device, dtype=torch.float32)

        for iteration in range(n_iter):
            w_exp = weights.unsqueeze(-1)  # (bs, L, 1)
            w_sum = weights.sum(dim=1, keepdim=True).clamp(min=1).unsqueeze(-1)

            # Weighted centroids
            center_m = (mobile * w_exp).sum(dim=1, keepdim=True) / w_sum
            center_t = (target * w_exp).sum(dim=1, keepdim=True) / w_sum

            m_c = mobile - center_m
            t_c = target - center_t

            # Weighted Kabsch
            sqrt_w = torch.sqrt(w_exp)
            H = torch.bmm((m_c * sqrt_w).transpose(1, 2), t_c * sqrt_w)
            U, S, Vt = torch.linalg.svd(H)
            R = torch.bmm(Vt.transpose(1, 2), U.transpose(1, 2))
            det = torch.linalg.det(R)
            Vt_fix = Vt.clone()
            Vt_fix[det < 0, -1, :] *= -1
            R = torch.bmm(Vt_fix.transpose(1, 2), U.transpose(1, 2))

            # Apply rotation
            m_rot = torch.bmm(m_c, R.transpose(1, 2))

            # Per-residue distances
            dist = torch.norm(m_rot - t_c, dim=2)  # (bs, L)

            # TM-weights
            tm_w = 1.0 / (1.0 + (dist / d0) ** 2)  # (bs, L)

            # Update weights for next iteration
            weights = tm_w

        # Store final weights and distances
        all_weights[batch_start:batch_end] = tm_w.cpu().numpy()
        all_distances[batch_start:batch_end] = dist.cpu().numpy()

    # Aggregate: mean weight per residue across all pairs
    mean_weight = all_weights.mean(axis=0)
    mean_distance = all_distances.mean(axis=0)

    # Also compute RMSF for comparison
    # First do the final alignment and compute coordinate deviations
    all_aligned = [ref_coords.copy()]
    for pair_idx, struct_idx in enumerate(others):
        # Use the final weights for alignment
        w = all_weights[pair_idx]
        mobile_np = coords_list[struct_idx].copy()
        # Quick Kabsch with weights on CPU
        w_3d = w[:, None]
        cm = (mobile_np * w_3d).sum(0) / w.sum()
        ct = (ref_coords * w_3d).sum(0) / w.sum()
        P = (mobile_np - cm) * np.sqrt(w_3d)
        Q = (ref_coords - ct) * np.sqrt(w_3d)
        H = P.T @ Q
        U, S, Vt = np.linalg.svd(H)
        d = np.linalg.det(Vt.T @ U.T)
        sign = np.eye(3)
        sign[2, 2] = np.sign(d)
        R = Vt.T @ sign @ U.T
        aligned = (mobile_np - cm) @ R.T + ct
        all_aligned.append(aligned)

    all_aligned = np.stack(all_aligned)
    mean_coords = all_aligned.mean(axis=0)
    deviations = all_aligned - mean_coords[None]
    rmsf = np.sqrt(np.mean(np.sum(deviations**2, axis=2), axis=0))

    # Core mask
    core_mask = mean_weight > weight_threshold
    n_core = int(core_mask.sum())

    return SuperpositionCore(
        n_structures=n_total,
        n_residues=n_residues,
        mean_weight=mean_weight,
        core_mask=core_mask,
        n_core=n_core,
        core_fraction=n_core / n_residues if n_residues > 0 else 0.0,
        mean_distance=mean_distance,
        weight_threshold=weight_threshold,
        d0=d0,
        rmsf=rmsf,
    )


def find_core_segments(
    core: SuperpositionCore,
    min_length: int = 5,
    gap_tolerance: int = 2,
) -> List[CoreSegment]:
    """
    Find contiguous core segments from superposition core mask.

    Args:
        core: SuperpositionCore result
        min_length: Minimum segment length
        gap_tolerance: Bridge short gaps within core regions

    Returns:
        List of CoreSegment sorted by start position
    """
    mask = core.core_mask.copy()
    n = len(mask)

    # Bridge short gaps
    if gap_tolerance > 0:
        i = 0
        while i < n:
            if not mask[i]:
                gap_start = i
                while i < n and not mask[i]:
                    i += 1
                gap_len = i - gap_start
                if gap_len <= gap_tolerance and gap_start > 0 and i < n:
                    mask[gap_start:i] = True
            else:
                i += 1

    # Extract segments
    segments = []
    seg_id = 0
    i = 0
    while i < n:
        if mask[i]:
            start = i
            while i < n and mask[i]:
                i += 1
            end = i
            length = end - start
            if length >= min_length:
                segments.append(CoreSegment(
                    segment_id=seg_id,
                    start=start,
                    end=end,
                    length=length,
                    mean_weight=float(core.mean_weight[start:end].mean()),
                    min_weight=float(core.mean_weight[start:end].min()),
                    mean_distance=float(core.mean_distance[start:end].mean()),
                ))
                seg_id += 1
        else:
            i += 1

    return segments


def extract_cross_cluster_core(
    cluster_cores: Dict[int, SuperpositionCore],
    weight_threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find universally conserved core across all clusters.

    Args:
        cluster_cores: {cluster_id: SuperpositionCore}
        weight_threshold: Threshold for each cluster

    Returns:
        (universal_core_mask, mean_weight_across_clusters)
        - universal_core_mask: True where ALL clusters agree it's core
        - mean_weight: average weight across clusters per residue
    """
    if not cluster_cores:
        raise ValueError("No cluster cores provided")

    n_residues = next(iter(cluster_cores.values())).n_residues
    n_clusters = len(cluster_cores)

    # Stack weights from all clusters
    weights = np.zeros((n_clusters, n_residues), dtype=np.float32)
    masks = np.zeros((n_clusters, n_residues), dtype=bool)

    for i, (cid, core) in enumerate(sorted(cluster_cores.items())):
        if core.n_residues != n_residues:
            raise ValueError(f"Cluster {cid} has {core.n_residues} residues, expected {n_residues}")
        weights[i] = core.mean_weight
        masks[i] = core.core_mask

    # Universal core: ALL clusters agree
    universal_mask = masks.all(axis=0)

    # Mean weight across clusters
    mean_weight = weights.mean(axis=0)

    return universal_mask, mean_weight
