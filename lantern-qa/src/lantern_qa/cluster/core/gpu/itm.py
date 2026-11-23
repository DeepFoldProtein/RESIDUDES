#!/usr/bin/env python3
"""
GPU-accelerated interface TM-score (iTM) calculation for multimers

iTM focuses on interface residues between chains, providing better
assessment of protein-protein docking quality.
"""

import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Union, Dict
from scipy.spatial.distance import cdist
from tqdm import tqdm

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def parse_pdb_with_chains(
    pdb_path: Union[str, Path],
    exclude_indices: Optional[dict] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse PDB or CIF, return (coords, chain_ids, residue_ids).

    `exclude_indices` is `{chain_id: set_of_zero_based_residue_positions}` —
    drops those residues before returning. Used by CDR-aware clustering.
    """
    from lantern_qa.cluster.core.gpu._loader import parse_structure_with_chains
    return parse_structure_with_chains(pdb_path, exclude_indices=exclude_indices)


def find_interface_residues(
    coords: np.ndarray,
    chain_ids: np.ndarray,
    cutoff: float = 8.0,
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Find interface residues between chains.

    Args:
        coords: CA coordinates (n_atoms, 3)
        chain_ids: Chain identifiers (n_atoms,)
        cutoff: Distance cutoff for interface definition (default: 8.0 Angstroms)

    Returns:
        Tuple of (interface_mask, interface_pairs)
    """
    unique_chains = np.unique(chain_ids)
    n_residues = len(coords)
    interface_mask = np.zeros(n_residues, dtype=bool)
    interface_pairs = []

    for i, chain_i in enumerate(unique_chains):
        for chain_j in unique_chains[i+1:]:
            mask_i = chain_ids == chain_i
            mask_j = chain_ids == chain_j

            coords_i = coords[mask_i]
            coords_j = coords[mask_j]

            # Distance matrix between chains
            dist_matrix = cdist(coords_i, coords_j)

            # Interface residues
            contact_i = np.any(dist_matrix < cutoff, axis=1)
            contact_j = np.any(dist_matrix < cutoff, axis=0)

            if np.any(contact_i) or np.any(contact_j):
                idx_i = np.where(mask_i)[0]
                idx_j = np.where(mask_j)[0]

                interface_mask[idx_i[contact_i]] = True
                interface_mask[idx_j[contact_j]] = True

                interface_pairs.append({
                    'chain_i': chain_i,
                    'chain_j': chain_j,
                    'residues_i': idx_i[contact_i],
                    'residues_j': idx_j[contact_j],
                    'n_contacts': int(np.sum(dist_matrix < cutoff)),
                })

    return interface_mask, interface_pairs


def compute_d0(L: int) -> float:
    """Compute d0 for TM-score normalization."""
    if L <= 21:
        return 0.5
    else:
        return 1.24 * ((L - 15) ** (1.0/3.0)) - 1.8


def compute_itm_single(
    coords1: np.ndarray,
    coords2: np.ndarray,
    interface_mask: np.ndarray,
    n_iter: int = 5,
) -> float:
    """
    Compute interface TM-score between two structures.

    Args:
        coords1: Coordinates of structure 1 (n_atoms, 3)
        coords2: Coordinates of structure 2 (n_atoms, 3)
        interface_mask: Boolean mask for interface residues
        n_iter: Number of refinement iterations

    Returns:
        Interface TM-score
    """
    # Extract interface residues
    iface_coords1 = coords1[interface_mask]
    iface_coords2 = coords2[interface_mask]

    L = len(iface_coords1)
    if L < 5:
        return 0.0

    d0 = compute_d0(L)

    # Iterative superposition
    weights = np.ones(L, dtype=np.float32)
    best_tm = 0.0

    for _ in range(n_iter):
        # Weighted center
        w_sum = weights.sum()
        center1 = (iface_coords1 * weights[:, None]).sum(axis=0) / w_sum
        center2 = (iface_coords2 * weights[:, None]).sum(axis=0) / w_sum

        c1 = iface_coords1 - center1
        c2 = iface_coords2 - center2

        # Weighted Kabsch
        sqrt_w = np.sqrt(weights)
        H = (c1 * sqrt_w[:, None]).T @ (c2 * sqrt_w[:, None])
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        if np.linalg.det(R) < 0:
            Vt[-1] *= -1
            R = Vt.T @ U.T

        # Apply rotation
        c1_rotated = c1 @ R

        # Distances
        distances = np.linalg.norm(c1_rotated - c2, axis=1)

        # TM-score
        tm_terms = 1.0 / (1.0 + (distances / d0) ** 2)
        tm_score = tm_terms.sum() / L

        best_tm = max(best_tm, tm_score)

        # Update weights
        weights = tm_terms

    return best_tm


def compute_itm_matrix_gpu(
    pdb_files: List[Union[str, Path]],
    batch_size: int = 1024,
    gpu_id: int = 0,
    interface_cutoff: float = 8.0,
    output_path: Optional[Union[str, Path]] = None,
    show_progress: bool = True,
    exclude_indices: Optional[dict] = None,
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """
    Compute all-vs-all interface TM-score matrix using GPU.

    Args:
        pdb_files: List of PDB file paths
        batch_size: Batch size for GPU computation
        gpu_id: GPU device ID
        interface_cutoff: Distance cutoff for interface definition
        output_path: Optional path to save results
        show_progress: Show progress bar

    Returns:
        Tuple of (iTM matrix, structure names, interface_mask)

    Example:
        >>> pdbs = list(Path("multimer_predictions/").glob("*.pdb"))
        >>> matrix, names, mask = compute_itm_matrix_gpu(pdbs)
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for GPU computation")

    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    n = len(pdb_files)

    # Load structures
    names = []
    all_coords = []
    all_chains = []

    from lantern_qa.cluster.core.gpu._loader import parse_structure_with_chains_batch
    names = [Path(p).name for p in pdb_files]
    _loaded = parse_structure_with_chains_batch(
        pdb_files, exclude_indices=exclude_indices, show_progress=show_progress)
    all_coords = [c for (c, _ch, _r) in _loaded]
    all_chains = [ch for (_c, ch, _r) in _loaded]

    # Use first structure to define interface (or use consensus)
    if show_progress:
        print("Finding interface residues...")
    interface_mask, interface_pairs = find_interface_residues(
        all_coords[0], all_chains[0], cutoff=interface_cutoff
    )

    n_interface = interface_mask.sum()
    if show_progress:
        print(f"  Interface residues: {n_interface}")

    if n_interface < 5:
        print("  Warning: Very few interface residues found")
        return np.zeros((n, n), dtype=np.float32), names, interface_mask

    # Extract interface coordinates
    iface_coords_list = [coords[interface_mask] for coords in all_coords]

    # Pad coordinates
    max_len = max(len(c) for c in iface_coords_list)
    padded_coords = np.zeros((n, max_len, 3), dtype=np.float32)
    lengths = np.zeros(n, dtype=np.int32)

    for i, coords in enumerate(iface_coords_list):
        L = len(coords)
        padded_coords[i, :L] = coords
        lengths[i] = L

    # Compute d0 values
    d0_values = np.array([compute_d0(L) for L in lengths], dtype=np.float32)

    # Transfer to GPU
    coords_gpu = torch.tensor(padded_coords, device=device, dtype=torch.float32)
    lengths_gpu = torch.tensor(lengths, device=device, dtype=torch.float32)
    d0_gpu = torch.tensor(d0_values, device=device, dtype=torch.float32)

    # Generate pairs
    pairs = [(i, j) for i in range(n) for j in range(i+1, n)]

    # Process in batches
    results = np.zeros((n, n), dtype=np.float32)

    # Import TM-score computation function
    from lantern_qa.cluster.core.gpu.tmscore import compute_tmscore_batch_iterative

    pair_iterator = range(0, len(pairs), batch_size)
    if show_progress:
        pair_iterator = tqdm(pair_iterator, desc="Computing iTM",
                            total=len(pairs)//batch_size + 1)

    for batch_start in pair_iterator:
        batch_end = min(batch_start + batch_size, len(pairs))
        batch_pairs = pairs[batch_start:batch_end]

        # Prepare batch
        idx1 = [p[0] for p in batch_pairs]
        idx2 = [p[1] for p in batch_pairs]

        coords1_batch = coords_gpu[idx1]
        coords2_batch = coords_gpu[idx2]

        # Use minimum length for normalization
        batch_lengths = torch.minimum(lengths_gpu[idx1], lengths_gpu[idx2])
        batch_d0 = torch.tensor([compute_d0(int(L.item())) for L in batch_lengths],
                                device=device, dtype=torch.float32)

        # Compute iTM scores
        with torch.no_grad():
            itm_scores = compute_tmscore_batch_iterative(
                coords1_batch, coords2_batch, batch_lengths, batch_d0, n_iter=5
            )

        # Store results
        itm_np = itm_scores.cpu().numpy()
        for idx, (i, j) in enumerate(batch_pairs):
            results[i, j] = itm_np[idx]
            results[j, i] = itm_np[idx]

    # Fill diagonal
    np.fill_diagonal(results, 1.0)

    # Clean up
    del coords_gpu, lengths_gpu, d0_gpu
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Save
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, results)
        np.save(output_path.with_name(output_path.stem + '_names.npy'), np.array(names))
        np.save(output_path.with_name(output_path.stem + '_interface_mask.npy'), interface_mask)

    return results, names, interface_mask
