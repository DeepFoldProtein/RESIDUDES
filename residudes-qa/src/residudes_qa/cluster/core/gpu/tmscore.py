#!/usr/bin/env python3
"""
GPU-accelerated TM-score calculation using PyTorch

Features:
- Batched Kabsch alignment on GPU
- Iterative refinement for improved accuracy
- Parallel TM-score computation for thousands of pairs
"""

import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Union
from tqdm import tqdm

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def check_gpu_available() -> bool:
    """Check if GPU is available."""
    if not TORCH_AVAILABLE:
        return False
    return torch.cuda.is_available()


def get_device(gpu_id: int = 0) -> "torch.device":
    """Get PyTorch device."""
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for GPU computation")

    if torch.cuda.is_available():
        device = torch.device(f'cuda:{gpu_id}')
    else:
        device = torch.device('cpu')
    return device


def parse_pdb_ca(
    pdb_path: Union[str, Path],
    exclude_indices: Optional[dict] = None,
) -> np.ndarray:
    """Extract CA atom coordinates from PDB or CIF file.

    `exclude_indices` is `{chain_id: set_of_zero_based_residue_positions}` —
    drops those residues before returning. Used by CDR-aware clustering.
    """
    from residudes_qa.cluster.core.gpu._loader import parse_structure_ca
    return parse_structure_ca(pdb_path, exclude_indices=exclude_indices)


def compute_d0(L: int) -> float:
    """Compute d0 for TM-score normalization."""
    if L <= 21:
        return 0.5
    else:
        return 1.24 * ((L - 15) ** (1.0/3.0)) - 1.8


def _kabsch_R(
    P_centered: "torch.Tensor",
    Q_centered: "torch.Tensor",
    weights: "torch.Tensor | None" = None,
) -> "torch.Tensor":
    """
    Batched weighted Kabsch SVD core, returning the rotation that maps
    P_centered onto Q_centered. Inputs are assumed already centred on
    their (weighted) centroids.

    Args:
        P_centered: (batch, n_atoms, 3) — centred query coords
        Q_centered: (batch, n_atoms, 3) — centred target coords
        weights:    (batch, n_atoms) per-atom weights, or None for uniform.
                    Atoms with weight 0 contribute nothing to the SVD
                    (use this to mask padding).

    Returns:
        R: (batch, 3, 3) rotation with det = +1 (reflection corrected).

    Shared by `compute_tmscore_batch_iterative` and the anchored kernel.
    """
    if weights is None:
        H = torch.bmm(P_centered.transpose(1, 2), Q_centered)
    else:
        sqrt_w = torch.sqrt(weights).unsqueeze(-1)
        H = torch.bmm((P_centered * sqrt_w).transpose(1, 2), Q_centered * sqrt_w)

    U, S, Vt = torch.linalg.svd(H)

    R = torch.bmm(Vt.transpose(1, 2), U.transpose(1, 2))
    det = torch.linalg.det(R)
    Vt_corrected = Vt.clone()
    Vt_corrected[det < 0, -1, :] *= -1
    R = torch.bmm(Vt_corrected.transpose(1, 2), U.transpose(1, 2))

    return R


def kabsch_rotation_batch(P: "torch.Tensor", Q: "torch.Tensor") -> "torch.Tensor":
    """
    Batched Kabsch algorithm to find optimal rotation matrix (uniform weights).

    Args:
        P: Query coordinates (batch, n_atoms, 3)
        Q: Target coordinates (batch, n_atoms, 3)

    Returns:
        Rotation matrices (batch, 3, 3)
    """
    P_centered = P - P.mean(dim=1, keepdim=True)
    Q_centered = Q - Q.mean(dim=1, keepdim=True)
    return _kabsch_R(P_centered, Q_centered, weights=None)


def compute_tmscore_batch_iterative(
    coords1_batch: "torch.Tensor",
    coords2_batch: "torch.Tensor",
    lengths: "torch.Tensor",
    d0_batch: "torch.Tensor",
    n_iter: int = 5,
) -> "torch.Tensor":
    """
    Compute TM-score with iterative refinement.

    Args:
        coords1_batch: Query coordinates (batch, n_atoms, 3)
        coords2_batch: Target coordinates (batch, n_atoms, 3)
        lengths: Actual lengths for each pair (batch,)
        d0_batch: d0 values for normalization (batch,)
        n_iter: Number of refinement iterations

    Returns:
        TM-scores (batch,)
    """
    batch_size = coords1_batch.shape[0]
    max_len = coords1_batch.shape[1]

    # Create mask for valid positions
    indices = torch.arange(max_len, device=coords1_batch.device).unsqueeze(0)
    valid_mask = indices < lengths.unsqueeze(1)

    # Initialize weights
    weights = valid_mask.float()

    best_tm_scores = torch.zeros(batch_size, device=coords1_batch.device, dtype=torch.float32)

    for iteration in range(n_iter):
        # Weighted centroid
        weights_sum = weights.sum(dim=1, keepdim=True).clamp(min=1)
        weights_expanded = weights.unsqueeze(-1)

        center1 = (coords1_batch * weights_expanded).sum(dim=1, keepdim=True) / weights_sum.unsqueeze(-1)
        center2 = (coords2_batch * weights_expanded).sum(dim=1, keepdim=True) / weights_sum.unsqueeze(-1)

        coords1_centered = coords1_batch - center1
        coords2_centered = coords2_batch - center2

        # Weighted Kabsch (shared util)
        R = _kabsch_R(coords1_centered, coords2_centered, weights=weights)

        # Apply rotation
        coords1_rotated = torch.bmm(coords1_centered, R.transpose(1, 2))

        # Compute distances
        distances = torch.norm(coords1_rotated - coords2_centered, dim=2)

        # TM-score
        d0_expanded = d0_batch.unsqueeze(1)
        tm_terms = 1.0 / (1.0 + (distances / d0_expanded) ** 2)
        tm_terms_masked = tm_terms * valid_mask.float()
        tm_scores = tm_terms_masked.sum(dim=1) / lengths.float()

        # Update best scores
        best_tm_scores = torch.maximum(best_tm_scores, tm_scores)

        # Update weights using TM-term
        weights = tm_terms * valid_mask.float()

    return best_tm_scores


def compute_tm_matrix_gpu(
    pdb_files: List[Union[str, Path]],
    batch_size: int = 2048,
    gpu_id: int = 0,
    output_path: Optional[Union[str, Path]] = None,
    show_progress: bool = True,
    exclude_indices: Optional[dict] = None,
    homo_permute: bool = False,
    homo_max_full: int = 24,
) -> Tuple[np.ndarray, List[str]]:
    """
    Compute all-vs-all TM-score matrix using GPU.

    homo_permute: for a HOMO-oligomer (all models share the same chain count k and
        chains are interchangeable), the global-Kabsch TM is sensitive to chain
        LABELING — a transposition of two chains collapses the score even though the
        3D structure is identical. When True, each pair is scored over all chain
        orderings of the second model and the MAX is kept (true quaternary distance).
        Full enumeration is used when k! <= homo_max_full (covers k<=4); for larger k
        a greedy Hungarian chain assignment is used. Requires uniform chain count
        across the pool; falls back to the plain path otherwise (with a warning).

    Args:
        pdb_files: List of PDB file paths
        batch_size: Batch size for GPU computation
        gpu_id: GPU device ID
        output_path: Optional path to save results
        show_progress: Show progress bar

    Returns:
        Tuple of (TM-score matrix, structure names)

    Example:
        >>> from pathlib import Path
        >>> pdbs = list(Path("predictions/").glob("*.pdb"))
        >>> matrix, names = compute_tm_matrix_gpu(pdbs, batch_size=1024)
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for GPU computation. Install with: pip install torch")

    device = get_device(gpu_id)
    n = len(pdb_files)

    # Load structures (index-keyed, not name-keyed: many input flavours, e.g.
    # AF3 unified `model.cif`, share a basename so a name-keyed dict collapses
    # to one entry).
    coords_cache: list = []
    lengths: list = []
    names: list = []

    names = [Path(p).name for p in pdb_files]
    chain_lists = None
    if homo_permute:
        from residudes_qa.cluster.core.gpu._loader import parse_structure_with_chains_batch
        parsed = parse_structure_with_chains_batch(
            pdb_files, exclude_indices=exclude_indices, show_progress=show_progress)
        coords_cache = [np.asarray(c, dtype=np.float32) for (c, _ch, _ri) in parsed]
        chain_lists = [list(_ch) for (_c, _ch, _ri) in parsed]
    else:
        from residudes_qa.cluster.core.gpu._loader import parse_structure_ca_batch
        coords_cache = parse_structure_ca_batch(
            pdb_files, exclude_indices=exclude_indices, show_progress=show_progress)
    lengths = [len(c) for c in coords_cache]

    # Find max length for padding
    max_len = max(lengths)

    # Pre-pad all coordinates
    padded_coords: list = []
    for coords in coords_cache:
        padded = np.zeros((max_len, 3), dtype=np.float32)
        padded[:len(coords)] = coords
        padded_coords.append(padded)

    # ---- homo-oligomer chain-permutation setup ----------------------------------
    # perm_row_idx[s] = list of int arrays (len max_len); each reorders structure s's
    # chain blocks to a given ordering, so gathering padded_coords[s] by it swaps
    # chains. Slot 0 = identity. Scoring keeps the MAX over these orderings.
    import itertools
    from math import factorial as _factorial
    perm_row_idx = None
    n_perm = 1
    if homo_permute:
        def _blocks(ch):
            bl, s = [], 0
            for i in range(1, len(ch) + 1):
                if i == len(ch) or ch[i] != ch[s]:
                    bl.append((s, i - s))
                    s = i
            return bl
        blks = [_blocks(ch) for ch in chain_lists]
        kset = {len(b) for b in blks}
        if len(kset) != 1 or kset == {1}:
            print(f"  [homo_permute] non-uniform / monomeric chain counts {sorted(kset)} "
                  f"-> falling back to plain chain-fixed TM")
            homo_permute = False
        else:
            k = kset.pop()
            slot_perms = list(itertools.permutations(range(k)))
            n_perm = len(slot_perms)
            if _factorial(k) > homo_max_full:
                print(f"  [homo_permute] WARNING k!={_factorial(k)} > homo_max_full={homo_max_full}: "
                      f"full enumeration is {n_perm}x work/memory — proceeding (raise homo_max_full to silence).")
            # int16 row-index (max_len < 32767) keeps k! copies cheap in RAM
            assert max_len < 32767, "max_len too large for int16 permutation index"
            base = np.arange(max_len, dtype=np.int16)
            perm_row_idx = []
            for bl in blks:
                arrs = []
                for sp in slot_perms:
                    order = np.concatenate([np.arange(bl[t][0], bl[t][0] + bl[t][1]) for t in sp])
                    full = base.copy()
                    full[:len(order)] = order.astype(np.int16)
                    arrs.append(full)
                perm_row_idx.append(np.stack(arrs))  # (n_perm, max_len) int16
            print(f"  [homo_permute] k={k} chains -> full enumeration of {n_perm} chain orderings per pair")

    # Generate all pairs
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((i, j))

    # Process in batches
    results = np.zeros((n, n), dtype=np.float32)

    pair_iterator = range(0, len(pairs), batch_size)
    if show_progress:
        pair_iterator = tqdm(pair_iterator, desc="Computing TM-scores",
                            total=len(pairs)//batch_size + 1)

    for batch_start in pair_iterator:
        batch_end = min(batch_start + batch_size, len(pairs))
        batch_pairs = pairs[batch_start:batch_end]

        # Prepare batch tensors
        coords1_list = []
        coords2_list = []
        lengths_list = []
        d0_list = []

        for i, j in batch_pairs:
            coords1_list.append(padded_coords[i])
            coords2_list.append(padded_coords[j])

            L = min(lengths[i], lengths[j])
            lengths_list.append(L)
            d0_list.append(compute_d0(L))

        # Convert to tensors
        coords1_batch = torch.tensor(np.array(coords1_list), device=device, dtype=torch.float32)
        coords2_batch = torch.tensor(np.array(coords2_list), device=device, dtype=torch.float32)
        lengths_batch = torch.tensor(lengths_list, device=device, dtype=torch.float32)
        d0_batch = torch.tensor(d0_list, device=device, dtype=torch.float32)

        # Compute TM-scores (homo_permute: keep MAX over all chain orderings of model 2)
        with torch.no_grad():
            if homo_permute and n_perm > 1:
                # gather each pair's model-2 chains into every ordering, score, take max
                j_idx = np.array([j for (_i, j) in batch_pairs])
                ridx = np.stack([perm_row_idx[j] for j in j_idx])              # (B, n_perm, max_len)
                ridx_t = torch.tensor(ridx.astype(np.int64), device=device)
                best = None
                for pi in range(n_perm):
                    gather = ridx_t[:, pi, :].unsqueeze(-1).expand(-1, -1, 3)  # (B, max_len, 3)
                    coords2_perm = torch.gather(coords2_batch, 1, gather)
                    tm_p = compute_tmscore_batch_iterative(
                        coords1_batch, coords2_perm, lengths_batch, d0_batch, n_iter=5)
                    best = tm_p if best is None else torch.maximum(best, tm_p)
                tm_scores = best
            else:
                tm_scores = compute_tmscore_batch_iterative(
                    coords1_batch, coords2_batch, lengths_batch, d0_batch, n_iter=5
                )

        # Store results
        tm_scores_np = tm_scores.cpu().numpy()
        for idx, (i, j) in enumerate(batch_pairs):
            results[i, j] = tm_scores_np[idx]
            results[j, i] = tm_scores_np[idx]

    # Fill diagonal
    np.fill_diagonal(results, 1.0)

    # Save if output path specified
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, results)
        np.save(output_path.with_name(output_path.stem + '_names.npy'), np.array(names))

    return results, names
