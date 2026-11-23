#!/usr/bin/env python3
"""
GPU-accelerated dRMSD (distance-based RMSD) calculation

dRMSD computes RMSD of internal distance matrices without alignment.
Useful for comparing local structure similarity.
"""

import os
import math
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Union
from scipy.spatial.distance import pdist
from tqdm import tqdm

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def parse_pdb_ca(
    pdb_path: Union[str, Path],
    exclude_indices: Optional[dict] = None,
) -> np.ndarray:
    """Extract CA atom coordinates from PDB or CIF (gemmi dispatch).

    `exclude_indices` is `{chain_id: set_of_zero_based_residue_positions}` —
    drops those residues before returning. Used by CDR-aware clustering.
    """
    from lantern_qa.cluster.core.gpu._loader import parse_structure_ca
    return parse_structure_ca(pdb_path, exclude_indices=exclude_indices)


def compute_distance_vector(coords: np.ndarray) -> np.ndarray:
    """
    Compute condensed pairwise distance vector.

    Args:
        coords: Atom coordinates (n_atoms, 3)

    Returns:
        Condensed distance vector (n_atoms * (n_atoms-1) / 2,)
    """
    return pdist(coords).astype(np.float32)


def compute_drmsd_cpu(
    dist_vec1: np.ndarray,
    dist_vec2: np.ndarray,
) -> float:
    """
    Compute dRMSD between two distance vectors.

    Args:
        dist_vec1: Distance vector 1
        dist_vec2: Distance vector 2

    Returns:
        dRMSD value
    """
    diff = dist_vec1 - dist_vec2
    return np.sqrt(np.mean(diff ** 2))


def compute_drmsd_matrix_gpu(
    pdb_files: List[Union[str, Path]],
    batch_size: int = 50,
    gpu_id: int = 0,
    output_path: Optional[Union[str, Path]] = None,
    show_progress: bool = True,
    residue_mask: Optional[np.ndarray] = None,
    exclude_indices: Optional[dict] = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    Compute all-vs-all dRMSD matrix using GPU.

    Args:
        pdb_files: List of PDB file paths
        batch_size: Row batch size for GPU computation
        gpu_id: GPU device ID
        output_path: Optional path to save results
        show_progress: Show progress bar
        residue_mask: Optional boolean mask for selecting residues

    Returns:
        Tuple of (dRMSD matrix, structure names)

    Example:
        >>> pdbs = list(Path("predictions/").glob("*.pdb"))
        >>> matrix, names = compute_drmsd_matrix_gpu(pdbs, batch_size=50)
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for GPU computation")

    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    n = len(pdb_files)

    # Load structures and compute distance vectors
    names = []
    coords_list = []

    from lantern_qa.cluster.core.gpu._loader import parse_structure_ca_batch
    names = [Path(p).name for p in pdb_files]
    coords_list = parse_structure_ca_batch(
        pdb_files, exclude_indices=exclude_indices, show_progress=show_progress)

    # Apply residue mask if provided
    if residue_mask is not None:
        coords_list = [coords[residue_mask] for coords in coords_list]

    # --- Host-RAM guard --------------------------------------------------
    # The condensed internal-distance vector is O(L^2) per model, and we
    # materialise one per structure in a single host array. For large
    # structures (e.g. a 1455-res x3 spike trimer => L=4365 => ~38 MB/model
    # => ~347 GB for 9k models) this OOM-kills the process before the GPU is
    # ever touched. Uniformly subsample CA atoms by a stride so the
    # dist_vectors array fits a host-RAM budget (override via
    # DRMSD_HOST_BUDGET_GB). Stride is positional, hence identical across
    # structures => dRMSD stays comparable (just coarser).
    n_atoms = coords_list[0].shape[0]
    n_dist_full = n_atoms * (n_atoms - 1) // 2
    host_budget_gb = float(os.environ.get("DRMSD_HOST_BUDGET_GB", "32"))
    proj_gb = n * n_dist_full * 4 / 1e9
    if proj_gb > host_budget_gb:
        target_atoms = max(64, int(math.sqrt(2 * host_budget_gb * 1e9 / (n * 4))))
        stride = max(1, math.ceil(n_atoms / target_atoms))
        if stride > 1:
            coords_list = [coords[::stride] for coords in coords_list]
            if show_progress:
                print(f"  ! dist_vectors would need ~{proj_gb:.0f} GB host RAM "
                      f"(> {host_budget_gb:.0f} GB budget) — subsampling CA every "
                      f"{stride} residues: {n_atoms} -> {coords_list[0].shape[0]} atoms")

    # Compute distance vectors
    if show_progress:
        print("Computing distance vectors...")
    dist_vectors = np.array([compute_distance_vector(coords) for coords in coords_list])

    n_dist = dist_vectors.shape[1]

    # Check memory and pick chunked vs full-upload path
    mem_gb = n * n_dist * 4 / 1e9
    if show_progress:
        print(f"  Data size: {mem_gb:.2f} GB")

    # Per-GPU budget based on **currently free** memory, not total — Slurm
    # binds one GPU per job but if multiple jobs land on the same physical
    # GPU (e.g. via co-scheduling) we'd OOM on the upload. Use free with a
    # 25% headroom for cuBLAS workspace + the temporary inner-product.
    if torch.cuda.is_available():
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        free_gb = free_bytes / 1e9
        total_gb = total_bytes / 1e9
        # Need 2x data size on GPU (upload tensor + inner-product output),
        # so budget = 75% of free divided by ~2.
        budget_gb = free_gb * 0.75 / 2.0
        if show_progress:
            print(f"  GPU mem: free={free_gb:.1f}/{total_gb:.1f} GB → "
                  f"budget={budget_gb:.1f} GB")
    else:
        budget_gb = 16.0  # CPU fallback assumption

    if mem_gb < budget_gb:
        # ---- Full-upload path (original) ----
        if show_progress:
            print(f"  full-upload path (fits in {budget_gb:.1f} GB budget)")
        all_data = torch.tensor(dist_vectors, device=device, dtype=torch.float32)
        # gram-matrix identity:  ||a-b||^2 = ||a||^2 + ||b||^2 - 2<a,b>
        norms = (all_data * all_data).sum(dim=1)
        inner = all_data @ all_data.t()
        sq_diff = norms.unsqueeze(0) + norms.unsqueeze(1) - 2.0 * inner
        sq_diff.clamp_(min=0.0)
        drmsd_gpu = torch.sqrt(sq_diff / float(n_dist))
        result = drmsd_gpu.cpu().numpy().astype(np.float32)
        del all_data, inner, sq_diff, drmsd_gpu, norms
    else:
        # ---- Chunked path: tile the matmul ----
        # Pick chunk_rows so that (chunk_rows × n_dist × 4) ≤ ~25% of budget
        # (we need to hold 2 such row-blocks on GPU during the (i,j) matmul.)
        max_rows = int(budget_gb * 1e9 * 0.25 / (n_dist * 4))
        chunk = max(64, min(max_rows, n))
        n_tiles = (n + chunk - 1) // chunk
        if show_progress:
            print(f"  chunked path: {n_tiles} row-tiles × {chunk} rows × "
                  f"{n_dist} dims  (budget {budget_gb:.1f} GB)")

        # Pre-compute per-row norms in chunks
        norms_np = np.zeros(n, dtype=np.float32)
        for i in range(0, n, chunk):
            ii = min(i + chunk, n)
            arr = torch.tensor(dist_vectors[i:ii], device=device, dtype=torch.float32)
            norms_np[i:ii] = (arr * arr).sum(dim=1).cpu().numpy()
            del arr
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Compute dRMSD block-by-block via gram identity
        result = np.zeros((n, n), dtype=np.float32)
        for i in range(0, n, chunk):
            ii = min(i + chunk, n)
            arr_i = torch.tensor(dist_vectors[i:ii], device=device, dtype=torch.float32)
            for j in range(0, n, chunk):
                jj = min(j + chunk, n)
                if j == i:
                    arr_j = arr_i
                else:
                    arr_j = torch.tensor(
                        dist_vectors[j:jj], device=device, dtype=torch.float32
                    )
                inner_block = (arr_i @ arr_j.t()).cpu().numpy()
                sq_diff = (
                    norms_np[i:ii, None] + norms_np[None, j:jj]
                    - 2.0 * inner_block
                )
                np.maximum(sq_diff, 0.0, out=sq_diff)
                result[i:ii, j:jj] = np.sqrt(sq_diff / float(n_dist))
                if j != i:
                    del arr_j
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            del arr_i
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if show_progress:
                done = ii
                print(f"    chunked rows {done}/{n}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Symmetrise (matmul + clamp can break exact symmetry by ULPs) + zero diag
    result = (result + result.T) / 2.0
    np.fill_diagonal(result, 0)

    # Save if output path specified
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, result)
        np.save(output_path.with_name(output_path.stem + '_names.npy'), np.array(names))

    return result, names


def compute_drmsd_matrix_multi_gpu(
    pdb_files: List[Union[str, Path]],
    n_gpus: int = 4,
    batch_size: int = 50,
    output_path: Optional[Union[str, Path]] = None,
    show_progress: bool = True,
) -> Tuple[np.ndarray, List[str]]:
    """
    Compute dRMSD matrix using multiple GPUs.

    Splits rows across GPUs for parallel processing.

    Args:
        pdb_files: List of PDB file paths
        n_gpus: Number of GPUs to use
        batch_size: Batch size per GPU
        output_path: Optional path to save results
        show_progress: Show progress bar

    Returns:
        Tuple of (dRMSD matrix, structure names)
    """
    import threading

    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for GPU computation")

    n = len(pdb_files)

    # Load structures (parallel; also fixes a latent NameError — this fn never
    # had an `exclude_indices` parameter yet referenced it here).
    from lantern_qa.cluster.core.gpu._loader import parse_structure_ca_batch
    names = [Path(p).name for p in pdb_files]
    coords_list = parse_structure_ca_batch(pdb_files, show_progress=show_progress)

    # Compute distance vectors
    if show_progress:
        print("Computing distance vectors...")
    dist_vectors = np.array([compute_distance_vector(coords) for coords in coords_list])

    # Split across GPUs
    result_dict = {}
    rows_per_gpu = (n + n_gpus - 1) // n_gpus

    def process_gpu(gpu_id, row_start, row_end):
        device = torch.device(f'cuda:{gpu_id}')

        # Transfer all data to GPU
        all_data = torch.tensor(dist_vectors, device=device, dtype=torch.float32)

        n_rows = row_end - row_start
        partial_result = torch.zeros((n_rows, n), device=device, dtype=torch.float32)

        for b_start in range(0, n_rows, batch_size):
            b_end = min(b_start + batch_size, n_rows)
            actual_start = row_start + b_start
            actual_end = row_start + b_end

            batch_rows = all_data[actual_start:actual_end]
            diff = batch_rows.unsqueeze(1) - all_data.unsqueeze(0)
            drmsd = torch.sqrt((diff ** 2).mean(dim=2))
            partial_result[b_start:b_end] = drmsd

            del diff, drmsd

        result_dict[gpu_id] = (row_start, row_end, partial_result.cpu().numpy())

        del all_data, partial_result
        torch.cuda.empty_cache()

    # Launch threads
    threads = []
    for gpu_id in range(n_gpus):
        row_start = gpu_id * rows_per_gpu
        row_end = min((gpu_id + 1) * rows_per_gpu, n)
        if row_start < n:
            t = threading.Thread(target=process_gpu, args=(gpu_id, row_start, row_end))
            threads.append(t)
            t.start()

    for t in threads:
        t.join()

    # Combine results
    result = np.zeros((n, n), dtype=np.float32)
    for gpu_id, (row_start, row_end, partial) in result_dict.items():
        result[row_start:row_end] = partial

    result = (result + result.T) / 2
    np.fill_diagonal(result, 0)

    # Save
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, result)
        np.save(output_path.with_name(output_path.stem + '_names.npy'), np.array(names))

    return result, names
