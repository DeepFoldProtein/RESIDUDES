"""
Anchored COM (center-of-mass) distance — Option C-1.

For each pair of structures (i, j):
    1. Kabsch-align model j's anchor onto model i's anchor (or vice versa)
    2. Apply the same transform to each probe chain in model j
    3. For each probe chain, compute its centroid (Cα-COM) in i's anchor frame
    4. Distance(i, j) = sqrt( Σ_chain ||COM_chain(i) - COM_chain(j)||² )

Concretely, all medoids share the same anchor frame and per-chain COMs
become a 3D point per chain. Distance is then L2 across stacked probe
COMs. This is a strict metric and very cheap (few KB per model, no SVD
per pair after pre-computation).

Used for:
  - Ab-antigen: anchor on antigen, measure heavy/light chain placement
  - TCR-pMHC : anchor on MHC, measure TCR αβ positioning
  - generic  : anchor on receptor, measure ligand chain(s) rigid-body pose

Output side-products:
  - feature_matrix : (N, K*3) per-model per-chain COMs in shared anchor frame
                     (saved as `com_features.npy` for 3D visualisation)
  - distance_matrix: (N, N) L2 distance in Å
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Dict, List, Set, Tuple, Union

from tqdm import tqdm

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def _load_anchor_and_probe_chains(
    path: Path,
    anchor_residues: Dict[str, Set[int]],
    probe_chains: List[str],
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Read structure once → return:
      anchor_xyz : (N_anchor, 3) — CA coords for residues listed in anchor_residues
      probe_xyz_by_chain : {chain_id: (N_probe_chain, 3)} — full chain CAs

    Reuses the project's CIF/PDB iterator to keep parsing semantics consistent
    with the rest of the GPU kernel set.
    """
    from lantern_qa.cluster.core.gpu._loader import _gemmi_iter, _pdb_iter, _is_cif

    iterator = _gemmi_iter(path) if _is_cif(path) else _pdb_iter(path)
    anchor_xyz: list = []
    probe_by_chain: Dict[str, list] = {c: [] for c in probe_chains}
    probe_set = set(probe_chains)
    for chain, idx_in_chain, _resid, xyz in iterator:
        if chain in anchor_residues and idx_in_chain in anchor_residues[chain]:
            anchor_xyz.append(xyz)
        if chain in probe_set:
            probe_by_chain[chain].append(xyz)
    return (
        np.array(anchor_xyz, dtype=np.float32),
        {c: np.array(v, dtype=np.float32) for c, v in probe_by_chain.items()},
    )


def _com_load_one(path, anchor_residues, probe_chains):
    """Parse worker (picklable) -> ((anchor_xyz, probe_by_chain), None) on
    success, or (None, err-string) on failure. Lets the parse step run in a
    parallel pool while the downstream Kabsch align stays serial (it needs the
    first valid model as reference frame)."""
    try:
        return _load_anchor_and_probe_chains(path, anchor_residues, probe_chains), None
    except Exception as exc:  # noqa: BLE001
        return None, repr(exc)


def _kabsch_Rt(P: np.ndarray, Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """R, t with P @ R + t ≈ Q (P rotates onto Q)."""
    Pc = P.mean(0); Qc = Q.mean(0)
    H = (P - Pc).T @ (Q - Qc)
    U, _, Vt = np.linalg.svd(H)
    d = float(np.sign(np.linalg.det(Vt.T @ U.T)))
    R = U @ np.diag([1.0, 1.0, d]) @ Vt
    t = Qc - Pc @ R
    return R, t


def compute_com_features(
    pdb_files: List[Union[str, Path]],
    anchor_residues: Dict[str, Set[int]],
    probe_chains: List[str],
    show_progress: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Build per-model COM features in a shared anchor frame.

    Args:
        pdb_files : list of CIF/PDB paths.
        anchor_residues : {chain_id: set(in-chain idx)} — anchor residue mask.
        probe_chains : list of probe chain ids (whole chain; residue-level
                       subset not supported here, use anchored-rmsd for that).
        show_progress : tqdm bar.

    Returns:
        feature_matrix : (N_models, n_probe_chains * 3) — per-model concat
                         of per-chain Cα COMs after Kabsch align of anchor
                         to the FIRST model's anchor.
        com_3d         : (N_models, n_probe_chains, 3) — reshape of features
                         (kept separately for visualisation).
        structure_names: list of input file basenames (matrix row order).
    """
    pdb_files = [Path(p) for p in pdb_files]
    names: List[str] = []
    probe_chains_sorted = sorted(probe_chains)
    n_probe = len(probe_chains_sorted)
    if n_probe == 0:
        raise ValueError("probe_chains must contain at least one chain")

    # First model = reference frame (anchor-Kabsch-aligned to itself == identity)
    # Parse step runs in a parallel pool (pure-CPU I/O); the Kabsch align below
    # stays serial because it references the first valid model as the frame.
    from functools import partial
    from lantern_qa.cluster.core.gpu._loader import _parallel_map
    loaded = _parallel_map(
        partial(_com_load_one, anchor_residues=anchor_residues,
                probe_chains=probe_chains_sorted),
        pdb_files, None, show_progress, "Loading anchor/probe")

    com_buf: List[np.ndarray] = []  # per-model (n_probe, 3)
    ref_anchor: np.ndarray | None = None
    bad: List[Tuple[str, str]] = []

    for path, (res, err) in zip(pdb_files, loaded):
        if res is None:
            bad.append((path.name, err))
            continue
        anchor_xyz, probe_xyz_by_chain = res
        if anchor_xyz.shape[0] < 3:
            bad.append((path.name, f"anchor_xyz too short: {anchor_xyz.shape}"))
            continue
        if any(probe_xyz_by_chain[c].shape[0] == 0 for c in probe_chains_sorted):
            missing = [c for c in probe_chains_sorted if probe_xyz_by_chain[c].shape[0] == 0]
            bad.append((path.name, f"empty probe chain(s): {missing}"))
            continue

        if ref_anchor is None:
            # First model → reference. No rotation needed.
            ref_anchor = anchor_xyz.copy()
            ref_anchor_center = ref_anchor.mean(0)
            # Place model so ref_anchor centroid = origin (clean coordinate frame).
            R = np.eye(3, dtype=np.float32)
            t = -ref_anchor_center
        else:
            n = min(anchor_xyz.shape[0], ref_anchor.shape[0])
            R, t = _kabsch_Rt(anchor_xyz[:n], ref_anchor[:n] - ref_anchor.mean(0))

        coms = np.zeros((n_probe, 3), dtype=np.float32)
        for k, ch in enumerate(probe_chains_sorted):
            xyz = probe_xyz_by_chain[ch]
            aligned = xyz @ R + t
            coms[k] = aligned.mean(0)
        com_buf.append(coms)
        # If file name is "model.cif" (common in AF3-style nested layouts), pick
        # a more informative key from the parent directories: <gen>_seedX_sampleY
        if path.name == "model.cif" and len(path.parts) >= 3:
            try:
                gen = path.parts[-4]   # .../<gen>/seed_X/sample_Y/model.cif
                seed = path.parts[-3]
                sample = path.parts[-2]
                names.append(f"{gen}_{seed}_{sample}".replace("seed_", "seed").replace("sample_", "sample"))
            except Exception:
                names.append(path.name)
        else:
            names.append(path.name)

    if bad and show_progress:
        print(f"  WARN: dropped {len(bad)} models with parse / chain issues "
              f"(first 3: {bad[:3]})")

    com_3d = np.stack(com_buf, axis=0).astype(np.float32)
    feature_matrix = com_3d.reshape(com_3d.shape[0], -1)
    return feature_matrix, com_3d, names


def compute_com_matrix_gpu(
    pdb_files: List[Union[str, Path]],
    anchor_residues: Dict[str, Set[int]],
    probe_chains: List[str],
    batch_size: int = 1024,
    gpu_id: int = 0,
    output_path: Union[str, Path, None] = None,
    show_progress: bool = True,
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """
    All-vs-all anchored-COM distance matrix.

    Returns:
        distance_matrix : (N, N) float32 L2 in Å
        structure_names : list of model file basenames (row/col order)
        feature_matrix  : (N, n_probe_chains * 3) — saved as side-product

    GPU is used only for the pairwise L2 (cheap). The expensive part is
    per-model anchor Kabsch (CPU, vectorised by gemmi+numpy).
    """
    feature_matrix, com_3d, names = compute_com_features(
        pdb_files, anchor_residues, probe_chains,
        show_progress=show_progress,
    )
    N = feature_matrix.shape[0]
    if N == 0:
        raise RuntimeError("No models passed loading / anchor checks")

    # Pairwise L2 on the concatenated (n_probe * 3)-D feature.
    if TORCH_AVAILABLE and torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_id}")
        F = torch.tensor(feature_matrix, device=device, dtype=torch.float32)
        # cdist works in batches; on a 14k-pool feature this is < 1 s.
        if show_progress:
            print(f"  Pairwise L2 on GPU: ({N}, {N})")
        D = torch.cdist(F, F)
        D_np = D.cpu().numpy().astype(np.float32)
        del F, D
        torch.cuda.empty_cache()
    else:
        from scipy.spatial.distance import cdist
        if show_progress:
            print(f"  Pairwise L2 on CPU: ({N}, {N})")
        D_np = cdist(feature_matrix, feature_matrix).astype(np.float32)

    np.fill_diagonal(D_np, 0.0)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, D_np)
        np.save(output_path.with_name(output_path.stem + "_features.npy"),
                feature_matrix)
        np.save(output_path.with_name(output_path.stem + "_com3d.npy"),
                com_3d)
        np.save(output_path.with_name(output_path.stem + "_names.npy"),
                np.array(names))

    return D_np, names, feature_matrix
