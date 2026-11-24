"""
Anchor-aligned probe RMSD / TM-score on GPU.

For each pair of structures (i, j):
    1. Kabsch alignment on `anchor_residues` only -> rotation R, anchor centroids
    2. Apply R + anchor-centroid translation to `probe_residues` of i
    3. Compute either RMSD (Å) or TM-score on the now-aligned probe coords
       (the same `compute_d0(L_probe)` formula used elsewhere in the package)

The metric decouples *alignment frame* (anchor) from *measurement target*
(probe), which the existing TM/iTM/dRMSD kernels conflate. Useful for:

- pMHC: anchor on MHC+β2m, measure peptide pose variability
- TCR-pMHC: anchor on MHC, measure TCR docking variability
- antibody / TCR: anchor on framework, measure CDR loop position

Anchor and probe must be disjoint chain sets. Within each chain we keep
the *full* chain (`anchor_residues` / `probe_residues` are sets of in-chain
indices to KEEP; v1 just keeps the whole chain — residue-level selection
is a v2 feature).

Reuses `_kabsch_R` from tmscore.py and `compute_d0` for d0 normalisation.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

from tqdm import tqdm

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def _split_anchor_probe(
    path: Path,
    anchor_residues: Dict[str, Set[int]],
    probe_residues: Dict[str, Set[int]],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read structure once and emit (anchor_coords, probe_coords) as two separate
    (N, 3) float32 arrays. Uses the same iterator the existing _loader uses,
    so PDB / CIF / .gz are all supported.

    Residues are kept iff their (chain_id, in_chain_index) is in the
    corresponding KEEP set. Chains absent from the KEEP map contribute
    nothing to that side.
    """
    from residudes_qa.cluster.core.gpu._loader import _gemmi_iter, _pdb_iter, _is_cif

    iterator = _gemmi_iter(path) if _is_cif(path) else _pdb_iter(path)
    anchor_xyz: list[tuple[float, float, float]] = []
    probe_xyz: list[tuple[float, float, float]] = []
    for chain, idx_in_chain, _resid, xyz in iterator:
        if chain in anchor_residues and idx_in_chain in anchor_residues[chain]:
            anchor_xyz.append(xyz)
        # Note: not elif — when anchor and probe sets overlap (allowed via
        # --allow-self-overlap for chain-restricted self comparison), the same
        # residue contributes to BOTH lists. For disjoint sets this is
        # equivalent to the original elif branch.
        if chain in probe_residues and idx_in_chain in probe_residues[chain]:
            probe_xyz.append(xyz)
        # Residues outside both sets are silently skipped (not in anchor, not in probe).
    return (
        np.array(anchor_xyz, dtype=np.float32),
        np.array(probe_xyz, dtype=np.float32),
    )


def compute_anchored_matrix_gpu(
    pdb_files: List[Union[str, Path]],
    anchor_residues: Dict[str, Set[int]],
    probe_residues: Dict[str, Set[int]],
    metric: str = "rmsd",
    batch_size: int = 1024,
    gpu_id: int = 0,
    output_path: Optional[Union[str, Path]] = None,
    show_progress: bool = True,
) -> Tuple[np.ndarray, List[str]]:
    """
    Compute the (N, N) anchor-aligned probe matrix.

    Args:
        pdb_files: list of PDB or CIF paths.
        anchor_residues: {chain_id: set(in_chain_indices)} — residues used for
            Kabsch alignment. Set the whole chain via `set(range(chain_len))`.
        probe_residues: {chain_id: set(in_chain_indices)} — residues whose
            distance gets measured after alignment. Disjoint from anchor.
        metric: "rmsd" -> probe RMSD in Å (distance, lower=more similar)
                "tm"   -> anchored TM-score (similarity, 0..1; use 1 - x as
                          a distance for clustering)
        batch_size, gpu_id, output_path, show_progress: standard.

    Returns:
        (matrix, names) — matrix is float32 (N, N), names is the list of
        Path(p).name for each input.
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for GPU computation.")
    if metric not in ("rmsd", "tm"):
        raise ValueError(f"metric must be 'rmsd' or 'tm', got {metric!r}")

    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    n = len(pdb_files)

    # ─── Load structures, split each into anchor/probe coords ───────────────
    names: list[str] = []
    anchor_list: list[np.ndarray] = []
    probe_list: list[np.ndarray] = []

    from functools import partial
    from residudes_qa.cluster.core.gpu._loader import _parallel_map
    names = [Path(f).name for f in pdb_files]
    _ap = _parallel_map(
        partial(_split_anchor_probe, anchor_residues=anchor_residues,
                probe_residues=probe_residues),
        [Path(f) for f in pdb_files], None, show_progress, "Loading anchor/probe coords")
    anchor_list = [a for (a, _b) in _ap]
    probe_list = [b for (_a, b) in _ap]

    # Sanity: every structure must contribute at least 3 anchor atoms (for
    # Kabsch to be defined) and >=1 probe atom. Same shapes are required so
    # that batched tensor ops work — caller must already have run the
    # sequence-consistency guard.
    n_anchor = anchor_list[0].shape[0]
    n_probe = probe_list[0].shape[0]
    if n_anchor < 3:
        raise ValueError(
            f"anchor has {n_anchor} atoms — need >= 3 for Kabsch. "
            f"Did anchor_residues match the chain composition?"
        )
    if n_probe < 1:
        raise ValueError(
            f"probe has {n_probe} atoms — nothing to measure. "
            f"Did probe_residues match the chain composition?"
        )
    for i in range(1, n):
        if anchor_list[i].shape[0] != n_anchor or probe_list[i].shape[0] != n_probe:
            raise ValueError(
                f"structure {i} ({names[i]}) has anchor={anchor_list[i].shape[0]}, "
                f"probe={probe_list[i].shape[0]} but expected anchor={n_anchor}, "
                f"probe={n_probe}. Run the sequence-consistency guard first."
            )

    if show_progress:
        print(f"  anchor: {n_anchor} atoms,  probe: {n_probe} atoms,  N={n} structures")

    # ─── Move to GPU as one big tensor ──────────────────────────────────────
    anchor_all = torch.tensor(np.stack(anchor_list), device=device, dtype=torch.float32)
    probe_all = torch.tensor(np.stack(probe_list), device=device, dtype=torch.float32)

    # Anchor centroids — one per structure
    anchor_centroids = anchor_all.mean(dim=1, keepdim=True)  # (N, 1, 3)

    # Centred anchor coords (will be reused across all pairs involving it).
    anchor_centred = anchor_all - anchor_centroids  # (N, n_anchor, 3)

    # Centre probe coords by the SAME anchor centroid so probe stays in the
    # anchor's local frame.
    probe_centred = probe_all - anchor_centroids  # (N, n_probe, 3)

    del anchor_all, probe_all  # the centred versions are what we need

    # d0 for TM
    if metric == "tm":
        from residudes_qa.cluster.core.gpu.tmscore import compute_d0
        d0_value = float(compute_d0(n_probe))

    # ─── Pair iteration ─────────────────────────────────────────────────────
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    results = np.zeros((n, n), dtype=np.float32)

    pair_iterator = range(0, len(pairs), batch_size)
    if show_progress:
        pair_iterator = tqdm(
            pair_iterator,
            desc=f"Computing anchored-{metric}",
            total=(len(pairs) + batch_size - 1) // batch_size,
        )

    from residudes_qa.cluster.core.gpu.tmscore import _kabsch_R

    for batch_start in pair_iterator:
        batch_end = min(batch_start + batch_size, len(pairs))
        batch = pairs[batch_start:batch_end]
        if not batch:
            continue

        idx_i = torch.tensor([p[0] for p in batch], device=device, dtype=torch.long)
        idx_j = torch.tensor([p[1] for p in batch], device=device, dtype=torch.long)

        anchor_i = anchor_centred[idx_i]  # (B, n_anchor, 3)
        anchor_j = anchor_centred[idx_j]
        probe_i = probe_centred[idx_i]    # (B, n_probe, 3)
        probe_j = probe_centred[idx_j]

        # Kabsch on anchor (uniform weights — anchor is treated as rigid)
        R = _kabsch_R(anchor_i, anchor_j, weights=None)  # (B, 3, 3)

        # Rotate probe_i into the j-anchor frame.
        # anchor_i and anchor_j are both already centred at their own centroid.
        # Kabsch R rotates anchor_i (centred) onto anchor_j (centred), so
        # applying it to probe_i (which sits at anchor_i's local origin) puts
        # probe_i in the same coordinate system as probe_j.
        probe_i_aligned = torch.bmm(probe_i, R.transpose(1, 2))  # (B, n_probe, 3)

        diff = probe_i_aligned - probe_j  # (B, n_probe, 3)

        if metric == "rmsd":
            # sqrt(mean(|diff|^2)) per pair
            sq = (diff * diff).sum(dim=2)  # (B, n_probe)
            metric_vals = torch.sqrt(sq.mean(dim=1))  # (B,)
        else:  # tm
            d_per = torch.norm(diff, dim=2)  # (B, n_probe)
            tm_terms = 1.0 / (1.0 + (d_per / d0_value) ** 2)
            metric_vals = tm_terms.mean(dim=1)  # (B,)

        vals_np = metric_vals.cpu().numpy()
        for k, (i, j) in enumerate(batch):
            results[i, j] = vals_np[k]
            results[j, i] = vals_np[k]

        del anchor_i, anchor_j, probe_i, probe_j, R, probe_i_aligned, diff, metric_vals

    # Diagonal — identical pair: RMSD = 0, TM = 1
    if metric == "rmsd":
        np.fill_diagonal(results, 0.0)
    else:
        np.fill_diagonal(results, 1.0)

    # ─── Save ───────────────────────────────────────────────────────────────
    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out, results)
        np.save(out.with_name(out.stem + "_names.npy"), np.array(names))

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results, names
