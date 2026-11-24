"""
Unified CA-atom loader for the GPU distance-matrix kernels.

Dispatches by file extension:
    .pdb / .ent          -> fast text parser (column-fixed PDB ATOM CA)
    .cif / .mmcif / .gz  -> gemmi (handles CIF + gzipped variants)

Supports an optional `exclude_indices` mask of the form
    {chain_id: set_of_zero_based_residue_positions}
which drops residues *before* coordinates are returned. Used by the
CDR-aware clustering wrapper (analysis/MassiveClustering_tool/run_clustering.py).
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Union

import numpy as np


def _is_cif(path: Path) -> bool:
    name = path.name.lower()
    return (
        name.endswith(".cif")
        or name.endswith(".cif.gz")
        or name.endswith(".mmcif")
        or name.endswith(".mmcif.gz")
    )


def _gemmi_iter(path: Path):
    """
    Yield (chain_id, residue_idx_in_chain, residue_id, ca_xyz) for every
    polymer residue with a CA atom in the first model.
    """
    import gemmi  # local import keeps gemmi a soft dep at module-import time

    try:
        st = gemmi.read_structure(str(path))
    except (IndexError, RuntimeError, ValueError, PermissionError, FileNotFoundError, OSError):
        # Empty or corrupted CIF (e.g. 0-byte file from failed inference).
        # Yield nothing so caller treats as a zero-atom structure and skips.
        return
    if len(st) == 0:
        return
    model = st[0]
    for chain in model:
        for i, residue in enumerate(chain):
            ca = residue.find_atom("CA", "*")
            if ca is None:
                # Some CIFs use altloc; fall back to any CA-named atom
                for atom in residue:
                    if atom.name == "CA":
                        ca = atom
                        break
            if ca is None:
                continue
            yield chain.name, i, residue.seqid.num, (ca.pos.x, ca.pos.y, ca.pos.z)


def _pdb_iter(path: Path):
    """Replicates the original column-fixed PDB ATOM CA parser."""
    chain_idx_counter: dict[str, int] = {}
    with open(path, "r") as f:
        last_resseq: dict[str, int | None] = {}
        for line in f:
            if not line.startswith("ATOM"):
                continue
            if line[12:16].strip() != "CA":
                continue
            chain = line[21]
            try:
                res_id = int(line[22:26])
            except ValueError:
                continue
            # Per-chain in-chain index — increments only on residue change.
            prev = last_resseq.get(chain)
            if prev != res_id:
                chain_idx_counter[chain] = chain_idx_counter.get(chain, -1) + 1
                last_resseq[chain] = res_id
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue
            yield chain, chain_idx_counter[chain], res_id, (x, y, z)


def parse_structure_ca(
    path: Union[str, Path],
    exclude_indices: dict[str, set[int]] | None = None,
) -> np.ndarray:
    """Return (N, 3) float32 array of CA coordinates."""
    p = Path(path)
    iterator = _gemmi_iter(p) if _is_cif(p) else _pdb_iter(p)
    coords = []
    for chain, idx_in_chain, _resid, xyz in iterator:
        if exclude_indices and idx_in_chain in exclude_indices.get(chain, ()):
            continue
        coords.append(xyz)
    if not coords:
        return np.zeros((0, 3), dtype=np.float32)
    return np.array(coords, dtype=np.float32)


def _resolve_workers(n_items: int, workers: int | None = None) -> int:
    """How many parse workers: explicit > $SLURM_CPUS_PER_TASK > cpu_count,
    capped at the number of items (never more workers than files)."""
    import os
    import multiprocessing as mp
    if workers is None:
        workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "0") or "0") or mp.cpu_count()
    return max(1, min(int(workers), max(1, n_items)))


def _parallel_map(fn, paths, workers, show_progress, desc):
    """Order-preserving parallel map (fork Pool + imap) with a serial fallback.

    Structure parsing is pure-CPU gemmi/text I/O (no CUDA in the workers), so a
    fork pool is safe even after the parent has initialised a CUDA context. Any
    failure (pickling, fork, ...) transparently falls back to the serial path,
    so results are always identical to the single-thread version.
    """
    from tqdm import tqdm
    n = len(paths)
    workers = _resolve_workers(n, workers)
    if workers > 1 and n > 1:
        try:
            import multiprocessing as mp
            ctx = mp.get_context("fork")
            with ctx.Pool(workers) as pool:
                it = pool.imap(fn, paths, chunksize=16)   # imap => input order preserved
                if show_progress:
                    it = tqdm(it, total=n, desc=f"{desc} ({workers}w)")
                return list(it)
        except Exception as exc:  # noqa: BLE001 — any failure => safe serial path
            print(f"  [loader] parallel load failed ({exc!r}); serial fallback")
    it = tqdm(paths, desc=desc) if show_progress else paths
    return [fn(p) for p in it]


def parse_structure_ca_batch(
    paths,
    exclude_indices: dict[str, set[int]] | None = None,
    workers: int | None = None,
    show_progress: bool = True,
) -> list:
    """Parallel CA-coord load. Returns list of (N,3) arrays in input order."""
    from functools import partial
    fn = partial(parse_structure_ca, exclude_indices=exclude_indices)
    return _parallel_map(fn, [Path(p) for p in paths], workers, show_progress,
                         "Loading structures")


def parse_structure_with_chains_batch(
    paths,
    exclude_indices: dict[str, set[int]] | None = None,
    workers: int | None = None,
    show_progress: bool = True,
) -> list:
    """Parallel load of (coords, chain_ids, residue_ids) tuples, input order."""
    from functools import partial
    fn = partial(parse_structure_with_chains, exclude_indices=exclude_indices)
    return _parallel_map(fn, [Path(p) for p in paths], workers, show_progress,
                         "Loading structures")


def parse_structure_with_chains(
    path: Union[str, Path],
    exclude_indices: dict[str, set[int]] | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (coords, chain_ids, residue_ids) — the inputs that interface-TM
    needs to detect chain boundaries."""
    p = Path(path)
    iterator = _gemmi_iter(p) if _is_cif(p) else _pdb_iter(p)
    coords = []
    chain_ids = []
    residue_ids = []
    for chain, idx_in_chain, resid, xyz in iterator:
        if exclude_indices and idx_in_chain in exclude_indices.get(chain, ()):
            continue
        coords.append(xyz)
        chain_ids.append(chain)
        residue_ids.append(resid)
    if not coords:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.array([], dtype=object),
            np.array([], dtype=np.int32),
        )
    return (
        np.array(coords, dtype=np.float32),
        np.array(chain_ids),
        np.array(residue_ids),
    )
