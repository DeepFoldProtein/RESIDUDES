"""ProteinMPNN-based scorers — global NLL and interface-only NLL.

Interface = residues with cross-chain CB–CB distance < cutoff (default 8 Å).
Returns mean per-residue NLL on the interface mask. Lower is better.

Reference: Dauparas et al. 2022 (Science) — ProteinMPNN.
Validated finding (CASP15/16, MF, Ab-Ag): MPNN > all physics terms for cross-pred
lDDT; mpnn_iface adds DockQ/ICS signal where global NLL plateaus.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import gemmi

AA20 = {'ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE',
        'LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL'}


def interface_mask(pdb_path: str | Path, cutoff: float = 8.0) -> np.ndarray:
    """Boolean mask aligned with ProteinMPNN per-residue order.

    Residue is interface if its CB (or CA for GLY) lies within `cutoff` Å of
    any CB on a different chain. Order matches MPNN's chain-then-residue
    iteration over residues that pass the backbone-completeness check
    (N, CA, C, O present).
    """
    st = gemmi.read_structure(str(pdb_path))
    coords, chains = [], []
    for model in st:
        for ch in model:
            for r in ch:
                if r.name not in AA20:
                    continue
                bb = [r.find_atom(a, '*') for a in ('N', 'CA', 'C', 'O')]
                if not all(bb):
                    continue
                cb = r.find_atom('CB', '*') or bb[1]  # GLY → CA
                coords.append([cb.pos.x, cb.pos.y, cb.pos.z])
                chains.append(ch.name)
        break  # first model only

    n = len(coords)
    if n == 0:
        return np.zeros(0, dtype=bool)
    coords = np.asarray(coords)
    chains = np.asarray(chains)
    mask = np.zeros(n, dtype=bool)
    for i in range(n):
        diff = chains != chains[i]
        if not diff.any():
            continue
        d = np.linalg.norm(coords[diff] - coords[i], axis=1)
        if (d < cutoff).any():
            mask[i] = True
    return mask


def mpnn_iface_score(per_residue_nll: np.ndarray, mask: np.ndarray) -> float:
    """Mean per-residue NLL on the interface mask. NaN if no interface."""
    per_residue_nll = np.asarray(per_residue_nll)
    if len(per_residue_nll) != len(mask) or not mask.any():
        return float('nan')
    return float(per_residue_nll[mask].mean())


def score_pdb(pdb_path: str | Path, mpnn_scorer=None, cutoff: float = 8.0) -> dict:
    """Score a PDB with MPNN (global + interface).

    If `mpnn_scorer` is None, an `MPNNScorer` is auto-constructed using the
    bundled MassiveClustering_tool/proteinmpnn/run/ directory (requires
    `bash MassiveClustering_tool/proteinmpnn/setup_weights.sh` to have run
    once).
    """
    if mpnn_scorer is None:
        from lantern_qa.ema.scorers.proteinmpnn_score import MPNNScorer
        mpnn_scorer = MPNNScorer()
    res = mpnn_scorer.score_structure(str(pdb_path))
    prs = np.asarray(res.per_residue_score)
    mask = interface_mask(pdb_path, cutoff=cutoff)
    return {
        'mpnn_global': float(prs.mean()),
        'mpnn_iface': mpnn_iface_score(prs, mask),
        'n_iface': int(mask.sum()),
        'n_total': int(len(mask)),
    }
