"""Numeric metric computation for AF3 results."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import gemmi
import numpy as np

from .reader import SampleFiles

# AF3 distogram bin configuration:
# boundaries = linspace(0, 31, 63) + half-step + catch-all → 64 bins
# centers: 0.25, 0.75, ..., 31.75 Å  (step = 0.5 Å)
_N_BINS = 64
_BIN_MIDS = np.linspace(0.25, 31.75, _N_BINS)  # shape (64,)

# One-letter amino acid codes (index 0-19 = standard AAs, 20 = X/unknown)
_AA_LETTERS = "ACDEFGHIKLMNPQRSTVWY"
_CHAR_TO_IDX: dict[str, int] = {c: i for i, c in enumerate(_AA_LETTERS)}


# ---------------------------------------------------------------------------
# pLDDT & chain utilities
# ---------------------------------------------------------------------------

def chain_boundaries(confidences: dict) -> list[tuple[str, int, int]]:
    """Return [(chain_label, start_token_idx, end_token_idx), ...] 0-based inclusive."""
    token_chain_ids: list[str] = confidences["token_chain_ids"]
    result: list[tuple[str, int, int]] = []
    if not token_chain_ids:
        return result
    current_chain = token_chain_ids[0]
    start = 0
    for i, cid in enumerate(token_chain_ids):
        if cid != current_chain:
            result.append((current_chain, start, i - 1))
            current_chain = cid
            start = i
    result.append((current_chain, start, len(token_chain_ids) - 1))
    return result


def residue_plddt(confidences: dict) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Extract per-residue pLDDT.

    Returns:
        residue_indices:   shape (N_residues,) 1-based global index
        plddt_values:      shape (N_residues,) 0-100
        chain_per_residue: list[str] chain label per residue
    """
    token_chain_ids: list[str] = confidences["token_chain_ids"]
    # Prefer pre-computed per-token pLDDT (from load_confidences in new schema)
    token_plddts = confidences.get("token_plddts")
    if token_plddts is None:
        token_plddts = np.array(confidences["atom_plddts"], dtype=np.float32)
    token_plddts = np.asarray(token_plddts, dtype=np.float32)
    residue_indices = np.arange(1, len(token_chain_ids) + 1, dtype=np.int32)
    return residue_indices, token_plddts, token_chain_ids


def compute_plddt_stats(confidences: dict) -> dict:
    _, plddt, chain_ids = residue_plddt(confidences)
    stats: dict = {
        "mean_plddt": float(plddt.mean()),
        "median_plddt": float(np.median(plddt)),
        "pct_very_low": float((plddt < 50).mean() * 100),
        "pct_low": float(((plddt >= 50) & (plddt < 70)).mean() * 100),
        "pct_confident": float(((plddt >= 70) & (plddt < 90)).mean() * 100),
        "pct_very_high": float((plddt >= 90).mean() * 100),
        "per_chain_mean": {},
    }
    for cid in dict.fromkeys(chain_ids):
        mask = np.array([c == cid for c in chain_ids])
        stats["per_chain_mean"][cid] = float(plddt[mask].mean())
    return stats


# ---------------------------------------------------------------------------
# PAE / contact matrices
# ---------------------------------------------------------------------------

def pae_matrix(confidences: dict) -> np.ndarray:
    return np.array(confidences["pae"], dtype=np.float32)


def contact_prob_matrix(confidences: dict) -> np.ndarray:
    return np.array(confidences["contact_probs"], dtype=np.float32)


def mean_pae_per_chain_pair(
    pae: np.ndarray,
    boundaries: list[tuple[str, int, int]],
) -> dict[tuple[str, str], float]:
    result: dict[tuple[str, str], float] = {}
    for ci, si, ei in boundaries:
        for cj, sj, ej in boundaries:
            block = pae[si:ei + 1, sj:ej + 1]
            result[(ci, cj)] = float(block.mean())
    return result


def compute_pae_stats(confidences: dict) -> dict:
    pae = pae_matrix(confidences)
    bounds = chain_boundaries(confidences)
    off_diag_vals: list[float] = []
    pair_means: dict = {}
    for i, (ci, si, ei) in enumerate(bounds):
        for j, (cj, sj, ej) in enumerate(bounds):
            block = pae[si:ei + 1, sj:ej + 1]
            pair_means[(ci, cj)] = float(block.mean())
            if i != j:
                off_diag_vals.extend(block.flatten().tolist())
    return {
        "mean_pae": float(pae.mean()),
        "max_pae": float(pae.max()),
        "mean_off_diagonal_pae": float(np.mean(off_diag_vals)) if off_diag_vals else float("nan"),
        "per_chain_pair_mean": pair_means,
    }


# ---------------------------------------------------------------------------
# PDE matrix
# ---------------------------------------------------------------------------

def pde_matrix(confidences: dict) -> np.ndarray:
    """(N×N) Predicted Distance Error matrix in Å. Symmetric (unlike PAE)."""
    return np.array(confidences["pde"], dtype=np.float32)


def mean_pde_per_chain_pair(
    pde: np.ndarray,
    boundaries: list[tuple[str, int, int]],
) -> dict[tuple[str, str], float]:
    result: dict[tuple[str, str], float] = {}
    for ci, si, ei in boundaries:
        for cj, sj, ej in boundaries:
            block = pde[si:ei + 1, sj:ej + 1]
            result[(ci, cj)] = float(block.mean())
    return result


def compute_pde_stats(confidences: dict) -> dict:
    pde = pde_matrix(confidences)
    bounds = chain_boundaries(confidences)
    pair_means: dict = {}
    for ci, si, ei in bounds:
        for cj, sj, ej in bounds:
            block = pde[si:ei + 1, sj:ej + 1]
            pair_means[(ci, cj)] = float(block.mean())
    return {
        "mean_pde": float(pde.mean()),
        "max_pde": float(pde.max()),
        "per_chain_pair_mean": pair_means,
    }


# ---------------------------------------------------------------------------
# pDockQ
# ---------------------------------------------------------------------------

def compute_pdockq_for_complex(
    best_sample: SampleFiles,
    summary: dict,
) -> Optional[float]:
    """Compute pDockQ for complex using scoring.pdockq and contact counting."""
    if summary.get("iptm") is None:
        return None

    try:
        from stoichiometry_predictor.subcomponent import _count_chain_pair_contacts
        from stoichiometry_predictor.scoring import pdockq
    except ImportError:
        return None

    contacts = _count_chain_pair_contacts(best_sample.model_cif_path)
    total_contacts = sum(contacts.values())

    cp_iptm = summary.get("chain_pair_iptm", [[]])
    n = len(cp_iptm)
    inter_vals: list[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            v = cp_iptm[i][j]
            if v is not None:
                inter_vals.append(float(v))
    if not inter_vals:
        return None
    avg_inter_iptm = sum(inter_vals) / len(inter_vals)

    return pdockq(total_contacts, avg_inter_iptm)


# ---------------------------------------------------------------------------
# Distogram
# ---------------------------------------------------------------------------

def compute_distogram(npy_path: Path) -> dict:
    """Load distogram_probs.npy (uint8 log-quantized) → decode → expected/argmax.

    Encoding (writer): q = round(-log10(p) / 7.0 × 255), stored as uint8
    Decode: p ≈ 10^(-q / 255 × 7), then renormalize quantization noise.

    Returns:
        distogram_expectation: (N, N) expected CA-CA distance in Å
        distogram_argmax:      (N, N) most probable distance bin centre in Å
        distogram_entropy:     (N, N) Shannon entropy in nats
    """
    q = np.load(npy_path).astype(np.float32)  # uint8 (N, N, 64) → float32
    probs = 10.0 ** (-q / 255.0 * 7.0)         # decode
    probs /= probs.sum(axis=-1, keepdims=True)  # renormalize quantization noise

    dist_expectation = (probs * _BIN_MIDS).sum(axis=-1)
    dist_argmax = _BIN_MIDS[probs.argmax(axis=-1)]

    eps = 1e-9
    entropy = -(probs * np.log(probs + eps)).sum(axis=-1)

    return {
        "distogram_expectation": dist_expectation,
        "distogram_argmax": dist_argmax,
        "distogram_entropy": entropy,
    }


# ---------------------------------------------------------------------------
# Ramachandran
# ---------------------------------------------------------------------------

def compute_ramachandran(
    cif_path: Path,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Compute phi/psi angles using gemmi.calculate_phi_psi (per-residue API).

    Returns phi_degrees, psi_degrees, chain_ids per residue.
    """
    st = gemmi.read_structure(str(cif_path))
    model = st[0]

    phi_vals: list[float] = []
    psi_vals: list[float] = []
    chain_ids: list[str] = []

    for chain in model:
        residues = list(chain)
        n = len(residues)
        for i in range(1, n - 1):
            prev_res = residues[i - 1]
            curr_res = residues[i]
            next_res = residues[i + 1]
            try:
                angles = gemmi.calculate_phi_psi(prev_res, curr_res, next_res)
                if len(angles) >= 2:
                    phi = ((math.degrees(angles[0]) + 180) % 360) - 180
                    psi = ((math.degrees(angles[1]) + 180) % 360) - 180
                    phi_vals.append(phi)
                    psi_vals.append(psi)
                    chain_ids.append(chain.name)
            except Exception:
                continue

    return np.array(phi_vals), np.array(psi_vals), chain_ids


# ---------------------------------------------------------------------------
# MSA
# ---------------------------------------------------------------------------

def parse_a3m(a3m_path: Path) -> np.ndarray:
    """Parse .a3m file into integer matrix (N_seqs x seq_len).

    Lowercase (insertions) removed. '-' = 21, AA = 0-19, unknown = 20.
    """
    sequences: list[str] = []
    current_seq: list[str] = []
    with open(a3m_path) as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if current_seq:
                    sequences.append("".join(current_seq))
                    current_seq = []
            else:
                # Remove lowercase insertions
                current_seq.append("".join(c for c in line if c == "-" or c.isupper()))
    if current_seq:
        sequences.append("".join(current_seq))

    if not sequences:
        return np.zeros((0, 0), dtype=np.int8)

    seq_len = len(sequences[0])
    mat = np.full((len(sequences), seq_len), 21, dtype=np.int8)
    for i, seq in enumerate(sequences):
        for j, c in enumerate(seq[:seq_len]):
            if c == "-":
                mat[i, j] = 21
            else:
                mat[i, j] = _CHAR_TO_IDX.get(c, 20)

    return mat


def compute_msa_visualization(
    a3m_paths: list[Path],
    chain_boundary_positions: list[int] | None = None,
) -> dict:
    """Compute MSA visualization data following CASP-Visualization-demo plot_msa.py logic.

    Each row of `lines` is colored with a UNIFORM scalar = that sequence's identity
    to the query. Gap positions are NaN (displayed white). Sequences are clustered by
    coverage pattern (which chain segments have any non-gap residues), then sorted
    ascending by identity within each cluster (low identity = bottom row).

    Returns dict:
        lines:    (N_seqs, seq_len) float matrix (seqid per non-gap cell, NaN for gaps)
        ln:       chain boundary positions [0, boundary1, ..., seq_len]
        nn:       cluster group cumulative boundaries [0, g1, g1+g2, ...]
        coverage: (seq_len,) int — number of non-gap sequences per position
        neff:     float — log2 of effective number of sequences
    """
    if not a3m_paths:
        return {"lines": [], "ln": [], "nn": [], "coverage": [], "neff": 0.0}

    # Load and concatenate MSA matrices
    mats: list[np.ndarray] = []
    for path in a3m_paths:
        m = parse_a3m(path)
        if m.size > 0:
            mats.append(m)

    if not mats:
        return {"lines": [], "ln": [], "nn": [], "coverage": [], "neff": 0.0}

    if len(mats) == 1:
        msa = mats[0]
    else:
        # Complex: horizontally concatenate paired MSAs (same number of sequences)
        max_seqs = max(m.shape[0] for m in mats)
        padded: list[np.ndarray] = []
        for m in mats:
            if m.shape[0] < max_seqs:
                pad = np.full((max_seqs - m.shape[0], m.shape[1]), 21, dtype=np.int8)
                m = np.vstack([m, pad])
            padded.append(m)
        msa = np.hstack(padded)

    n_seqs, seq_len = msa.shape
    query = msa[0]

    # Determine chain segment lengths for gapid computation
    if chain_boundary_positions:
        boundaries = [0] + list(chain_boundary_positions) + [seq_len]
        ls = [boundaries[i + 1] - boundaries[i] for i in range(len(boundaries) - 1)]
    else:
        ls = [seq_len]
    ln_arr = np.cumsum([0] + ls)  # length = len(ls)+1

    gap = msa != 21   # (n_seqs, seq_len) bool — True = non-gap
    qid = msa == query[None, :]  # (n_seqs, seq_len) bool — True = matches query

    # gapid[i] = binary vector indicating which chain segments have ≥1 non-gap residue
    # This is used to cluster sequences by their coverage pattern (same as original code)
    gapid = np.stack(
        [gap[:, ln_arr[i]:ln_arr[i + 1]].max(-1) for i in range(len(ls))],
        axis=-1,
    )  # (n_seqs, n_chains)

    lines_parts: list[np.ndarray] = []
    nn: list[int] = [0]

    for g in np.unique(gapid, axis=0):
        # Sequences belonging to this coverage pattern
        mask = np.where((gapid == g).all(axis=-1))[0]
        qid_g = qid[mask]
        gap_g = gap[mask]

        # Per-sequence identity: mean(match & non-gap) across all covered chain segments
        # Averaged only over segments with coverage (g[i] = 1)
        g_sum = float(g.sum()) + 1e-8
        seqid = np.stack(
            [qid_g[:, ln_arr[i]:ln_arr[i + 1]].mean(-1) for i in range(len(ls))],
            axis=-1,
        ).sum(-1) / g_sum  # (n_group,) scalar identity per sequence

        # Sort ascending by seqid (low identity = row 0 = bottom of plot)
        sorted_idx = seqid.argsort()
        seqid_sorted = seqid[sorted_idx]   # ascending (low → high)
        gap_sorted = gap_g[sorted_idx].astype(np.float32)
        gap_sorted[gap_sorted == 0.0] = np.nan  # NaN for gaps

        # Each row colored uniformly by its OWN seqid value
        lines_g = gap_sorted * seqid_sorted[:, None]

        lines_parts.append(lines_g)
        nn.append(nn[-1] + len(lines_g))

    lines_mat = np.concatenate(lines_parts, axis=0)  # (n_seqs, seq_len)

    # Coverage: number of non-gap sequences per position (from original msa)
    coverage = gap.sum(axis=0).tolist()

    n_eff = _compute_neff(msa[:min(n_seqs, 500)], theta=0.38)

    return {
        "lines": lines_mat.tolist(),
        "ln": ln_arr.tolist(),
        "nn": nn,
        "coverage": coverage,
        "neff": float(n_eff),
    }


def _compute_neff(msa: np.ndarray, theta: float = 0.38) -> float:
    """Compute log2 of effective sequences with Hamming similarity clustering."""
    n_seqs, seq_len = msa.shape
    if n_seqs == 0:
        return 0.0

    # For large MSAs, subsample to avoid O(N^2) compute
    max_seqs = min(n_seqs, 500)
    if n_seqs > max_seqs:
        indices = np.random.choice(n_seqs, max_seqs, replace=False)
        msa = msa[indices]
        n_seqs = max_seqs

    # Hamming distance (excluding gaps)
    non_gap = (msa != 21).astype(np.float32)
    matches = (msa[:, None, :] == msa[None, :, :]).astype(np.float32)
    shared = non_gap[:, None, :] * non_gap[None, :, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        sim = np.where(
            shared.sum(axis=-1) > 0,
            (matches * shared).sum(axis=-1) / shared.sum(axis=-1),
            0.0,
        )
    similar = (sim >= (1.0 - theta)).astype(np.float32)
    weights = 1.0 / similar.sum(axis=-1)
    neff = math.log2(weights.sum()) if weights.sum() > 1 else 0.0
    return neff
