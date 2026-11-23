"""
ipSAE — refactored callable API.

Upstream source: https://github.com/DunbrackLab/IPSAE (ipsae.py v4, 2026-01-03)
Reference:       Dunbrack, "Interface-Specific pTM", biorxiv 637595

Structure:
  1. Math helpers      — ptm_func, calc_d0
  2. Atom parsers      — parse_pdb_atom_line, parse_cif_atom_line
  3. Structure loader  — parse_structure(path)
  4. PAE loader        — load_pae(pae_path, fmt, ...)
  5. Score compute     — compute_ipsae(pae, coords, chains, ...)
  6. File wrapper      — run_ipsae(structure_path, pae_path, ...)
  7. CLI (legacy)      — argv parsing preserved

All numerical logic is copied verbatim from ipsae_original.py (MIT license,
Roland Dunbrack). Only the wrapping into functions + dict returns is new.
"""

from __future__ import annotations

import json
import math
import os
import sys
from typing import Optional

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════
# 1. Math helpers (verbatim from upstream)
# ═══════════════════════════════════════════════════════════════════════════

def ptm_func(x, d0):
    return 1.0 / (1 + (x / d0) ** 2.0)


ptm_func_vec = np.vectorize(ptm_func)


def calc_d0(L, pair_type):
    L = float(L)
    min_value = 1.0
    if pair_type == "nucleic_acid":
        min_value = 2.0
    if L > 27:
        d0 = 1.24 * (L - 15) ** (1.0 / 3.0) - 1.8
    else:
        d0 = 1.0
    return max(min_value, d0)


def calc_d0_array(L, pair_type):
    L = np.array(L, dtype=float)
    L = np.maximum(26, L)
    min_value = 1.0
    if pair_type == "nucleic_acid":
        min_value = 2.0
    return np.maximum(min_value, 1.24 * (L - 15) ** (1.0 / 3.0) - 1.8)


# ═══════════════════════════════════════════════════════════════════════════
# 2. Atom line parsers (verbatim)
# ═══════════════════════════════════════════════════════════════════════════

def parse_pdb_atom_line(line):
    atom_num = line[6:11].strip()
    atom_name = line[12:16].strip()
    residue_name = line[17:20].strip()
    if residue_name == "LIG":
        return None
    chain_id = line[21].strip()
    residue_seq_num = line[22:26].strip()
    x = line[30:38].strip()
    y = line[38:46].strip()
    z = line[46:54].strip()
    return {
        "atom_num": int(atom_num),
        "atom_name": atom_name,
        "residue_name": residue_name,
        "chain_id": chain_id,
        "residue_seq_num": int(residue_seq_num),
        "x": float(x),
        "y": float(y),
        "z": float(z),
    }


def parse_cif_atom_line(line, fielddict):
    parts = line.split()
    atom_num = parts[fielddict["id"]]
    atom_name = parts[fielddict["label_atom_id"]]
    residue_name = parts[fielddict["label_comp_id"]]
    if "auth_asym_id" in fielddict:
        chain_id = parts[fielddict["auth_asym_id"]]
    else:
        chain_id = parts[fielddict["label_asym_id"]]
    residue_seq_num = parts[fielddict["label_seq_id"]]
    x = parts[fielddict["Cartn_x"]]
    y = parts[fielddict["Cartn_y"]]
    z = parts[fielddict["Cartn_z"]]
    if residue_seq_num == ".":
        return None
    return {
        "atom_num": int(atom_num),
        "atom_name": atom_name,
        "residue_name": residue_name,
        "chain_id": chain_id,
        "residue_seq_num": int(residue_seq_num),
        "x": float(x),
        "y": float(y),
        "z": float(z),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 3. Helpers for nested chain-pair dicts
# ═══════════════════════════════════════════════════════════════════════════

def contiguous_ranges(numbers) -> Optional[str]:
    """Match upstream: return None for empty input (callers treat None as 'skip')."""
    if not numbers:
        return None
    sorted_numbers = sorted(numbers)
    start = sorted_numbers[0]
    end = start
    ranges = []

    def _fmt(a, b):
        return f"{a}" if a == b else f"{a}-{b}"

    for n in sorted_numbers[1:]:
        if n == end + 1:
            end = n
        else:
            ranges.append(_fmt(start, end))
            start = end = n
    ranges.append(_fmt(start, end))
    return "+".join(ranges)


def init_chainpairdict_zeros(chainlist):
    return {c1: {c2: 0 for c2 in chainlist if c1 != c2} for c1 in chainlist}


def init_chainpairdict_npzeros(chainlist, size):
    return {c1: {c2: np.zeros(size) for c2 in chainlist if c1 != c2} for c1 in chainlist}


def init_chainpairdict_set(chainlist):
    return {c1: {c2: set() for c2 in chainlist if c1 != c2} for c1 in chainlist}


def classify_chains(chains, residue_types):
    nuc_residue_set = {"DA", "DC", "DT", "DG", "A", "C", "U", "G"}
    chain_types = {}
    _, first_idx = np.unique(chains, return_index=True)
    unique_chains = chains[np.sort(first_idx)]
    for chain in unique_chains:
        indices = np.where(chains == chain)[0]
        chain_residues = residue_types[indices]
        nuc_count = sum(r in nuc_residue_set for r in chain_residues)
        chain_types[chain] = "nucleic_acid" if nuc_count > 0 else "protein"
    return chain_types


# ═══════════════════════════════════════════════════════════════════════════
# 4. Structure loader
# ═══════════════════════════════════════════════════════════════════════════

RESIDUE_SET = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "DA", "DC", "DT", "DG", "A", "C", "U", "G",
}


def parse_structure(path: str):
    """Parse PDB or mmCIF file; return dict with coordinates, chains, etc.

    Returns
    -------
    dict with keys:
        residues         — list of CA/C1-token dicts
        cb_residues      — list of CB/C3-token dicts (Gly uses CA)
        chains           — np.array[str] per-residue chain id
        coordinates      — np.array[numres, 3] CB-like coords
        distances        — np.array[numres, numres] pairwise CB distances
        token_mask       — np.array[ntotal] token→residue mask
        unique_chains    — np.array[str] ordered unique chain ids
        residue_types    — np.array[str]
        chain_pair_type  — dict[dict[str]]
        numres           — int
        CA_atom_num      — np.array[int] CA atom indices (for AF3 atom_plddt lookup)
        CB_atom_num      — np.array[int] CB atom indices
    """
    is_cif = path.endswith(".cif")

    residues = []
    cb_residues = []
    chains = []
    atomsitefield_num = 0
    atomsitefield_dict = {}
    token_mask = []

    with open(path, "r") as pdb:
        for line in pdb:
            if line.startswith("_atom_site."):
                line = line.strip()
                (_, fieldname) = line.split(".")
                atomsitefield_dict[fieldname] = atomsitefield_num
                atomsitefield_num += 1
                continue
            if line.startswith("ATOM") or line.startswith("HETATM"):
                if is_cif:
                    atom = parse_cif_atom_line(line, atomsitefield_dict)
                else:
                    atom = parse_pdb_atom_line(line)
                if atom is None:
                    token_mask.append(0)
                    continue
                if atom["atom_name"] == "CA" or "C1" in atom["atom_name"]:
                    token_mask.append(1)
                    residues.append({
                        "atom_num": atom["atom_num"],
                        "coor": np.array([atom["x"], atom["y"], atom["z"]]),
                        "res": atom["residue_name"],
                        "chainid": atom["chain_id"],
                        "resnum": atom["residue_seq_num"],
                        "residue": f"{atom['residue_name']:3}   {atom['chain_id']:3} {atom['residue_seq_num']:4}",
                    })
                    chains.append(atom["chain_id"])
                if (
                    atom["atom_name"] == "CB"
                    or "C3" in atom["atom_name"]
                    or (atom["residue_name"] == "GLY" and atom["atom_name"] == "CA")
                ):
                    cb_residues.append({
                        "atom_num": atom["atom_num"],
                        "coor": np.array([atom["x"], atom["y"], atom["z"]]),
                        "res": atom["residue_name"],
                        "chainid": atom["chain_id"],
                        "resnum": atom["residue_seq_num"],
                        "residue": f"{atom['residue_name']:3}   {atom['chain_id']:3} {atom['residue_seq_num']:4}",
                    })
                if (
                    atom["atom_name"] != "CA"
                    and "C1" not in atom["atom_name"]
                    and atom["residue_name"] not in RESIDUE_SET
                ):
                    token_mask.append(0)

    numres = len(residues)
    CA_atom_num = np.array([r["atom_num"] - 1 for r in residues])
    CB_atom_num = np.array([r["atom_num"] - 1 for r in cb_residues])
    coordinates = np.array([r["coor"] for r in cb_residues])
    chains = np.array(chains)
    _, first_idx = np.unique(chains, return_index=True)
    unique_chains = chains[np.sort(first_idx)]
    token_array = np.array(token_mask)
    residue_types = np.array([r["res"] for r in residues])

    # chain pair types
    chain_dict = classify_chains(chains, residue_types)
    chain_pair_type = init_chainpairdict_zeros(unique_chains)
    for c1 in unique_chains:
        for c2 in unique_chains:
            if c1 == c2:
                continue
            if chain_dict[c1] == "nucleic_acid" or chain_dict[c2] == "nucleic_acid":
                chain_pair_type[c1][c2] = "nucleic_acid"
            else:
                chain_pair_type[c1][c2] = "protein"

    distances = np.sqrt(((coordinates[:, None, :] - coordinates[None, :, :]) ** 2).sum(axis=2))

    return {
        "residues": residues,
        "cb_residues": cb_residues,
        "chains": chains,
        "coordinates": coordinates,
        "distances": distances,
        "token_mask": token_array,
        "unique_chains": unique_chains,
        "residue_types": residue_types,
        "chain_pair_type": chain_pair_type,
        "numres": numres,
        "CA_atom_num": CA_atom_num,
        "CB_atom_num": CB_atom_num,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 5. PAE / plddt loader (af2 / af3 / boltz)
# ═══════════════════════════════════════════════════════════════════════════

def load_pae(
    pae_path: str,
    fmt: str,
    structure: dict,
):
    """Load PAE matrix (+ plddt, per-chain iptm if available).

    fmt ∈ {"af2", "af3", "boltz"}
    """
    numres = structure["numres"]
    token_array = structure["token_mask"]
    CA_atom_num = structure["CA_atom_num"]
    CB_atom_num = structure["CB_atom_num"]
    unique_chains = structure["unique_chains"]

    iptm_af = None        # af2 scalar, af3/boltz per-chain-pair dict
    iptm_af2_scalar = -1.0
    ptm_af2_scalar = -1.0
    plddt = None
    cb_plddt = None
    pae_matrix = None

    if fmt == "af2":
        if not os.path.exists(pae_path):
            raise FileNotFoundError(f"AF2 PAE file not found: {pae_path}")
        if pae_path.endswith(".pkl"):
            data = np.load(pae_path, allow_pickle=True)
        else:
            with open(pae_path, "r") as f:
                data = json.load(f)
        iptm_af2_scalar = float(data["iptm"]) if "iptm" in data else -1.0
        ptm_af2_scalar = float(data["ptm"]) if "ptm" in data else -1.0
        if "plddt" in data:
            plddt = np.array(data["plddt"])
            cb_plddt = np.array(data["plddt"])
        else:
            plddt = np.zeros(numres)
            cb_plddt = np.zeros(numres)
        if "pae" in data:
            pae_matrix = np.array(data["pae"])
        elif "predicted_aligned_error" in data:
            pae_matrix = np.array(data["predicted_aligned_error"])
        else:
            raise ValueError(
                f"AF2 PAE data in {pae_path} is missing required key "
                "'pae' or 'predicted_aligned_error'"
            )
        iptm_af = iptm_af2_scalar

    elif fmt == "boltz":
        plddt_path = pae_path.replace("pae", "plddt")
        if os.path.exists(plddt_path):
            data_plddt = np.load(plddt_path)
            raw = data_plddt["plddt"]
            plddt_boltz = np.array(100.0 * raw) if np.max(raw) <= 1.0 else np.array(raw)
            plddt = plddt_boltz[np.ix_(token_array.astype(bool))]
            cb_plddt = plddt_boltz[np.ix_(token_array.astype(bool))]
        else:
            ntokens = int(token_array.sum())
            plddt = np.zeros(ntokens)
            cb_plddt = np.zeros(ntokens)
        if not os.path.exists(pae_path):
            raise FileNotFoundError(f"Boltz PAE file not found: {pae_path}")
        data_pae = np.load(pae_path)
        pae_matrix_boltz = np.array(data_pae["pae"])
        pae_matrix = pae_matrix_boltz[np.ix_(token_array.astype(bool), token_array.astype(bool))]

        summary_path = pae_path.replace("pae", "confidence").replace(".npz", ".json")
        iptm_boltz = {c1: {c2: 0 for c2 in unique_chains if c1 != c2} for c1 in unique_chains}
        if os.path.exists(summary_path):
            with open(summary_path, "r") as f:
                data_summary = json.load(f)
            if "pair_chains_iptm" in data_summary:
                pairs_data = data_summary["pair_chains_iptm"]
                for n1, c1 in enumerate(unique_chains):
                    for n2, c2 in enumerate(unique_chains):
                        if c1 == c2:
                            continue
                        iptm_boltz[c1][c2] = pairs_data[str(n1)][str(n2)]
        iptm_af = iptm_boltz

    elif fmt == "af3":
        if not os.path.exists(pae_path):
            raise FileNotFoundError(f"AF3 PAE file not found: {pae_path}")
        with open(pae_path, "r") as f:
            data = json.load(f)
        if "atom_plddts" in data:
            atom_plddts = np.array(data["atom_plddts"])
            plddt = atom_plddts[CA_atom_num]
            cb_plddt = atom_plddts[CB_atom_num]
        else:
            plddt = np.zeros(numres)
            cb_plddt = np.zeros(numres)
        if "pae" not in data:
            raise ValueError("no PAE data in AF3 json file")
        pae_matrix_af3 = np.array(data["pae"])
        pae_matrix = pae_matrix_af3[np.ix_(token_array.astype(bool), token_array.astype(bool))]

        iptm_af3 = {c1: {c2: 0 for c2 in unique_chains if c1 != c2} for c1 in unique_chains}
        summary_path = None
        if "confidences" in pae_path:
            summary_path = pae_path.replace("confidences", "summary_confidences")
        elif "full_data" in pae_path:
            summary_path = pae_path.replace("full_data", "summary_confidences")
        if summary_path and os.path.exists(summary_path):
            with open(summary_path, "r") as f:
                data_summary = json.load(f)
            af3_pairs = data_summary["chain_pair_iptm"]
            for n1, c1 in enumerate(unique_chains):
                for n2, c2 in enumerate(unique_chains):
                    if c1 == c2:
                        continue
                    iptm_af3[c1][c2] = af3_pairs[n1][n2]
        iptm_af = iptm_af3
    else:
        raise ValueError(f"Unknown fmt: {fmt!r}")

    return {
        "pae_matrix": pae_matrix,
        "plddt": plddt,
        "cb_plddt": cb_plddt,
        "iptm_af": iptm_af,
        "iptm_af2_scalar": iptm_af2_scalar,
        "ptm_af2_scalar": ptm_af2_scalar,
        "fmt": fmt,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 6. Core score computation
# ═══════════════════════════════════════════════════════════════════════════

def compute_ipsae(structure: dict, pae_data: dict, pae_cutoff: float, dist_cutoff: float):
    """Compute ipSAE, ipTM_d0chn, pDockQ, pDockQ2, LIS per chain pair.

    Returns a dict with:
      byres: per-residue arrays
      asym:  {c1: {c2: float}} asymmetric score
      max:   {c1: {c2: float}} max of (A→B, B→A)
      *_res: residue string identifiers for asym/max
      n0/d0: normalization values
      pDockQ, pDockQ2, LIS, valid_pair_counts, unique_residues_*
    """
    chains = structure["chains"]
    unique_chains = structure["unique_chains"]
    residues = structure["residues"]
    distances = structure["distances"]
    numres = structure["numres"]
    chain_pair_type = structure["chain_pair_type"]

    pae_matrix = pae_data["pae_matrix"]
    cb_plddt = pae_data["cb_plddt"]

    # init
    iptm_d0chn_byres = init_chainpairdict_npzeros(unique_chains, numres)
    ipsae_d0chn_byres = init_chainpairdict_npzeros(unique_chains, numres)
    ipsae_d0dom_byres = init_chainpairdict_npzeros(unique_chains, numres)
    ipsae_d0res_byres = init_chainpairdict_npzeros(unique_chains, numres)

    iptm_d0chn_asym = init_chainpairdict_zeros(unique_chains)
    ipsae_d0chn_asym = init_chainpairdict_zeros(unique_chains)
    ipsae_d0dom_asym = init_chainpairdict_zeros(unique_chains)
    ipsae_d0res_asym = init_chainpairdict_zeros(unique_chains)

    iptm_d0chn_max = init_chainpairdict_zeros(unique_chains)
    ipsae_d0chn_max = init_chainpairdict_zeros(unique_chains)
    ipsae_d0dom_max = init_chainpairdict_zeros(unique_chains)
    ipsae_d0res_max = init_chainpairdict_zeros(unique_chains)

    iptm_d0chn_asymres = init_chainpairdict_zeros(unique_chains)
    ipsae_d0chn_asymres = init_chainpairdict_zeros(unique_chains)
    ipsae_d0dom_asymres = init_chainpairdict_zeros(unique_chains)
    ipsae_d0res_asymres = init_chainpairdict_zeros(unique_chains)

    iptm_d0chn_maxres = init_chainpairdict_zeros(unique_chains)
    ipsae_d0chn_maxres = init_chainpairdict_zeros(unique_chains)
    ipsae_d0dom_maxres = init_chainpairdict_zeros(unique_chains)
    ipsae_d0res_maxres = init_chainpairdict_zeros(unique_chains)

    n0chn = init_chainpairdict_zeros(unique_chains)
    n0dom = init_chainpairdict_zeros(unique_chains)
    n0dom_max = init_chainpairdict_zeros(unique_chains)
    n0res = init_chainpairdict_zeros(unique_chains)
    n0res_max = init_chainpairdict_zeros(unique_chains)
    n0res_byres = init_chainpairdict_npzeros(unique_chains, numres)

    d0chn = init_chainpairdict_zeros(unique_chains)
    d0dom = init_chainpairdict_zeros(unique_chains)
    d0dom_max = init_chainpairdict_zeros(unique_chains)
    d0res = init_chainpairdict_zeros(unique_chains)
    d0res_max = init_chainpairdict_zeros(unique_chains)
    d0res_byres = init_chainpairdict_npzeros(unique_chains, numres)

    valid_pair_counts = init_chainpairdict_zeros(unique_chains)
    dist_valid_pair_counts = init_chainpairdict_zeros(unique_chains)
    unique_residues_chain1 = init_chainpairdict_set(unique_chains)
    unique_residues_chain2 = init_chainpairdict_set(unique_chains)
    dist_unique_residues_chain1 = init_chainpairdict_set(unique_chains)
    dist_unique_residues_chain2 = init_chainpairdict_set(unique_chains)
    pDockQ_unique_residues = init_chainpairdict_set(unique_chains)

    pDockQ = init_chainpairdict_zeros(unique_chains)
    pDockQ2 = init_chainpairdict_zeros(unique_chains)
    LIS = init_chainpairdict_zeros(unique_chains)

    pDockQ_cutoff = 8.0

    # pDockQ
    for c1 in unique_chains:
        for c2 in unique_chains:
            if c1 == c2:
                continue
            npairs = 0
            for i in range(numres):
                if chains[i] != c1:
                    continue
                valid_pairs = (chains == c2) & (distances[i] <= pDockQ_cutoff)
                npairs += int(np.sum(valid_pairs))
                if valid_pairs.any():
                    pDockQ_unique_residues[c1][c2].add(i)
                    for j in np.where(valid_pairs)[0]:
                        pDockQ_unique_residues[c1][c2].add(j)
            if npairs > 0:
                mean_plddt = cb_plddt[list(pDockQ_unique_residues[c1][c2])].mean()
                x = mean_plddt * math.log10(npairs)
                pDockQ[c1][c2] = 0.724 / (1 + math.exp(-0.052 * (x - 152.611))) + 0.018
            else:
                pDockQ[c1][c2] = 0.0

    # pDockQ2
    for c1 in unique_chains:
        for c2 in unique_chains:
            if c1 == c2:
                continue
            npairs = 0
            s = 0.0
            for i in range(numres):
                if chains[i] != c1:
                    continue
                valid_pairs = (chains == c2) & (distances[i] <= pDockQ_cutoff)
                if valid_pairs.any():
                    npairs += int(np.sum(valid_pairs))
                    pae_list = pae_matrix[i][valid_pairs]
                    s += ptm_func_vec(pae_list, 10.0).sum()
            if npairs > 0:
                mean_plddt = cb_plddt[list(pDockQ_unique_residues[c1][c2])].mean()
                mean_ptm = s / npairs
                x = mean_plddt * mean_ptm
                pDockQ2[c1][c2] = 1.31 / (1 + math.exp(-0.075 * (x - 84.733))) + 0.005
            else:
                pDockQ2[c1][c2] = 0.0

    # LIS
    for c1 in unique_chains:
        for c2 in unique_chains:
            if c1 == c2:
                continue
            mask = (chains[:, None] == c1) & (chains[None, :] == c2)
            sel = pae_matrix[mask]
            if sel.size > 0:
                valid = sel[sel < 12]
                if valid.size > 0:
                    LIS[c1][c2] = float(((12 - valid) / 12).mean())
                else:
                    LIS[c1][c2] = 0.0
            else:
                LIS[c1][c2] = 0.0

    # iptm_d0chn, ipsae_d0chn byres
    for c1 in unique_chains:
        for c2 in unique_chains:
            if c1 == c2:
                continue
            n0chn[c1][c2] = int(np.sum(chains == c1) + np.sum(chains == c2))
            d0chn[c1][c2] = calc_d0(n0chn[c1][c2], chain_pair_type[c1][c2])
            ptm_matrix_d0chn = ptm_func_vec(pae_matrix, d0chn[c1][c2])
            valid_pairs_iptm = (chains == c2)
            valid_pairs_matrix = np.outer(chains == c1, chains == c2) & (pae_matrix < pae_cutoff)
            for i in range(numres):
                if chains[i] != c1:
                    continue
                valid_pairs_ipsae = valid_pairs_matrix[i]
                iptm_d0chn_byres[c1][c2][i] = (
                    ptm_matrix_d0chn[i, valid_pairs_iptm].mean() if valid_pairs_iptm.any() else 0.0
                )
                ipsae_d0chn_byres[c1][c2][i] = (
                    ptm_matrix_d0chn[i, valid_pairs_ipsae].mean() if valid_pairs_ipsae.any() else 0.0
                )
                valid_pair_counts[c1][c2] += int(np.sum(valid_pairs_ipsae))
                if valid_pairs_ipsae.any():
                    iresnum = residues[i]["resnum"]
                    unique_residues_chain1[c1][c2].add(iresnum)
                    for j in np.where(valid_pairs_ipsae)[0]:
                        unique_residues_chain2[c1][c2].add(residues[j]["resnum"])
                valid_pairs_dist = (
                    (chains == c2) & (pae_matrix[i] < pae_cutoff) & (distances[i] < dist_cutoff)
                )
                dist_valid_pair_counts[c1][c2] += int(np.sum(valid_pairs_dist))
                if valid_pairs_dist.any():
                    iresnum = residues[i]["resnum"]
                    dist_unique_residues_chain1[c1][c2].add(iresnum)
                    for j in np.where(valid_pairs_dist)[0]:
                        dist_unique_residues_chain2[c1][c2].add(residues[j]["resnum"])

    # ipsae_d0dom, ipsae_d0res byres
    for c1 in unique_chains:
        for c2 in unique_chains:
            if c1 == c2:
                continue
            r1 = len(unique_residues_chain1[c1][c2])
            r2 = len(unique_residues_chain2[c1][c2])
            n0dom[c1][c2] = r1 + r2
            d0dom[c1][c2] = calc_d0(n0dom[c1][c2], chain_pair_type[c1][c2])
            ptm_matrix_d0dom = ptm_func_vec(pae_matrix, d0dom[c1][c2])
            valid_pairs_matrix = np.outer(chains == c1, chains == c2) & (pae_matrix < pae_cutoff)
            n0res_byres_all = np.sum(valid_pairs_matrix, axis=1)
            d0res_byres_all = calc_d0_array(n0res_byres_all, chain_pair_type[c1][c2])
            n0res_byres[c1][c2] = n0res_byres_all
            d0res_byres[c1][c2] = d0res_byres_all
            for i in range(numres):
                if chains[i] != c1:
                    continue
                valid_pairs = valid_pairs_matrix[i]
                ipsae_d0dom_byres[c1][c2][i] = (
                    ptm_matrix_d0dom[i, valid_pairs].mean() if valid_pairs.any() else 0.0
                )
                ptm_row_d0res = ptm_func_vec(pae_matrix[i], d0res_byres[c1][c2][i])
                ipsae_d0res_byres[c1][c2][i] = (
                    ptm_row_d0res[valid_pairs].mean() if valid_pairs.any() else 0.0
                )

    # asym aggregation
    for c1 in unique_chains:
        for c2 in unique_chains:
            if c1 == c2:
                continue
            vals = iptm_d0chn_byres[c1][c2]
            mi = int(np.argmax(vals))
            iptm_d0chn_asym[c1][c2] = float(vals[mi])
            iptm_d0chn_asymres[c1][c2] = residues[mi]["residue"]

            vals = ipsae_d0chn_byres[c1][c2]
            mi = int(np.argmax(vals))
            ipsae_d0chn_asym[c1][c2] = float(vals[mi])
            ipsae_d0chn_asymres[c1][c2] = residues[mi]["residue"]

            vals = ipsae_d0dom_byres[c1][c2]
            mi = int(np.argmax(vals))
            ipsae_d0dom_asym[c1][c2] = float(vals[mi])
            ipsae_d0dom_asymres[c1][c2] = residues[mi]["residue"]

            vals = ipsae_d0res_byres[c1][c2]
            mi = int(np.argmax(vals))
            ipsae_d0res_asym[c1][c2] = float(vals[mi])
            ipsae_d0res_asymres[c1][c2] = residues[mi]["residue"]
            n0res[c1][c2] = int(n0res_byres[c1][c2][mi])
            d0res[c1][c2] = float(d0res_byres[c1][c2][mi])

            if c1 > c2:
                # iptm_d0chn max
                mv = max(iptm_d0chn_asym[c1][c2], iptm_d0chn_asym[c2][c1])
                mres = iptm_d0chn_asymres[c1][c2] if mv == iptm_d0chn_asym[c1][c2] else iptm_d0chn_asymres[c2][c1]
                iptm_d0chn_max[c1][c2] = iptm_d0chn_max[c2][c1] = mv
                iptm_d0chn_maxres[c1][c2] = iptm_d0chn_maxres[c2][c1] = mres
                # ipsae_d0chn max
                mv = max(ipsae_d0chn_asym[c1][c2], ipsae_d0chn_asym[c2][c1])
                mres = ipsae_d0chn_asymres[c1][c2] if mv == ipsae_d0chn_asym[c1][c2] else ipsae_d0chn_asymres[c2][c1]
                ipsae_d0chn_max[c1][c2] = ipsae_d0chn_max[c2][c1] = mv
                ipsae_d0chn_maxres[c1][c2] = ipsae_d0chn_maxres[c2][c1] = mres
                # ipsae_d0dom max
                mv = max(ipsae_d0dom_asym[c1][c2], ipsae_d0dom_asym[c2][c1])
                if mv == ipsae_d0dom_asym[c1][c2]:
                    mres = ipsae_d0dom_asymres[c1][c2]
                    mn0 = n0dom[c1][c2]
                    md0 = d0dom[c1][c2]
                else:
                    mres = ipsae_d0dom_asymres[c2][c1]
                    mn0 = n0dom[c2][c1]
                    md0 = d0dom[c2][c1]
                ipsae_d0dom_max[c1][c2] = ipsae_d0dom_max[c2][c1] = mv
                ipsae_d0dom_maxres[c1][c2] = ipsae_d0dom_maxres[c2][c1] = mres
                n0dom_max[c1][c2] = n0dom_max[c2][c1] = mn0
                d0dom_max[c1][c2] = d0dom_max[c2][c1] = md0
                # ipsae_d0res max
                mv = max(ipsae_d0res_asym[c1][c2], ipsae_d0res_asym[c2][c1])
                if mv == ipsae_d0res_asym[c1][c2]:
                    mres = ipsae_d0res_asymres[c1][c2]
                    mn0 = n0res[c1][c2]
                    md0 = d0res[c1][c2]
                else:
                    mres = ipsae_d0res_asymres[c2][c1]
                    mn0 = n0res[c2][c1]
                    md0 = d0res[c2][c1]
                ipsae_d0res_max[c1][c2] = ipsae_d0res_max[c2][c1] = mv
                ipsae_d0res_maxres[c1][c2] = ipsae_d0res_maxres[c2][c1] = mres
                n0res_max[c1][c2] = n0res_max[c2][c1] = mn0
                d0res_max[c1][c2] = d0res_max[c2][c1] = md0

    return {
        "pae_cutoff": pae_cutoff,
        "dist_cutoff": dist_cutoff,
        "unique_chains": unique_chains,
        "iptm_d0chn_byres": iptm_d0chn_byres,
        "ipsae_d0chn_byres": ipsae_d0chn_byres,
        "ipsae_d0dom_byres": ipsae_d0dom_byres,
        "ipsae_d0res_byres": ipsae_d0res_byres,
        "iptm_d0chn_asym": iptm_d0chn_asym,
        "ipsae_d0chn_asym": ipsae_d0chn_asym,
        "ipsae_d0dom_asym": ipsae_d0dom_asym,
        "ipsae_d0res_asym": ipsae_d0res_asym,
        "iptm_d0chn_max": iptm_d0chn_max,
        "ipsae_d0chn_max": ipsae_d0chn_max,
        "ipsae_d0dom_max": ipsae_d0dom_max,
        "ipsae_d0res_max": ipsae_d0res_max,
        "iptm_d0chn_asymres": iptm_d0chn_asymres,
        "ipsae_d0chn_asymres": ipsae_d0chn_asymres,
        "ipsae_d0dom_asymres": ipsae_d0dom_asymres,
        "ipsae_d0res_asymres": ipsae_d0res_asymres,
        "iptm_d0chn_maxres": iptm_d0chn_maxres,
        "ipsae_d0chn_maxres": ipsae_d0chn_maxres,
        "ipsae_d0dom_maxres": ipsae_d0dom_maxres,
        "ipsae_d0res_maxres": ipsae_d0res_maxres,
        "n0chn": n0chn, "n0dom": n0dom, "n0dom_max": n0dom_max,
        "n0res": n0res, "n0res_max": n0res_max, "n0res_byres": n0res_byres,
        "d0chn": d0chn, "d0dom": d0dom, "d0dom_max": d0dom_max,
        "d0res": d0res, "d0res_max": d0res_max, "d0res_byres": d0res_byres,
        "valid_pair_counts": valid_pair_counts,
        "dist_valid_pair_counts": dist_valid_pair_counts,
        "unique_residues_chain1": unique_residues_chain1,
        "unique_residues_chain2": unique_residues_chain2,
        "dist_unique_residues_chain1": dist_unique_residues_chain1,
        "dist_unique_residues_chain2": dist_unique_residues_chain2,
        "pDockQ": pDockQ,
        "pDockQ2": pDockQ2,
        "LIS": LIS,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 7. Output writers (txt / pml) — preserved from upstream
# ═══════════════════════════════════════════════════════════════════════════

_CHAINCOLOR = {
    "A": "magenta", "B": "marine", "C": "lime", "D": "orange",
    "E": "yellow", "F": "cyan", "G": "lightorange", "H": "pink",
    "I": "deepteal", "J": "forest", "K": "lightblue", "L": "slate",
    "M": "violet", "N": "arsenic", "O": "iodine", "P": "silver",
    "Q": "red", "R": "sulfur", "S": "purple", "T": "olive",
    "U": "palegreen", "V": "green", "W": "blue", "X": "palecyan",
    "Y": "limon", "Z": "chocolate",
}


def _format_cutoff_string(cutoff: float) -> str:
    s = str(int(cutoff))
    return "0" + s if cutoff < 10 else s


def _write_byres_tsv(path: str, structure: dict, scores: dict, pae_data: dict):
    with open(path, "w") as f:
        f.write(
            "i   AlignChn ScoredChain  AlignResNum  AlignResType  AlignRespLDDT      "
            "n0chn  n0dom  n0res    d0chn     d0dom     d0res   ipTM_pae  ipSAE_d0chn "
            "ipSAE_d0dom    ipSAE \n"
        )
        uc = structure["unique_chains"]
        residues = structure["residues"]
        chains = structure["chains"]
        plddt = pae_data["plddt"]
        numres = structure["numres"]
        for c1 in uc:
            for c2 in uc:
                if c1 == c2:
                    continue
                for i in range(numres):
                    if chains[i] != c1:
                        continue
                    f.write(
                        f"{i+1:<4d}    {c1:4}      {c2:4}      "
                        f"{residues[i]['resnum']:4d}           "
                        f"{residues[i]['res']:3}        "
                        f"{plddt[i]:8.2f}         "
                        f"{int(scores['n0chn'][c1][c2]):5d}  "
                        f"{int(scores['n0dom'][c1][c2]):5d}  "
                        f"{int(scores['n0res_byres'][c1][c2][i]):5d}  "
                        f"{scores['d0chn'][c1][c2]:8.3f}  "
                        f"{scores['d0dom'][c1][c2]:8.3f}  "
                        f"{scores['d0res_byres'][c1][c2][i]:8.3f}   "
                        f"{scores['iptm_d0chn_byres'][c1][c2][i]:8.4f}    "
                        f"{scores['ipsae_d0chn_byres'][c1][c2][i]:8.4f}    "
                        f"{scores['ipsae_d0dom_byres'][c1][c2][i]:8.4f}    "
                        f"{scores['ipsae_d0res_byres'][c1][c2][i]:8.4f}\n"
                    )


def _write_summary_txt_and_pml(
    txt_path: str, pml_path: str, structure: dict, scores: dict, pae_data: dict,
    pae_cutoff: float, dist_cutoff: float, pdb_stem: str,
):
    pae_string = _format_cutoff_string(pae_cutoff)
    dist_string = _format_cutoff_string(dist_cutoff)
    uc = structure["unique_chains"]
    fmt = pae_data["fmt"]
    iptm_af = pae_data["iptm_af"]

    chainpairs = set()
    for c1 in uc:
        for c2 in uc:
            if c1 >= c2:
                continue
            chainpairs.add(f"{c1}-{c2}")

    header = (
        "\nChn1 Chn2  PAE Dist  Type   ipSAE    ipSAE_d0chn ipSAE_d0dom  ipTM_af  "
        "ipTM_d0chn     pDockQ     pDockQ2    LIS       n0res  n0chn  n0dom   d0res   "
        "d0chn   d0dom  nres1   nres2   dist1   dist2  Model\n"
    )
    with open(txt_path, "w") as OUT, open(pml_path, "w") as PML:
        OUT.write(header)
        PML.write("# " + header.lstrip("\n"))
        for pair in sorted(chainpairs):
            ca, cb = pair.split("-")
            for c1, c2 in ((ca, cb), (cb, ca)):
                color1 = _CHAINCOLOR.get(c1, "magenta")
                color2 = _CHAINCOLOR.get(c2, "marine")
                r1 = len(scores["unique_residues_chain1"][c1][c2])
                r2 = len(scores["unique_residues_chain2"][c1][c2])
                dr1 = len(scores["dist_unique_residues_chain1"][c1][c2])
                dr2 = len(scores["dist_unique_residues_chain2"][c1][c2])
                if fmt == "af2":
                    iptm_af_val = iptm_af  # scalar
                else:
                    iptm_af_val = iptm_af[c1][c2]
                outstring = (
                    f"{c1}    {c2}     {pae_string:3}  {dist_string:3}  {'asym':5} "
                    f"{scores['ipsae_d0res_asym'][c1][c2]:8.6f}    "
                    f"{scores['ipsae_d0chn_asym'][c1][c2]:8.6f}    "
                    f"{scores['ipsae_d0dom_asym'][c1][c2]:8.6f}    "
                    f"{iptm_af_val:5.3f}    "
                    f"{scores['iptm_d0chn_asym'][c1][c2]:8.6f}    "
                    f"{scores['pDockQ'][c1][c2]:8.4f}   "
                    f"{scores['pDockQ2'][c1][c2]:8.4f}   "
                    f"{scores['LIS'][c1][c2]:8.4f}   "
                    f"{int(scores['n0res'][c1][c2]):5d}  "
                    f"{int(scores['n0chn'][c1][c2]):5d}  "
                    f"{int(scores['n0dom'][c1][c2]):5d}  "
                    f"{scores['d0res'][c1][c2]:6.2f}  "
                    f"{scores['d0chn'][c1][c2]:6.2f}  "
                    f"{scores['d0dom'][c1][c2]:6.2f}  "
                    f"{r1:5d}   {r2:5d}   {dr1:5d}   {dr2:5d}   {pdb_stem}\n"
                )
                OUT.write(outstring)
                PML.write("# " + outstring)
                if c1 > c2:
                    r1m = max(
                        len(scores["unique_residues_chain2"][c1][c2]),
                        len(scores["unique_residues_chain1"][c2][c1]),
                    )
                    r2m = max(
                        len(scores["unique_residues_chain1"][c1][c2]),
                        len(scores["unique_residues_chain2"][c2][c1]),
                    )
                    dr1m = max(
                        len(scores["dist_unique_residues_chain2"][c1][c2]),
                        len(scores["dist_unique_residues_chain1"][c2][c1]),
                    )
                    dr2m = max(
                        len(scores["dist_unique_residues_chain1"][c1][c2]),
                        len(scores["dist_unique_residues_chain2"][c2][c1]),
                    )
                    pdq2v = max(scores["pDockQ2"][c1][c2], scores["pDockQ2"][c2][c1])
                    if fmt == "boltz":
                        iptm_val = max(iptm_af[c1][c2], iptm_af[c2][c1])
                    else:
                        iptm_val = iptm_af_val
                    lis_v = (scores["LIS"][c1][c2] + scores["LIS"][c2][c1]) / 2.0
                    outstring = (
                        f"{c2}    {c1}     {pae_string:3}  {dist_string:3}  {'max':5} "
                        f"{scores['ipsae_d0res_max'][c1][c2]:8.6f}    "
                        f"{scores['ipsae_d0chn_max'][c1][c2]:8.6f}    "
                        f"{scores['ipsae_d0dom_max'][c1][c2]:8.6f}    "
                        f"{iptm_val:5.3f}    "
                        f"{scores['iptm_d0chn_max'][c1][c2]:8.6f}    "
                        f"{scores['pDockQ'][c1][c2]:8.4f}   "
                        f"{pdq2v:8.4f}   {lis_v:8.4f}   "
                        f"{int(scores['n0res_max'][c1][c2]):5d}  "
                        f"{int(scores['n0chn'][c1][c2]):5d}  "
                        f"{int(scores['n0dom_max'][c1][c2]):5d}  "
                        f"{scores['d0res_max'][c1][c2]:6.2f}  "
                        f"{scores['d0chn'][c1][c2]:6.2f}  "
                        f"{scores['d0dom_max'][c1][c2]:6.2f}  "
                        f"{r1m:5d}   {r2m:5d}   {dr1m:5d}   {dr2m:5d}   {pdb_stem}\n"
                    )
                    OUT.write(outstring)
                    PML.write("# " + outstring)
                r1_range = contiguous_ranges(scores["unique_residues_chain1"][c1][c2])
                r2_range = contiguous_ranges(scores["unique_residues_chain2"][c1][c2])
                if r1_range is not None and r2_range is not None:
                    PML.write(
                        f"alias color_{c1}_{c2}, color gray80, all; "
                        f"color {color1}, chain  {c1} and resi {r1_range}; "
                        f"color {color2}, chain  {c2} and resi {r2_range}\n\n"
                    )
            OUT.write("\n")


# ═══════════════════════════════════════════════════════════════════════════
# 8. Top-level callable API
# ═══════════════════════════════════════════════════════════════════════════

def _detect_format(structure_path: str, pae_path: str) -> tuple[str, bool]:
    """Return (fmt, is_cif). fmt ∈ {af2, af3, boltz}."""
    is_cif = structure_path.endswith(".cif")
    if pae_path.endswith(".json") and not is_cif:
        return "af2", False
    if pae_path.endswith(".json") and is_cif:
        return "af3", True
    if pae_path.endswith(".npz"):
        return "boltz", is_cif
    raise ValueError(f"Cannot detect format from {pae_path=} {structure_path=}")


def run_ipsae(
    structure_path: str,
    pae_path: str,
    pae_cutoff: float = 10.0,
    dist_cutoff: float = 15.0,
    output_dir: Optional[str] = None,
    write_files: bool = False,
    fmt: Optional[str] = None,
) -> dict:
    """Top-level callable.

    Parameters
    ----------
    structure_path : str   — path to .pdb or .cif
    pae_path       : str   — path to PAE file (.json for AF2/AF3, .npz for Boltz)
    pae_cutoff     : float — interface PAE cutoff (default 10 Å)
    dist_cutoff    : float — interface Cβ-Cβ cutoff (default 15 Å)
    output_dir     : str   — where to write .txt/.pml (default: next to structure)
    write_files    : bool  — if True, write legacy .txt / _byres.txt / .pml files
    fmt            : str   — override format detection ("af2","af3","boltz")

    Returns
    -------
    dict with structure, pae_data, scores, and output paths (if written)
    """
    if fmt is None:
        fmt, _ = _detect_format(structure_path, pae_path)

    structure = parse_structure(structure_path)
    pae_data = load_pae(pae_path, fmt, structure)
    scores = compute_ipsae(structure, pae_data, pae_cutoff, dist_cutoff)

    output_paths = {}
    if write_files:
        pae_s = _format_cutoff_string(pae_cutoff)
        dist_s = _format_cutoff_string(dist_cutoff)
        base = structure_path[:-4] if structure_path.endswith((".cif", ".pdb")) else structure_path
        stem = f"{base}_{pae_s}_{dist_s}"
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            stem = os.path.join(output_dir, os.path.basename(stem))
        txt_path = f"{stem}.txt"
        byres_path = f"{stem}_byres.txt"
        pml_path = f"{stem}.pml"
        _write_byres_tsv(byres_path, structure, scores, pae_data)
        _write_summary_txt_and_pml(
            txt_path, pml_path, structure, scores, pae_data,
            pae_cutoff, dist_cutoff, os.path.basename(base),
        )
        output_paths = {"txt": txt_path, "byres": byres_path, "pml": pml_path}

    return {
        "structure": structure,
        "pae_data": pae_data,
        "scores": scores,
        "output_paths": output_paths,
        "fmt": fmt,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 9. CLI — preserved behavior
# ═══════════════════════════════════════════════════════════════════════════

def _cli():
    if len(sys.argv) < 5:
        print(
            "Usage:\n"
            "  python ipsae.py <pae_file> <pdb_or_cif_file> <pae_cutoff> <dist_cutoff>\n\n"
            "  AF2:   <pae_file>=.json  <structure>=.pdb\n"
            "  AF3:   <pae_file>=.json  <structure>=.cif\n"
            "  Boltz: <pae_file>=.npz   <structure>=.cif|.pdb\n"
        )
        sys.exit(1)
    pae_path = sys.argv[1]
    pdb_path = sys.argv[2]
    pae_cutoff = float(sys.argv[3])
    dist_cutoff = float(sys.argv[4])
    run_ipsae(
        structure_path=pdb_path,
        pae_path=pae_path,
        pae_cutoff=pae_cutoff,
        dist_cutoff=dist_cutoff,
        write_files=True,
    )


if __name__ == "__main__":
    _cli()
