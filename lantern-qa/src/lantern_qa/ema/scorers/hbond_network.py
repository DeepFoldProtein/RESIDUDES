#!/usr/bin/env python3
"""
H-bond Network Analysis — Novel QA metric.

Instead of just summing H-bond energies, analyze the NETWORK TOPOLOGY:
  1. Helix continuity: length of consecutive i→i+4 H-bond chains
  2. Sheet pairing: number of parallel/antiparallel strand pairs
  3. Network connectivity: largest connected component
  4. H-bond density: hbonds per secondary structure residue

These capture fold quality beyond simple energy sums.
"""

import numpy as np
import gemmi
from typing import Dict, List, Tuple
from pathlib import Path
import sys

# hbond_energy is vendored next to this module in the EMA tree (originally from
# the personal Ramachadra work dir).  Add our own directory to sys.path so the
# import resolves whether this module is imported as lantern_qa.ema.scorers.hbond_network
# or run directly as a script — no hardcoded personal path that breaks on other
# nodes/users.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from hbond_energy import find_backbone_hbonds, HBondEnergy


def analyze_hbond_network(pdb_path: str) -> Dict:
    """
    Analyze H-bond network topology.

    Returns dict with:
        n_hbonds: total backbone H-bonds
        n_helix_hbonds: H-bonds with i→i+4 pattern
        max_helix_run: longest consecutive helix H-bond chain
        n_helix_runs: number of helix segments (≥3 consecutive)
        n_sheet_hbonds: H-bonds with |i-j| > 4 (long-range)
        n_sheet_pairs: number of paired sheet H-bonds (i→j and j±2→i±2)
        largest_component: size of largest connected H-bond component
        hbond_quality: average energy of H-bonds (more negative = better)
        network_score: combined topology score (higher = better fold)
    """
    result = {
        'n_hbonds': 0,
        'n_helix_hbonds': 0,
        'max_helix_run': 0,
        'n_helix_runs': 0,
        'n_sheet_hbonds': 0,
        'n_sheet_pairs': 0,
        'largest_component': 0,
        'hbond_quality': 0.0,
        'network_score': 0.0,
    }

    hbonds = find_backbone_hbonds(pdb_path)
    if not hbonds:
        return result

    calc = HBondEnergy()
    result['n_hbonds'] = len(hbonds)

    # Compute energies
    energies = []
    for hb in hbonds:
        hb.energy = calc.compute_hbond_energy(hb)
        energies.append(hb.energy)
    result['hbond_quality'] = float(np.mean(energies)) if energies else 0.0

    # ====== Helix analysis (i→i+4 pattern) ======
    helix_hbonds = []
    for hb in hbonds:
        sep = hb.acceptor_res_idx - hb.donor_res_idx
        if sep == 4 or sep == -4:  # α-helix i→i+4
            helix_hbonds.append(hb)
        elif sep == 3 or sep == -3:  # 3_10 helix
            helix_hbonds.append(hb)

    result['n_helix_hbonds'] = len(helix_hbonds)

    # Find consecutive helix runs
    if helix_hbonds:
        donor_positions = sorted(set(min(hb.donor_res_idx, hb.acceptor_res_idx) for hb in helix_hbonds))
        runs = []
        current_run = [donor_positions[0]] if donor_positions else []
        for i in range(1, len(donor_positions)):
            if donor_positions[i] - donor_positions[i-1] <= 2:  # consecutive or near
                current_run.append(donor_positions[i])
            else:
                if len(current_run) >= 3:
                    runs.append(current_run)
                current_run = [donor_positions[i]]
        if len(current_run) >= 3:
            runs.append(current_run)

        result['n_helix_runs'] = len(runs)
        result['max_helix_run'] = max(len(r) for r in runs) if runs else 0

    # ====== Sheet analysis (long-range H-bonds) ======
    sheet_hbonds = []
    for hb in hbonds:
        sep = abs(hb.acceptor_res_idx - hb.donor_res_idx)
        if sep > 4:
            sheet_hbonds.append(hb)

    result['n_sheet_hbonds'] = len(sheet_hbonds)

    # Count paired sheet H-bonds (i→j and nearby j→i = strand pair)
    sheet_set = set((hb.donor_res_idx, hb.acceptor_res_idx) for hb in sheet_hbonds)
    n_pairs = 0
    for hb in sheet_hbonds:
        d, a = hb.donor_res_idx, hb.acceptor_res_idx
        # Check for reverse H-bond nearby
        for dd in range(-2, 3):
            for da in range(-2, 3):
                if (a + da, d + dd) in sheet_set:
                    n_pairs += 1
                    break
    result['n_sheet_pairs'] = n_pairs // 2  # each pair counted twice

    # ====== Connected components ======
    # Build adjacency from H-bonds (residue-level graph)
    adj = {}
    for hb in hbonds:
        d, a = hb.donor_res_idx, hb.acceptor_res_idx
        adj.setdefault(d, set()).add(a)
        adj.setdefault(a, set()).add(d)

    # BFS to find components
    visited = set()
    components = []
    for node in adj:
        if node in visited:
            continue
        component = set()
        queue = [node]
        while queue:
            n = queue.pop(0)
            if n in visited:
                continue
            visited.add(n)
            component.add(n)
            for neighbor in adj.get(n, []):
                if neighbor not in visited:
                    queue.append(neighbor)
        components.append(component)

    result['largest_component'] = max(len(c) for c in components) if components else 0

    # ====== Combined network score ======
    # Higher = better structured protein
    # Normalize by approximate number of residues
    n_res_approx = max(
        max(hb.donor_res_idx for hb in hbonds) + 1,
        max(hb.acceptor_res_idx for hb in hbonds) + 1
    ) if hbonds else 1

    network_score = (
        # H-bond density (more = better)
        result['n_hbonds'] / n_res_approx * 2.0 +
        # Helix quality (longer runs = better)
        min(result['max_helix_run'] / 10.0, 1.0) * 1.5 +
        # Sheet quality (more pairs = better)
        min(result['n_sheet_pairs'] / 5.0, 1.0) * 1.5 +
        # Connectivity (larger component = better)
        result['largest_component'] / n_res_approx * 2.0 +
        # H-bond energy quality (more negative = better)
        min(-result['hbond_quality'] / 0.5, 1.0) * 1.0
    )
    result['network_score'] = float(network_score)

    return result


if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument('pdb')
    args = p.parse_args()

    r = analyze_hbond_network(args.pdb)
    print(f"H-bond Network Analysis: {Path(args.pdb).name}")
    for k, v in r.items():
        if isinstance(v, float):
            print(f"  {k:<22} {v:>10.3f}")
        else:
            print(f"  {k:<22} {v:>10}")
