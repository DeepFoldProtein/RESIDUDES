#!/usr/bin/env python3
"""Ab-Ag QA via aADARS with per-CDR decomposition.

Loads trained aADARS potential [20 AG × 4 CDR × 13 dist bins] and scores
a predicted complex returning:
  - E_total: full aADARS energy
  - E_H3: contribution from CDR-H3 × Ag pairs
  - E_H12: from CDR-H1/H2
  - E_L:   from CDR-L (all light chain CDRs)
  - E_FW:  from Framework × Ag
  - n_contacts_{H3, H12, L, FW}

Expected pattern: E_H3 should dominate in well-docked poses;
E_FW should be small (framework shouldn't touch Ag).
"""
import sys, json
from pathlib import Path
import numpy as np
import torch
import gemmi

AA = ['ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE',
      'LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL']
AA2I = {a: i for i, a in enumerate(AA)}

# CDR definitions (Chothia) matching fft-docking-torch/aadars.py
CDR_H = {'H3': (95, 102), 'H1': (26, 32), 'H2': (52, 56)}
CDR_L = {'L1': (24, 34), 'L2': (50, 56), 'L3': (89, 97)}

# CDR types (0=H3, 1=H12, 2=L, 3=FW)
CDR_NAMES = ['H3', 'H12', 'L', 'FW']


def cdr_type(chain_role, res_num):
    """Fallback (no ANARCI) — assumes PDB already uses Chothia numbering."""
    if chain_role == 'H':
        h3 = CDR_H['H3']; h1 = CDR_H['H1']; h2 = CDR_H['H2']
        if h3[0] <= res_num <= h3[1]: return 0
        if h1[0] <= res_num <= h1[1] or h2[0] <= res_num <= h2[1]: return 1
        return 3
    if chain_role == 'L':
        for lo, hi in CDR_L.values():
            if lo <= res_num <= hi: return 2
        return 3
    return None


THREE_TO_ONE = {
    'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E',
    'GLY':'G','HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F',
    'PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V',
}


def anarci_cdr_types_for_chain(seq, chain_type):
    """Use ANARCI to map each sequence position to CDR type (0-3).
    Returns list[int] of length len(seq)."""
    from anarci import anarci as run_anarci
    n = len(seq)
    types = [3] * n
    try:
        res, det, _ = run_anarci([("chain", seq)], scheme='chothia', output=False)
        if res[0] is None: return types
        numbered, seq_start, _ = res[0][0]
        idx = seq_start
        for (cn, ins), aa in numbered:
            if aa == '-': continue
            if idx >= n: break
            if chain_type == 'H':
                if 95 <= cn <= 102: types[idx] = 0
                elif (26 <= cn <= 32) or (52 <= cn <= 56): types[idx] = 1
            elif chain_type == 'L':
                if (24 <= cn <= 34) or (50 <= cn <= 56) or (89 <= cn <= 97):
                    types[idx] = 2
            idx += 1
    except Exception:
        pass
    return types


def get_cb(res):
    cb = res.find_atom('CB', '*')
    return cb or res.find_atom('CA', '*')


def _find_aadars_pot() -> str:
    """Resolve the legacy aADARS potential (env var → bundled dir).

    Priority: $EMA_AADARS_POT → EMA/data/potentials/aadars_fft_decoy_445_v2.pt.
    No hardcoded personal path; raises a clear error if not found.
    """
    import os
    env = os.environ.get('EMA_AADARS_POT')
    if env and Path(env).exists():
        return env
    bundled = Path(__file__).resolve().parents[1] / 'data' / 'potentials' / 'aadars_fft_decoy_445_v2.pt'
    if bundled.exists():
        return str(bundled)
    raise FileNotFoundError(
        'aADARS potential not found. Set EMA_AADARS_POT or place '
        'aadars_fft_decoy_445_v2.pt under EMA/data/potentials/. '
        '(The newer XGBDimer pipeline uses AADARSv3Scorer, not this legacy class.)'
    )


class AADARSScorer:
    def __init__(self, pot_path=None):
        if pot_path is None:
            pot_path = _find_aadars_pot()
        d = torch.load(pot_path, weights_only=False)
        self.pot = d['potential']  # [20, 4, 13]
        self.cfg = d['config']
        self.bw = self.cfg['BIN_WIDTH']
        self.r_min = self.cfg['R_MIN']
        self.r_max = self.cfg['R_MAX']
        self.n_bins = self.cfg['N_BINS']

    def score(self, pdb_path, heavy, light, ag_chains, cutoff=15.0, use_anarci=True):
        st = gemmi.read_structure(pdb_path)

        ab_atoms = []  # (cb_pos, cdr_type)
        ag_atoms = []
        # Collect per-chain residue sequences for ANARCI
        chain_residues = {}  # chain_id -> list of (res, aa_1letter)
        for model in st:
            for chain in model:
                is_h = chain.name == heavy
                is_l = light and chain.name == light
                is_ag = chain.name in ag_chains
                if not (is_h or is_l or is_ag): continue
                residues = []
                for res in chain:
                    if res.name not in AA2I: continue
                    cb = get_cb(res)
                    if cb is None: continue
                    residues.append(res)
                if not residues: continue
                if is_ag:
                    for res in residues:
                        cb = get_cb(res)
                        pos = np.array([cb.pos.x, cb.pos.y, cb.pos.z])
                        ag_atoms.append((pos, AA2I[res.name]))
                else:
                    role = 'H' if is_h else 'L'
                    if use_anarci:
                        seq = ''.join(THREE_TO_ONE[r.name] for r in residues)
                        types_list = anarci_cdr_types_for_chain(seq, role)
                    else:
                        types_list = [cdr_type(role, r.seqid.num) for r in residues]
                    for res, t in zip(residues, types_list):
                        if t is None: t = 3
                        cb = get_cb(res)
                        pos = np.array([cb.pos.x, cb.pos.y, cb.pos.z])
                        ab_atoms.append((pos, t))
            break

        if not ab_atoms or not ag_atoms:
            return None

        ab_pos = np.array([a[0] for a in ab_atoms])
        ab_cdr = np.array([a[1] for a in ab_atoms])
        ag_pos = np.array([a[0] for a in ag_atoms])
        ag_type = np.array([a[1] for a in ag_atoms])

        D = np.sqrt(((ag_pos[:, None] - ab_pos[None, :]) ** 2).sum(-1))
        # Bin distances
        dbin = np.floor((D - self.r_min) / self.bw).astype(int)
        valid = (dbin >= 0) & (dbin < self.n_bins)

        pot = self.pot.cpu().numpy()  # [20, 4, 13]

        result = {'E_total': 0.0}
        for ct_idx, name in enumerate(CDR_NAMES):
            mask = valid & (ab_cdr[None, :] == ct_idx)
            if not mask.any():
                result[f'E_{name}'] = 0.0
                result[f'n_{name}'] = 0
                continue
            ii, jj = np.where(mask)
            e = pot[ag_type[ii], ct_idx, dbin[ii, jj]].sum()
            result[f'E_{name}'] = float(e)
            result[f'n_{name}'] = int(mask.sum())
            result['E_total'] += float(e)

        # Ratios (potential discriminators)
        total_n = sum(result[f'n_{n}'] for n in CDR_NAMES)
        result['frac_H3'] = result['n_H3'] / max(total_n, 1)
        result['frac_FW'] = result['n_FW'] / max(total_n, 1)
        # Lower FW fraction = better docking (FW shouldn't touch Ag)
        return result


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('pdb')
    ap.add_argument('--heavy', default='H')
    ap.add_argument('--light', default='L')
    ap.add_argument('--ag', default='A')
    args = ap.parse_args()
    s = AADARSScorer()
    r = s.score(args.pdb, args.heavy, args.light, args.ag.split(','))
    for k, v in r.items():
        print(f'  {k:<12} {v}')
