#!/usr/bin/env python3
"""aADARS v3 scorer — 7 CDR types × 26 bins."""
import os
import sys
from pathlib import Path
import numpy as np
import torch
import gemmi

# Sibling scorer module — works once package is installed (`pip install -e .`).
# Fall back to legacy dev path if running outside the installed package.
try:
    from residudes_qa.ema.scorers.aadars_per_cdr import (
        THREE_TO_ONE, AA2I, get_cb
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from aadars_per_cdr import (  # type: ignore
        THREE_TO_ONE, AA2I, get_cb
    )


def _find_v3_potential() -> str:
    """Resolve aADARS v3 potential path.

    Priority: $EMA_AADARS_V3_POT env var → bundled EMA/data/potentials/.
    """
    env = os.environ.get('EMA_AADARS_V3_POT')
    if env and Path(env).exists():
        return env
    bundled = Path(__file__).resolve().parents[1] / 'data' / 'potentials' / 'aadars_v3_full.pt'
    if bundled.exists():
        return str(bundled)
    raise FileNotFoundError(
        'aADARS v3 potential not found. Set EMA_AADARS_V3_POT or place '
        'aadars_v3_full.pt under EMA/data/potentials/.'
    )

CDR_NAMES_V3 = ['CDR_H1', 'CDR_H2', 'CDR_H3', 'CDR_L1', 'CDR_L2', 'CDR_L3', 'FRAMEWORK']
# Mapping from Chothia number → v3 type index (by chain role)
def cdr_type_v3(chain_role, cn):
    if chain_role == 'H':
        if 26 <= cn <= 32:
            return 0  # CDR_H1
        if 52 <= cn <= 56:
            return 1  # CDR_H2
        if 95 <= cn <= 102:
            return 2 # CDR_H3
        return 6  # FW
    if chain_role == 'L':
        if 24 <= cn <= 34:
            return 3  # CDR_L1
        if 50 <= cn <= 56:
            return 4  # CDR_L2
        if 89 <= cn <= 97:
            return 5  # CDR_L3
        return 6  # FW
    return None


def anarci_cdr_types_v3(seq, chain_type):
    """Return list of v3 type indices per sequence position."""
    from anarci import anarci as run_anarci
    n = len(seq)
    types = [6] * n  # default FRAMEWORK
    try:
        res, det, _ = run_anarci([("chain", seq)], scheme='chothia', output=False)
        if res[0] is None:
            return types
        numbered, seq_start, _ = res[0][0]
        idx = seq_start
        for (cn, ins), aa in numbered:
            if aa == '-':
                continue
            if idx >= n:
                break
            t = cdr_type_v3(chain_type, cn)
            if t is not None:
                types[idx] = t
            idx += 1
    except Exception:
        pass
    return types


class AADARSv3Scorer:
    def __init__(self, pot_path: str | None = None):
        if pot_path is None:
            pot_path = _find_v3_potential()
        d = torch.load(pot_path, weights_only=False, map_location='cpu')
        self.pot = d['potential'].numpy()  # [20, 7, 26]
        cfg = d['config']
        self.bw = cfg['BIN_WIDTH']
        self.r_min = cfg['R_MIN']
        self.r_max = cfg['R_MAX']
        self.n_bins = cfg['N_BINS']
        self.ab_names = d['ab_type_names']

    def score(self, pdb_path, heavy, light, ag_chains):
        st = gemmi.read_structure(str(pdb_path))

        ab_atoms = []   # (pos, ab_type)
        ag_atoms = []   # (pos, ag_type)
        for model in st:
            for chain in model:
                is_h = chain.name == heavy
                is_l = light and chain.name == light
                is_ag = chain.name in ag_chains
                if not (is_h or is_l or is_ag):
                    continue
                residues = [r for r in chain if r.name in AA2I and get_cb(r) is not None]
                if not residues:
                    continue
                if is_ag:
                    for res in residues:
                        cb = get_cb(res)
                        ag_atoms.append((np.array([cb.pos.x, cb.pos.y, cb.pos.z]), AA2I[res.name]))
                else:
                    role = 'H' if is_h else 'L'
                    seq = ''.join(THREE_TO_ONE[r.name] for r in residues)
                    types = anarci_cdr_types_v3(seq, role)
                    for res, t in zip(residues, types):
                        cb = get_cb(res)
                        ab_atoms.append((np.array([cb.pos.x, cb.pos.y, cb.pos.z]), t))
            break

        if not ab_atoms or not ag_atoms:
            return None

        ab_pos = np.array([a[0] for a in ab_atoms])
        ab_t = np.array([a[1] for a in ab_atoms])
        ag_pos = np.array([a[0] for a in ag_atoms])
        ag_t = np.array([a[1] for a in ag_atoms])
        D = np.sqrt(((ag_pos[:,None]-ab_pos[None,:])**2).sum(-1))
        dbin = np.floor((D - self.r_min) / self.bw).astype(int)
        valid = (dbin >= 0) & (dbin < self.n_bins)

        result = {'E_total_v3': 0.0}
        for ct_idx, name in enumerate(self.ab_names):
            mask = valid & (ab_t[None,:] == ct_idx)
            if not mask.any():
                result[f'E_{name}'] = 0.0
                result[f'n_{name}'] = 0
                continue
            ii, jj = np.where(mask)
            e = self.pot[ag_t[ii], ct_idx, dbin[ii, jj]].sum()
            result[f'E_{name}'] = float(e)
            result[f'n_{name}'] = int(mask.sum())
            result['E_total_v3'] += float(e)

        total_n = sum(result[f'n_{n}'] for n in self.ab_names)
        for name in self.ab_names:
            result[f'frac_{name}'] = result[f'n_{name}'] / max(total_n, 1)
        return result


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('pdb')
    ap.add_argument('--heavy', default='H')
    ap.add_argument('--light', default='L')
    ap.add_argument('--ag', default='A')
    a = ap.parse_args()
    s = AADARSv3Scorer()
    r = s.score(a.pdb, a.heavy, a.light, a.ag.split(','))
    for k, v in r.items():
        print(f'  {k:<22} {v}')
