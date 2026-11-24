#!/usr/bin/env python3
"""NDRD (Neighbor-Dependent Ramachandran Distribution) QA scorer.

Per-residue energy: E = -log(P(phi, psi | residue, neighbor))
  - Uses 'left' direction (previous neighbor) by default
  - Sum over backbone residues (skip terminals)

Source: Dunbrack 2010 NDRD_TCBIG (Turn+Coil+Bridge+PIHelix+310Helix)
Best for loop/turn quality assessment.
"""
import gzip, pickle, os, sys
from pathlib import Path
import numpy as np
import gemmi

def _find_ndrd_path() -> str:
    """Resolve NDRD TCBIG file location.

    Priority:
      1. $EMA_NDRD_PATH env var
      2. EMA/data/dunbrack/NDRD_TCBIG.txt.gz (bundled, no space in name)
    """
    candidates = []
    env = os.environ.get('EMA_NDRD_PATH')
    if env:
        candidates.append(env)
    here = Path(__file__).resolve()
    candidates.append(here.parents[1] / 'data' / 'dunbrack' / 'NDRD_TCBIG.txt.gz')
    for c in candidates:
        if Path(c).exists():
            return str(c)
    raise FileNotFoundError(
        'NDRD TCBIG file not found. Set EMA_NDRD_PATH or place the file under '
        'EMA/data/dunbrack/NDRD_TCBIG.txt.gz (see EMA/data/dunbrack/README.md).'
    )


# Cache lives next to the data file so it's reusable across users
def _cache_path() -> str:
    import os
    env = os.environ.get('EMA_NDRD_PATH')
    if env:  # residudes-qa: cache next to the NDRD file in the data root
        return str(Path(env).resolve().parent / 'ndrd_tcbig_table.pkl')
    here = Path(__file__).resolve()
    return str(here.parents[1] / 'data' / 'dunbrack' / 'ndrd_tcbig_table.pkl')


# Resolved at first use to avoid hard failure on import when data isn't installed.
NDRD_PATH = None
CACHE_PATH = None


def _ensure_paths():
    global NDRD_PATH, CACHE_PATH
    if NDRD_PATH is None:
        NDRD_PATH = _find_ndrd_path()
    if CACHE_PATH is None:
        CACHE_PATH = _cache_path()


def parse_ndrd():
    """Parse NDRD TCBIG file → dict[(res, dir, neigh)] = (logp_grid, phi_min, psi_min, step)"""
    _ensure_paths()
    if Path(CACHE_PATH).exists():
        with open(CACHE_PATH, 'rb') as f: return pickle.load(f)

    print('Parsing NDRD TCBIG (one-time, ~30s)...')
    table = {}  # (res, dir, neigh) → 73x73 array of logp
    with gzip.open(NDRD_PATH, 'rt') as f:
        for line in f:
            if line.startswith('#') or not line.strip(): continue
            parts = line.split()
            if len(parts) < 7: continue
            res, dr, neigh = parts[0], parts[1], parts[2]
            phi, psi = int(parts[3]), int(parts[4])
            logp = float(parts[6])  # log(prob)
            key = (res, dr, neigh)
            if key not in table:
                # 73 bins from -180 to 180 step 5
                table[key] = np.full((73, 73), 25.0, dtype=np.float32)  # default = high energy
            i = (phi + 180) // 5
            j = (psi + 180) // 5
            if 0 <= i < 73 and 0 <= j < 73:
                table[key][i, j] = logp
    print(f'  Parsed {len(table)} (res,dir,neigh) maps')
    Path(CACHE_PATH).parent.mkdir(exist_ok=True)
    with open(CACHE_PATH, 'wb') as f: pickle.dump(table, f)
    return table


_TABLE = None
def get_table():
    global _TABLE
    if _TABLE is None: _TABLE = parse_ndrd()
    return _TABLE


def get_logp(res, neigh, phi_deg, psi_deg, direction='left', table=None):
    """Get -log(P) from NDRD. Returns 25 (high energy) if no entry."""
    if table is None: table = get_table()
    # Try specific neighbor first, fallback to ALL
    key = (res, direction, neigh)
    if key not in table:
        key = (res, direction, 'ALL')
        if key not in table: return 25.0
    grid = table[key]
    i = max(0, min(72, int((phi_deg + 180) // 5)))
    j = max(0, min(72, int((psi_deg + 180) // 5)))
    return float(grid[i, j])


AA_SET = {'ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE',
          'LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL'}


def score_ndrd(pdb_path, direction='left'):
    """Compute total NDRD energy for a structure."""
    table = get_table()
    try:
        st = gemmi.read_structure(str(pdb_path))
    except Exception:
        import tempfile, shutil
        tmp = tempfile.NamedTemporaryFile(suffix='.pdb', delete=False)
        shutil.copy(str(pdb_path), tmp.name); tmp.close()
        st = gemmi.read_structure(tmp.name)
        os.unlink(tmp.name)

    total_E = 0.0
    n = 0
    n_outliers = 0  # E > 10 (very low prob)
    for model in st:
        for ch in model:
            residues = [r for r in ch if r.name in AA_SET]
            for i, r in enumerate(residues):
                if i == 0 or i == len(residues) - 1: continue  # skip terminals
                # Find prev/next residues
                prev_r = residues[i-1] if direction == 'left' else residues[i+1]
                neigh = prev_r.name
                # Compute phi/psi
                try:
                    n_atom = r.find_atom('N', '*'); ca = r.find_atom('CA', '*'); c = r.find_atom('C', '*')
                    c_prev = residues[i-1].find_atom('C', '*')
                    n_next = residues[i+1].find_atom('N', '*')
                    if not all([n_atom, ca, c, c_prev, n_next]): continue
                    phi = np.degrees(gemmi.calculate_dihedral(c_prev.pos, n_atom.pos, ca.pos, c.pos))
                    psi = np.degrees(gemmi.calculate_dihedral(n_atom.pos, ca.pos, c.pos, n_next.pos))
                except Exception:
                    continue
                E = get_logp(r.name, neigh, phi, psi, direction=direction, table=table)
                total_E += E
                n += 1
                if E > 10: n_outliers += 1
        break

    if n == 0: return {'E_ndrd': np.nan, 'E_ndrd_per_res': np.nan, 'n_residues': 0}
    return {
        'E_ndrd': float(total_E),
        'E_ndrd_per_res': float(total_E / n),
        'n_ndrd_outliers': int(n_outliers),
        'ndrd_outlier_frac': float(n_outliers / n),
        'n_residues': n,
    }


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument('pdb')
    a = ap.parse_args()
    r = score_ndrd(a.pdb)
    for k, v in r.items(): print(f'  {k:<22} {v}')
