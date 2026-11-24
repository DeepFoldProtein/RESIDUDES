#!/usr/bin/env python3
"""BBDep Chi Energy with bilinear interpolation on rotamer probs.

Keeps original E_chi = -log Σ_r P(r|φ,ψ) × VM(χ_obs|μ_r, κ_r) formula
but interpolates P(r|φ,ψ) linearly across 4 corner (φ,ψ) bins
instead of snapping to single bin.
"""
import sys, os
from pathlib import Path
import numpy as np
import gemmi
from scipy.special import i0 as bessel_i0


def _find_dunbrack_lib() -> str:
    """Resolve BBDep rotamer library location.

    Priority:
      1. $EMA_BBDEP_LIB env var
      2. EMA/data/dunbrack/ALL.bbdep.rotamers.lib (bundled)
    """
    candidates = []
    env = os.environ.get('EMA_BBDEP_LIB')
    if env:
        candidates.append(env)
    here = Path(__file__).resolve()
    candidates.append(here.parents[1] / 'data' / 'dunbrack' / 'ALL.bbdep.rotamers.lib')
    for c in candidates:
        if Path(c).exists():
            return str(c)
    raise FileNotFoundError(
        'BBDep rotamer library not found. Set EMA_BBDEP_LIB or place the file '
        'under EMA/data/dunbrack/ALL.bbdep.rotamers.lib (see EMA/data/dunbrack/README.md).'
    )


# bbdep_chi_energy (the rotamer-library parser) is vendored next to this module
# in the EMA tree.  Try the installed-package import first, then fall back to a
# same-directory import based on __file__ so it resolves whether this module is
# loaded as residudes_qa.ema.scorers.bbdep_chi_spline_v2 or run directly as a script — no
# hardcoded personal path that breaks on other nodes/users.
try:
    from residudes_qa.ema.scorers.bbdep_chi_energy import BBDepRotamerLibrary  # type: ignore
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        from bbdep_chi_energy import BBDepRotamerLibrary  # type: ignore
    except ImportError:
        BBDepRotamerLibrary = None  # set when needed; tests for import-only still pass


LIB_PATH = _find_dunbrack_lib() if BBDepRotamerLibrary is not None else None

AA_SET = {'ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE',
          'LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL'}

CHI_ATOMS = {
    'ARG': [('N','CA','CB','CG'), ('CA','CB','CG','CD'),
            ('CB','CG','CD','NE'), ('CG','CD','NE','CZ')],
    'ASN': [('N','CA','CB','CG'), ('CA','CB','CG','OD1')],
    'ASP': [('N','CA','CB','CG'), ('CA','CB','CG','OD1')],
    'CYS': [('N','CA','CB','SG')],
    'GLN': [('N','CA','CB','CG'), ('CA','CB','CG','CD'), ('CB','CG','CD','OE1')],
    'GLU': [('N','CA','CB','CG'), ('CA','CB','CG','CD'), ('CB','CG','CD','OE1')],
    'HIS': [('N','CA','CB','CG'), ('CA','CB','CG','ND1')],
    'ILE': [('N','CA','CB','CG1'), ('CA','CB','CG1','CD1')],
    'LEU': [('N','CA','CB','CG'), ('CA','CB','CG','CD1')],
    'LYS': [('N','CA','CB','CG'), ('CA','CB','CG','CD'),
            ('CB','CG','CD','CE'), ('CG','CD','CE','NZ')],
    'MET': [('N','CA','CB','CG'), ('CA','CB','CG','SD'), ('CB','CG','SD','CE')],
    'PHE': [('N','CA','CB','CG'), ('CA','CB','CG','CD1')],
    'PRO': [('N','CA','CB','CG'), ('CA','CB','CG','CD')],
    'SER': [('N','CA','CB','OG')],
    'THR': [('N','CA','CB','OG1')],
    'TRP': [('N','CA','CB','CG'), ('CA','CB','CG','CD1')],
    'TYR': [('N','CA','CB','CG'), ('CA','CB','CG','CD1')],
    'VAL': [('N','CA','CB','CG1')],
}


def get_chi_angles(residue):
    """Return list of χ angles in degrees."""
    name = residue.name
    if name not in CHI_ATOMS: return []
    chis = []
    for atoms in CHI_ATOMS[name]:
        pos = []
        for a in atoms:
            atom = residue.find_atom(a, '*')
            if atom is None: return chis  # missing atom
            pos.append(atom.pos)
        if len(pos) == 4:
            chi = np.degrees(gemmi.calculate_dihedral(*pos))
            chis.append(chi)
    return chis


def vm_logprob(chi, mu, kappa):
    """Von Mises log-probability (normalized, in radians)."""
    chi_r = np.radians(chi)
    mu_r = np.radians(mu)
    return kappa * np.cos(chi_r - mu_r) - np.log(2 * np.pi * bessel_i0(kappa))


def get_rotamers_interpolated(lib, restype, phi, psi, method='bilinear'):
    """Get rotamer list with weighted averaging over 4 corner bins."""
    if restype not in lib.data: return []

    # 4 corner bins (10° spacing)
    phi_floor = int(np.floor(phi / 10) * 10)
    psi_floor = int(np.floor(psi / 10) * 10)
    phi_ceil = phi_floor + 10
    psi_ceil = psi_floor + 10

    # Wrap
    def wrap(b):
        if b >= 180: return b - 360
        if b < -180: return b + 360
        return b

    corners = [(wrap(phi_floor), wrap(psi_floor)),
               (wrap(phi_ceil),  wrap(psi_floor)),
               (wrap(phi_floor), wrap(psi_ceil)),
               (wrap(phi_ceil),  wrap(psi_ceil))]

    # Bilinear weights
    wx = (phi - phi_floor) / 10.0
    wy = (psi - psi_floor) / 10.0
    weights = [(1-wx)*(1-wy), wx*(1-wy), (1-wx)*wy, wx*wy]

    if method == 'grid':
        # Nearest corner
        idx = int(weights.index(max(weights)))
        key = corners[idx]
        return lib.data[restype].get(key, [])

    # Collect rotamers from all 4 corners with signature (μ vector)
    # Different bins may have different top-50 rotamer sets — key by tuple(mu rounded)
    combined = {}  # signature → (prob, chi_mean, chi_sig)
    for (pb, sb), w in zip(corners, weights):
        rots = lib.data[restype].get((pb, sb), [])
        for r in rots:
            sig = tuple(round(m / 10) * 10 for m in r['chi_mean'])  # 10° rotamer signature
            if sig not in combined:
                combined[sig] = {'prob': 0.0, 'chi_mean': r['chi_mean'], 'chi_sig': r['chi_sig']}
            combined[sig]['prob'] += w * r['prob']
    return list(combined.values())


def compute_residue_chi_E(lib, restype, phi, psi, chi_obs, method='bilinear'):
    """E_chi for one residue with interpolated rotamers."""
    if not chi_obs: return 0.0
    rots = get_rotamers_interpolated(lib, restype, phi, psi, method=method)
    if not rots: return 10.0

    total = 0.0
    for r in rots:
        if r['prob'] < 1e-6: continue
        n_chi = min(len(chi_obs), len(r['chi_mean']))
        log_prob = np.log(r['prob'] + 1e-10)
        for i in range(n_chi):
            chi = chi_obs[i]
            mu = r['chi_mean'][i]
            sig = max(r['chi_sig'][i], 1.0)
            sig_r = np.radians(sig)
            kappa = min(1.0 / (sig_r ** 2), 500.0)
            log_prob += vm_logprob(chi, mu, kappa)
        total += np.exp(log_prob)

    if total < 1e-10: return 10.0
    return -np.log(total)


_LIB = None
def get_lib():
    global _LIB
    if _LIB is None: _LIB = BBDepRotamerLibrary(LIB_PATH)
    return _LIB


def score_bbdep_chi_v2(pdb_path, methods=('grid', 'bilinear')):
    lib = get_lib()
    try:
        st = gemmi.read_structure(str(pdb_path))
    except Exception:
        import tempfile, shutil
        tmp = tempfile.NamedTemporaryFile(suffix='.pdb', delete=False)
        shutil.copy(str(pdb_path), tmp.name); tmp.close()
        st = gemmi.read_structure(tmp.name)
        os.unlink(tmp.name)

    totals = {m: 0.0 for m in methods}
    n = 0
    for model in st:
        for ch in model:
            residues = [r for r in ch if r.name in AA_SET]
            for i in range(1, len(residues) - 1):
                r = residues[i]
                try:
                    N = r.find_atom('N','*'); CA = r.find_atom('CA','*'); C = r.find_atom('C','*')
                    C_prev = residues[i-1].find_atom('C','*')
                    N_next = residues[i+1].find_atom('N','*')
                    phi = np.degrees(gemmi.calculate_dihedral(C_prev.pos, N.pos, CA.pos, C.pos))
                    psi = np.degrees(gemmi.calculate_dihedral(N.pos, CA.pos, C.pos, N_next.pos))
                    chi_obs = get_chi_angles(r)
                except Exception: continue
                if not chi_obs: continue
                for m in methods:
                    totals[m] += compute_residue_chi_E(lib, r.name, phi, psi, chi_obs, m)
                n += 1
        break
    if n == 0: return None
    return {f'E_bbdep_chi_v2_{m}_per_res': totals[m] / n for m in methods}


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument('pdb')
    a = ap.parse_args()
    r = score_bbdep_chi_v2(a.pdb)
    for k, v in r.items(): print(f'  {k:<42} {v:.4f}')
