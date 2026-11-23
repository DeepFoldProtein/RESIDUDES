#!/usr/bin/env python3
"""DFIRE v2 scorer (167 atom types, trained on PDB natives + CASP14/15 decoys).
Energy = E_pos - alpha * E_neg (alpha=1.57 from training).
"""
import os, sys, torch, numpy as np
import gemmi
from pathlib import Path


def _find_dfire_table() -> str:
    """Resolve DFIRE v2 energy table file.

    Priority:
      1. $EMA_DFIRE_V2_TABLE env var
      2. EMA/data/dfire/dfire_v2_energy_table_with_types.pt (bundled)
    """
    env = os.environ.get('EMA_DFIRE_V2_TABLE')
    if env and Path(env).exists():
        return env
    here = Path(__file__).resolve()
    bundled = here.parents[1] / 'data' / 'dfire' / 'dfire_v2_energy_table_with_types.pt'
    if bundled.exists():
        return str(bundled)
    raise FileNotFoundError(
        'DFIRE v2 energy table not found. Set EMA_DFIRE_V2_TABLE or place '
        'dfire_v2_energy_table_with_types.pt under EMA/data/dfire/.'
    )


class DFIREv2Scorer:
    def __init__(self, path: str | None = None, device='cuda'):
        if path is None:
            path = _find_dfire_table()
        # CPU fallback if no CUDA
        if device == 'cuda' and not torch.cuda.is_available():
            device = 'cpu'
        d = torch.load(path, weights_only=False, map_location=device)
        self.E_pos = torch.as_tensor(d['E_pos'], dtype=torch.float32).to(device)
        self.E_neg = torch.as_tensor(d['E_neg'], dtype=torch.float32).to(device)
        self.alpha = float(d['alpha'])
        self.types = d['types']
        self.type_index = d['type_index']
        self.valid_pairs = d['valid_pairs']   # set of (i,j) valid pair tuples
        self.bin_centers = d['bin_centers']
        self.n_bins = d['num_bins']
        self.r_min = float(self.bin_centers[0]) - 0.25
        self.r_max = float(self.bin_centers[-1]) + 0.25
        self.bw = 0.5  # uniform
        self.cut_bin = int(d['cut_bin']) if not isinstance(d['cut_bin'], int) else d['cut_bin']
        self.device = device
        # Precompute valid pair mask tensor
        vp = torch.zeros(len(self.types), len(self.types), dtype=torch.bool, device=device)
        for (i, j) in self.valid_pairs:
            vp[i, j] = True
            vp[j, i] = True
        self.valid_mask = vp

    def _ensure_pdb(self, path):
        p = str(path)
        if p.endswith('.cif'): return p
        try:
            with open(p, 'rb') as f: head = f.read(10)
            if b'PFRMAT TS' in head or not head.startswith(b'ATOM'):
                # CASP TS file, write as-is but read with gemmi pdb
                pass
        except Exception: pass
        return p

    def score(self, struct_path):
        try:
            st = gemmi.read_structure(str(struct_path))
        except Exception:
            # CASP TS fallback: copy to .pdb extension
            import tempfile, shutil
            tmp = tempfile.NamedTemporaryFile(suffix='.pdb', delete=False)
            shutil.copy(str(struct_path), tmp.name); tmp.close()
            st = gemmi.read_structure(tmp.name)
            import os; os.unlink(tmp.name)

        atoms = []
        for model in st:
            for chain in model:
                for res in chain:
                    if res.name not in {'ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY',
                                         'HIS','ILE','LEU','LYS','MET','PHE','PRO','SER',
                                         'THR','TRP','TYR','VAL'}: continue
                    for atom in res:
                        key = f'{res.name} {atom.name}'
                        if key in self.type_index:
                            atoms.append((self.type_index[key],
                                          atom.pos.x, atom.pos.y, atom.pos.z))
            break
        if len(atoms) < 2: return {'E_dfire_v2': np.nan, 'n_atoms': 0}

        idx = torch.tensor([a[0] for a in atoms], dtype=torch.long, device=self.device)
        xyz = torch.tensor([a[1:] for a in atoms], dtype=torch.float32, device=self.device)
        N = xyz.shape[0]
        diff = xyz.unsqueeze(0) - xyz.unsqueeze(1)
        dist = torch.sqrt((diff ** 2).sum(-1) + 1e-8)
        triu = torch.triu_indices(N, N, offset=1, device=self.device)
        i, j = triu[0], triu[1]
        d = dist[i, j]
        mask_d = (d >= self.r_min) & (d < self.r_max)
        ti = idx[i]; tj = idx[j]
        mask_v = self.valid_mask[ti, tj]
        final = mask_d & mask_v
        if not final.any(): return {'E_dfire_v2': 0.0, 'n_atoms': N}
        vd = d[final]; vti = ti[final]; vtj = tj[final]
        # Bin index
        dbin = torch.clamp(torch.floor((vd - self.r_min) / self.bw).long(), 0, self.n_bins - 1)
        e_pos = self.E_pos[vti, vtj, dbin]
        e_neg = self.E_neg[vti, vtj, dbin]
        E_total = (e_pos - self.alpha * e_neg).sum().item()
        E_pos_sum = e_pos.sum().item()
        E_neg_sum = e_neg.sum().item()
        return {
            'E_dfire_v2': float(E_total),
            'E_dfire_v2_pos': float(E_pos_sum),
            'E_dfire_v2_neg': float(E_neg_sum),
            'E_dfire_v2_per_atom': float(E_total / N),
            'n_atoms': int(N),
            'n_pairs': int(final.sum().item()),
        }


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument('pdb')
    a = ap.parse_args()
    s = DFIREv2Scorer()
    r = s.score(a.pdb)
    for k, v in r.items(): print(f'  {k:<24} {v}')
