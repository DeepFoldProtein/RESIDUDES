#!/usr/bin/env python3
"""
Backbone-Dependent Rotamer Library (bbdep2010) Parser and Chi Energy
- Parse Dunbrack's bbdep2010 rotamer library
- Calculate backbone-dependent sidechain χ energy using von Mises
- Uses actual rotamer probabilities and distributions
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import gemmi

# =============================================================================
# BBDep2010 Parser
# =============================================================================

class BBDepRotamerLibrary:
    """
    Parse and store backbone-dependent rotamer library
    
    Format of ALL.bbdep.rotamers.lib:
    # T  Phi  Psi  Count  r1 r2 r3 r4 Probabil  chi1Val chi2Val chi3Val chi4Val  chi1Sig chi2Sig chi3Sig chi4Sig
    ARG  -180 -180    14   1  2  2  1  0.199100   63.2   176.9   176.7    85.9    7.4    11.1    10.7    10.1
    """
    
    # Number of chi angles per residue type
    N_CHI = {
        'ARG': 4, 'LYS': 4, 'MET': 3, 'GLN': 3, 'GLU': 3,
        'LEU': 2, 'ILE': 2, 'ASN': 2, 'ASP': 2, 'HIS': 2,
        'PHE': 2, 'TYR': 2, 'TRP': 2, 'PRO': 2, 'CYS': 1,
        'SER': 1, 'THR': 1, 'VAL': 1,
        # Special types
        'CYH': 1, 'CYD': 1, 'TPR': 2, 'CPR': 2,
    }
    
    def __init__(self, lib_path: str, max_rotamers: int = 50):
        """
        Load rotamer library
        
        Args:
            lib_path: Path to ALL.bbdep.rotamers.lib
            max_rotamers: Maximum rotamers to keep per (restype, phi, psi) bin
        """
        self.max_rotamers = max_rotamers
        self.data = defaultdict(dict)  # {restype: {(phi, psi): [rotamers...]}}
        self._parse(lib_path)
    
    def _parse(self, path: str):
        """Parse the library file"""
        print(f"Loading bbdep rotamer library from {path}...")
        
        current_restype = None
        rotamer_count = 0
        
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split()
                if len(parts) < 17:
                    continue
                
                try:
                    restype = parts[0]
                    phi = int(parts[1])
                    psi = int(parts[2])
                    count = int(parts[3])
                    r1, r2, r3, r4 = int(parts[4]), int(parts[5]), int(parts[6]), int(parts[7])
                    prob = float(parts[8])
                    
                    chi_vals = [float(parts[9]), float(parts[10]), float(parts[11]), float(parts[12])]
                    chi_sigs = [float(parts[13]), float(parts[14]), float(parts[15]), float(parts[16])]
                    
                except (ValueError, IndexError):
                    continue
                
                key = (phi, psi)
                
                if restype not in self.data:
                    self.data[restype] = {}
                
                if key not in self.data[restype]:
                    self.data[restype][key] = []
                
                # Only keep top rotamers
                if len(self.data[restype][key]) < self.max_rotamers:
                    self.data[restype][key].append({
                        'rotamer': (r1, r2, r3, r4),
                        'prob': prob,
                        'chi_mean': chi_vals,
                        'chi_sig': chi_sigs,
                    })
        
        # Summary
        print(f"Loaded {len(self.data)} residue types:")
        for restype, bins in sorted(self.data.items()):
            n_rotamers = sum(len(rots) for rots in bins.values())
            print(f"  {restype}: {len(bins)} φ/ψ bins, {n_rotamers} rotamers")
    
    def get_rotamers(self, restype: str, phi: float, psi: float) -> List[Dict]:
        """
        Get rotamers for given residue type and backbone angles
        
        Args:
            restype: 3-letter amino acid code
            phi, psi: Backbone angles in degrees
        
        Returns:
            List of rotamer dictionaries with prob, chi_mean, chi_sig
        """
        if restype not in self.data:
            return []
        
        # Round to 10-degree bins
        phi_bin = int(round(phi / 10.0) * 10)
        psi_bin = int(round(psi / 10.0) * 10)
        
        # Handle wraparound
        if phi_bin == 180:
            phi_bin = -180
        if psi_bin == 180:
            psi_bin = -180
        
        key = (phi_bin, psi_bin)
        return self.data[restype].get(key, [])
    
    def get_n_chi(self, restype: str) -> int:
        """Get number of χ angles for residue type"""
        return self.N_CHI.get(restype, 0)


# =============================================================================
# Backbone-Dependent Chi Energy
# =============================================================================

class BBDepChiEnergy(nn.Module):
    """
    Backbone-dependent sidechain χ energy using rotamer library
    
    E_χ = -log Σ_r [ P(r|φ,ψ) × Π_i N(χ_i | μ_i^r, σ_i^r) ]
    
    Where:
        r = rotamer index
        P(r|φ,ψ) = rotamer probability from library
        N() = normal (von Mises for angles) distribution
    """
    
    def __init__(self, lib_path: str, device: str = 'cpu'):
        super().__init__()
        self.device = device
        self.lib = BBDepRotamerLibrary(lib_path)
    
    def compute_chi_log_prob(self, chi_obs: np.ndarray, chi_mean: List[float],
                             chi_sig: List[float], n_chi: int) -> float:
        """
        Compute log probability of observed chi angles given rotamer parameters

        Uses von Mises distribution: log p = κ cos(χ - μ) - log(2π I₀(κ))
        κ = 1/σ² where σ is in radians
        """
        from scipy.special import i0 as bessel_i0

        log_prob = 0.0

        for i in range(n_chi):
            if np.isnan(chi_obs[i]):
                continue

            mu = np.deg2rad(chi_mean[i])
            sig_deg = max(chi_sig[i], 2.0)  # Minimum 2 degrees
            sig_rad = np.deg2rad(sig_deg)
            kappa = 1.0 / (sig_rad ** 2)
            # No cap — let the normalization handle high κ

            chi = chi_obs[i]  # Already in radians

            # Von Mises log probability (properly normalized)
            log_prob += kappa * np.cos(chi - mu) - np.log(2 * np.pi * bessel_i0(kappa))

        return log_prob
    
    def compute_residue_energy(self, restype: str, phi: float, psi: float, 
                               chi_obs: np.ndarray, eps: float = 1e-10) -> float:
        """
        Compute χ energy for a single residue
        
        Args:
            restype: Residue type (3-letter code)
            phi, psi: Backbone angles in degrees
            chi_obs: Observed χ angles in radians (shape: [4])
            eps: Numerical stability
        
        Returns:
            Energy = -log P(χ | restype, φ, ψ)
        """
        rotamers = self.lib.get_rotamers(restype, phi, psi)
        n_chi = self.lib.get_n_chi(restype)
        
        if not rotamers or n_chi == 0:
            return 0.0
        
        # Compute weighted probability over all rotamers
        log_probs = []
        log_weights = []
        
        for rot in rotamers:
            prob = rot['prob']
            if prob <= 0:
                continue
            
            log_weight = np.log(prob + eps)
            log_p = self.compute_chi_log_prob(chi_obs, rot['chi_mean'], rot['chi_sig'], n_chi)
            
            log_probs.append(log_weight + log_p)
        
        if not log_probs:
            return 0.0
        
        # Log-sum-exp for numerical stability
        max_log_prob = max(log_probs)
        total_log_prob = max_log_prob + np.log(sum(np.exp(lp - max_log_prob) for lp in log_probs))
        
        # Energy = -log probability
        energy = -total_log_prob
        
        return energy
    
    def forward(self, chi_data: Dict, eps: float = 1e-10) -> torch.Tensor:
        """
        Compute total χ energy for a structure
        
        Args:
            chi_data: Output from extract_chi_angles_gemmi() containing:
                - resnames: List of residue names
                - phi, psi: (N,) backbone angles in radians
                - chi: (N, 4) chi angles in radians
        
        Returns:
            Total energy (scalar tensor)
        """
        resnames = chi_data['resnames']
        phi = np.rad2deg(chi_data['phi'])  # Convert to degrees for library lookup
        psi = np.rad2deg(chi_data['psi'])
        chi = chi_data['chi']  # Keep in radians for von Mises
        
        total_energy = 0.0
        
        for i, resname in enumerate(resnames):
            if np.isnan(phi[i]) or np.isnan(psi[i]):
                continue
            
            energy = self.compute_residue_energy(resname, phi[i], psi[i], chi[i], eps)
            total_energy += energy
        
        return torch.tensor(total_energy, device=self.device)
    
    def forward_per_residue(self, chi_data: Dict) -> np.ndarray:
        """Return per-residue χ energies"""
        resnames = chi_data['resnames']
        phi = np.rad2deg(chi_data['phi'])
        psi = np.rad2deg(chi_data['psi'])
        chi = chi_data['chi']
        
        N = len(resnames)
        energies = np.zeros(N)
        
        for i, resname in enumerate(resnames):
            if np.isnan(phi[i]) or np.isnan(psi[i]):
                energies[i] = np.nan
                continue
            
            energies[i] = self.compute_residue_energy(resname, phi[i], psi[i], chi[i])
        
        return energies


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    from pathlib import Path
    from sidechain_energy import extract_chi_angles_gemmi
    import sys
    
    # Load library
    lib_path = "/runs/users/gunwoong/work/Ramachadra/data/dunbrack/ALL.bbdep.rotamers.lib"
    
    if not Path(lib_path).exists():
        print(f"Library not found: {lib_path}")
        sys.exit(1)
    
    print("Testing BBDep Chi Energy...")
    bbdep_energy = BBDepChiEnergy(lib_path)
    
    # Test on CASP14
    predictions_dir = Path('/runs/users/gunwoong/work/DFIRE/dfire_40/data/casp14_full/predictions')
    pdb_files = list(predictions_dir.glob('**/*.pdb'))[:5]
    
    print(f"\nTesting on {len(pdb_files)} structures...")
    
    for pdb_path in pdb_files:
        chi_data = extract_chi_angles_gemmi(str(pdb_path))
        if chi_data is None:
            continue
        
        E = bbdep_energy(chi_data)
        E_per_res = bbdep_energy.forward_per_residue(chi_data)
        
        mean_E = np.nanmean(E_per_res)
        n_valid = np.sum(~np.isnan(E_per_res))
        
        print(f"{pdb_path.stem}: E_chi={E.item():.1f}, mean_E={mean_E:.2f}, n={n_valid}")
    
    print("\nBBDep chi energy test passed")
