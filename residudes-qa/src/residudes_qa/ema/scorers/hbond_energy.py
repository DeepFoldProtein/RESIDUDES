#!/usr/bin/env python3
"""
H-bond Energy Module for Protein Structure QA

Based on Rosetta-style hydrogen bond geometry:
- Distance: d_HA (H...A distance)
- Angle: θ_AHD (A-H-D angle)
- Angle: θ_BAH (B-A-H angle for sp2 acceptors)

Phase 1: Backbone H-bonds (alpha-helix, beta-sheet)
"""

import numpy as np
import gemmi
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass


@dataclass
class HBond:
    """Hydrogen bond data structure"""
    donor_res_idx: int
    acceptor_res_idx: int
    donor_res_name: str
    acceptor_res_name: str
    d_HA: float  # H...A distance
    theta_AHD: float  # A-H-D angle (degrees)
    theta_BAH: float  # B-A-H angle (degrees)
    energy: float = 0.0


class HBondEnergy:
    """
    Backbone hydrogen bond energy calculator.
    
    Energy function:
        E_hbond = f(d) × g(θ_AHD) × h(θ_BAH)
    
    Where:
        f(d) = LJ-like distance function
        g(θ) = cos²(θ - 180°) for AHD angle
        h(θ) = cos²(θ - 120°) for BAH angle (sp2)
    """
    
    def __init__(self, d_opt: float = 1.9, d_max: float = 2.5):
        """
        Args:
            d_opt: Optimal H...A distance (Å)
            d_max: Maximum H...A distance (Å)
        """
        self.d_opt = d_opt
        self.d_max = d_max
        
    def _energy_distance(self, d: float) -> float:
        """Distance-dependent energy (LJ-like)

        Favorable H-bond (d ~ 1.9Å) → negative energy (-1.0)
        Unfavorable (too far) → ~0 energy
        Too close (clash) → positive energy
        """
        if d > self.d_max:
            return 0.0
        if d < 1.4:  # Hard clash
            return 5.0
        if d <= self.d_opt:
            # Flat at optimal — no extra reward for being too close
            # Mild repulsion ramp from 1.4 to 1.6Å
            if d < 1.6:
                return -1.0 + (1.6 - d) / 0.2 * 6.0  # ramp from -1.0 to +5.0
            return -1.0
        # Decay from d_opt to d_max
        x = (d - self.d_opt) / (self.d_max - self.d_opt)
        return -1.0 * (1 - x**2)
    
    def _energy_angle_AHD(self, theta: float) -> float:
        """Angular energy for A-H-D angle (optimal at 180°)"""
        # cos²(θ - 180°) = cos²(θ + 180°)
        return np.cos(np.radians(theta - 180))**2
    
    def _energy_angle_BAH(self, theta: float) -> float:
        """Angular energy for B-A-H angle (optimal at ~120° for sp2)"""
        return np.cos(np.radians(theta - 120))**2
    
    def compute_hbond_energy(self, hbond: HBond) -> float:
        """Compute energy for a single H-bond"""
        e_dist = self._energy_distance(hbond.d_HA)
        e_ahd = self._energy_angle_AHD(hbond.theta_AHD)
        e_bah = self._energy_angle_BAH(hbond.theta_BAH)
        
        return e_dist * e_ahd * e_bah


def calculate_h_position(n_pos: gemmi.Position, ca_pos: gemmi.Position, 
                         c_prev_pos: Optional[gemmi.Position] = None) -> gemmi.Position:
    """
    Calculate H position from N, CA, and previous C.
    
    If C_prev is not available, estimate H along N-CA direction.
    N-H bond length ~1.0 Å
    """
    NH_LENGTH = 1.0
    
    if c_prev_pos is not None:
        # H is approximately opposite to C_prev from N
        # Direction: N - C_prev (normalized) + N - CA (normalized)
        vec_nc = gemmi.Position(
            n_pos.x - c_prev_pos.x,
            n_pos.y - c_prev_pos.y,
            n_pos.z - c_prev_pos.z
        )
        vec_nca = gemmi.Position(
            ca_pos.x - n_pos.x,
            ca_pos.y - n_pos.y,
            ca_pos.z - n_pos.z
        )
        
        # Normalize and combine
        len_nc = (vec_nc.x**2 + vec_nc.y**2 + vec_nc.z**2)**0.5
        len_nca = (vec_nca.x**2 + vec_nca.y**2 + vec_nca.z**2)**0.5
        
        if len_nc < 0.1 or len_nca < 0.1:
            return None
        
        # H direction is roughly opposite to bisector of C-N-CA
        h_dir = gemmi.Position(
            vec_nc.x/len_nc - vec_nca.x/len_nca,
            vec_nc.y/len_nc - vec_nca.y/len_nca,
            vec_nc.z/len_nc - vec_nca.z/len_nca
        )
        len_h = (h_dir.x**2 + h_dir.y**2 + h_dir.z**2)**0.5
        if len_h < 0.1:
            return None
        
        h_pos = gemmi.Position(
            n_pos.x + NH_LENGTH * h_dir.x / len_h,
            n_pos.y + NH_LENGTH * h_dir.y / len_h,
            n_pos.z + NH_LENGTH * h_dir.z / len_h
        )
        return h_pos
    else:
        # Simple fallback: H along N to CA direction (opposite)
        vec = gemmi.Position(
            n_pos.x - ca_pos.x,
            n_pos.y - ca_pos.y,
            n_pos.z - ca_pos.z
        )
        length = (vec.x**2 + vec.y**2 + vec.z**2)**0.5
        if length < 0.1:
            return None
        
        h_pos = gemmi.Position(
            n_pos.x + NH_LENGTH * vec.x / length,
            n_pos.y + NH_LENGTH * vec.y / length,
            n_pos.z + NH_LENGTH * vec.z / length
        )
        return h_pos


def find_backbone_hbonds(pdb_path: str, d_max: float = 3.5) -> List[HBond]:
    """
    Find backbone hydrogen bonds in a protein structure.
    
    Backbone H-bond: N-H (donor) → O=C (acceptor)
    
    Args:
        pdb_path: Path to PDB file
        d_max: Maximum H...O distance to consider
        
    Returns:
        List of HBond objects
    """
    st = gemmi.read_structure(pdb_path)
    model = st[0]
    
    hbonds = []
    
    for chain in model:
        residues = [r for r in chain]
        len(residues)
        
        # Extract backbone atoms
        backbone = []
        for i, res in enumerate(residues):
            n = res.find_atom('N', '*')
            ca = res.find_atom('CA', '*')
            c = res.find_atom('C', '*')
            o = res.find_atom('O', '*')
            
            c_prev = residues[i-1].find_atom('C', '*') if i > 0 else None
            
            backbone.append({
                'res_idx': i,
                'res_name': res.name,
                'N': n.pos if n else None,
                'CA': ca.pos if ca else None,
                'C': c.pos if c else None,
                'O': o.pos if o else None,
                'C_prev': c_prev.pos if c_prev else None,
            })
        
        # Find H-bonds
        for i, donor in enumerate(backbone):
            if donor['N'] is None or donor['CA'] is None:
                continue
            
            # Calculate H position
            h_pos = calculate_h_position(
                donor['N'], donor['CA'], donor['C_prev']
            )
            if h_pos is None:
                continue
            
            # Look for acceptors (must be at least 3 residues away)
            for j, acceptor in enumerate(backbone):
                if abs(i - j) < 3:
                    continue
                    
                if acceptor['O'] is None or acceptor['C'] is None:
                    continue
                
                # Calculate H...O distance
                d_HO = h_pos.dist(acceptor['O'])
                
                if d_HO > d_max:
                    continue
                
                # Calculate angles
                # θ_AHD = O-H-N angle
                vec_HO = gemmi.Position(
                    acceptor['O'].x - h_pos.x,
                    acceptor['O'].y - h_pos.y,
                    acceptor['O'].z - h_pos.z
                )
                vec_HN = gemmi.Position(
                    donor['N'].x - h_pos.x,
                    donor['N'].y - h_pos.y,
                    donor['N'].z - h_pos.z
                )
                
                theta_OHN = _angle_between(vec_HO, vec_HN)
                
                # θ_BAH = C-O-H angle
                vec_OC = gemmi.Position(
                    acceptor['C'].x - acceptor['O'].x,
                    acceptor['C'].y - acceptor['O'].y,
                    acceptor['C'].z - acceptor['O'].z
                )
                vec_OH = gemmi.Position(
                    h_pos.x - acceptor['O'].x,
                    h_pos.y - acceptor['O'].y,
                    h_pos.z - acceptor['O'].z
                )
                
                theta_COH = _angle_between(vec_OC, vec_OH)
                
                # Filter by angle (O-H-N should be > 120°)
                if theta_OHN < 120:
                    continue
                
                hbond = HBond(
                    donor_res_idx=i,
                    acceptor_res_idx=j,
                    donor_res_name=donor['res_name'],
                    acceptor_res_name=acceptor['res_name'],
                    d_HA=d_HO,
                    theta_AHD=theta_OHN,
                    theta_BAH=theta_COH,
                )
                hbonds.append(hbond)
    
    return hbonds


def _angle_between(v1: gemmi.Position, v2: gemmi.Position) -> float:
    """Calculate angle between two vectors (in degrees)"""
    dot = v1.x*v2.x + v1.y*v2.y + v1.z*v2.z
    len1 = (v1.x**2 + v1.y**2 + v1.z**2)**0.5
    len2 = (v2.x**2 + v2.y**2 + v2.z**2)**0.5
    
    if len1 < 1e-6 or len2 < 1e-6:
        return 0.0
    
    cos_angle = dot / (len1 * len2)
    cos_angle = max(-1.0, min(1.0, cos_angle))  # Clamp for numerical stability
    
    return np.degrees(np.arccos(cos_angle))


def compute_total_hbond_energy(pdb_path: str) -> Tuple[float, Dict]:
    """
    Compute total backbone H-bond energy for a structure.
    
    Returns:
        (total_energy, details_dict)
    """
    hbonds = find_backbone_hbonds(pdb_path)
    
    if len(hbonds) == 0:
        return 0.0, {'n_hbonds': 0, 'hbonds': []}
    
    calculator = HBondEnergy()
    
    total_energy = 0.0
    for hb in hbonds:
        hb.energy = calculator.compute_hbond_energy(hb)
        total_energy += hb.energy
    
    return total_energy, {
        'n_hbonds': len(hbonds),
        'energy_per_hbond': total_energy / len(hbonds) if hbonds else 0,
        'hbonds': hbonds,
    }


# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    from pathlib import Path
    
    print("=" * 70)
    print("H-bond Energy Test")
    print("=" * 70)
    
    # Test on CASP14 structures
    CASP14_DIR = Path("/runs/users/gunwoong/work/DFIRE/dfire_40/data/casp14_full")
    
    test_pdbs = list((CASP14_DIR / "predictions" / "T1030-D2").glob("*.pdb"))[:5]
    
    for pdb in test_pdbs:
        energy, details = compute_total_hbond_energy(str(pdb))
        print(f"{pdb.name[:40]}: E={energy:.2f}, n_hbonds={details['n_hbonds']}")
    
    # Test on native
    native = CASP14_DIR / "targets" / "T1030-D2.pdb"
    if native.exists():
        energy, details = compute_total_hbond_energy(str(native))
        print(f"\nNative: E={energy:.2f}, n_hbonds={details['n_hbonds']}")
