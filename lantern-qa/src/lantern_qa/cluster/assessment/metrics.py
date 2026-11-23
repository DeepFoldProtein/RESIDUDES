"""
Quality metrics computation module

Computes confidence and quality metrics for structure predictions:
- pLDDT: predicted Local Distance Difference Test
- pTM: predicted TM-score
- piTM: predicted interface TM-score (multimer)
- pDockQ: predicted DockQ score (multimer)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, List, Union
from dataclasses import dataclass


@dataclass
class QualityMetrics:
    """Container for structure quality metrics."""
    structure_name: str
    plddt: Optional[float] = None
    ptm: Optional[float] = None
    pitm: Optional[float] = None  # Multimer only
    pdockq: Optional[float] = None  # Multimer only
    iptm: Optional[float] = None  # Interface pTM


def compute_quality_metrics(
    structure_path: Union[str, Path],
    scores_file: Optional[Union[str, Path]] = None,
) -> QualityMetrics:
    """
    Compute quality metrics for a single structure.

    Args:
        structure_path: Path to PDB file
        scores_file: Optional path to JSON scores file from prediction

    Returns:
        QualityMetrics object

    Example:
        >>> metrics = compute_quality_metrics("prediction.pdb")
        >>> print(metrics.plddt)
    """
    structure_path = Path(structure_path)
    name = structure_path.name

    plddt = None
    ptm = None
    pitm = None

    # Try to extract pLDDT from B-factor column
    if structure_path.exists():
        plddt = _extract_plddt_from_pdb(structure_path)

    # Try to load from scores file
    if scores_file and Path(scores_file).exists():
        scores = _load_scores_file(scores_file)
        ptm = scores.get('ptm')
        pitm = scores.get('iptm')

    return QualityMetrics(
        structure_name=name,
        plddt=plddt,
        ptm=ptm,
        pitm=pitm,
    )


def compute_batch_metrics(
    structure_paths: List[Union[str, Path]],
    scores_dir: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """
    Compute quality metrics for multiple structures.

    Args:
        structure_paths: List of PDB file paths
        scores_dir: Optional directory containing score files

    Returns:
        DataFrame with metrics for all structures
    """
    results = []

    for path in structure_paths:
        path = Path(path)
        scores_file = None

        if scores_dir:
            # Try to find matching scores file
            scores_dir = Path(scores_dir)
            possible_names = [
                path.stem + "_scores.json",
                path.stem + ".json",
            ]
            for name in possible_names:
                if (scores_dir / name).exists():
                    scores_file = scores_dir / name
                    break

        metrics = compute_quality_metrics(path, scores_file)
        results.append({
            'structure': metrics.structure_name,
            'plddt': metrics.plddt,
            'ptm': metrics.ptm,
            'pitm': metrics.pitm,
            'pdockq': metrics.pdockq,
        })

    return pd.DataFrame(results)


def _extract_plddt_from_pdb(pdb_path: Path) -> Optional[float]:
    """Extract average pLDDT from B-factor column of PDB file."""
    b_factors = []

    try:
        with open(pdb_path) as f:
            for line in f:
                if line.startswith(('ATOM', 'HETATM')):
                    try:
                        b_factor = float(line[60:66].strip())
                        b_factors.append(b_factor)
                    except (ValueError, IndexError):
                        continue

        if b_factors:
            return np.mean(b_factors)
    except Exception:
        pass

    return None


def _load_scores_file(scores_path: Path) -> Dict:
    """Load scores from JSON file."""
    import json

    try:
        with open(scores_path) as f:
            return json.load(f)
    except Exception:
        return {}


def compute_pdockq(
    pitm: float,
    n_contacts: int,
) -> float:
    """
    Compute predicted DockQ score from piTM and interface contacts.

    Based on the pDockQ formula from Evans et al.

    Args:
        pitm: predicted interface TM-score
        n_contacts: Number of interface contacts

    Returns:
        Predicted DockQ score
    """
    # Simplified pDockQ approximation
    # Full formula requires more detailed interface analysis
    if pitm is None or n_contacts < 1:
        return 0.0

    # Sigmoid-based approximation
    L = 0.724
    x0 = 152.611
    k = 0.052
    b = 0.018

    pdockq = L / (1 + np.exp(-k * (n_contacts - x0))) + b
    pdockq = pdockq * pitm  # Scale by piTM

    return min(max(pdockq, 0.0), 1.0)


def classify_dockq_quality(dockq: float) -> str:
    """
    Classify DockQ score into quality categories.

    Categories:
    - Incorrect: < 0.23
    - Acceptable: 0.23 - 0.49
    - Medium: 0.49 - 0.80
    - High: >= 0.80

    Args:
        dockq: DockQ score

    Returns:
        Quality category string
    """
    if dockq >= 0.80:
        return "High"
    elif dockq >= 0.49:
        return "Medium"
    elif dockq >= 0.23:
        return "Acceptable"
    else:
        return "Incorrect"
