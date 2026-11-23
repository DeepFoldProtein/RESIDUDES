"""
Confidence scoring module for MassiveFoldClustering.

Provides backbone quality assessment via ProteinMPNN sequence recovery.
"""

from .mpnn_scoring import MPNNScorer
from .analysis import compare_recovery_with_rmsf, generate_confidence_report

__all__ = [
    "MPNNScorer",
    "compare_recovery_with_rmsf",
    "generate_confidence_report",
]
