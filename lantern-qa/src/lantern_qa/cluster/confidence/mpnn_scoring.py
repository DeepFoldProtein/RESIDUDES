"""
Backwards-compatible re-export stub.

ProteinMPNN scoring is now a top-level EMA scorer (it's a general QA tool,
not MassiveClustering-specific). This stub keeps existing callers working:

    # legacy (still works)
    from lantern_qa.cluster.confidence.mpnn_scoring import MPNNScorer

    # preferred (new code)
    from lantern_qa.ema.scorers.proteinmpnn_score import MPNNScorer

The actual implementation lives in
``EMA/scorers/proteinmpnn_score.py``.
"""

from lantern_qa.ema.scorers.proteinmpnn_score import (  # noqa: F401
    MPNNScorer,
    MPNNResult,
    ClusterConfidence,
    save_mpnn_results,
)

__all__ = ["MPNNScorer", "MPNNResult", "ClusterConfidence", "save_mpnn_results"]
