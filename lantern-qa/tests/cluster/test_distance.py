"""Tests for distance module."""

import numpy as np
import pytest

from lantern_qa.cluster.core.distance import similarity_to_distance


class TestSimilarityToDistance:
    def test_tm_conversion(self):
        """TM similarity (1=identical) -> distance (0=identical)."""
        sim = np.array([[1.0, 0.8, 0.3], [0.8, 1.0, 0.5], [0.3, 0.5, 1.0]])
        dist = similarity_to_distance(sim, "tm")
        assert dist[0, 0] == 0.0
        assert dist[0, 1] == pytest.approx(0.2)
        assert dist[0, 2] == pytest.approx(0.7)
        assert np.allclose(dist, dist.T)

    def test_drmsd_keeps_distance(self):
        """dRMSD is already a distance — diagonal should stay 0, off-diag > 0."""
        d = np.array([[0.0, 5.0, 10.0], [5.0, 0.0, 3.0], [10.0, 3.0, 0.0]])
        result = similarity_to_distance(d, "drmsd")
        assert result[0, 0] == 0.0
        assert result[0, 1] > 0
        assert result[0, 2] > result[0, 1]  # ordering preserved

    def test_roundtrip(self, small_distance_matrix):
        """Convert distance->similarity->distance should roundtrip."""
        dist = small_distance_matrix
        sim = 1.0 - dist
        recovered = similarity_to_distance(sim, "tm")
        assert np.allclose(dist, recovered, atol=1e-10)
