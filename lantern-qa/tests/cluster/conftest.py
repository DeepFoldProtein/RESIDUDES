"""Shared fixtures for MassiveClustering tests."""

import numpy as np
import pytest


@pytest.fixture
def small_distance_matrix():
    """5x5 symmetric distance matrix with zero diagonal."""
    np.random.seed(42)
    n = 5
    d = np.random.rand(n, n)
    d = (d + d.T) / 2
    np.fill_diagonal(d, 0)
    return d


@pytest.fixture
def medium_distance_matrix():
    """50x50 distance matrix for clustering tests."""
    np.random.seed(42)
    n = 50
    d = np.random.rand(n, n)
    d = (d + d.T) / 2
    np.fill_diagonal(d, 0)
    return d


@pytest.fixture
def large_distance_matrix():
    """200x200 distance matrix for stress tests."""
    np.random.seed(42)
    n = 200
    d = np.random.rand(n, n)
    d = (d + d.T) / 2
    np.fill_diagonal(d, 0)
    return d


@pytest.fixture
def clustered_distance_matrix():
    """100x100 distance matrix with 3 clear clusters."""
    np.random.seed(42)
    n = 100
    d = np.ones((n, n)) * 0.8
    # Cluster 0: indices 0-33
    d[:34, :34] = np.random.rand(34, 34) * 0.2
    # Cluster 1: indices 34-66
    d[34:67, 34:67] = np.random.rand(33, 33) * 0.2
    # Cluster 2: indices 67-99
    d[67:, 67:] = np.random.rand(33, 33) * 0.2
    d = (d + d.T) / 2
    np.fill_diagonal(d, 0)
    return d


@pytest.fixture
def sample_labels():
    """Labels for 50 structures, 3 clusters."""
    np.random.seed(42)
    return np.random.randint(0, 3, 50)


@pytest.fixture
def sample_names():
    """Structure names for 50 structures."""
    return [f"struct_{i}" for i in range(50)]


@pytest.fixture
def tmp_output(tmp_path):
    """Temporary output directory."""
    return tmp_path
