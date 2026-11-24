"""
Dimensionality reduction and embedding visualization

Supports UMAP and t-SNE for visualizing high-dimensional structure relationships.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Union, Literal, Tuple
from pathlib import Path

EmbeddingMethod = Literal["umap", "tsne"]


def compute_embedding(
    distance_matrix: np.ndarray,
    method: EmbeddingMethod = "umap",
    n_components: int = 2,
    **kwargs,
) -> np.ndarray:
    """
    Compute 2D/3D embedding from distance matrix.

    Args:
        distance_matrix: Precomputed distance matrix (N x N)
        method: 'umap' or 'tsne'
        n_components: Number of dimensions (2 or 3)
        **kwargs: Additional arguments for the embedding method

    Returns:
        Embedding coordinates (N x n_components)

    Example:
        >>> coords = compute_embedding(dist_matrix, method="umap")
        >>> print(coords.shape)  # (N, 2)
    """
    if method == "umap":
        return _compute_umap(distance_matrix, n_components, **kwargs)
    elif method == "tsne":
        return _compute_tsne(distance_matrix, n_components, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'umap' or 'tsne'")


def _compute_umap(
    distance_matrix: np.ndarray,
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
    **kwargs,
) -> np.ndarray:
    """Compute UMAP embedding."""
    try:
        import umap
    except ImportError:
        raise ImportError("umap-learn is required. Install with: pip install umap-learn")

    n = len(distance_matrix)
    n_neighbors = min(n_neighbors, n - 1)

    reducer = umap.UMAP(
        n_components=n_components,
        metric='precomputed',
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
        **kwargs,
    )

    import warnings
    with warnings.catch_warnings():
        # a fixed random_state (reproducible layouts) makes UMAP warn that it runs single-threaded, and the
        # precomputed metric that inverse_transform is unavailable: both are intended
        warnings.filterwarnings("ignore", message=".*(precomputed metric|overridden to 1 by setting random_state).*")
        return reducer.fit_transform(distance_matrix)


def _compute_tsne(
    distance_matrix: np.ndarray,
    n_components: int = 2,
    perplexity: float = 30.0,
    random_state: int = 42,
    **kwargs,
) -> np.ndarray:
    """Compute t-SNE embedding."""
    from sklearn.manifold import TSNE

    n = len(distance_matrix)
    perplexity = min(perplexity, (n - 1) / 3)  # Perplexity must be < n/3

    tsne = TSNE(
        n_components=n_components,
        metric='precomputed',
        perplexity=perplexity,
        random_state=random_state,
        init='random',  # Required for precomputed metric
        **kwargs,
    )

    return tsne.fit_transform(distance_matrix)


def plot_umap(
    distance_matrix: np.ndarray,
    labels: Optional[np.ndarray] = None,
    colors: Optional[np.ndarray] = None,
    color_label: str = "Value",
    title: str = "UMAP Embedding",
    output_path: Optional[Union[str, Path]] = None,
    figsize: tuple = (10, 8),
    **umap_kwargs,
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Create UMAP visualization with optional coloring.

    Args:
        distance_matrix: Precomputed distance matrix
        labels: Cluster labels for discrete coloring
        colors: Continuous values for colormap
        color_label: Label for colorbar
        title: Plot title
        output_path: Path to save figure
        figsize: Figure size
        **umap_kwargs: Additional arguments for UMAP

    Returns:
        Tuple of (Figure, embedding_coordinates)
    """
    embedding = compute_embedding(distance_matrix, method="umap", **umap_kwargs)

    fig, ax = plt.subplots(figsize=figsize)

    if labels is not None:
        # Discrete coloring by cluster
        unique_labels = np.unique(labels)
        cmap = plt.cm.tab20(np.linspace(0, 1, max(len(unique_labels), 20)))

        for i, label in enumerate(unique_labels):
            mask = labels == label
            ax.scatter(
                embedding[mask, 0], embedding[mask, 1],
                c=[cmap[i % len(cmap)]], alpha=0.6, s=20,
                label=f'Cluster {label} (n={mask.sum()})'
            )
        ax.legend(loc='best', fontsize=8)

    elif colors is not None:
        # Continuous coloring
        scatter = ax.scatter(
            embedding[:, 0], embedding[:, 1],
            c=colors, cmap='RdYlGn', alpha=0.6, s=20
        )
        plt.colorbar(scatter, ax=ax, label=color_label)

    else:
        # No coloring
        ax.scatter(embedding[:, 0], embedding[:, 1], alpha=0.6, s=20)

    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')

    return fig, embedding


def plot_tsne(
    distance_matrix: np.ndarray,
    labels: Optional[np.ndarray] = None,
    colors: Optional[np.ndarray] = None,
    color_label: str = "Value",
    title: str = "t-SNE Embedding",
    output_path: Optional[Union[str, Path]] = None,
    figsize: tuple = (10, 8),
    **tsne_kwargs,
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Create t-SNE visualization with optional coloring.

    Args:
        distance_matrix: Precomputed distance matrix
        labels: Cluster labels for discrete coloring
        colors: Continuous values for colormap
        color_label: Label for colorbar
        title: Plot title
        output_path: Path to save figure
        figsize: Figure size
        **tsne_kwargs: Additional arguments for t-SNE

    Returns:
        Tuple of (Figure, embedding_coordinates)
    """
    embedding = compute_embedding(distance_matrix, method="tsne", **tsne_kwargs)

    fig, ax = plt.subplots(figsize=figsize)

    if labels is not None:
        unique_labels = np.unique(labels)
        cmap = plt.cm.tab20(np.linspace(0, 1, max(len(unique_labels), 20)))

        for i, label in enumerate(unique_labels):
            mask = labels == label
            ax.scatter(
                embedding[mask, 0], embedding[mask, 1],
                c=[cmap[i % len(cmap)]], alpha=0.6, s=20,
                label=f'Cluster {label} (n={mask.sum()})'
            )
        ax.legend(loc='best', fontsize=8)

    elif colors is not None:
        scatter = ax.scatter(
            embedding[:, 0], embedding[:, 1],
            c=colors, cmap='RdYlGn', alpha=0.6, s=20
        )
        plt.colorbar(scatter, ax=ax, label=color_label)

    else:
        ax.scatter(embedding[:, 0], embedding[:, 1], alpha=0.6, s=20)

    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')

    return fig, embedding
