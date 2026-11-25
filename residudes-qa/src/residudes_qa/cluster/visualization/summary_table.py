"""
Cluster Summary Table Module

Generates summary tables showing cluster statistics:
- Cluster ID, Size, Percentage
- Intra-cluster similarity per metric (TM, dRMSD, iTM, etc.)
- Native/Best score
- Medoid structure name

Outputs both PNG table image and CSV file.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Union


# Metric display configuration
_METRIC_DISPLAY = {
    'tm':    {'label': 'Intra-TM',    'fmt': '.3f', 'is_distance': False},
    'itm':   {'label': 'Intra-iTM',   'fmt': '.3f', 'is_distance': False},
    'drmsd': {'label': 'Intra-dRMSD', 'fmt': '.2f', 'is_distance': True},
}


def _compute_intra_cluster_mean(matrix: np.ndarray, cluster_indices: np.ndarray) -> float:
    """Vectorized mean of upper-triangle pairwise values within a cluster."""
    if len(cluster_indices) <= 1:
        return 0.0 if _is_likely_distance(matrix) else 1.0
    sub = matrix[np.ix_(cluster_indices, cluster_indices)]
    triu = np.triu_indices(len(cluster_indices), k=1)
    return float(np.mean(sub[triu]))


def _is_likely_distance(matrix: np.ndarray) -> bool:
    """Heuristic: if diagonal is ~0, it's a distance matrix; if ~1, similarity."""
    diag_mean = np.mean(np.diag(matrix[:min(10, len(matrix))]))
    return diag_mean < 0.5


def create_cluster_summary_table(
    labels: np.ndarray,
    similarity_matrices: Optional[Dict[str, np.ndarray]] = None,
    structure_names: Optional[List[str]] = None,
    medoid_indices: Optional[List[int]] = None,
    native_idx: Optional[int] = None,
    scores_df: Optional[pd.DataFrame] = None,
    score_column: str = 'tm_score',
    target_name: str = "Target",
    output_dir: Optional[Union[str, Path]] = None,
    is_native: bool = True,
    clustering_metric: str = "tm",
    # Backward compatibility: single matrix
    similarity_matrix: Optional[np.ndarray] = None,
) -> Tuple[Optional[plt.Figure], pd.DataFrame]:
    """
    Create cluster summary table as PNG and CSV with multi-metric support.

    Shows intra-cluster statistics for each available metric (TM, dRMSD, iTM, etc.).

    Args:
        labels: Cluster labels (N,)
        similarity_matrices: Dict mapping metric_name -> raw matrix (N x N).
            TM/iTM matrices are similarity (0-1, diagonal=1).
            dRMSD matrices are distance (Angstroms, diagonal=0).
        structure_names: List of structure names
        medoid_indices: Indices of cluster medoids
        native_idx: Index of native/best structure
        scores_df: DataFrame with assessment scores
        score_column: Column name for native comparison score
        target_name: Name of target
        output_dir: Directory to save outputs
        is_native: True if native_idx is real native, False if best model fallback
        clustering_metric: Which metric was used for clustering (shown in header)
        similarity_matrix: DEPRECATED single matrix (backward compat)

    Returns:
        Tuple of (Figure, DataFrame)
    """
    # Backward compatibility
    if similarity_matrices is None and similarity_matrix is not None:
        similarity_matrices = {'tm': similarity_matrix}
    if similarity_matrices is None:
        similarity_matrices = {}

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_total = len(labels)

    if structure_names is None:
        structure_names = [f'struct_{i}' for i in range(n_total)]
    if medoid_indices is None:
        medoid_indices = []

    # Determine metric order for display
    metric_order = []
    for m in ['tm', 'itm', 'drmsd']:
        if m in similarity_matrices:
            metric_order.append(m)
    # Any extra metrics not in the predefined list
    for m in similarity_matrices:
        if m not in metric_order:
            metric_order.append(m)

    # Build score array if available
    score_array = None
    if scores_df is not None and score_column in scores_df.columns:
        score_array = _build_score_array(scores_df, structure_names, score_column)

    # Calculate cluster statistics
    cluster_stats = []

    for cluster_id in range(n_clusters):
        mask = labels == cluster_id
        cluster_indices = np.where(mask)[0]
        size = len(cluster_indices)

        if size == 0:
            continue

        # Medoid info
        medoid_idx = medoid_indices[cluster_id] if cluster_id < len(medoid_indices) else -1
        if 0 <= medoid_idx < len(structure_names):
            medoid_name = structure_names[medoid_idx]
            medoid_display = medoid_name[:25] + "..." if len(medoid_name) > 28 else medoid_name
        else:
            medoid_name = "N/A"
            medoid_display = "N/A"

        # Native/Best in cluster?
        has_native = native_idx in cluster_indices if native_idx is not None else False

        # Marker
        marker = ""
        if has_native:
            marker = " (Best)" if not is_native else " (Native)"

        row = {
            'Cluster': f"C{cluster_id}{marker}",
            'Size': size,
            '%': f"{size/n_total*100:.1f}%",
            'cluster_id': cluster_id,
            'has_native': has_native,
            'Medoid': medoid_display,
            'Medoid_Full': medoid_name,
        }

        # Compute intra-cluster stats for each metric
        for m in metric_order:
            mat = similarity_matrices[m]
            val = _compute_intra_cluster_mean(mat, cluster_indices)
            cfg = _METRIC_DISPLAY.get(m, {'label': f'Intra-{m.upper()}', 'fmt': '.3f'})
            row[cfg['label']] = f"{val:{cfg['fmt']}}"

        # Native/Best score from assessment
        if score_array is not None:
            cluster_scores = score_array[cluster_indices]
            mean_native_score = np.nanmean(cluster_scores)
            row['Native-Score'] = f"{mean_native_score:.3f}" if not np.isnan(mean_native_score) else "N/A"
        else:
            row['Native-Score'] = "N/A"

        cluster_stats.append(row)

    if not cluster_stats:
        return None, pd.DataFrame()

    # Build column list for display
    intra_cols = [_METRIC_DISPLAY.get(m, {'label': f'Intra-{m.upper()}'})['label'] for m in metric_order]
    display_columns = ['Cluster', 'Size', '%'] + intra_cols + ['Native-Score', 'Medoid']
    csv_columns = ['Cluster', 'Size', '%'] + intra_cols + ['Native-Score', 'Medoid_Full']

    # Create figure with table
    n_cols = len(display_columns)
    fig_width = max(12, 6 + n_cols * 1.5)
    fig_height = max(4, len(cluster_stats) * 0.4 + 2)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis('off')

    table_data = [[s.get(c, 'N/A') for c in display_columns] for s in cluster_stats]

    # Column widths: adapt to number of metrics
    base_widths = [0.10, 0.06, 0.06]
    metric_width = 0.10
    tail_widths = [0.12, 0.30]
    col_widths = base_widths + [metric_width] * len(intra_cols) + tail_widths
    total = sum(col_widths)
    col_widths = [w / total for w in col_widths]

    table = ax.table(
        cellText=table_data,
        colLabels=display_columns,
        cellLoc='center',
        loc='center',
        colWidths=col_widths,
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.8)

    # Style header
    for i in range(n_cols):
        cell = table[(0, i)]
        cell.set_facecolor('#2c3e50')
        cell.set_text_props(weight='bold', color='white')

    # Style data rows
    for i, stats in enumerate(cluster_stats):
        for j in range(n_cols):
            cell = table[(i + 1, j)]
            if stats['has_native']:
                cell.set_facecolor('#ffe6e6' if is_native else '#e6f2ff')
            else:
                cell.set_facecolor('#f8f9fa' if i % 2 == 0 else 'white')

    # Title
    ref_type = "Native Structure" if is_native else "Best Scoring Model"
    metrics_str = ", ".join(m.upper() for m in metric_order)
    plt.title(
        f'{target_name} - Cluster Summary\n'
        f'{n_total} structures | {n_clusters} clusters | '
        f'Clustered by: {clustering_metric.upper()} | Metrics: {metrics_str}',
        fontsize=12, fontweight='bold', pad=20,
    )

    plt.tight_layout()

    # Save outputs
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save PNG
        png_file = output_dir / f"{target_name}_cluster_summary.png"
        fig.savefig(png_file, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"  Saved: {png_file}")

        # Save CSV
        csv_df = pd.DataFrame([{k: s.get(k, 'N/A') for k in csv_columns} for s in cluster_stats])
        # Rename Medoid_Full -> Medoid for CSV
        csv_df.columns = ['Cluster', 'Size', '%'] + intra_cols + ['Native-Score', 'Medoid']

        csv_file = output_dir / f"{target_name}_cluster_summary.csv"
        with open(csv_file, 'w') as f:
            f.write(f"# Target: {target_name}\n")
            f.write(f"# Reference Type: {ref_type}\n")
            f.write(f"# Clustering Metric: {clustering_metric}\n")
            f.write(f"# Metrics: {metrics_str}\n")
            f.write(f"# Total Structures: {n_total}\n")
            f.write(f"# Clusters: {n_clusters}\n")
            csv_df.to_csv(f, index=False)
        print(f"  Saved: {csv_file}")

    df = pd.DataFrame(cluster_stats)
    return fig, df


def _build_score_array(scores_df, structure_names, score_column):
    """Build score array matching structure order."""
    if 'file' in scores_df.columns:
        key_col = 'file'
    elif 'model_name' in scores_df.columns:
        key_col = 'model_name'
    elif 'structure' in scores_df.columns:
        key_col = 'structure'
    else:
        return None

    scores_df = scores_df.copy()
    scores_df['_key'] = scores_df[key_col].str.replace('.pdb', '', regex=False)
    score_map = dict(zip(scores_df['_key'], scores_df[score_column]))

    score_array = np.array([
        score_map.get(s.replace('.pdb', ''), np.nan)
        for s in structure_names
    ])

    return score_array


def create_medoid_summary_table(
    medoid_indices: List[int],
    structure_names: List[str],
    labels: np.ndarray,
    scores_df: Optional[pd.DataFrame] = None,
    score_columns: Optional[List[str]] = None,
    target_name: str = "Target",
    output_path: Optional[Union[str, Path]] = None,
) -> Tuple[Optional[plt.Figure], pd.DataFrame]:
    """
    Create summary table for cluster medoids only.

    Args:
        medoid_indices: List of medoid indices
        structure_names: List of all structure names
        labels: Cluster labels
        scores_df: DataFrame with assessment scores
        score_columns: Score columns to include
        target_name: Target name
        output_path: Output path for PNG

    Returns:
        Tuple of (Figure, DataFrame)
    """
    if score_columns is None:
        score_columns = ['plddt', 'tm_score', 'native_tm', 'native_dockq', 'native_lddt']

    medoid_data = []

    for cluster_id, medoid_idx in enumerate(medoid_indices):
        if medoid_idx < 0 or medoid_idx >= len(structure_names):
            continue

        medoid_name = structure_names[medoid_idx]
        cluster_size = (labels == cluster_id).sum()

        row = {
            'Cluster': f'C{cluster_id}',
            'Size': cluster_size,
            'Medoid': medoid_name[:30] + "..." if len(medoid_name) > 33 else medoid_name,
        }

        if scores_df is not None:
            score_map = _get_score_for_structure(scores_df, medoid_name)
            for col in score_columns:
                if col in score_map:
                    val = score_map[col]
                    row[col] = f"{val:.3f}" if isinstance(val, float) and not np.isnan(val) else "N/A"

        medoid_data.append(row)

    if not medoid_data:
        return None, pd.DataFrame()

    df = pd.DataFrame(medoid_data)

    fig, ax = plt.subplots(figsize=(14, max(3, len(medoid_data) * 0.5)))
    ax.axis('off')

    columns = ['Cluster', 'Size', 'Medoid'] + [c for c in score_columns if c in df.columns]
    table_data = df[columns].values.tolist()

    table = ax.table(
        cellText=table_data,
        colLabels=columns,
        cellLoc='center',
        loc='center',
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.6)

    for i in range(len(columns)):
        table[(0, i)].set_facecolor('#34495e')
        table[(0, i)].set_text_props(weight='bold', color='white')

    plt.title(f'{target_name} - Medoid Summary', fontsize=12, fontweight='bold')
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')

    return fig, df


def _get_score_for_structure(scores_df, structure_name):
    """Get scores for a specific structure."""
    clean_name = structure_name.replace('.pdb', '')

    for key_col in ['file', 'model_name', 'structure']:
        if key_col in scores_df.columns:
            df_copy = scores_df.copy()
            df_copy['_key'] = df_copy[key_col].str.replace('.pdb', '', regex=False)

            match = df_copy[df_copy['_key'] == clean_name]
            if len(match) > 0:
                return match.iloc[0].to_dict()

    return {}


def print_cluster_summary(
    labels: np.ndarray,
    similarity_matrices: Optional[Dict[str, np.ndarray]] = None,
    medoid_indices: Optional[List[int]] = None,
    structure_names: Optional[List[str]] = None,
    native_idx: Optional[int] = None,
    # Backward compat
    similarity_matrix: Optional[np.ndarray] = None,
):
    """
    Print cluster summary to console.

    Args:
        labels: Cluster labels
        similarity_matrices: Dict of metric_name -> matrix
        medoid_indices: Medoid indices
        structure_names: Structure names
        native_idx: Native structure index
        similarity_matrix: DEPRECATED single matrix (backward compat)
    """
    if similarity_matrices is None and similarity_matrix is not None:
        similarity_matrices = {'tm': similarity_matrix}
    if similarity_matrices is None:
        similarity_matrices = {}

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_total = len(labels)

    if structure_names is None:
        structure_names = [f'struct_{i}' for i in range(n_total)]
    if medoid_indices is None:
        medoid_indices = []

    metric_order = [m for m in ['tm', 'itm', 'drmsd'] if m in similarity_matrices]
    metric_order += [m for m in similarity_matrices if m not in metric_order]

    print("\n" + "=" * 70)
    print("CLUSTER SUMMARY")
    print("=" * 70)
    print(f"Total structures: {n_total}")
    print(f"Number of clusters: {n_clusters}")
    print("-" * 70)

    # Header
    header = f"{'Cluster':<10} {'Size':<8} {'%':<8}"
    for m in metric_order:
        cfg = _METRIC_DISPLAY.get(m, {'label': f'Intra-{m.upper()}'})
        header += f" {cfg['label']:<12}"
    header += f" {'Medoid':<30}"
    print(header)
    print("-" * 70)

    for cluster_id in range(n_clusters):
        mask = labels == cluster_id
        cluster_indices = np.where(mask)[0]
        size = len(cluster_indices)

        medoid_idx = medoid_indices[cluster_id] if cluster_id < len(medoid_indices) else -1
        medoid_name = structure_names[medoid_idx][:28] if 0 <= medoid_idx < len(structure_names) else "N/A"

        marker = ""
        if native_idx is not None and native_idx in cluster_indices:
            marker = " *"

        pct = size / n_total * 100
        line = f"C{cluster_id}{marker:<8} {size:<8} {pct:<7.1f}%"

        for m in metric_order:
            val = _compute_intra_cluster_mean(similarity_matrices[m], cluster_indices)
            cfg = _METRIC_DISPLAY.get(m, {'fmt': '.3f'})
            line += f" {val:<12{cfg['fmt']}}"

        line += f" {medoid_name}"
        print(line)

    print("=" * 70)
    if native_idx is not None:
        print("* = Contains Native/Best structure")
    print()
