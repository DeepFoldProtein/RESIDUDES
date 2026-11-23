"""
Native Structure Comparison and Confidence Analysis

Integrates assessment scores (TM-to-native, lDDT, DockQ) and
confidence scores (pLDDT, pTM, iPTM) with clustering results.
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Union
from pathlib import Path


def load_assessment_scores(
    scores_path: Union[str, Path],
) -> pd.DataFrame:
    """
    Load assessment scores CSV.

    Expected columns (monomers): model_name, tm_score, rmsd, lddt, oligo_gdtts, oligo_gdtha, parameters, file
    Expected columns (multimers): + dockq_ave, dockq_wave, ips, ics, qs_best, qs_global, ilddt
    """
    df = pd.read_csv(scores_path)
    # Normalize filename column
    if 'file' in df.columns:
        df['structure_name'] = df['file'].apply(lambda x: str(x).strip())
    return df


def load_ranking_scores(
    ranking_path: Union[str, Path],
) -> pd.DataFrame:
    """
    Load ranking scores CSV.

    Expected columns (monomers): global_rank, plddts, parameters, file, model_name, ptm
    Expected columns (multimers): + iptm+ptm, iptm
    """
    df = pd.read_csv(ranking_path)
    if 'file' in df.columns:
        df['structure_name'] = df['file'].apply(lambda x: str(x).strip())
    return df


def analyze_native_scores(
    labels: np.ndarray,
    structure_names: List[str],
    scores_df: Optional[pd.DataFrame] = None,
    ranking_df: Optional[pd.DataFrame] = None,
    medoid_indices: Optional[List[int]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    target_name: str = "Target",
) -> pd.DataFrame:
    """
    Analyze assessment and confidence scores per cluster.

    For each cluster computes:
    - Mean/max TM-to-native, lDDT, DockQ (if available)
    - Mean/max pLDDT, pTM, iPTM (if available)
    - Medoid scores
    - Best structure per metric

    Args:
        labels: Cluster labels
        structure_names: Structure filenames
        scores_df: Assessment scores (TM, lDDT, DockQ etc.)
        ranking_df: Ranking/confidence scores (pLDDT, pTM, iPTM)
        medoid_indices: Medoid indices per cluster
        output_dir: Save results if provided
        target_name: Target name

    Returns:
        DataFrame with per-cluster score summary
    """
    n = len(labels)
    unique_clusters = sorted(np.unique(labels))

    # Build name-to-index mapping for quick lookup
    name_to_idx = {name: i for i, name in enumerate(structure_names)}

    # Merge scores with structure index
    score_map = {}  # idx -> {metric: value}
    if scores_df is not None:
        for _, row in scores_df.iterrows():
            name = row.get('structure_name', row.get('file', ''))
            name = str(name).strip()
            idx = name_to_idx.get(name)
            if idx is not None:
                score_map.setdefault(idx, {})
                for col in ['tm_score', 'rmsd', 'lddt', 'dockq_ave', 'dockq_wave',
                            'ips', 'ics', 'qs_global', 'ilddt', 'oligo_gdtts']:
                    if col in row and pd.notna(row[col]):
                        try:
                            score_map[idx][col] = float(row[col])
                        except (ValueError, TypeError):
                            pass

    rank_map = {}  # idx -> {metric: value}
    if ranking_df is not None:
        for _, row in ranking_df.iterrows():
            name = row.get('structure_name', row.get('file', ''))
            name = str(name).strip()
            idx = name_to_idx.get(name)
            if idx is not None:
                rank_map.setdefault(idx, {})
                # Confidence columns we collect from ranking.csv. Includes
                # legacy aliases (plddts) and CASP17 unified-output names
                # (plddt, ranking_score, gpde, fraction_disordered).
                for col in ['plddts', 'plddt', 'ptm', 'iptm', 'iptm+ptm',
                            'ranking_score', 'gpde', 'fraction_disordered',
                            'global_rank']:
                    if col in row and pd.notna(row[col]):
                        try:
                            rank_map[idx][col] = float(row[col])
                        except (ValueError, TypeError):
                            pass

    # Determine available metrics
    all_score_metrics = set()
    for v in score_map.values():
        all_score_metrics.update(v.keys())
    all_rank_metrics = set()
    for v in rank_map.values():
        all_rank_metrics.update(v.keys())

    # Per-cluster statistics
    rows = []
    for cid in unique_clusters:
        mask = labels == cid
        indices = np.where(mask)[0]

        row = {'cluster': cid, 'size': len(indices)}

        # Assessment scores
        for metric in sorted(all_score_metrics):
            vals = [score_map[i][metric] for i in indices
                    if i in score_map and metric in score_map[i]]
            if vals:
                row[f'{metric}_mean'] = float(np.mean(vals))
                row[f'{metric}_min'] = float(np.min(vals))
                row[f'{metric}_max'] = float(np.max(vals))
                row[f'{metric}_std'] = float(np.std(vals))
                # Best structure
                best_idx = indices[np.argmax([
                    score_map.get(i, {}).get(metric, -999) for i in indices
                ])]
                row[f'{metric}_best'] = structure_names[best_idx]

        # Confidence scores
        for metric in sorted(all_rank_metrics):
            vals = [rank_map[i][metric] for i in indices
                    if i in rank_map and metric in rank_map[i]]
            if vals:
                row[f'{metric}_mean'] = float(np.mean(vals))
                row[f'{metric}_min'] = float(np.min(vals))
                row[f'{metric}_max'] = float(np.max(vals))

        # Medoid scores
        if medoid_indices and cid < len(medoid_indices):
            midx = medoid_indices[cid]
            if midx in score_map:
                for metric, val in score_map[midx].items():
                    row[f'medoid_{metric}'] = val
            if midx in rank_map:
                for metric, val in rank_map[midx].items():
                    row[f'medoid_{metric}'] = val

        rows.append(row)

    df = pd.DataFrame(rows)

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        csv_path = output_dir / f"{target_name}_native_scores.csv"
        df.to_csv(csv_path, index=False, float_format='%.4f')

        # Plot if we have TM scores
        try:
            _plot_native_scores(df, target_name, output_dir)
        except Exception:
            pass

    return df


def _plot_native_scores(
    df: pd.DataFrame,
    target_name: str,
    output_dir: Path,
):
    """Plot per-cluster native scores."""
    import matplotlib.pyplot as plt

    # Determine what metrics are available
    score_cols = [c for c in df.columns if c.endswith('_mean') and
                  not c.startswith('medoid_')]
    if not score_cols:
        return

    # Key metrics to plot
    priority = ['tm_score_mean', 'lddt_mean', 'dockq_ave_mean',
                'plddts_mean', 'ptm_mean', 'iptm_mean']
    plot_cols = [c for c in priority if c in df.columns]
    if not plot_cols:
        plot_cols = score_cols[:4]

    n_plots = len(plot_cols)
    if n_plots == 0:
        return

    fig, axes = plt.subplots(n_plots, 1, figsize=(12, 3 * n_plots))
    if n_plots == 1:
        axes = [axes]

    clusters = df['cluster'].values

    for ax, col in zip(axes, plot_cols):
        vals = df[col].values
        colors = plt.cm.viridis(vals / (vals.max() + 1e-8))
        ax.bar(range(len(clusters)), vals, color=colors, width=0.8)
        ax.set_ylabel(col.replace('_mean', '').replace('_', ' ').title())
        ax.set_xticks(range(len(clusters)))
        ax.set_xticklabels([f'C{c}' for c in clusters], rotation=45, fontsize=7)

        # Mark max
        best = np.argmax(vals)
        ax.bar(best, vals[best], color='red', width=0.8, alpha=0.7)

    axes[0].set_title(f'{target_name} — Scores per Cluster')
    axes[-1].set_xlabel('Cluster')

    plt.tight_layout()
    fig.savefig(output_dir / f"{target_name}_native_scores.png",
                dpi=150, bbox_inches='tight')
    plt.close(fig)
