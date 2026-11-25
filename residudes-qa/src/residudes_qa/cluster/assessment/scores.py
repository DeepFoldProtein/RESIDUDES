"""
Score loading and merging utilities

Load pre-computed assessment scores and merge with clustering results.
"""

import pandas as pd
from pathlib import Path
from typing import Optional, List, Union


def load_assessment_scores(
    scores_file: Union[str, Path],
    structure_column: str = "file",
) -> pd.DataFrame:
    """
    Load pre-computed assessment scores from CSV file.

    Expected columns: structure/file, tm_score, rmsd, dockq_ave, lddt, etc.

    Args:
        scores_file: Path to CSV file with assessment scores
        structure_column: Column name containing structure names

    Returns:
        DataFrame with standardized column names
    """
    df = pd.read_csv(scores_file)

    # Standardize column names
    column_mapping = {
        'tm_score': 'native_tm',
        'tm': 'native_tm',
        'rmsd': 'native_rmsd',
        'dockq_ave': 'native_dockq',
        'dockq': 'native_dockq',
        'lddt': 'native_lddt',
        'ilddt': 'native_ilddt',
        'ics': 'native_ics',
        'ips': 'native_ips',
    }

    for old_name, new_name in column_mapping.items():
        if old_name in df.columns and new_name not in df.columns:
            df[new_name] = df[old_name]

    # Rename structure column
    if structure_column in df.columns and structure_column != 'structure':
        df['structure'] = df[structure_column]

    return df


def merge_with_clustering(
    clustering_df: pd.DataFrame,
    assessment_df: pd.DataFrame,
    on: str = "structure",
) -> pd.DataFrame:
    """
    Merge clustering results with assessment scores.

    Args:
        clustering_df: DataFrame with clustering results
        assessment_df: DataFrame with assessment scores
        on: Column to merge on

    Returns:
        Merged DataFrame
    """
    # Ensure consistent structure naming (remove path, keep filename stem)
    clustering_df = clustering_df.copy()
    assessment_df = assessment_df.copy()

    if on in clustering_df.columns:
        clustering_df['_merge_key'] = clustering_df[on].apply(
            lambda x: Path(x).stem if isinstance(x, str) else x
        )
    else:
        clustering_df['_merge_key'] = clustering_df.index

    if on in assessment_df.columns:
        assessment_df['_merge_key'] = assessment_df[on].apply(
            lambda x: Path(x).stem if isinstance(x, str) else x
        )
    else:
        assessment_df['_merge_key'] = assessment_df.index

    merged = clustering_df.merge(
        assessment_df.drop(columns=[on], errors='ignore'),
        on='_merge_key',
        how='left'
    )

    merged = merged.drop(columns=['_merge_key'])

    return merged


def compute_cluster_stats(
    df: pd.DataFrame,
    cluster_column: str = "cluster",
    score_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Compute statistics per cluster for given score columns.

    Args:
        df: DataFrame with clustering and score data
        cluster_column: Column containing cluster labels
        score_columns: Columns to aggregate (auto-detect if None)

    Returns:
        DataFrame with per-cluster statistics
    """
    if score_columns is None:
        # Auto-detect score columns
        score_columns = [
            c for c in df.columns
            if c.startswith('native_') or c in ['plddt', 'ptm', 'pitm', 'pdockq']
        ]

    available_cols = [c for c in score_columns if c in df.columns]

    if not available_cols:
        return pd.DataFrame()

    agg_dict = {}
    for col in available_cols:
        agg_dict[col] = ['mean', 'std', 'min', 'max', 'count']

    stats = df.groupby(cluster_column).agg(agg_dict)
    stats.columns = ['_'.join(col).strip() for col in stats.columns]

    # Add cluster size
    stats['size'] = df.groupby(cluster_column).size()

    return stats.reset_index()


def classify_dockq_quality(dockq: float) -> str:
    """
    Classify DockQ score into quality categories.

    Categories (CAPRI criteria):
    - Incorrect: < 0.23
    - Acceptable: 0.23 - 0.49
    - Medium: 0.49 - 0.80
    - High: >= 0.80

    Args:
        dockq: DockQ score

    Returns:
        Quality category string
    """
    if pd.isna(dockq):
        return "Unknown"
    if dockq >= 0.80:
        return "High"
    elif dockq >= 0.49:
        return "Medium"
    elif dockq >= 0.23:
        return "Acceptable"
    else:
        return "Incorrect"


def add_dockq_categories(df: pd.DataFrame, dockq_column: str = "native_dockq") -> pd.DataFrame:
    """
    Add DockQ quality category column to DataFrame.

    Args:
        df: DataFrame with DockQ scores
        dockq_column: Column containing DockQ scores

    Returns:
        DataFrame with added 'dockq_category' column
    """
    df = df.copy()
    if dockq_column in df.columns:
        df['dockq_category'] = df[dockq_column].apply(classify_dockq_quality)
    return df
