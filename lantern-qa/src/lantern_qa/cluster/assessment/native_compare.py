"""
Native structure comparison module

Compares predictions with experimental/native structures using:
- TM-score
- iTM (interface TM-score)
- DockQ
- RMSD
- lDDT
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, List, Union
from dataclasses import dataclass


@dataclass
class NativeComparison:
    """Container for native structure comparison results."""
    structure_name: str
    native_tm: Optional[float] = None
    native_itm: Optional[float] = None
    native_dockq: Optional[float] = None
    native_rmsd: Optional[float] = None
    native_lddt: Optional[float] = None
    native_ilddt: Optional[float] = None  # Interface lDDT
    native_ics: Optional[float] = None  # Interface Contact Similarity
    native_ips: Optional[float] = None  # Interface Patch Similarity


def compare_with_native(
    prediction_path: Union[str, Path],
    native_path: Union[str, Path],
    metrics: List[str] = ["tm", "rmsd"],
) -> NativeComparison:
    """
    Compare a prediction with native structure.

    Args:
        prediction_path: Path to predicted PDB file
        native_path: Path to native/experimental PDB file
        metrics: List of metrics to compute

    Returns:
        NativeComparison object

    Example:
        >>> result = compare_with_native(
        ...     "prediction.pdb",
        ...     "native.pdb",
        ...     metrics=["tm", "dockq"]
        ... )
    """
    # TODO: Integrate with US-align, DockQ tools
    raise NotImplementedError(
        "Native comparison requires external tools (US-align, DockQ). "
        "Use load_assessment_scores() for pre-computed CASP scores."
    )


def load_assessment_scores(
    scores_file: Union[str, Path],
    structure_column: str = "file",
) -> pd.DataFrame:
    """
    Load pre-computed assessment scores from CSV file.

    Expected columns: structure/file, tm_score, rmsd, dockq_ave, etc.

    Args:
        scores_file: Path to CSV file with assessment scores
        structure_column: Column name containing structure names

    Returns:
        DataFrame with assessment scores
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
    Merge clustering results with native assessment scores.

    Args:
        clustering_df: DataFrame with clustering results
        assessment_df: DataFrame with assessment scores
        on: Column to merge on

    Returns:
        Merged DataFrame
    """
    # Ensure consistent structure naming
    if on in clustering_df.columns:
        clustering_df = clustering_df.copy()
        clustering_df['_merge_key'] = clustering_df[on].apply(
            lambda x: Path(x).stem if isinstance(x, str) else x
        )
    else:
        clustering_df['_merge_key'] = clustering_df.index

    if on in assessment_df.columns:
        assessment_df = assessment_df.copy()
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


def compute_cluster_native_stats(
    df: pd.DataFrame,
    cluster_column: str = "cluster",
    native_columns: List[str] = ["native_tm", "native_dockq"],
) -> pd.DataFrame:
    """
    Compute native comparison statistics per cluster.

    Args:
        df: DataFrame with clustering and native comparison data
        cluster_column: Column containing cluster labels
        native_columns: Columns to aggregate

    Returns:
        DataFrame with per-cluster statistics
    """
    available_cols = [c for c in native_columns if c in df.columns]

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
