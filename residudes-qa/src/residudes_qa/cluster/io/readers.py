"""
File readers module

Load various input formats:
- PDB structure files
- Precomputed distance matrices
- Assessment score files
- Clustering results
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Union, List, Tuple, Dict


def load_pdb_list(
    input_path: Union[str, Path],
    pattern: str = "*.pdb",
    recursive: bool = False,
) -> List[Path]:
    """
    Load list of PDB files from directory or file.

    Args:
        input_path: Directory path or text file with paths
        pattern: Glob pattern for PDB files
        recursive: Search recursively in subdirectories

    Returns:
        List of PDB file paths
    """
    input_path = Path(input_path)

    if input_path.is_file():
        with open(input_path) as f:
            paths = [Path(line.strip()) for line in f if line.strip()]
        return [p for p in paths if p.exists()]

    elif input_path.is_dir():
        if recursive:
            pattern = "**/" + pattern
        return sorted(input_path.glob(pattern))

    else:
        raise FileNotFoundError(f"Path not found: {input_path}")


def load_distance_matrix(
    matrix_path: Union[str, Path],
    names_path: Optional[Union[str, Path]] = None,
) -> Tuple[np.ndarray, Optional[List[str]]]:
    """
    Load precomputed distance/similarity matrix.

    Supports: .npy, .npz, .csv, .tsv

    Args:
        matrix_path: Path to matrix file
        names_path: Optional path to structure names file

    Returns:
        Tuple of (matrix, structure_names or None)
    """
    matrix_path = Path(matrix_path)

    if matrix_path.suffix == '.npy':
        matrix = np.load(matrix_path)
    elif matrix_path.suffix == '.npz':
        data = np.load(matrix_path)
        matrix = data['matrix'] if 'matrix' in data else data[data.files[0]]
    elif matrix_path.suffix in ['.csv', '.tsv']:
        sep = '\t' if matrix_path.suffix == '.tsv' else ','
        df = pd.read_csv(matrix_path, sep=sep, index_col=0)
        matrix = df.values
    else:
        matrix = np.load(matrix_path)

    names = None
    if names_path:
        names_path = Path(names_path)
        if names_path.exists():
            if names_path.suffix == '.npy':
                names = np.load(names_path, allow_pickle=True).tolist()
            else:
                with open(names_path) as f:
                    names = [line.strip() for line in f if line.strip()]

    return matrix, names


def load_scores(
    scores_path: Union[str, Path],
    structure_column: str = "file",
) -> pd.DataFrame:
    """
    Load assessment scores from CSV file.

    Args:
        scores_path: Path to CSV file
        structure_column: Column containing structure names

    Returns:
        DataFrame with scores
    """
    df = pd.read_csv(scores_path)

    if structure_column in df.columns and structure_column != 'structure':
        df['structure'] = df[structure_column]

    column_mapping = {
        'tm_score': 'native_tm',
        'dockq_ave': 'native_dockq',
        'dockq_wave': 'native_dockq_weighted',
        'ilddt': 'native_ilddt',
        'ics': 'native_ics',
        'ips': 'native_ips',
    }

    for old_name, new_name in column_mapping.items():
        if old_name in df.columns and new_name not in df.columns:
            df[new_name] = df[old_name]

    return df


def load_clustering_results(
    results_path: Union[str, Path],
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Load saved clustering results.

    Args:
        results_path: Path to results directory or CSV file

    Returns:
        Tuple of (labels, full_dataframe)
    """
    results_path = Path(results_path)

    if results_path.is_dir():
        csv_files = list(results_path.glob("*clustering*.csv"))
        if csv_files:
            results_path = csv_files[0]
        else:
            raise FileNotFoundError(f"No clustering CSV found in {results_path}")

    df = pd.read_csv(results_path)

    cluster_col = None
    for col in ['cluster', 'tm_cluster', 'itm_cluster', 'labels']:
        if col in df.columns:
            cluster_col = col
            break

    if cluster_col is None:
        raise ValueError("No cluster column found in results file")

    labels = df[cluster_col].values

    return labels, df


def load_config(config_path: Union[str, Path]) -> Dict:
    """
    Load configuration from YAML or JSON file.

    Args:
        config_path: Path to config file

    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)

    if config_path.suffix in ['.yaml', '.yml']:
        import yaml
        with open(config_path) as f:
            return yaml.safe_load(f)
    elif config_path.suffix == '.json':
        import json
        with open(config_path) as f:
            return json.load(f)
    else:
        raise ValueError(f"Unsupported config format: {config_path.suffix}")
