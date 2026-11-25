"""
Model Provenance Analysis

Analyzes which prediction methods (afm_basic, afm_dropout_full, cf_*, etc.)
contribute to each cluster. Identifies method-specific conformational preferences.
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Union
from pathlib import Path
import re


# Method parsing: extract prediction method from structure name.
# Patterns are evaluated in order; longest/most-specific prefixes must come
# first so they win against shorter overlapping ones (e.g. afm_dropout_full_
# before afm_).
_METHOD_PATTERNS = [
    # MassiveFold (CASP17-CAPRI flat bundles): files are named
    # `Model_<idx>_<engine>_<paramset>_..._pred_<i>.{pdb,cif}`. The Model_<digits>_
    # prefix disambiguates from our own pipeline's outputs (which never start
    # with `Model_`), so per-paramset attribution is preserved.
    # AF2 (6 sets):
    (r'^Model_\d+_afm_dropout_04_woTemplates_', 'mf_afm_dropout_04_woTemplates'),
    (r'^Model_\d+_afm_dropout_noSM_woTemplates_', 'mf_afm_dropout_noSM_woTemplates'),
    (r'^Model_\d+_afm_dropout_full_', 'mf_afm_dropout_full'),
    (r'^Model_\d+_afm_woTemplates_', 'mf_afm_woTemplates'),
    (r'^Model_\d+_afm_basic_', 'mf_afm_basic'),
    (r'^Model_\d+_cf_woTemplates_', 'mf_cf_woTemplates'),
    # AF3 (8 sets) — longest first:
    (r'^Model_\d+_af3_woUnpaired_woPaired_woTemplates_', 'mf_af3_woUnpaired_woPaired_woTemplates'),
    (r'^Model_\d+_af3_woUnpaired_woPaired_', 'mf_af3_woUnpaired_woPaired'),
    (r'^Model_\d+_af3_woUnpaired_woTemplates_', 'mf_af3_woUnpaired_woTemplates'),
    (r'^Model_\d+_af3_woUnpaired_', 'mf_af3_woUnpaired'),
    (r'^Model_\d+_af3_woPaired_woTemplates_', 'mf_af3_woPaired_woTemplates'),
    (r'^Model_\d+_af3_woPaired_', 'mf_af3_woPaired'),
    (r'^Model_\d+_af3_woTemplates_', 'mf_af3_woTemplates'),
    (r'^Model_\d+_af3_basic_', 'mf_af3_basic'),
    # In-house pipeline (single MSA build per variant directory)
    (r'^afm_dropout_full_woTemplates_r3_', 'afm_dropout_full_woTemplates_r3'),
    (r'^afm_dropout_full_woTemplates_', 'afm_dropout_full_woTemplates'),
    (r'^afm_dropout_full_', 'afm_dropout_full'),
    (r'^afm_dropout_noSM_woTemplates_', 'afm_dropout_noSM_woTemplates'),
    (r'^afm_woTemplates_', 'afm_woTemplates'),
    (r'^afm_basic_', 'afm_basic'),
    (r'^cf_dropout_full_woTemplates_', 'cf_dropout_full_woTemplates'),
    (r'^cf_dropout_full_', 'cf_dropout_full'),
    (r'^cf_woTemplates_', 'cf_woTemplates'),
    (r'^cf_', 'cf'),
    # CASP17 unified-output models (af3/bt2/of3/pt2 + seed/sample)
    (r'^af3_', 'af3'),
    (r'^bt2_', 'bt2'),
    (r'^of3_', 'of3'),
    (r'^pt2_', 'pt2'),
]


def parse_method(name: str) -> str:
    """Extract prediction method from structure filename."""
    for pattern, method in _METHOD_PATTERNS:
        if re.match(pattern, name):
            return method
    return 'unknown'


def analyze_provenance(
    labels: np.ndarray,
    structure_names: List[str],
    medoid_indices: Optional[List[int]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    target_name: str = "Target",
) -> pd.DataFrame:
    """
    Analyze prediction method distribution across clusters.

    For each cluster, computes:
    - Count and fraction of each method
    - Dominant method
    - Method diversity (entropy)

    Args:
        labels: Cluster labels for each structure
        structure_names: Structure filenames
        medoid_indices: Medoid indices per cluster
        output_dir: If provided, save CSV and plot
        target_name: Target name for labeling

    Returns:
        DataFrame with per-cluster method distribution
    """
    n = len(labels)
    methods = [parse_method(name) for name in structure_names]
    unique_methods = sorted(set(methods))
    unique_clusters = sorted(np.unique(labels))

    # Per-cluster method distribution
    rows = []
    for cid in unique_clusters:
        mask = labels == cid
        cluster_methods = [methods[i] for i in range(n) if mask[i]]
        cluster_size = len(cluster_methods)

        row = {'cluster': cid, 'size': cluster_size}

        # Count per method
        for m in unique_methods:
            count = cluster_methods.count(m)
            row[f'{m}_count'] = count
            row[f'{m}_frac'] = count / cluster_size if cluster_size > 0 else 0.0

        # Dominant method
        method_counts = {m: cluster_methods.count(m) for m in unique_methods}
        dominant = max(method_counts, key=method_counts.get)
        row['dominant_method'] = dominant
        row['dominant_frac'] = method_counts[dominant] / cluster_size

        # Method diversity: Shannon entropy
        fracs = np.array([method_counts[m] / cluster_size for m in unique_methods
                          if method_counts[m] > 0])
        entropy = -np.sum(fracs * np.log2(fracs + 1e-10))
        row['method_entropy'] = float(entropy)

        # Medoid method
        if medoid_indices and cid < len(medoid_indices):
            midx = medoid_indices[cid]
            row['medoid_method'] = methods[midx]

        rows.append(row)

    df = pd.DataFrame(rows)

    # Global method distribution
    global_counts = {m: methods.count(m) for m in unique_methods}

    # Enrichment: cluster fraction / global fraction
    for m in unique_methods:
        global_frac = global_counts[m] / n
        if global_frac > 0:
            df[f'{m}_enrichment'] = df[f'{m}_frac'] / global_frac

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save CSV
        csv_path = output_dir / f"{target_name}_provenance.csv"
        df.to_csv(csv_path, index=False)

        # Save summary
        summary_path = output_dir / f"{target_name}_method_summary.csv"
        summary_rows = []
        for m in unique_methods:
            summary_rows.append({
                'method': m,
                'total_count': global_counts[m],
                'total_frac': f"{global_counts[m]/n:.3f}",
                'n_clusters_dominant': int((df['dominant_method'] == m).sum()),
            })
        pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

        # Plot
        try:
            _plot_provenance(df, unique_methods, target_name, output_dir)
        except Exception:
            pass

    return df


def _plot_provenance(
    df: pd.DataFrame,
    methods: List[str],
    target_name: str,
    output_dir: Path,
):
    """Create stacked bar chart of method distribution per cluster."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Top: stacked bar of method fractions
    ax = axes[0]
    clusters = df['cluster'].values
    bottom = np.zeros(len(clusters))
    colors = plt.cm.Set3(np.linspace(0, 1, len(methods)))

    for i, m in enumerate(methods):
        col = f'{m}_frac'
        if col in df.columns:
            vals = df[col].values
            ax.bar(range(len(clusters)), vals, bottom=bottom,
                   label=m, color=colors[i], width=0.8)
            bottom += vals

    ax.set_xlabel('Cluster')
    ax.set_ylabel('Method Fraction')
    ax.set_title(f'{target_name} — Prediction Method Distribution per Cluster')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    ax.set_xticks(range(len(clusters)))
    ax.set_xticklabels([f'C{c}' for c in clusters], rotation=45, fontsize=7)

    # Bottom: enrichment heatmap
    ax2 = axes[1]
    enrichment_cols = [f'{m}_enrichment' for m in methods if f'{m}_enrichment' in df.columns]
    if enrichment_cols:
        enrichment_data = df[enrichment_cols].values.T
        method_labels = [c.replace('_enrichment', '') for c in enrichment_cols]
        im = ax2.imshow(enrichment_data, aspect='auto', cmap='RdBu_r',
                        vmin=0, vmax=2)
        ax2.set_yticks(range(len(method_labels)))
        ax2.set_yticklabels(method_labels, fontsize=8)
        ax2.set_xticks(range(len(clusters)))
        ax2.set_xticklabels([f'C{c}' for c in clusters], rotation=45, fontsize=7)
        ax2.set_title('Method Enrichment (cluster_frac / global_frac)')
        plt.colorbar(im, ax=ax2, shrink=0.6, label='Enrichment')

    plt.tight_layout()
    fig.savefig(output_dir / f"{target_name}_provenance.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
