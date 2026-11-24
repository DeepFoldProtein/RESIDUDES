#!/usr/bin/env python3
"""
MassiveClustering CLI

Command-line interface for protein structure clustering analysis.

Usage:
    massivecluster run --matrix tm.npy --output results/
    massivecluster comprehensive --matrix tm:tm.npy itm:itm.npy --scores scores.csv --output results/
    massivecluster network --matrix tm_dist.npy --labels labels.npy --output network/
    massivecluster consensus --pdbs pdb_dir/ --labels labels.npy --output consensus/
    massivecluster analyze --input results/ --scores scores.csv
    massivecluster report --input results/ --output report/
"""

import argparse
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        prog='massivecluster',
        description='MassiveClustering - Protein structure clustering analysis tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # End-to-end: auto K + clustering + visualization (recommended)
  massivecluster comprehensive --matrix tm:tm.npy itm:itm.npy -s scores.csv -o results/ -t T1212

  # Run clustering only
  massivecluster run --matrix tm_matrix.npy --output results/

  # Network analysis on clustered data
  massivecluster network --matrix tm_dist.npy --labels clustering.csv -o network/ -t T1212

  # Extract consensus structures per cluster
  massivecluster consensus --pdbs predictions/ --labels clustering.csv -o consensus/ -t T1212

  # Analyze clustering with assessment scores
  massivecluster analyze --input results/ --scores assessment.csv
"""
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Run command
    run_parser = subparsers.add_parser('run', help='Run clustering analysis')
    run_parser.add_argument('--matrix', '-m', required=True,
                            help='Path to distance matrix (.npy)')
    run_parser.add_argument('--names', '-n',
                            help='Path to structure names file')
    run_parser.add_argument('--metric', default='tm',
                            choices=['tm', 'itm', 'drmsd'],
                            help='Distance metric type (default: tm)')
    run_parser.add_argument('--method', default='fuzzy',
                            choices=['kmeans', 'hierarchical', 'fuzzy', 'leiden'],
                            help='Clustering method (default: fuzzy). leiden: community detection (auto K).')
    run_parser.add_argument('--k', type=int,
                            help='Number of clusters (auto-select if not specified)')
    run_parser.add_argument('--k-min', type=int, default=3,
                            help='Minimum K for auto-selection (default: 3)')
    run_parser.add_argument('--k-max', type=int, default=None,
                            help='Maximum K for auto-selection (default: sqrt(n))')
    run_parser.add_argument('--output', '-o', required=True,
                            help='Output directory')
    run_parser.add_argument('--target', '-t', default='Unknown',
                            help='Target name for labeling')

    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze clustering results')
    analyze_parser.add_argument('--input', '-i', required=True,
                                help='Path to clustering results')
    analyze_parser.add_argument('--scores', '-s',
                                help='Path to assessment scores CSV')
    analyze_parser.add_argument('--output', '-o',
                                help='Output directory (default: input dir)')

    # Report command
    report_parser = subparsers.add_parser('report', help='Generate analysis report')
    report_parser.add_argument('--input', '-i', required=True,
                               help='Path to clustering results')
    report_parser.add_argument('--matrix', '-m',
                               help='Path to distance matrix')
    report_parser.add_argument('--scores', '-s',
                               help='Path to assessment scores')
    report_parser.add_argument('--output', '-o', required=True,
                               help='Output directory')
    report_parser.add_argument('--target', '-t', default='Unknown',
                               help='Target name')

    # Heatmap command
    heatmap_parser = subparsers.add_parser('heatmap', help='Generate assessment heatmap')
    heatmap_parser.add_argument('--input', '-i', required=True,
                                help='Path to results CSV with metrics')
    heatmap_parser.add_argument('--output', '-o', required=True,
                                help='Output image path')
    heatmap_parser.add_argument('--metrics', nargs='+',
                                help='Metrics to include')
    heatmap_parser.add_argument('--cluster-col', default='cluster',
                                help='Cluster column name')

    # Comprehensive command
    comp_parser = subparsers.add_parser('comprehensive',
                                         help='End-to-end clustering + visualization')
    comp_parser.add_argument('--matrix', '-m', required=True, nargs='+',
                              help='Distance matrix .npy file(s). '
                                   'If multiple, provide as: tm:tm.npy itm:itm.npy drmsd:drmsd.npy')
    comp_parser.add_argument('--names', '-n',
                              help='Path to structure names file (.npy or .txt)')
    comp_parser.add_argument('--scores', '-s',
                              help='Path to assessment scores CSV')
    comp_parser.add_argument('--method', default='fuzzy',
                              choices=['fuzzy', 'kmeans', 'hierarchical', 'leiden'],
                              help='Clustering method (default: fuzzy). K-selection always uses hierarchical. '
                                   'leiden: community detection (auto K, ignores --k).')
    comp_parser.add_argument('--k', type=int, default=None,
                              help='Number of clusters (auto-select if not specified)')
    comp_parser.add_argument('--k-min', type=int, default=3,
                              help='Minimum K for auto-selection (default: 3)')
    comp_parser.add_argument('--k-max', type=int, default=None,
                              help='Maximum K for auto-selection (default: sqrt(n))')
    comp_parser.add_argument('--no-adaptive-k', action='store_true',
                              help='Disable adaptive K convergence detection')
    comp_parser.add_argument('--plateau-window', type=int, default=3,
                              help='Window size for silhouette plateau detection (default: 3)')
    comp_parser.add_argument('--plateau-threshold', type=float, default=0.01,
                              help='Threshold for silhouette plateau (default: 0.01)')
    comp_parser.add_argument('--pdbs', type=str, default=None,
                              help='Directory containing PDB files for consensus analysis '
                                   '(RMSF, conserved segments, representative CIF)')
    comp_parser.add_argument('--ranking', type=str, default=None,
                              help='Ranking CSV with confidence scores (pLDDT, pTM, iPTM)')
    comp_parser.add_argument('--output', '-o', required=True,
                              help='Output directory')
    comp_parser.add_argument('--target', '-t', default='Unknown',
                              help='Target name for labeling')

    # Network command
    net_parser = subparsers.add_parser('network',
                                        help='Network analysis on clustered data')
    net_parser.add_argument('--matrix', '-m', required=True,
                             help='Distance matrix .npy file')
    net_parser.add_argument('--labels', '-l', required=True,
                             help='Cluster labels file (.npy or CSV)')
    net_parser.add_argument('--names', '-n',
                             help='Structure names file')
    net_parser.add_argument('--betweenness', action='store_true',
                             help='Compute betweenness centrality (slow for N>5000)')
    net_parser.add_argument('--n-hubs', type=int, default=10,
                             help='Number of hub structures to identify (default: 10)')
    net_parser.add_argument('--community-method', default='auto',
                             choices=['auto', 'leiden', 'louvain'],
                             help='Community detection method (default: auto = Leiden with Louvain fallback)')
    net_parser.add_argument('--resolution', type=float, default=None,
                             help='Leiden CPM resolution parameter (default: None = modularity)')
    net_parser.add_argument('--resolution-scan', action='store_true',
                             help='Perform resolution scan to find optimal granularity')
    net_parser.add_argument('--stability', action='store_true',
                             help='Bootstrap community stability: subsample + re-detect, report ARI + co-association')
    net_parser.add_argument('--n-boot', type=int, default=30,
                             help='Bootstrap iterations for --stability (default 30)')
    net_parser.add_argument('--sample-frac', type=float, default=0.8,
                             help='Node subsample fraction per bootstrap (default 0.8)')
    net_parser.add_argument('--consensus-communities', action='store_true',
                             help='Fuse Leiden partitions across resolutions into one consensus partition')
    net_parser.add_argument('--n-seeds', type=int, default=0,
                             help='Run Leiden under N different RNG seeds on the full graph: '
                                  'keep best-modularity partition + report seed-stability (ARI/co-assoc). '
                                  '0 = off. Leiden only finds a local optimum per seed, so e.g. --n-seeds 20.')
    net_parser.add_argument('--pdbs', default=None,
                             help='Directory of structures (named by --names) → compute per-CLUSTER and '
                                  'per-COMMUNITY confidence (mean pLDDT from B-factor), save CSVs, and '
                                  'annotate the network graph with a per-community table.')
    net_parser.add_argument('--pdockq', action='store_true',
                             help='Also compute interface pDockQ (PAE-free interface confidence) with --pdbs.')
    net_parser.add_argument('--visualize', action='store_true',
                             help='Generate network visualization plots')
    net_parser.add_argument('--output', '-o', required=True,
                             help='Output directory')
    net_parser.add_argument('--target', '-t', default='Unknown',
                             help='Target name')

    # Interface distance matrix command (complementary metric for clustering)
    ifm_parser = subparsers.add_parser(
        'iface-matrix',
        help='Build an interface distance matrix (jaccard / chamfer / wasserstein) → .npy for clustering',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Interface-based distances capture *binding mode / contact pattern*, complementary
to global TM / dRMSD. Output .npy + structure_names.npy plug straight into
`run` / `comprehensive` / `network` (use --metric drmsd: it's already a distance).

  # contact-pattern (set-based) — needs a contacts_long table
  massivecluster iface-matrix --metric iface-jaccard-residue \\
      --contacts contacts_long.parquet -o out/iface

  # interface-shape (coord-based) — needs CIFs + chains
  massivecluster iface-matrix --metric iface-chamfer \\
      --clusters clusters.parquet --ab-chains DE --ag-chains ABC -o out/iface

Then: massivecluster comprehensive --matrix iface:out/iface/iface-*_matrix.npy \\
        --names out/iface/structure_names.npy -o results/ -t H1335
""")
    ifm_parser.add_argument('--metric', required=True,
                            choices=['iface-jaccard-residue', 'iface-jaccard-contact',
                                     'iface-chamfer', 'iface-wasserstein'])
    ifm_parser.add_argument('--contacts', default=None,
                            help='contacts_long parquet (jaccard metrics)')
    ifm_parser.add_argument('--clusters', default=None,
                            help='parquet with model_id (+ src_cif for chamfer/wasserstein) — fixes row order')
    ifm_parser.add_argument('--model-col', default='model_id')
    ifm_parser.add_argument('--cif-col', default='src_cif')
    ifm_parser.add_argument('--ab-chains', default=None, help='e.g. DE (chamfer/wasserstein)')
    ifm_parser.add_argument('--ag-chains', default=None, help='e.g. ABC (chamfer/wasserstein)')
    ifm_parser.add_argument('--anchor-chains', default=None,
                            help='chains to Kabsch-align on before chamfer/wasserstein '
                                 '(default: --ag-chains, i.e. align on antigen). Empty string disables alignment.')
    ifm_parser.add_argument('--side', default='epitope', choices=['epitope', 'paratope'])
    ifm_parser.add_argument('--cutoff', type=float, default=5.0)
    ifm_parser.add_argument('--reg', type=float, default=0.1, help='Sinkhorn reg (wasserstein)')
    ifm_parser.add_argument('--workers', type=int, default=8)
    ifm_parser.add_argument('--output', '-o', required=True, help='Output directory')

    # Consensus command
    cons_parser = subparsers.add_parser('consensus',
                                         help='Extract consensus structures from clusters')
    cons_parser.add_argument('--pdbs', '-p', required=True,
                              help='Directory containing PDB files, or text file with paths')
    cons_parser.add_argument('--labels', '-l', required=True,
                              help='Cluster labels file (.npy or CSV)')
    cons_parser.add_argument('--medoids',
                              help='Medoid indices file (.npy)')
    cons_parser.add_argument('--rmsf-threshold', type=float, default=None,
                              help='RMSF cutoff for conserved core (default: auto per cluster)')
    cons_parser.add_argument('--max-structures', type=int, default=200,
                              help='Max structures to align per cluster (default: 200)')
    cons_parser.add_argument('--output', '-o', required=True,
                              help='Output directory')
    cons_parser.add_argument('--target', '-t', default='Unknown',
                              help='Target name')

    # MPNN scoring command
    mpnn_parser = subparsers.add_parser('mpnn-score',
                                         help='Score structures using ProteinMPNN sequence recovery')
    mpnn_parser.add_argument('--pdbs', '-p', required=True,
                              help='Directory containing PDB files to score')
    mpnn_parser.add_argument('--proteinmpnn-dir', type=str, default=None,
                              help='Path to proteinmpnn/run/ directory '
                                   '(default: auto-detect in tool directory)')
    mpnn_parser.add_argument('--model', type=str, default='v_48_020',
                              help='ProteinMPNN model name (default: v_48_020)')
    mpnn_parser.add_argument('--num-sequences', type=int, default=8,
                              help='Number of sequences to sample per structure (default: 8)')
    mpnn_parser.add_argument('--sampling-temp', type=float, default=0.1,
                              help='Sampling temperature (default: 0.1, near-greedy)')
    mpnn_parser.add_argument('--rmsf-csv', type=str, default=None,
                              help='Per-residue RMSF CSV for recovery vs RMSF comparison')
    mpnn_parser.add_argument('--labels', '-l', type=str, default=None,
                              help='Cluster labels file for cluster-level scoring')
    mpnn_parser.add_argument('--max-structures', type=int, default=10,
                              help='Max structures to score (default: 10)')
    mpnn_parser.add_argument('--device', type=str, default='auto',
                              choices=['auto', 'cuda', 'cpu'],
                              help='Device for inference (default: auto)')
    mpnn_parser.add_argument('--output', '-o', required=True,
                              help='Output directory')
    mpnn_parser.add_argument('--target', '-t', default='Unknown',
                              help='Target name')

    # Superposition core command
    score_parser = subparsers.add_parser('superposition-core',
                                          help='TM-align superposition core analysis '
                                               '(per-residue structural conservation)')
    score_parser.add_argument('--pdbs', '-p', required=True,
                               help='Directory containing PDB files')
    score_parser.add_argument('--labels', '-l', required=True,
                               help='Cluster labels CSV (from clustering)')
    score_parser.add_argument('--medoids', type=str, default=None,
                               help='Cluster summary CSV with medoid info')
    score_parser.add_argument('--gpu', type=int, default=0,
                               help='GPU device ID (default: 0)')
    score_parser.add_argument('--max-clusters', type=int, default=10,
                               help='Max clusters to analyze (default: 10)')
    score_parser.add_argument('--max-per-cluster', type=int, default=150,
                               help='Max structures per cluster (default: 150)')
    score_parser.add_argument('--weight-threshold', type=float, default=0.5,
                               help='TM-weight threshold for core (default: 0.5)')
    score_parser.add_argument('--consensus-dir', type=str, default=None,
                               help='Existing consensus dir for RMSF comparison')
    score_parser.add_argument('--output', '-o', required=True,
                               help='Output directory')
    score_parser.add_argument('--target', '-t', default='Unknown',
                               help='Target name')

    # Inpaint command
    inpaint_parser = subparsers.add_parser('inpaint',
                                            help='Superposition core → PatchR inpainting pipeline')
    inpaint_parser.add_argument('--pdb', required=True,
                                 help='Medoid PDB file (template)')
    inpaint_parser.add_argument('--core', required=True,
                                 help='cross_cluster_core.csv from superposition-core')
    inpaint_parser.add_argument('--threshold', type=float, default=0.9,
                                 help='TM-weight threshold for fixed residues (default: 0.9)')
    inpaint_parser.add_argument('--weight-col', default='weight_mean',
                                 help='Column name for weight (default: weight_mean)')
    inpaint_parser.add_argument('--n-samples', type=int, default=5,
                                 help='Number of diffusion samples (default: 5)')
    inpaint_parser.add_argument('--gpu', type=int, default=0,
                                 help='GPU device for PatchR (default: 0)')
    inpaint_parser.add_argument('--native', type=str, default=None,
                                 help='Native PDB for DockQ evaluation (optional)')
    inpaint_parser.add_argument('--run-patchr', action='store_true',
                                 help='Actually run PatchR predict (requires patchr installed)')
    inpaint_parser.add_argument('--analyze-only', action='store_true',
                                 help='Skip generation, only analyze existing results')
    inpaint_parser.add_argument('--output', '-o', required=True,
                                 help='Output directory')
    inpaint_parser.add_argument('--target', '-t', required=True,
                                 help='Target name')

    # Cluster-analysis command (end-to-end Phase 1-3b contact-map analysis)
    ca_parser = subparsers.add_parser(
        'cluster-analysis',
        help='End-to-end CDR×epitope contact-map analysis (Phase 1-3b) for clustered Ab-Ag models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Runs, in one shot, into a standard folder layout under --output:
  phase1_contact_map/   epitope frequency + enrichment (per cluster)
  phase2_signatures/    per-cluster fingerprint MD + overview + signatures.json
  phase3a_1d_viz/        epitope bars, heatmaps, enrichment grid, freq-by-method
  phase3b_2d_viz/        CDR×epitope 2D heatmaps + physchem channel distribution

Example (H1335: paratope=Fab D,E; epitope=gHgLgO A,B,C):
  massivecluster cluster-analysis \\
    --clusters strict_atom_ok.parquet --contacts contacts_long.parquet \\
    --master master_h1335.parquet -o out/contact_analysis -t H1335 \\
    --paratope-chains D,E --epitope-chains A,B,C --heavy E --light D \\
    --anarci-bin /path/to/.venv-cdr/bin/ANARCI
""")
    ca_parser.add_argument('--clusters', required=True,
                           help='Cluster assignments parquet/csv (needs model_id, cluster; +src_cif for 3b)')
    ca_parser.add_argument('--contacts', default=None,
                           help='contacts_long parquet (model_id, abc_chain/abc_resnum, ...). '
                                'OPTIONAL — if omitted, derived from the cluster CIFs using '
                                '--paratope-chains/--epitope-chains.')
    ca_parser.add_argument('--master', required=True,
                           help='Master metrics parquet (per-cluster quality summary)')
    ca_parser.add_argument('--output', '-o', required=True, help='Output directory')
    ca_parser.add_argument('--target', '-t', default='target', help='Target name for labels')
    ca_parser.add_argument('--cluster-col', default='cluster')
    ca_parser.add_argument('--model-col', default='model_id')
    ca_parser.add_argument('--score-col', default='xgb_v8_pred',
                           help='Score column for recommended-pick (default: xgb_v8_pred)')
    ca_parser.add_argument('--cif-col', default='src_cif')
    # Phase toggles
    ca_parser.add_argument('--no-1d', action='store_true', help='Skip Phase 3a 1D viz')
    ca_parser.add_argument('--no-2d', action='store_true', help='Skip Phase 3b 2D viz (pair-level)')
    ca_parser.add_argument('--no-freq-by-method', action='store_true',
                           help='Skip per-method freq breakdown in Phase 3a')
    # Phase 3b pair-level params
    ca_parser.add_argument('--paratope-chains', help='Comma list, e.g. D,E (required for 2D)')
    ca_parser.add_argument('--epitope-chains', help='Comma list, e.g. A,B,C (required for 2D)')
    ca_parser.add_argument('--heavy', help='Heavy chain ID for ANARCI CDR labels')
    ca_parser.add_argument('--light', help='Light chain ID for ANARCI CDR labels')
    ca_parser.add_argument('--pair-cutoff', type=float, default=5.0)
    ca_parser.add_argument('--pair-workers', type=int, default=8)
    ca_parser.add_argument('--top-n-para', type=int, default=30)
    ca_parser.add_argument('--top-n-epi', type=int, default=30)
    ca_parser.add_argument('--anarci-bin', default='ANARCI',
                           help='ANARCI binary path (absolute path auto-prepends its bin dir to PATH)')
    ca_parser.add_argument('--max-models', type=int, default=None,
                           help='Subsample to N models per cluster (speeds up 3b on huge sets)')

    # Submission-picks command (RRF fusion + diversity-aware selection)
    sp_parser = subparsers.add_parser(
        'submission-picks',
        help='Reference picker: RRF-fuse quality scores + pick N models across clusters',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
A reference starting point — fuses scores by Reciprocal Rank Fusion (scale-free,
no weight tuning) and picks N models balancing cluster diversity with quality.

Example (H1335 strict set):
  massivecluster submission-picks \\
    --master strict_atom_ok.parquet -o out/picks -n 5 \\
    --scores xgb_v8_pred,json_iptm,json_ranking_score,iface_pdockq \\
    --strategy hybrid --sanity-cols has_atom_outlier --copy-cif
""")
    sp_parser.add_argument('--master', required=True,
                           help='parquet with cluster + score columns (+ src_cif for --copy-cif)')
    sp_parser.add_argument('--output', '-o', required=True, help='Output directory')
    sp_parser.add_argument('-n', '--n-picks', type=int, default=5)
    sp_parser.add_argument('--scores', required=True,
                           help='comma-separated score columns to fuse (e.g. xgb_v8_pred,json_iptm)')
    sp_parser.add_argument('--lower-is-better', default=None,
                           help='comma-separated subset of --scores where lower is better (e.g. dfire_v2_per_atom)')
    sp_parser.add_argument('--strategy', default='hybrid',
                           choices=['hybrid', 'top_rrf', 'per_cluster'])
    sp_parser.add_argument('--cluster-col', default='cluster')
    sp_parser.add_argument('--model-col', default='model_id')
    sp_parser.add_argument('--cif-col', default='src_cif')
    sp_parser.add_argument('--sanity-cols', default=None,
                           help='comma-separated boolean cols that must be False to keep a model (e.g. has_atom_outlier)')
    sp_parser.add_argument('--k', type=int, default=60, help='RRF damping constant (default 60)')
    sp_parser.add_argument('--copy-cif', action='store_true',
                           help='copy picked CIFs as model_1.cif … model_N.cif')

    # Join-clusters command (merge clustering output onto a master/score table)
    jc_parser = subparsers.add_parser(
        'join-clusters',
        help='Attach cluster labels from a clustering CSV onto a master parquet (handles model_id mismatch via CIF basename)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Removes the recurring friction where the clustering output and the xgb master use
different model_id conventions but reference the same files. Matches on the CIF
file basename by default.

  massivecluster join-clusters \\
    --master xgb_out/master_with_xgb.parquet \\
    --clustering clustering/.../results/com_clustering.csv \\
    -o out/clusters_with_xgb.parquet
  # → master rows + a `cluster` column, ready for cluster-analysis / submission-picks
""")
    jc_parser.add_argument('--master', required=True,
                           help='parquet/csv with src_cif (+ model_id, scores)')
    jc_parser.add_argument('--clustering', required=True,
                           help='clustering CSV/parquet with a structure + cluster column')
    jc_parser.add_argument('--on', default='cif-basename',
                           choices=['cif-basename', 'model-id'],
                           help="match key: cif-basename (basename of master.src_cif == clustering.structure) "
                                "or model-id (exact). Default cif-basename.")
    jc_parser.add_argument('--cif-col', default='src_cif')
    jc_parser.add_argument('--model-col', default='model_id')
    jc_parser.add_argument('--structure-col', default='structure')
    jc_parser.add_argument('--cluster-col', default='cluster')
    jc_parser.add_argument('--output', '-o', required=True)

    # Repair-terminal command (fix Protenix terminal-O placement, keep + flag)
    rt_parser = subparsers.add_parser(
        'repair-terminal',
        help='Reposition mis-placed terminal carboxyl O/OXT (Protenix bug); keeps models + records which were fixed',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Keeps every model — only snaps terminal O/OXT atoms farther than --max-outlier-dist
from their CA back onto sane carboxyl geometry. Writes repaired CIFs + a manifest
flagging exactly which models/atoms were touched (effective_cif = repaired path if
fixed, else original).

Example:
  massivecluster repair-terminal --master strict_atom_ok.parquet \\
    -o out/repaired --workers 4
""")
    rt_parser.add_argument('--master', required=True,
                           help='parquet with model_id + src_cif columns')
    rt_parser.add_argument('--output', '-o', required=True,
                           help='output dir (writes cifs/ + repair_manifest.csv)')
    rt_parser.add_argument('--model-col', default='model_id')
    rt_parser.add_argument('--cif-col', default='src_cif')
    rt_parser.add_argument('--max-outlier-dist', type=float, default=10.0,
                           help='atoms farther than this (Å) from CA are repositioned (default 10)')
    rt_parser.add_argument('--workers', type=int, default=4)

    # Canonicalize-chains command (homo-oligomer chain-permutation fix)
    cc_parser = subparsers.add_parser(
        'canonicalize-chains',
        help='Relabel homo-oligomer chains to a common frame (fixes transposition '
             'mislabeling that breaks TM/dRMSD); run BEFORE matrix/clustering',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
For a homo-oligomer, global-Kabsch TM/dRMSD are sensitive to chain labeling:
a cyclic relabel is a rigid rotation (fine) but a transposition collapses the
score. Predictors label interchangeable chains arbitrarily, so a big fraction of
a pool is mutually mis-scored -> fragmented clusters. This relabels every model
to the best-matching chain permutation vs a reference, then the usual pipeline
runs on the canonical structures.

  massivecluster canonicalize-chains -i pool_pdbs/ -o canon_pdbs/ \\
      --require-ref-length --workers 16
  # -> canon_pdbs/*.pdb (chains A,B,C...) + canon_manifest.csv (parity, tm_to_ref)
""")
    cc_parser.add_argument('--input', '-i', required=True, help='dir of structures (.pdb/.cif)')
    cc_parser.add_argument('--output', '-o', required=True, help='output dir for canonical .pdb')
    cc_parser.add_argument('--ref', default=None,
                           help='reference structure (default: auto, modal chain count)')
    cc_parser.add_argument('--ext', default='pdb,cif', help='comma list of extensions to read')
    cc_parser.add_argument('--max-perms', type=int, default=5040,
                           help='skip models whose (n_chains)! exceeds this (default 5040=7!)')
    cc_parser.add_argument('--require-ref-length', action='store_true',
                           help='only write models matching reference total length (needed for dRMSD)')
    cc_parser.add_argument('--workers', type=int, default=8)
    cc_parser.set_defaults(func=canonicalize_chains_cmd)

    # --- XGBDimerScorer bridge: filter-* + join-xgb subcommands ----------
    try:
        from residudes_qa.cluster.xgb_filters import register_filter_subcommands
        register_filter_subcommands(subparsers)
    except ImportError as e:
        # xgb_filters module missing — non-fatal, just disables the extension.
        print(f"  (xgb_filters not available: {e})", file=sys.stderr)

    # Version
    parser.add_argument('--version', '-v', action='version',
                        version='%(prog)s 0.5.0')

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    # If the subcommand registered its own callback via set_defaults(func=...),
    # dispatch through that — covers all xgb-filter / join-xgb subcommands.
    if hasattr(args, 'func'):
        args.func(args)
        return

    # Execute command (legacy clustering subcommands)
    if args.command == 'run':
        run_clustering(args)
    elif args.command == 'analyze':
        analyze_results(args)
    elif args.command == 'report':
        generate_report_cmd(args)
    elif args.command == 'heatmap':
        generate_heatmap_cmd(args)
    elif args.command == 'comprehensive':
        comprehensive_cmd(args)
    elif args.command == 'network':
        network_cmd(args)
    elif args.command == 'consensus':
        consensus_cmd(args)
    elif args.command == 'mpnn-score':
        mpnn_score_cmd(args)
    elif args.command == 'superposition-core':
        superposition_core_cmd(args)
    elif args.command == 'inpaint':
        inpaint_cmd(args)
    elif args.command == 'cluster-analysis':
        cluster_analysis_cmd(args)
    elif args.command == 'submission-picks':
        submission_picks_cmd(args)
    elif args.command == 'repair-terminal':
        repair_terminal_cmd(args)
    elif args.command == 'iface-matrix':
        iface_matrix_cmd(args)
    elif args.command == 'join-clusters':
        join_clusters_cmd(args)


def join_clusters_cmd(args):
    """Attach cluster labels onto a master, matching on CIF basename."""
    import os
    import pandas as pd

    def _load(p):
        return (pd.read_parquet(p) if str(p).endswith('.parquet') else pd.read_csv(p))

    master = _load(args.master)
    clust = _load(args.clustering)
    if args.cluster_col not in clust.columns:
        raise SystemExit(f"clustering file lacks '{args.cluster_col}' column "
                         f"(has: {list(clust.columns)[:8]}…)")

    if args.on == 'model-id':
        merged = master.merge(
            clust[[args.structure_col, args.cluster_col]].rename(
                columns={args.structure_col: args.model_col}),
            on=args.model_col, how='inner')
    else:  # cif-basename
        if args.cif_col not in master.columns:
            raise SystemExit(f"master lacks '{args.cif_col}' — needed for cif-basename match")
        if args.structure_col not in clust.columns:
            raise SystemExit(f"clustering lacks '{args.structure_col}' column")
        master = master.copy()
        master['_key'] = master[args.cif_col].map(lambda p: os.path.basename(str(p)))
        clust = clust[[args.structure_col, args.cluster_col]].copy()
        clust['_key'] = clust[args.structure_col].astype(str)
        merged = master.merge(clust[['_key', args.cluster_col]], on='_key', how='inner').drop(columns='_key')

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out, index=False) if str(out).endswith('.parquet') else merged.to_csv(out, index=False)
    print(f"  joined: {len(merged)} / master {len(master)} / clustering {len(clust)} "
          f"(match: {args.on})")
    print(f"  clusters: {merged[args.cluster_col].nunique()}")
    print(f"  wrote: {out}")
    if len(merged) == 0:
        print("  WARNING: 0 rows matched — check --on / column names / id conventions")


def iface_matrix_cmd(args):
    """Build an interface distance matrix (jaccard / chamfer / wasserstein)."""
    import numpy as np
    import pandas as pd
    from residudes_qa.cluster.core.iface_metrics import compute_iface_matrix

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    def _load(p):
        return (pd.read_parquet(p) if str(p).endswith('.parquet') else pd.read_csv(p))

    contacts = None
    model_ids = None
    cif_paths = None
    cluster_df = None
    if args.clusters:
        cluster_df = _load(args.clusters)
        model_ids = cluster_df[args.model_col].tolist()
        if args.cif_col in cluster_df.columns:
            cif_paths = cluster_df[args.cif_col].tolist()
    if args.contacts:
        contacts = _load(args.contacts)
        if model_ids is None:
            model_ids = sorted(contacts[args.model_col].unique().tolist())

    ab = tuple(args.ab_chains) if args.ab_chains else None
    ag = tuple(args.ag_chains) if args.ag_chains else None
    # anchor: None → default to ag inside the metric; "" → explicitly disable
    if args.anchor_chains is None:
        anchor = None
    elif args.anchor_chains == "":
        anchor = ()
    else:
        anchor = tuple(args.anchor_chains)

    # --- contact-pattern metrics: build the contact table the metric needs ---
    # iface_jaccard_contact wants pair-level cols ab_chain/ab_resnum/abc_chain/abc_resnum;
    # iface_jaccard_residue wants epitope-side abc_chain/abc_resnum. We can derive
    # both from a pair table (pair_contacts) or auto-extract it from the CIFs.
    if args.metric in ("iface-jaccard-residue", "iface-jaccard-contact"):
        pair_df = None
        if contacts is not None and {"paratope_chain", "epitope_chain"}.issubset(contacts.columns):
            pair_df = contacts  # already a pair_contacts table
        elif contacts is not None and {"ab_chain", "abc_chain"}.issubset(contacts.columns):
            pass  # already in ab_*/abc_* form — use directly below
        elif contacts is None and cif_paths is not None and ab and ag:
            from residudes_qa.cluster.analysis.pair_contacts import extract_pair_contacts_batch
            print(f"  no --contacts given — extracting pair contacts from "
                  f"{len(model_ids)} CIFs ({args.workers} workers)…")
            pair_df = extract_pair_contacts_batch(
                cluster_df[[args.cif_col, args.model_col]],
                paratope_chains=ab, epitope_chains=ag,
                cif_col=args.cif_col, model_id_col=args.model_col,
                cutoff=args.cutoff, workers=args.workers,
            )
            if "_err" in pair_df.columns:
                pair_df = pair_df[pair_df["_err"].isna()]
            pair_df = pair_df.dropna(subset=["paratope_chain"])

        if pair_df is not None:
            # map pair_contacts column names → the ab_*/abc_* the metric expects
            contacts = pair_df.rename(columns={
                "paratope_chain": "ab_chain", "paratope_resnum": "ab_resnum",
                "paratope_resname": "ab_resname",
                "epitope_chain": "abc_chain", "epitope_resnum": "abc_resnum",
                "epitope_resname": "abc_resname",
            })
        if contacts is None:
            raise SystemExit("jaccard metrics need --contacts (pair or epitope table) "
                             "or --clusters(+src_cif) with --ab-chains/--ag-chains to extract from CIFs")

    print(f"Building {args.metric} matrix for {len(model_ids) if model_ids else '?'} models…")
    mat = compute_iface_matrix(
        args.metric,
        contacts_long=contacts, cif_paths=cif_paths, model_ids=model_ids,
        ab_chains=ab, ag_chains=ag, anchor_chains=anchor, side=args.side,
        cutoff=args.cutoff, reg=args.reg, workers=args.workers,
    )
    npy = out_dir / f"{args.metric}_matrix.npy"
    np.save(npy, mat)
    np.save(out_dir / "structure_names.npy", np.array(model_ids, dtype=object))
    print(f"  matrix: {mat.shape}  range [{np.nanmin(mat):.3f}, {np.nanmax(mat):.3f}]")
    print(f"  wrote: {npy}")
    print(f"  names: {out_dir / 'structure_names.npy'}")
    print(f"\n  → cluster it: massivecluster comprehensive "
          f"--matrix iface:{npy} --names {out_dir/'structure_names.npy'} "
          f"--metric drmsd -o results/ -t <target>")


def repair_terminal_cmd(args):
    """Repair mis-placed terminal carboxyl oxygens; keep + flag."""
    import pandas as pd
    from residudes_qa.cluster.analysis.repair_terminal import repair_batch

    path = str(args.master)
    df = pd.read_parquet(path) if path.endswith('.parquet') else pd.read_csv(path)
    print(f"Loaded {len(df)} models from {path}")
    manifest = repair_batch(
        df, args.output,
        model_col=args.model_col, cif_col=args.cif_col,
        max_outlier_dist=args.max_outlier_dist, workers=args.workers,
    )
    n_rep = int(manifest['repaired'].sum())
    print(f"\n{'='*60}")
    print(f"Repaired {n_rep}/{len(manifest)} models "
          f"(atoms > {args.max_outlier_dist}Å from CA)")
    if n_rep:
        print(f"  max original dist among repaired: "
              f"{manifest.loc[manifest['repaired'], 'max_orig_dist'].max():.0f} Å")
    print(f"Manifest: {Path(args.output) / 'repair_manifest.csv'}")
    print(f"Repaired CIFs: {Path(args.output) / 'cifs'}/")
    print(f"{'='*60}")


def submission_picks_cmd(args):
    """RRF fusion + diversity-aware submission picker (reference)."""
    import pandas as pd
    from residudes_qa.cluster.analysis.submission_picks import run_submission_picks

    path = str(args.master)
    df = pd.read_parquet(path) if path.endswith('.parquet') else pd.read_csv(path)
    print(f"Loaded {len(df)} models from {path}")

    score_cols = [c.strip() for c in args.scores.split(',') if c.strip()]
    lower = ([c.strip() for c in args.lower_is_better.split(',') if c.strip()]
             if args.lower_is_better else None)
    sanity = ([c.strip() for c in args.sanity_cols.split(',') if c.strip()]
              if args.sanity_cols else None)

    manifest = run_submission_picks(
        df, args.output,
        score_cols=score_cols, n=args.n_picks,
        cluster_col=args.cluster_col, model_col=args.model_col,
        cif_col=args.cif_col, strategy=args.strategy, k=args.k,
        lower_is_better=lower, sanity_cols=sanity, copy_cif=args.copy_cif,
    )
    print(f"\n{'='*60}")
    print(f"Picks: {manifest['picks']}")
    print(f"Fusion table: {manifest['fusion_table']}")
    print(f"{'='*60}")


def cluster_analysis_cmd(args):
    """End-to-end Phase 1-3b contact-map analysis."""
    import pandas as pd
    from residudes_qa.cluster.analysis.pipeline import run_cluster_analysis

    def _load(path):
        path = str(path)
        if path.endswith('.parquet'):
            return pd.read_parquet(path)
        return pd.read_csv(path)

    print(f"Loading inputs…")
    clusters = _load(args.clusters)
    contacts = _load(args.contacts) if args.contacts else None
    master = _load(args.master)
    print(f"  clusters: {len(clusters)} rows, "
          f"{clusters[args.cluster_col].nunique()} clusters")
    print(f"  contacts_long: "
          + (f"{len(contacts)} rows" if contacts is not None
             else "(none — will derive from CIFs)"))
    print(f"  master: {len(master)} rows")

    # Optional subsample per cluster (keeps Phase 3b CIF parsing tractable)
    if args.max_models:
        parts = []
        for _, grp in clusters.groupby(args.cluster_col):
            n = min(args.max_models, len(grp))
            parts.append(grp.sample(n=n, random_state=42))
        clusters = pd.concat(parts, ignore_index=True)
        print(f"  subsampled to {len(clusters)} models "
              f"(<= {args.max_models}/cluster)")

    para = args.paratope_chains.split(',') if args.paratope_chains else None
    epi = args.epitope_chains.split(',') if args.epitope_chains else None

    manifest = run_cluster_analysis(
        cluster_assignments=clusters,
        contacts_long=contacts,
        master_df=master,
        output_dir=args.output,
        target_name=args.target,
        cluster_col=args.cluster_col,
        model_col=args.model_col,
        score_col=args.score_col,
        cif_col=args.cif_col,
        run_1d_viz=not args.no_1d,
        run_freq_by_method=not args.no_freq_by_method,
        run_2d_viz=not args.no_2d,
        paratope_chains=para,
        epitope_chains=epi,
        heavy_chain=args.heavy,
        light_chain=args.light,
        pair_cutoff=args.pair_cutoff,
        pair_workers=args.pair_workers,
        top_n_para=args.top_n_para,
        top_n_epi=args.top_n_epi,
        anarci_bin=args.anarci_bin,
    )
    print(f"\n{'='*60}")
    print(f"Done. Manifest: {Path(args.output) / 'manifest.json'}")
    for phase, info in manifest.get("phases", {}).items():
        elapsed = info.get("elapsed_sec", "?")
        print(f"  {phase}: {elapsed}s -> {info.get('dir')}")
    print(f"{'='*60}")


def run_clustering(args):
    """Execute clustering command."""
    import numpy as np
    from residudes_qa.cluster.core.distance import load_precomputed_matrix, similarity_to_distance
    from residudes_qa.cluster.core.clustering import run_clustering as cluster
    from residudes_qa.cluster.core.fuzzy import run_fuzzy_clustering
    from residudes_qa.cluster.io.writers import save_clustering_results, save_cluster_stats
    from residudes_qa.cluster.visualization.embeddings import compute_embedding

    print(f"\n{'='*60}")
    print(f"  MassiveClustering - Run")
    print(f"{'='*60}")

    # Load matrix
    print(f"\nLoading distance matrix: {args.matrix}")
    matrix, names = load_precomputed_matrix(args.matrix, args.names)
    print(f"  Matrix shape: {matrix.shape}")
    print(f"  Structures: {len(matrix)}")

    # Convert to distance if needed
    dist_matrix = similarity_to_distance(matrix, args.metric)

    # Run clustering
    print(f"\nRunning {args.method} clustering...")
    if args.method == 'fuzzy':
        n_clusters = args.k or 5
        result = run_fuzzy_clustering(
            dist_matrix, n_clusters=n_clusters,
            structure_names=names
        )
        labels = result.labels
        membership = result.membership
        print(f"  Clusters: {result.n_clusters}")
        print(f"  Fuzz coefficient: {result.fuzz_coefficient:.3f}")
    else:
        result = cluster(
            dist_matrix,
            n_clusters=args.k,
            method=args.method,
            k_range=(args.k_min, args.k_max),
            structure_names=names
        )
        labels = result.labels
        membership = None
        print(f"  Clusters: {result.n_clusters}")
        print(f"  Silhouette: {result.silhouette_score:.3f}" if result.silhouette_score else "")

    # Compute embedding
    print("\nComputing UMAP embedding...")
    embedding = compute_embedding(dist_matrix, method='umap')

    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving results to: {output_dir}")

    save_clustering_results(
        labels=labels,
        output_path=output_dir / f"{args.target}_clustering.csv",
        structure_names=names,
        membership=membership,
        embedding=embedding
    )

    save_cluster_stats(
        labels=labels,
        output_path=output_dir / f"{args.target}_cluster_stats.csv",
        medoid_indices=result.medoid_indices if hasattr(result, 'medoid_indices') else None,
        medoid_names=result.medoid_names if hasattr(result, 'medoid_names') else None,
        intra_distances=result.intra_cluster_distances if hasattr(result, 'intra_cluster_distances') else None,
    )

    print(f"\n{'='*60}")
    print("  done")
    print(f"{'='*60}")


def analyze_results(args):
    """Execute analyze command."""
    import pandas as pd
    from residudes_qa.cluster.io.readers import load_clustering_results, load_scores
    from residudes_qa.cluster.assessment.native_compare import merge_with_clustering, compute_cluster_native_stats

    print(f"\n{'='*60}")
    print(f"  MassiveClustering - Analyze")
    print(f"{'='*60}")

    # Load clustering results
    print(f"\nLoading clustering results: {args.input}")
    labels, cluster_df = load_clustering_results(args.input)
    print(f"  Structures: {len(labels)}")
    print(f"  Clusters: {len(set(labels))}")

    # Load and merge scores if provided
    if args.scores:
        print(f"\nLoading assessment scores: {args.scores}")
        scores_df = load_scores(args.scores)
        merged_df = merge_with_clustering(cluster_df, scores_df)

        # Compute cluster stats
        cluster_stats = compute_cluster_native_stats(merged_df)

        print("\n  Cluster Statistics:")
        print(cluster_stats.to_string())

        # Save
        output_dir = Path(args.output) if args.output else Path(args.input).parent
        merged_df.to_csv(output_dir / "clustering_with_scores.csv", index=False)
        cluster_stats.to_csv(output_dir / "cluster_native_stats.csv", index=False)
        print(f"\nSaved to: {output_dir}")

    print(f"\n{'='*60}")


def generate_report_cmd(args):
    """Execute report command."""
    import numpy as np
    from residudes_qa.cluster.io.readers import load_clustering_results, load_scores, load_distance_matrix
    from residudes_qa.cluster.visualization.report import generate_report

    print(f"\n{'='*60}")
    print(f"  MassiveClustering - Report")
    print(f"{'='*60}")

    # Load data
    labels, cluster_df = load_clustering_results(args.input)

    dist_matrix = None
    if args.matrix:
        dist_matrix, _ = load_distance_matrix(args.matrix)

    scores_df = None
    if args.scores:
        scores_df = load_scores(args.scores)
        # Merge with clustering
        from residudes_qa.cluster.assessment.native_compare import merge_with_clustering
        scores_df = merge_with_clustering(cluster_df, scores_df)

    # Generate report
    generate_report(
        distance_matrix=dist_matrix,
        labels=labels,
        structure_names=cluster_df.get('structure', cluster_df.index).tolist(),
        assessment_df=scores_df,
        target_name=args.target,
        output_dir=args.output,
    )


def generate_heatmap_cmd(args):
    """Execute heatmap command."""
    import pandas as pd
    from residudes_qa.cluster.assessment.heatmap import generate_heatmap
    import matplotlib.pyplot as plt

    print(f"\n{'='*60}")
    print(f"  MassiveClustering - Heatmap")
    print(f"{'='*60}")

    df = pd.read_csv(args.input)

    fig = generate_heatmap(
        df=df,
        metrics=args.metrics,
        cluster_column=args.cluster_col,
        output_path=args.output,
    )
    plt.close(fig)

    print(f"\nHeatmap saved to: {args.output}")


def comprehensive_cmd(args):
    """End-to-end clustering + visualization for multiple metrics."""
    import numpy as np
    from residudes_qa.cluster.core.distance import load_precomputed_matrix, similarity_to_distance
    from residudes_qa.cluster.core.clustering import run_clustering as cluster, select_optimal_k
    from residudes_qa.cluster.io.writers import save_clustering_results, save_cluster_stats, save_k_selection_scores
    from residudes_qa.cluster.visualization.comprehensive import (
        create_comprehensive_report,
        create_k_diagnostic_figure,
    )
    from residudes_qa.cluster.visualization.summary_table import create_cluster_summary_table
    import matplotlib.pyplot as plt

    print(f"\n{'='*60}")
    print(f"  MassiveClustering - Comprehensive Analysis")
    print(f"  Target: {args.target}")
    print(f"{'='*60}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse matrix arguments (support "metric:path" or plain paths)
    metrics = {}
    for m in args.matrix:
        if ':' in m:
            name, path = m.split(':', 1)
            metrics[name] = path
        else:
            # Infer metric name from filename
            p = Path(m)
            name = p.stem.split('_')[0]  # e.g., "tm_matrix.npy" → "tm"
            metrics[name] = m

    # Load structure names
    names = None
    if args.names:
        names_path = Path(args.names)
        if names_path.suffix == '.npy':
            names = np.load(names_path, allow_pickle=True).tolist()
        else:
            with open(names_path) as f:
                names = [line.strip() for line in f if line.strip()]

    # Load assessment scores
    scores_df = None
    native_idx = None
    if args.scores:
        import pandas as pd
        scores_df = pd.read_csv(args.scores)
        print(f"  Assessment scores: {len(scores_df)} entries")

        # Find native_idx: best TM-score structure's index in structure_names
        if names and 'file' in scores_df.columns:
            score_col = next((c for c in ['tm_score', 'lddt', 'dockq_ave']
                              if c in scores_df.columns), None)
            if score_col:
                best_row = scores_df.loc[scores_df[score_col].astype(float).idxmax()]
                best_file = best_row['file'].replace('.pdb', '')
                name_keys = [n.replace('.pdb', '') for n in names]
                if best_file in name_keys:
                    native_idx = name_keys.index(best_file)
                    print(f"  Native/Best structure: {best_file} (idx={native_idx}, "
                          f"{score_col}={best_row[score_col]:.3f})")

    # Process each metric
    all_results = {}
    for metric_name, matrix_path in metrics.items():
        print(f"\n--- Metric: {metric_name} ---")

        # Load matrix
        matrix = np.load(matrix_path)
        print(f"  Matrix: {matrix.shape}")

        # Convert to distance
        dist_matrix = similarity_to_distance(matrix, metric_name)

        # Leiden determines K automatically — skip K selection
        k_scores = None
        if args.method == 'leiden':
            n_clusters = None  # Leiden determines K
            print(f"  Method: Leiden (auto K via community detection)")
        elif args.k is None:
            n_clusters, k_scores = select_optimal_k(
                dist_matrix,
                k_range=(args.k_min, args.k_max),
                cluster_method=args.method,
                adaptive=not getattr(args, 'no_adaptive_k', False),
                plateau_window=getattr(args, 'plateau_window', 3),
                plateau_threshold=getattr(args, 'plateau_threshold', 0.01),
            )
            print(f"  Auto K = {n_clusters}")
            if k_scores.get('is_ascending_at_boundary', False):
                print(f"  WARNING: Silhouette still ascending at k_max={args.k_max}!")
                print(f"           Consider increasing --k-max for better K selection.")
            bic_k = k_scores.get('bic_optimal_k')
            if bic_k is not None and bic_k != n_clusters:
                print(f"  Note: BIC suggests K={bic_k}")

            # Save K diagnostic plot
            if k_scores and 'combined_scores' in k_scores:
                fig = create_k_diagnostic_figure(k_scores, target_name=args.target, metric_name=metric_name)
                if fig is not None:
                    fig.savefig(
                        output_dir / f"{args.target}_{metric_name}_k_diagnostic.png",
                        dpi=150, bbox_inches='tight'
                    )
                    plt.close(fig)
                    print(f"  K diagnostic plot saved")

                # Save K selection scores CSV
                save_k_selection_scores(
                    k_scores,
                    output_path=output_dir / f"{args.target}_{metric_name}_k_scores.csv",
                    target_name=args.target,
                    metric_name=metric_name,
                )
                print(f"  K selection scores saved")
        else:
            n_clusters = args.k
            print(f"  K = {n_clusters} (user specified)")

        # Run clustering
        result = cluster(
            dist_matrix,
            n_clusters=n_clusters,
            method=args.method,
            k_range=(args.k_min, args.k_max),
            structure_names=names,
        )
        print(f"  Clusters: {result.n_clusters}, Silhouette: "
              f"{result.silhouette_score:.3f}" if result.silhouette_score else "")

        all_results[metric_name] = {
            'result': result,
            'dist_matrix': dist_matrix,
            'sim_matrix': matrix,
            'k_scores': k_scores,
        }

        # Save per-metric results
        save_clustering_results(
            labels=result.labels,
            output_path=output_dir / f"{args.target}_{metric_name}_clustering.csv",
            structure_names=names,
            membership=result.membership,
            embedding=result.embedding,
        )

        save_cluster_stats(
            labels=result.labels,
            output_path=output_dir / f"{args.target}_{metric_name}_cluster_stats.csv",
            medoid_indices=result.medoid_indices,
            medoid_names=result.medoid_names,
            intra_distances=result.intra_cluster_distances,
        )

    # Generate comprehensive report per metric
    print(f"\nGenerating comprehensive visualization...")
    try:
        for metric_name, data in all_results.items():
            result = data['result']
            dist_matrix = data['dist_matrix']

            # best_idx: largest cluster's medoid (blue circle fallback)
            best_idx_for_plot = None
            if native_idx is None and result.medoid_indices:
                _, counts = np.unique(result.labels, return_counts=True)
                largest_cluster = np.argmax(counts)
                if largest_cluster < len(result.medoid_indices):
                    best_idx_for_plot = result.medoid_indices[largest_cluster]
                    if best_idx_for_plot is not None:
                        best_name = names[best_idx_for_plot] if names else f"idx={best_idx_for_plot}"
                        print(f"  Best predicted (largest cluster medoid): {best_name}")

            create_comprehensive_report(
                distance_matrix=dist_matrix,
                labels=result.labels,
                structure_names=names,
                assessment_df=scores_df,
                target_name=f"{args.target}_{metric_name}",
                output_dir=str(output_dir),
                embedding=result.embedding,
                medoid_indices=result.medoid_indices,
                membership_confidence=getattr(result, 'max_membership', None),
                native_idx=native_idx,
                best_idx=best_idx_for_plot,
            )

            # Cluster-level hierarchy: dendrogram + inter-cluster heatmap,
            # showing how clusters relate and merge. K leaves -> readable at
            # any pool size. (separate try so a failure here doesn't drop the
            # comprehensive report above.)
            try:
                from residudes_qa.cluster.visualization.dendrogram import (
                    plot_cluster_clustermap,
                )
                fig_dn = plot_cluster_clustermap(
                    distance_matrix=dist_matrix,
                    labels=result.labels,
                    structure_names=names,
                    medoid_indices=result.medoid_indices,
                    title=f"{args.target}_{metric_name}  Cluster Hierarchy",
                )
                if fig_dn is not None:
                    fig_dn.savefig(
                        output_dir / f"{args.target}_{metric_name}_dendrogram.png",
                        dpi=150, bbox_inches="tight", facecolor="white",
                    )
                    plt.close(fig_dn)
                    print(f"  Cluster dendrogram saved")
            except Exception as e:
                print(f"  Warning: Cluster dendrogram failed: {e}")
    except Exception as e:
        print(f"  Warning: Comprehensive report generation failed: {e}")

    # Generate combined cluster summary table (all metrics in one table)
    # Use the first metric for clustering labels, show intra-cluster stats for ALL metrics
    print(f"\nGenerating cluster summary table...")
    try:
        primary_metric = list(all_results.keys())[0]
        primary_result = all_results[primary_metric]['result']

        # Collect all raw similarity matrices
        sim_matrices = {m: data['sim_matrix'] for m, data in all_results.items()}

        fig, summary_df = create_cluster_summary_table(
            labels=primary_result.labels,
            similarity_matrices=sim_matrices,
            structure_names=names,
            medoid_indices=primary_result.medoid_indices,
            scores_df=scores_df,
            target_name=args.target,
            output_dir=str(output_dir),
            clustering_metric=primary_metric,
        )
        if fig is not None:
            plt.close(fig)
    except Exception as e:
        print(f"  Warning: Cluster summary table generation failed: {e}")
        import traceback
        traceback.print_exc()

    # Provenance analysis (method distribution per cluster)
    if names:
        print(f"\nRunning provenance analysis...")
        try:
            from residudes_qa.cluster.analysis.provenance import analyze_provenance
            primary_metric = list(all_results.keys())[0]
            primary_result = all_results[primary_metric]['result']
            prov_df = analyze_provenance(
                labels=primary_result.labels,
                structure_names=names,
                medoid_indices=primary_result.medoid_indices,
                output_dir=str(output_dir),
                target_name=args.target,
            )
            print(f"  Provenance: {len(prov_df.columns)} features across "
                  f"{len(prov_df)} clusters")
        except Exception as e:
            print(f"  Warning: Provenance analysis failed: {e}")

    # Native comparison + confidence analysis
    if scores_df is not None or getattr(args, 'ranking', None):
        print(f"\nRunning score analysis...")
        try:
            from residudes_qa.cluster.analysis.native_comparison import (
                analyze_native_scores, load_ranking_scores,
            )
            primary_metric = list(all_results.keys())[0]
            primary_result = all_results[primary_metric]['result']

            ranking_df = None
            if getattr(args, 'ranking', None):
                ranking_df = load_ranking_scores(args.ranking)
                print(f"  Ranking data: {len(ranking_df)} entries")

            score_df = analyze_native_scores(
                labels=primary_result.labels,
                structure_names=names,
                scores_df=scores_df,
                ranking_df=ranking_df,
                medoid_indices=primary_result.medoid_indices,
                output_dir=str(output_dir),
                target_name=args.target,
            )
            print(f"  Scores: {len(score_df)} clusters analyzed")
        except Exception as e:
            print(f"  Warning: Score analysis failed: {e}")

    # Consensus analysis (requires PDB files)
    if getattr(args, 'pdbs', None):
        pdb_dir = Path(args.pdbs)
        if pdb_dir.is_dir():
            print(f"\nRunning consensus analysis...")
            try:
                from residudes_qa.cluster.core.consensus import (
                    extract_all_cluster_consensus,
                    find_conserved_segments,
                    write_consensus_cif,
                )
                import shutil

                primary_metric = list(all_results.keys())[0]
                primary_result = all_results[primary_metric]['result']

                # Build PDB path list matching structure_names order
                pdb_paths = []
                missing = 0
                for name in names:
                    # structure_names may be bare basenames (no extension) while
                    # the pool holds <name>.pdb / <name>.cif (e.g. iface-matrix
                    # flat-dir mode). Resolve the real file by trying extensions.
                    p = pdb_dir / name
                    if not p.exists():
                        for _ext in (".pdb", ".cif"):
                            cand = pdb_dir / (name + _ext)
                            if cand.exists():
                                p = cand
                                break
                    pdb_paths.append(str(p))
                    if not p.exists():
                        missing += 1
                if missing > 0:
                    print(f"  Warning: {missing}/{len(names)} PDB files not found in {pdb_dir}")

                # -- Consensus folder --
                consensus_dir = output_dir / f"{args.target}_consensus"
                consensus_dir.mkdir(parents=True, exist_ok=True)

                consensus_list = extract_all_cluster_consensus(
                    pdb_paths=pdb_paths,
                    labels=primary_result.labels,
                    medoid_indices=primary_result.medoid_indices,
                    max_structures_per_cluster=200,
                )

                # Save per-cluster consensus CIF + RMSF + segments
                import pandas as pd
                summary_rows = []
                for cons in consensus_list:
                    cid = cons.cluster_id
                    prefix = f"cluster{cid}"

                    # Consensus CIF (mean coords, B-factor=RMSF)
                    template = pdb_paths[cons.medoid_idx]
                    if Path(template).exists():
                        write_consensus_cif(
                            consensus=cons,
                            template_pdb=template,
                            output_path=consensus_dir / f"{prefix}_consensus.cif",
                            target_name=args.target,
                        )

                    # RMSF CSV
                    rmsf_df = pd.DataFrame({
                        'residue_index': range(cons.n_residues),
                        'rmsf': cons.rmsf,
                        'conserved': cons.conserved_mask.astype(int),
                    })
                    rmsf_df.to_csv(consensus_dir / f"{prefix}_rmsf.csv", index=False)

                    # Conserved segments
                    segments = find_conserved_segments(cons)
                    if segments:
                        seg_rows = [{
                            'segment_id': s.segment_id,
                            'start': s.start, 'end': s.end,
                            'length': s.length,
                            'mean_rmsf': f"{s.mean_rmsf:.3f}",
                            'max_rmsf': f"{s.max_rmsf:.3f}",
                        } for s in segments]
                        pd.DataFrame(seg_rows).to_csv(
                            consensus_dir / f"{prefix}_conserved_segments.csv", index=False)

                    summary_rows.append({
                        'cluster': cid,
                        'medoid': Path(cons.medoid_name).name if cons.medoid_name else '',
                        'n_members': cons.n_members,
                        'n_residues': cons.n_residues,
                        'mean_rmsf': f"{cons.mean_rmsf:.3f}",
                        'median_rmsf': f"{cons.median_rmsf:.3f}",
                        'n_conserved': cons.n_conserved,
                        'frac_conserved': f"{cons.conserved_fraction:.2f}",
                        'n_segments': len(segments),
                    })

                pd.DataFrame(summary_rows).to_csv(
                    consensus_dir / f"{args.target}_consensus_summary.csv", index=False)
                print(f"  Consensus: {len(consensus_list)} clusters → {consensus_dir}")

                # -- Representatives folder --
                rep_dir = output_dir / f"{args.target}_representatives"
                rep_dir.mkdir(parents=True, exist_ok=True)

                rep_rows = []
                for i, midx in enumerate(primary_result.medoid_indices):
                    medoid_name = names[midx] if names and midx < len(names) else f"structure_{midx}"
                    src = pdb_dir / medoid_name
                    if src.exists():
                        shutil.copy2(str(src), str(rep_dir / f"cluster{i}_medoid.pdb"))
                    rep_rows.append({
                        'cluster': i,
                        'medoid_idx': midx,
                        'medoid_name': medoid_name,
                    })

                pd.DataFrame(rep_rows).to_csv(
                    rep_dir / f"{args.target}_representatives.csv", index=False)
                print(f"  Representatives: {len(rep_rows)} medoid PDBs → {rep_dir}")

            except Exception as e:
                print(f"  Warning: Consensus analysis failed: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"  Warning: PDB directory not found: {pdb_dir}")

    print(f"\n{'='*60}")
    print(f"  Results saved to: {output_dir}")
    print(f"{'='*60}")


def canonicalize_chains_cmd(args):
    """Chain-canonicalize a homo-oligomer ensemble (transposition fix)."""
    from pathlib import Path
    from residudes_qa.cluster.analysis.chain_canon import canonicalize_directory
    print(f"\n{'='*60}\n  MassiveClustering - Chain Canonicalization\n{'='*60}")
    mdf = canonicalize_directory(
        args.input, args.output, ref=args.ref,
        ext=tuple(e.strip() for e in args.ext.split(',') if e.strip()),
        max_perms=args.max_perms, require_ref_length=args.require_ref_length,
        workers=args.workers,
    )
    out = Path(args.output) / "canon_manifest.csv"
    mdf.to_csv(out, index=False)
    print(f"  manifest: {out}")
    print(f"  next: build a matrix on {args.output} then `comprehensive` / `network`")


def network_cmd(args):
    """Network analysis on clustered structures."""
    import numpy as np
    import json
    from residudes_qa.cluster.core.network import (
        analyze_cluster_network,
        compute_cluster_transition_matrix,
    )

    print(f"\n{'='*60}")
    print(f"  MassiveClustering - Network Analysis")
    print(f"  Target: {args.target}")
    print(f"{'='*60}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load distance matrix
    print(f"\nLoading distance matrix: {args.matrix}")
    dist_matrix = np.load(args.matrix)
    print(f"  Shape: {dist_matrix.shape}")

    # Load labels
    labels_path = Path(args.labels)
    if labels_path.suffix == '.npy':
        labels = np.load(labels_path)
    else:
        import pandas as pd
        df = pd.read_csv(labels_path)
        for col in ['cluster', 'tm_cluster', 'itm_cluster', 'labels']:
            if col in df.columns:
                labels = df[col].values
                break
        else:
            raise ValueError("No cluster column found in labels file")

    print(f"  Labels: {len(labels)} structures, {len(np.unique(labels))} clusters")

    # Load names
    names = None
    if args.names:
        names_path = Path(args.names)
        if names_path.suffix == '.npy':
            names = np.load(names_path, allow_pickle=True).tolist()
        else:
            with open(names_path) as f:
                names = [line.strip() for line in f if line.strip()]

    # Run network analysis
    print(f"\nRunning network analysis...")
    community_method = getattr(args, 'community_method', 'auto')
    resolution = getattr(args, 'resolution', None)

    result = analyze_cluster_network(
        distance_matrix=dist_matrix,
        labels=labels,
        structure_names=names,
        compute_betweenness=args.betweenness,
        n_hubs=args.n_hubs,
        community_method=community_method,
        resolution=resolution,
    )

    print(f"  Nodes: {result.n_nodes}")
    print(f"  Edges: {result.n_edges}")
    print(f"  Density: {result.density:.4f}")
    print(f"  Components: {result.metadata.get('graph_components', '?')}")
    print(f"  Community method: {result.community_method}")
    if result.modularity is not None:
        print(f"  Modularity: {result.modularity:.4f}")
    if result.communities is not None:
        print(f"  Communities: {len(np.unique(result.communities))}")

    # Compute transition matrix
    print(f"\nComputing transition matrix...")
    transition = compute_cluster_transition_matrix(dist_matrix, labels)

    # Resolution scan
    scan_results = None
    if getattr(args, 'resolution_scan', False):
        print(f"\nRunning Leiden resolution scan...")
        from residudes_qa.cluster.core.network import build_structure_graph, resolution_scan
        G = build_structure_graph(dist_matrix, structure_names=names)
        try:
            scan_results = resolution_scan(G)
            print(f"  Scanned {len(scan_results['resolutions'])} resolutions")
            print(f"  Suggested resolution: {scan_results.get('suggested_resolution', '?'):.4f}")

            # Save scan results
            import pandas as pd
            scan_df = pd.DataFrame({
                'resolution': scan_results['resolutions'],
                'n_communities': scan_results['n_communities'],
                'modularity': scan_results['modularity'],
            })
            scan_df.to_csv(output_dir / f"{args.target}_resolution_scan.csv", index=False)
        except ImportError:
            print(f"  WARNING: leidenalg not installed. Install with: pip install leidenalg igraph")

    # Bootstrap community stability (robustness — not a single run)
    if getattr(args, 'stability', False):
        print(f"\nBootstrap community stability ({args.n_boot} iters, "
              f"frac={args.sample_frac})...")
        from residudes_qa.cluster.core.network import build_structure_graph, bootstrap_community_stability
        import pandas as pd
        G = build_structure_graph(dist_matrix, structure_names=names)
        try:
            stab = bootstrap_community_stability(
                G, n_boot=args.n_boot, sample_frac=args.sample_frac,
                resolution=getattr(args, 'resolution', None),
            )
            print(f"  mean ARI vs full: {stab['mean_ari']:.3f} ± {stab['std_ari']:.3f}")
            print(f"  (→1.0 = communities reproducible; low/variable = fragile)")
            np.save(output_dir / f"{args.target}_coassociation.npy", stab['coassoc'])
            pd.DataFrame({'ari': stab['ari_values']}).to_csv(
                output_dir / f"{args.target}_bootstrap_ari.csv", index=False)
            pd.DataFrame({
                'node': stab['node_order'],
                'name': [names[i] for i in stab['node_order']] if names else stab['node_order'],
                'stability': stab['per_node_stability'],
            }).to_csv(output_dir / f"{args.target}_node_stability.csv", index=False)
        except ImportError:
            print(f"  WARNING: leidenalg/sklearn not installed.")

    # Multi-seed Leiden on the full graph: best-modularity partition + seed-stability
    if getattr(args, 'n_seeds', 0) and args.n_seeds > 1:
        print(f"\nMulti-seed Leiden ({args.n_seeds} seeds on full graph)...")
        from residudes_qa.cluster.core.network import build_structure_graph, multiseed_communities
        import pandas as pd
        G = build_structure_graph(dist_matrix, structure_names=names)
        try:
            ms = multiseed_communities(
                G, n_seeds=args.n_seeds,
                resolution=getattr(args, 'resolution', None),
            )
            print(f"  best modularity: {ms['best_modularity']:.4f} (seed {ms['best_seed']})")
            print(f"  n_communities per seed: min={min(ms['n_communities_per_seed'])} "
                  f"max={max(ms['n_communities_per_seed'])}")
            print(f"  mean pairwise ARI across seeds: {ms['mean_ari']:.3f} ± {ms['std_ari']:.3f}")
            print(f"  (→1.0 = partition seed-robust; lower = single-run split is a seed artifact)")
            np.save(output_dir / f"{args.target}_multiseed_coassoc.npy", ms['coassoc'])
            np.save(output_dir / f"{args.target}_multiseed_best_labels.npy", ms['best_labels'])
            pd.DataFrame({
                'seed': list(range(len(ms['modularities']))),
                'modularity': ms['modularities'],
                'n_communities': ms['n_communities_per_seed'],
            }).to_csv(output_dir / f"{args.target}_multiseed_per_seed.csv", index=False)
            pd.DataFrame({'ari_pair': ms['ari_pairs']}).to_csv(
                output_dir / f"{args.target}_multiseed_ari.csv", index=False)
            pd.DataFrame({
                'node': ms['node_order'],
                'name': [names[i] for i in ms['node_order']] if names else ms['node_order'],
                'best_community': ms['best_labels'],
            }).to_csv(output_dir / f"{args.target}_multiseed_communities.csv", index=False)
        except ImportError:
            print(f"  WARNING: leidenalg/sklearn not installed.")

    # Consensus communities across resolutions
    if getattr(args, 'consensus_communities', False):
        print(f"\nConsensus communities across resolutions...")
        from residudes_qa.cluster.core.network import build_structure_graph, consensus_communities
        import pandas as pd
        G = build_structure_graph(dist_matrix, structure_names=names)
        try:
            con = consensus_communities(G)
            print(f"  consensus communities: {con['n_consensus']} "
                  f"(fused {len(con['resolutions'])} resolutions)")
            np.save(output_dir / f"{args.target}_consensus_coassoc.npy", con['coassoc'])
            pd.DataFrame({
                'node': con['node_order'],
                'name': [names[i] for i in con['node_order']] if names else con['node_order'],
                'consensus_community': con['consensus_labels'],
            }).to_csv(output_dir / f"{args.target}_consensus_communities.csv", index=False)
        except ImportError:
            print(f"  WARNING: leidenalg not installed.")

    # Save results
    np.save(output_dir / f"{args.target}_inter_cluster_dist.npy", result.inter_cluster_matrix)
    np.save(output_dir / f"{args.target}_transition_matrix.npy", transition)
    np.save(output_dir / f"{args.target}_degree_centrality.npy", result.degree_centrality)

    if result.betweenness_centrality is not None:
        np.save(output_dir / f"{args.target}_betweenness.npy", result.betweenness_centrality)

    if result.communities is not None:
        np.save(output_dir / f"{args.target}_communities.npy", result.communities)

    # Save hub info
    hub_info = {
        "hub_indices": result.hub_indices,
        "hub_names": [names[i] for i in result.hub_indices] if names else None,
        "n_nodes": result.n_nodes,
        "n_edges": result.n_edges,
        "density": result.density,
        "modularity": result.modularity,
        "community_method": result.community_method,
        "n_communities": int(len(np.unique(result.communities))) if result.communities is not None else None,
        "n_components": result.metadata.get('graph_components'),
    }
    with open(output_dir / f"{args.target}_network_summary.json", 'w') as f:
        json.dump(hub_info, f, indent=2, default=str)

    # Save connectivity
    connectivity_data = {}
    for k, v in result.cluster_connectivity.items():
        connectivity_data[str(k)] = {str(k2): int(v2) for k2, v2 in v.items()}
    with open(output_dir / f"{args.target}_cluster_connectivity.json", 'w') as f:
        json.dump(connectivity_data, f, indent=2)

    print(f"\n  Hub structures (top {args.n_hubs}):")
    for rank, idx in enumerate(result.hub_indices[:5], 1):
        name = names[idx] if names else f"s{idx}"
        print(f"    {rank}. {name} (degree={result.degree_centrality[idx]:.3f})")

    # Per-cluster + per-community confidence (pLDDT from B-factor [+ pDockQ]).
    # Cluster summary pairs with silhouette; community summary with modularity.
    corner_table_str = None
    if getattr(args, 'pdbs', None):
        from residudes_qa.cluster.analysis.group_confidence import (
            confidence_from_paths, summarize_by_group, format_group_table,
            plot_confidence_by_group,
        )
        import matplotlib.pyplot as _plt
        pdir = Path(args.pdbs)

        def _resolve(nm):
            for cand in (pdir / str(nm), pdir / f"{nm}.pdb", pdir / f"{nm}.cif"):
                if cand.exists():
                    return cand
            return None

        paths = [_resolve(n) for n in names] if names else None
        if paths and all(p is not None for p in paths):
            print(f"\nComputing per-group confidence (pLDDT"
                  f"{' + pDockQ' if getattr(args, 'pdockq', False) else ''}) ...")
            vals = confidence_from_paths(paths, pdockq=getattr(args, 'pdockq', False))
            cdf = summarize_by_group(labels, vals, group_name='cluster',
                                     names=names, highlight_prefix='OURS_')
            cdf.to_csv(output_dir / f"{args.target}_cluster_confidence.csv", index=False)
            print(f"  saved {args.target}_cluster_confidence.csv ({len(cdf)} clusters)")
            comm = result.communities[:len(labels)] if result.communities is not None else None
            if comm is not None:
                mdf = summarize_by_group(comm, vals, group_name='community',
                                         names=names, highlight_prefix='OURS_')
                mdf.to_csv(output_dir / f"{args.target}_community_confidence.csv", index=False)
                print(f"  saved {args.target}_community_confidence.csv ({len(mdf)} communities)")
                footer = (f"modularity {result.modularity:.3f} | {len(mdf)} comm"
                          if result.modularity is not None else None)
                corner_table_str = format_group_table(mdf, group_name='community', footer=footer)
            # violin/strip plots per metric (useful even before picking models)
            for vname, arr in vals.items():
                for gname, glabels in [("cluster", labels), ("community", comm)]:
                    if glabels is None:
                        continue
                    fig_v = plot_confidence_by_group(
                        glabels, arr, value_name=vname, group_name=gname,
                        names=names, highlight_prefix='OURS_',
                        title=f"{args.target} — {vname} by {gname}",
                        output_path=output_dir / f"{args.target}_{vname}_by_{gname}.png")
                    _plt.close(fig_v)
            print(f"  saved confidence violin plots")
        else:
            n_missing = sum(1 for p in (paths or []) if p is None)
            print(f"  WARNING: --pdbs set but {n_missing} structures not found in {pdir}; "
                  f"skipping confidence")

    # Generate visualization (always generate report; --visualize adds individual plots)
    print(f"\nGenerating network visualization...")
    import matplotlib.pyplot as plt
    from residudes_qa.cluster.visualization.network_viz import create_network_report
    try:
        fig = create_network_report(
            distance_matrix=dist_matrix,
            labels=labels,
            network_result=result,
            transition_matrix=transition,
            scan_results=scan_results,
            target_name=args.target,
            output_dir=str(output_dir),
        )
        plt.close(fig)
        print(f"  Network report saved")
    except Exception as e:
        print(f"  WARNING: Network visualization failed: {e}")

    # Additional individual plots if --visualize
    if getattr(args, 'visualize', False):
        from residudes_qa.cluster.visualization.network_viz import (
            plot_network_graph, plot_community_comparison,
            plot_transition_heatmap, plot_resolution_scan,
        )
        try:
            fig = plot_network_graph(
                dist_matrix, labels,
                communities=result.communities,
                degree_centrality=result.degree_centrality,
                hub_indices=result.hub_indices,
                title=f'{args.target} Network',
                corner_table=corner_table_str,
                output_path=output_dir / f"{args.target}_network_graph.png",
            )
            plt.close(fig)
            if result.communities is not None:
                fig = plot_community_comparison(
                    labels, result.communities[:len(labels)],
                    title=f'{args.target} Cluster vs Community',
                    output_path=output_dir / f"{args.target}_community_comparison.png",
                )
                plt.close(fig)
            fig = plot_transition_heatmap(
                transition,
                title=f'{args.target} Transition',
                output_path=output_dir / f"{args.target}_transition_heatmap.png",
            )
            plt.close(fig)
            if scan_results:
                fig = plot_resolution_scan(
                    scan_results,
                    title=f'{args.target} Resolution Scan',
                    output_path=output_dir / f"{args.target}_resolution_scan.png",
                )
                plt.close(fig)
            print(f"  Individual plots saved")
        except Exception as e:
            print(f"  WARNING: Individual plots failed: {e}")

    print(f"\n{'='*60}")
    print(f"  Results saved to: {output_dir}")
    print(f"{'='*60}")


def consensus_cmd(args):
    """Extract consensus structures and local segments from clusters."""
    import numpy as np
    from residudes_qa.cluster.core.consensus import (
        extract_all_cluster_consensus,
        write_consensus_pdb,
        write_consensus_cif,
        find_conserved_segments,
        write_segment_pdb,
        write_segment_cif,
        compare_segments_across_clusters,
    )
    from residudes_qa.cluster.io.readers import load_pdb_list

    print(f"\n{'='*60}")
    print(f"  MassiveClustering - Consensus & Local Substructure Extraction")
    print(f"  Target: {args.target}")
    print(f"{'='*60}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load PDB paths
    print(f"\nLoading PDB files: {args.pdbs}")
    pdb_paths = load_pdb_list(args.pdbs)
    print(f"  Found {len(pdb_paths)} PDB files")

    # Load labels
    labels_path = Path(args.labels)
    if labels_path.suffix == '.npy':
        labels = np.load(labels_path)
    else:
        import pandas as pd
        df = pd.read_csv(labels_path)
        for col in ['cluster', 'tm_cluster', 'itm_cluster', 'labels']:
            if col in df.columns:
                labels = df[col].values
                break
        else:
            raise ValueError("No cluster column found in labels file")

    if len(pdb_paths) != len(labels):
        print(f"  Warning: {len(pdb_paths)} PDBs but {len(labels)} labels")
        min_n = min(len(pdb_paths), len(labels))
        pdb_paths = pdb_paths[:min_n]
        labels = labels[:min_n]

    n_clusters = len(np.unique(labels[labels >= 0]))
    print(f"  Structures: {len(labels)}, Clusters: {n_clusters}")

    # Load medoids if provided
    medoid_indices = None
    if args.medoids:
        medoid_indices = np.load(args.medoids).tolist()
        print(f"  Medoids loaded: {len(medoid_indices)}")

    # Extract consensus for all clusters
    print(f"\nExtracting consensus structures...")
    results = extract_all_cluster_consensus(
        pdb_paths=[str(p) for p in pdb_paths],
        labels=labels,
        medoid_indices=medoid_indices,
        rmsf_threshold=args.rmsf_threshold,
        max_structures_per_cluster=args.max_structures,
    )

    # Save results
    print(f"\n  Results per cluster:")
    summary_rows = []
    all_segment_rows = []

    for res in results:
        medoid_stem = Path(res.medoid_name).stem if res.medoid_name else "unknown"
        print(f"    Cluster {res.cluster_id}: {res.n_members} members, "
              f"{res.n_conserved}/{res.n_residues} conserved "
              f"({res.conserved_fraction:.1%}), "
              f"mean RMSF={res.mean_rmsf:.2f} A, medoid={medoid_stem}")

        prefix = f"{args.target}_cluster{res.cluster_id}"

        # Write consensus PDB + CIF
        try:
            write_consensus_pdb(
                consensus=res, template_pdb=res.medoid_name,
                output_path=output_dir / f"{prefix}_consensus.pdb",
            )
            write_consensus_cif(
                consensus=res, template_pdb=res.medoid_name,
                output_path=output_dir / f"{prefix}_consensus.cif",
                target_name=args.target,
            )
        except Exception as e:
            print(f"      Write failed: {e}")

        # Save RMSF
        np.save(output_dir / f"{prefix}_rmsf.npy", res.rmsf)

        # Find conserved segments (local substructures)
        segments = find_conserved_segments(
            res, min_segment_length=5, rmsf_threshold=args.rmsf_threshold,
        )

        if segments:
            print(f"      Local segments: {len(segments)}")
            for seg in segments:
                seg_prefix = f"{prefix}_seg{seg.segment_id}"
                print(f"        seg{seg.segment_id}: res {seg.start}-{seg.end-1} "
                      f"({seg.length} res), RMSF={seg.mean_rmsf:.2f} A")

                # Write segment PDB + CIF
                try:
                    write_segment_pdb(
                        consensus=res, segment=seg,
                        template_pdb=res.medoid_name,
                        output_path=output_dir / f"{seg_prefix}.pdb",
                    )
                    write_segment_cif(
                        consensus=res, segment=seg,
                        template_pdb=res.medoid_name,
                        output_path=output_dir / f"{seg_prefix}.cif",
                        target_name=args.target,
                    )
                except Exception as e:
                    print(f"          Segment write failed: {e}")

                all_segment_rows.append({
                    'cluster_id': res.cluster_id,
                    'segment_id': seg.segment_id,
                    'start': seg.start,
                    'end': seg.end - 1,
                    'length': seg.length,
                    'mean_rmsf': seg.mean_rmsf,
                    'max_rmsf': seg.max_rmsf,
                    'medoid': medoid_stem,
                })

        summary_rows.append({
            'cluster_id': res.cluster_id,
            'n_members': res.n_members,
            'n_residues': res.n_residues,
            'n_conserved': res.n_conserved,
            'conserved_fraction': res.conserved_fraction,
            'mean_rmsf': res.mean_rmsf,
            'median_rmsf': res.median_rmsf,
            'rmsf_threshold': res.rmsf_threshold,
            'n_segments': len(segments),
            'medoid': res.medoid_name,
        })

    # Cross-cluster segment comparison
    print(f"\n  Cross-cluster segment comparison...")
    comparisons = compare_segments_across_clusters(results)
    shared = [c for c in comparisons if c.rmsd < 2.0]
    print(f"    Total comparisons: {len(comparisons)}")
    print(f"    Shared cores (RMSD < 2.0 A): {len(shared)}")

    for comp in shared[:10]:
        print(f"      Cluster {comp.cluster_a} seg{comp.segment_a_id} <-> "
              f"Cluster {comp.cluster_b} seg{comp.segment_b_id}: "
              f"overlap res {comp.overlap_start}-{comp.overlap_end-1} "
              f"({comp.overlap_length} res), RMSD={comp.rmsd:.2f} A")

    # Save summaries
    import pandas as pd
    pd.DataFrame(summary_rows).to_csv(
        output_dir / f"{args.target}_consensus_summary.csv", index=False)

    if all_segment_rows:
        pd.DataFrame(all_segment_rows).to_csv(
            output_dir / f"{args.target}_segments.csv", index=False)

    if comparisons:
        comp_rows = [{
            'cluster_a': c.cluster_a, 'cluster_b': c.cluster_b,
            'seg_a': c.segment_a_id, 'seg_b': c.segment_b_id,
            'overlap_start': c.overlap_start, 'overlap_end': c.overlap_end,
            'overlap_length': c.overlap_length,
            'overlap_frac_a': c.overlap_fraction_a,
            'overlap_frac_b': c.overlap_fraction_b,
            'rmsd': c.rmsd,
        } for c in comparisons]
        pd.DataFrame(comp_rows).to_csv(
            output_dir / f"{args.target}_segment_comparisons.csv", index=False)

    print(f"\n{'='*60}")
    print(f"  Results saved to: {output_dir}")
    print(f"  Files: consensus PDB/CIF, segment PDB/CIF, RMSF arrays,")
    print(f"         summary CSV, segments CSV, comparisons CSV")
    print(f"{'='*60}")


def mpnn_score_cmd(args):
    """Score structures using ProteinMPNN sequence recovery."""
    import numpy as np
    from pathlib import Path
    from residudes_qa.cluster.confidence.mpnn_scoring import MPNNScorer, save_mpnn_results
    from residudes_qa.cluster.confidence.analysis import (
        compare_recovery_with_rmsf, generate_confidence_report,
    )

    print(f"\n{'='*60}")
    print(f"  MassiveClustering - ProteinMPNN Confidence Scoring")
    print(f"{'='*60}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Auto-detect proteinmpnn directory
    mpnn_dir = args.proteinmpnn_dir
    if mpnn_dir is None:
        # Look in tool directory
        # residudes-qa: vendored ProteinMPNN code (weights via $EMA_PROTEINMPNN_WEIGHTS)
        candidate = Path(os.environ.get('EMA_PROTEINMPNN_DIR', Path(__file__).resolve().parents[1] / 'mpnn' / 'run'))
        if candidate.is_dir():
            mpnn_dir = str(candidate)
            print(f"  Auto-detected ProteinMPNN: {mpnn_dir}")
        else:
            print(f"  ERROR: ProteinMPNN not found at {candidate}")
            print(f"  Please specify --proteinmpnn-dir")
            sys.exit(1)

    # Initialize scorer
    scorer = MPNNScorer(
        proteinmpnn_dir=mpnn_dir,
        model_name=args.model,
        device=args.device,
        num_sequences=args.num_sequences,
        sampling_temp=args.sampling_temp,
    )

    # Collect PDB files
    pdbs_path = Path(args.pdbs)
    if pdbs_path.is_dir():
        pdb_files = sorted(pdbs_path.glob('*.pdb'))
    elif pdbs_path.is_file() and pdbs_path.suffix == '.txt':
        with open(pdbs_path) as f:
            pdb_files = [Path(line.strip()) for line in f if line.strip()]
    else:
        pdb_files = [pdbs_path]

    if not pdb_files:
        print(f"  ERROR: No PDB files found in {args.pdbs}")
        sys.exit(1)

    # Limit number of structures
    if len(pdb_files) > args.max_structures:
        print(f"  Found {len(pdb_files)} PDBs, scoring first {args.max_structures}")
        pdb_files = pdb_files[:args.max_structures]
    else:
        print(f"  Found {len(pdb_files)} PDB files to score")

    # Score structures
    print(f"\nScoring structures with ProteinMPNN ({args.model})...")
    results = scorer.score_multiple([str(p) for p in pdb_files])

    if not results:
        print("  ERROR: No structures were successfully scored")
        sys.exit(1)

    # Save results
    print(f"\nSaving results...")
    save_mpnn_results(results, str(output_dir), target_name=args.target)

    # Print summary
    print(f"\n{'='*60}")
    print(f"  ProteinMPNN Scoring Summary")
    print(f"{'='*60}")
    print(f"  Structures scored: {len(results)}")
    recoveries = [r.overall_recovery for r in results]
    scores = [r.mpnn_score for r in results]
    print(f"  Recovery: mean={np.mean(recoveries):.4f}, "
          f"std={np.std(recoveries):.4f}, "
          f"range=[{np.min(recoveries):.4f}, {np.max(recoveries):.4f}]")
    print(f"  MPNN Score (NLL): mean={np.mean(scores):.4f}, "
          f"std={np.std(scores):.4f}")
    print()

    for r in results:
        print(f"  {Path(r.pdb_path).name:40s} "
              f"recovery={r.overall_recovery:.4f}  "
              f"score={r.mpnn_score:.4f}  "
              f"L={r.n_residues}")

    # RMSF comparison if provided
    if args.rmsf_csv:
        import pandas as pd
        print(f"\nComparing with RMSF profile: {args.rmsf_csv}")
        rmsf_df = pd.read_csv(args.rmsf_csv)
        # Expect column 'rmsf' or second column
        if 'rmsf' in rmsf_df.columns:
            rmsf_values = rmsf_df['rmsf'].values
        else:
            rmsf_values = rmsf_df.iloc[:, 1].values

        for r in results:
            comparison = compare_recovery_with_rmsf(
                r.per_residue_recovery, rmsf_values
            )
            q = comparison['quadrants']
            print(f"\n  {Path(r.pdb_path).name}:")
            print(f"    Q1 Stable core:       {q['Q1_stable_core']['count']:4d} "
                  f"({q['Q1_stable_core']['fraction']:.1%})")
            print(f"    Q2 Flexible valid:    {q['Q2_flexible_valid']['count']:4d} "
                  f"({q['Q2_flexible_valid']['fraction']:.1%})")
            print(f"    Q3 Suspicious fixed:  {q['Q3_suspicious_fixed']['count']:4d} "
                  f"({q['Q3_suspicious_fixed']['fraction']:.1%})")
            print(f"    Q4 Poorly predicted:  {q['Q4_poorly_predicted']['count']:4d} "
                  f"({q['Q4_poorly_predicted']['fraction']:.1%})")
            print(f"    Recovery-RMSF corr:   {comparison['correlation']['pearson']:.4f}")

    print(f"\n{'='*60}")
    print(f"  Results saved to: {output_dir}")
    print(f"{'='*60}")


def superposition_core_cmd(args):
    """Execute TM-align superposition core analysis."""
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from residudes_qa.cluster.core.gpu.superposition_core import (
        extract_superposition_core_gpu,
        find_core_segments,
        extract_cross_cluster_core,
        _parse_pdb_ca,
    )

    target = args.target
    pdb_dir = Path(args.pdbs)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  MassiveClustering - Superposition Core Analysis")
    print(f"  Target: {target}")
    print(f"{'='*60}")

    # Load cluster labels
    labels_path = Path(args.labels)
    cluster_df = pd.read_csv(labels_path)
    print(f"\nLoaded cluster labels: {labels_path}")
    print(f"  Structures: {len(cluster_df)}")

    # Load medoids if provided
    medoids = {}
    if args.medoids:
        medoid_path = Path(args.medoids)
        if medoid_path.exists():
            with open(medoid_path) as f:
                for line in f:
                    if line.startswith('#') or line.startswith('Cluster'):
                        continue
                    parts = line.strip().split(',')
                    if len(parts) >= 2:
                        cid = int(parts[0].replace('C', ''))
                        medoids[cid] = parts[-1]

    # Get clusters sorted by size
    cluster_counts = cluster_df['cluster'].value_counts().sort_values(ascending=False)
    clusters_to_analyze = list(cluster_counts.index[:args.max_clusters])
    n_clusters_total = len(cluster_counts)

    print(f"  Clusters: {n_clusters_total} total, analyzing top {len(clusters_to_analyze)}")

    # Load RMSF dir if provided
    rmsf_dir = Path(args.consensus_dir) if args.consensus_dir else None

    cluster_cores = {}
    results_rows = []

    for cid in clusters_to_analyze:
        cluster_members = cluster_df[cluster_df['cluster'] == cid]['structure'].tolist()
        medoid_name = medoids.get(cid, cluster_members[0])
        medoid_path = pdb_dir / medoid_name

        if not medoid_path.exists():
            print(f"  Cluster {cid}: SKIP (medoid not found)")
            continue

        # Load coordinates
        coords_list = []
        try:
            mc = _parse_pdb_ca(medoid_path)
            coords_list.append(mc)
        except Exception:
            continue

        n_residues = len(mc)
        np.random.seed(42)
        sample = [m for m in cluster_members if m != medoid_name]
        if len(sample) > args.max_per_cluster - 1:
            sample = list(np.random.choice(sample, args.max_per_cluster - 1, replace=False))

        for name in sample:
            pp = pdb_dir / name
            if not pp.exists():
                continue
            try:
                c = _parse_pdb_ca(pp)
                if len(c) == n_residues:
                    coords_list.append(c)
            except Exception:
                continue

        if len(coords_list) < 5:
            continue

        core = extract_superposition_core_gpu(
            coords_list, reference_idx=0,
            weight_threshold=args.weight_threshold,
            n_iter=5, batch_size=512, gpu_id=args.gpu,
        )
        cluster_cores[cid] = core
        segments = find_core_segments(core, min_length=5, gap_tolerance=2)

        print(f"  Cluster {cid}: {len(coords_list)} structs, "
              f"core={core.n_core}/{core.n_residues} ({core.core_fraction:.1%}), "
              f"{len(segments)} segments")

        # Save per-residue data
        res_df = pd.DataFrame({
            'residue': range(n_residues),
            'tm_weight': core.mean_weight,
            'tm_core': core.core_mask.astype(int),
            'mean_dist_A': core.mean_distance,
            'rmsf_A': core.rmsf,
        })
        if rmsf_dir and (rmsf_dir / f"cluster{cid}_rmsf.csv").exists():
            rmsf_data = pd.read_csv(rmsf_dir / f"cluster{cid}_rmsf.csv")
            if len(rmsf_data) == n_residues and 'conserved' in rmsf_data.columns:
                res_df['rmsf_core'] = rmsf_data['conserved'].values
        res_df.to_csv(out_dir / f"cluster{cid}_core_residues.csv", index=False)

        results_rows.append({
            'cluster': cid, 'n_members': len(coords_list),
            'n_residues': n_residues, 'tm_core_n': core.n_core,
            'tm_core_frac': core.core_fraction,
            'mean_weight': float(core.mean_weight.mean()),
            'mean_dist': float(core.mean_distance.mean()),
            'n_segments': len(segments),
        })

    if not cluster_cores:
        print("No clusters analyzed successfully.")
        return

    pd.DataFrame(results_rows).to_csv(out_dir / "superposition_core_summary.csv", index=False)

    # Cross-cluster analysis
    n_residues = next(iter(cluster_cores.values())).n_residues
    n_clusters = len(cluster_cores)
    weight_stack = np.stack([c.mean_weight for c in cluster_cores.values()])
    core_stack = np.stack([c.core_mask for c in cluster_cores.values()])

    universal_mask, mean_weight = extract_cross_cluster_core(cluster_cores)
    majority_mask = core_stack.sum(axis=0) > n_clusters / 2
    never_core = ~core_stack.any(axis=0)

    print(f"\n  Cross-cluster core:")
    print(f"    Universal (ALL): {universal_mask.sum()}/{n_residues} "
          f"({universal_mask.sum()/n_residues:.1%})")
    print(f"    Majority (>{n_clusters//2}): {majority_mask.sum()}/{n_residues} "
          f"({majority_mask.sum()/n_residues:.1%})")
    print(f"    Never core: {never_core.sum()}/{n_residues} "
          f"({never_core.sum()/n_residues:.1%})")

    # Multi-threshold
    print(f"  Multi-threshold (ALL clusters):")
    for thresh, name in [(0.5, 'aligned'), (0.8, 'core'),
                          (0.9, 'stable'), (0.95, 'rigid')]:
        n = int((weight_stack > thresh).all(axis=0).sum())
        print(f"    τ>{thresh} ({name:>7s}): {n}/{n_residues} ({n/n_residues:.1%})")

    var_w = weight_stack.var(axis=0)
    high_var = (var_w > 0.01).sum()
    print(f"  High-variance: {high_var}/{n_residues} ({high_var/n_residues:.1%})")

    # Save cross-cluster
    cross_df = pd.DataFrame({
        'residue': range(n_residues),
        'weight_mean': mean_weight,
        'weight_std': weight_stack.std(axis=0),
        'weight_var': var_w,
        'n_clusters_core': core_stack.sum(axis=0).astype(int),
        'universal_core': universal_mask.astype(int),
        'majority_core': majority_mask.astype(int),
        'never_core': never_core.astype(int),
    })
    for thresh in [0.5, 0.8, 0.9, 0.95]:
        cross_df[f'all_gt_{thresh}'] = (weight_stack > thresh).all(axis=0).astype(int)
    cross_df.to_csv(out_dir / "cross_cluster_core.csv", index=False)

    # ── 6-panel visualization ──
    residues = np.arange(n_residues)
    fig, axes = plt.subplots(6, 1, figsize=(16, 22), sharex=True,
                             gridspec_kw={'height_ratios': [2, 1.5, 1.5, 2, 1, 1]})
    fig.suptitle(f"{target}: TM-align Superposition Core "
                 f"({n_residues} res, {n_clusters} clusters)", fontsize=14, y=0.99)

    # P1: Heatmap
    ax = axes[0]
    im = ax.imshow(weight_stack, aspect='auto', cmap='RdYlBu',
                   vmin=0, vmax=1, interpolation='nearest',
                   extent=[0, n_residues, n_clusters-0.5, -0.5])
    ax.set_ylabel('Cluster')
    ax.set_title('Per-residue TM-weight')
    ax.set_yticks(range(n_clusters))
    ax.set_yticklabels([f'C{cid}' for cid in sorted(cluster_cores.keys())], fontsize=7)
    plt.colorbar(im, ax=ax, label='TM-weight', shrink=0.8, pad=0.01)

    # P2: Mean ± std
    ax = axes[1]
    mean_w = weight_stack.mean(axis=0)
    std_w = weight_stack.std(axis=0)
    ax.fill_between(residues, mean_w - std_w, mean_w + std_w,
                    alpha=0.3, color='steelblue')
    ax.plot(residues, mean_w, 'b-', linewidth=1.0, label='mean')
    for thresh, ls, c in [(0.95, '-', 'darkgreen'), (0.9, '--', 'green'),
                           (0.8, '-.', 'orange'), (0.5, ':', 'red')]:
        ax.axhline(thresh, color=c, linestyle=ls, alpha=0.5, linewidth=0.8,
                   label=f'τ={thresh}')
    ax.set_ylabel('Mean TM-weight')
    ax.set_title('Mean ± std across clusters')
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=7, ncol=6, loc='lower left')

    # P3: Variance
    ax = axes[2]
    colors_var = np.where(var_w > 0.05, 'red',
                 np.where(var_w > 0.01, 'orange',
                 np.where(var_w > 0.001, 'gold', 'steelblue')))
    ax.bar(residues, var_w, width=1.0, color=colors_var)
    ax.set_ylabel('Weight var')
    ax.set_title('Weight variance (red=switch, blue=rigid)')
    if var_w.max() > 0.1:
        ax.set_yscale('log')

    # P4: Multi-threshold layers
    ax = axes[3]
    thresholds = [0.95, 0.9, 0.8, 0.5]
    colors_t = ['#1a237e', '#1565c0', '#42a5f5', '#bbdefb']
    labels_t = ['rigid(>0.95)', 'stable(>0.9)', 'core(>0.8)', 'aligned(>0.5)']
    for thresh, color, label in zip(thresholds, colors_t, labels_t):
        frac = (weight_stack > thresh).mean(axis=0)
        ax.fill_between(residues, 0, frac, alpha=0.6, color=color, label=label)
    ax.set_ylabel('Fraction')
    ax.set_ylim(0, 1.05)
    ax.set_title('Multi-threshold core layers')
    ax.legend(fontsize=8, loc='lower left')

    # P5: TM-weight vs RMSF
    ax = axes[4]
    c0 = list(cluster_cores.values())[0]
    ax.plot(residues, c0.mean_weight, 'b-', alpha=0.7, linewidth=0.8)
    ax.set_ylabel('TM-weight', color='blue', fontsize=9)
    ax2 = ax.twinx()
    if c0.rmsf is not None:
        ax2.plot(residues, np.clip(c0.rmsf, 0, np.percentile(c0.rmsf, 98)),
                 'r-', alpha=0.7, linewidth=0.8)
        ax2.set_ylabel('RMSF (Å)', color='red', fontsize=9)
    ax.set_title(f'Cluster {sorted(cluster_cores.keys())[0]}: TM-weight vs RMSF')

    # P6: Core level map
    ax = axes[5]
    core_level = np.zeros(n_residues)
    for i, thresh in enumerate([0.5, 0.8, 0.9, 0.95]):
        core_level[(weight_stack > thresh).all(axis=0)] = i + 1
    core_level[(weight_stack < 0.5).all(axis=0)] = -1
    level_colors = {-1: '#d32f2f', 0: '#e0e0e0', 1: '#bbdefb',
                    2: '#42a5f5', 3: '#1565c0', 4: '#1a237e'}
    ax.bar(residues, np.ones(n_residues), width=1.0,
           color=[level_colors[int(l)] for l in core_level])
    ax.set_yticks([])
    ax.set_xlabel('Residue index')
    counts = {k: int((core_level == k).sum()) for k in [-1, 0, 1, 2, 3, 4]}
    ax.set_title(f'rigid={counts[4]} | stable={counts[3]} | core={counts[2]} | '
                 f'aligned={counts[1]} | partial={counts[0]} | variable={counts[-1]}',
                 fontsize=9)

    plt.tight_layout()
    fig_path = out_dir / f"{target}_superposition_core.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n  Figure: {fig_path}")
    print(f"  Results: {out_dir}")
    print(f"{'='*60}")


def inpaint_cmd(args):
    """Superposition core → PatchR inpainting pipeline with analysis."""
    import numpy as np
    from residudes_qa.cluster.core.inpainting import (
        generate_patchr_input, run_patchr, analyze_inpainting,
    )

    target = args.target
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  MassiveClustering - Inpainting Pipeline")
    print(f"  Target: {target}")
    print(f"{'='*60}")

    # ── Step 1: Generate PatchR input ──────────────────────────
    if not args.analyze_only:
        print(f"\n[Step 1] Generating PatchR input (threshold={args.threshold})")
        yaml_path, cif_path, meta_path = generate_patchr_input(
            pdb_path=args.pdb,
            core_csv=args.core,
            threshold=args.threshold,
            target=target,
            output_dir=str(out),
            weight_col=args.weight_col,
            n_samples=args.n_samples,
        )
        print(f"  Files: {yaml_path}, {cif_path}, {meta_path}")

    # ── Step 2: Run PatchR predict ─────────────────────────────
    results_dir = out / "results"
    yaml_path = out / f"{target}.yaml"

    if args.run_patchr and not args.analyze_only:
        print(f"\n[Step 2] Running PatchR predict (GPU {args.gpu}, {args.n_samples} samples)")
        rc = run_patchr(str(yaml_path), str(results_dir), args.n_samples, args.gpu)
        if rc != 0:
            print(f"  [ERROR] PatchR failed with return code {rc}")
            return
        print(f"  PatchR completed successfully")
    elif not args.analyze_only:
        print(f"\n[Step 2] Skipped (use --run-patchr to execute)")
        print(f"  Run manually:")
        print(f"    CUDA_VISIBLE_DEVICES={args.gpu} patchr predict {yaml_path} "
              f"--out_dir {results_dir} --diffusion_samples {args.n_samples}")

    # ── Step 3: Analyze results ────────────────────────────────
    pred_dir = None
    for candidate in [
        results_dir / f"patchr_results_{target}" / "predictions" / target,
        results_dir / f"boltz_results_{target}" / "predictions" / target,
    ]:
        if candidate.exists():
            pred_dir = candidate
            break

    if pred_dir is None:
        print(f"\n[Step 3] No prediction results found. Run PatchR first.")
        return

    print(f"\n[Step 3] Analyzing inpainting results")
    meta_path = out / f"{target}_inpainting_metadata.json"
    summary = analyze_inpainting(
        pred_dir=str(pred_dir),
        metadata_path=str(meta_path),
        target=target,
        output_dir=str(out),
        native_pdb=args.native,
    )

    # ── Print summary ──────────────────────────────────────────
    if summary:
        print(f"\n{'='*60}")
        print(f"  Inpainting Analysis: {target}")
        print(f"{'='*60}")
        print(f"  Threshold: τ>{args.threshold}")
        print(f"  Residues: {summary['n_fixed']} fixed + {summary['n_inpainted']} inpainted = {summary['n_total']} total")
        print(f"\n  Structural Diversity ({summary['n_models']} models, pairwise after superpose):")
        print(f"    Fixed RMSD:     {summary['rmsd_fixed_mean']:.3f}Å")
        print(f"    Inpainted RMSD: {summary['rmsd_inpainted_mean']:.3f}Å")
        print(f"\n  Per-region RMSF:")
        print(f"    Fixed region:     {summary['rmsf_fixed_mean']:.3f}Å")
        print(f"    Inpainted region: {summary['rmsf_inpainted_mean']:.3f}Å")
        print(f"\n  Confidence Scores: {summary['confidence_scores']}")
        if 'dockq_scores' in summary:
            print(f"  DockQ Scores: {summary['dockq_scores']}")
        print(f"\n  Output files:")
        print(f"    Analysis CSV: {summary['analysis_csv']}")
        print(f"    RMSF CSV:     {summary['rmsf_csv']}")
        print(f"    Figure:       {summary['figure']}")
        print(f"{'='*60}")


if __name__ == '__main__':
    main()
