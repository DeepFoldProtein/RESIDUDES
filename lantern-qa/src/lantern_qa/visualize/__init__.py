"""lantern_qa.visualize — per-target prediction plots and rankings (ex casp17/visualization, af3-viz).

pLDDT / PAE / PDE / distogram / contact probability / Ramachandran / MSA images and
all_rankings.csv, summary.csv, per_seed_summary.csv for every stoichiometry x MSA variant x model.
"""
from .reader import (
    SampleFiles,
    PredictionRun,
    discover_prediction_types,
    find_best_sample,
    load_confidences,
    load_summary_confidences,
    load_all_ranking_scores,
    get_msa_paths,
)
from .metrics import (
    residue_plddt,
    chain_boundaries,
    pae_matrix,
    contact_prob_matrix,
    mean_pae_per_chain_pair,
    pde_matrix,
    mean_pde_per_chain_pair,
    compute_plddt_stats,
    compute_pae_stats,
    compute_pde_stats,
    compute_pdockq_for_complex,
    compute_distogram,
    compute_ramachandran,
    parse_a3m,
    compute_msa_visualization,
)
from .ranking import (
    build_ranking_dataframe,
    build_per_seed_summary,
    find_global_best,
)
from .plots import (
    plot_plddt,
    plot_pae_heatmap,
    plot_pde_heatmap,
    plot_contact_heatmap,
    plot_ranking_bars,
    plot_metrics_comparison,
    plot_distogram,
    plot_ramachandran,
    plot_msa,
    save_figure,
)

