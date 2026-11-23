#!/usr/bin/env python3
"""
Analysis utilities for ProteinMPNN confidence scoring.

Compares sequence recovery with RMSF profiles to identify:
- Well-predicted stable cores (high recovery + low RMSF)
- Flexible but valid regions (high recovery + high RMSF)
- Suspicious fixed regions (low recovery + low RMSF)
- Poorly predicted flexible regions (low recovery + high RMSF)
"""

import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def compare_recovery_with_rmsf(
    recovery: np.ndarray,
    rmsf: np.ndarray,
    recovery_threshold: float = 0.5,
    rmsf_threshold: float = 2.0,
) -> Dict:
    """
    Classify residues into four quadrants based on recovery and RMSF.

    Quadrant classification:
        Q1: High recovery + Low RMSF  → Stable, well-predicted core
        Q2: High recovery + High RMSF → Flexible but physically valid
        Q3: Low recovery + Low RMSF   → Fixed but unnatural backbone (WARNING)
        Q4: Low recovery + High RMSF  → Poorly predicted, unreliable

    Args:
        recovery: Per-residue sequence recovery (0-1), shape (L,)
        rmsf: Per-residue RMSF values (Å), shape (L,)
        recovery_threshold: Cutoff for high vs low recovery
        rmsf_threshold: Cutoff for high vs low RMSF (Å)

    Returns:
        Dict with quadrant masks, counts, and statistics
    """
    n = min(len(recovery), len(rmsf))
    recovery = recovery[:n]
    rmsf = rmsf[:n]

    high_rec = recovery >= recovery_threshold
    low_rec = ~high_rec
    high_rmsf = rmsf >= rmsf_threshold
    low_rmsf = ~high_rmsf

    q1_mask = high_rec & low_rmsf   # Stable core
    q2_mask = high_rec & high_rmsf  # Flexible but valid
    q3_mask = low_rec & low_rmsf    # Suspicious fixed
    q4_mask = low_rec & high_rmsf   # Poorly predicted

    result = {
        "n_residues": n,
        "recovery_threshold": recovery_threshold,
        "rmsf_threshold": rmsf_threshold,
        "quadrants": {
            "Q1_stable_core": {
                "mask": q1_mask,
                "count": int(q1_mask.sum()),
                "fraction": float(q1_mask.mean()),
                "mean_recovery": float(recovery[q1_mask].mean()) if q1_mask.any() else 0.0,
                "mean_rmsf": float(rmsf[q1_mask].mean()) if q1_mask.any() else 0.0,
            },
            "Q2_flexible_valid": {
                "mask": q2_mask,
                "count": int(q2_mask.sum()),
                "fraction": float(q2_mask.mean()),
                "mean_recovery": float(recovery[q2_mask].mean()) if q2_mask.any() else 0.0,
                "mean_rmsf": float(rmsf[q2_mask].mean()) if q2_mask.any() else 0.0,
            },
            "Q3_suspicious_fixed": {
                "mask": q3_mask,
                "count": int(q3_mask.sum()),
                "fraction": float(q3_mask.mean()),
                "mean_recovery": float(recovery[q3_mask].mean()) if q3_mask.any() else 0.0,
                "mean_rmsf": float(rmsf[q3_mask].mean()) if q3_mask.any() else 0.0,
            },
            "Q4_poorly_predicted": {
                "mask": q4_mask,
                "count": int(q4_mask.sum()),
                "fraction": float(q4_mask.mean()),
                "mean_recovery": float(recovery[q4_mask].mean()) if q4_mask.any() else 0.0,
                "mean_rmsf": float(rmsf[q4_mask].mean()) if q4_mask.any() else 0.0,
            },
        },
        "correlation": {
            "pearson": float(np.corrcoef(recovery, rmsf)[0, 1])
            if n > 1
            else 0.0,
        },
    }

    return result


def generate_confidence_report(
    cluster_scores: Dict,
    rmsf_profiles: Optional[Dict[int, np.ndarray]] = None,
    output_dir: str = ".",
    target_name: str = "unknown",
) -> str:
    """
    Generate comprehensive confidence report combining MPNN recovery with RMSF.

    Args:
        cluster_scores: Dict mapping cluster_id → ClusterConfidence
        rmsf_profiles: Dict mapping cluster_id → per-residue RMSF array
        output_dir: Directory for output files
        target_name: Target identifier

    Returns:
        Path to summary CSV file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Summary CSV
    summary_path = output_dir / f"{target_name}_confidence_report.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = [
            "cluster_id", "n_residues",
            "consensus_recovery", "medoid_recovery",
            "mean_member_recovery", "std_member_recovery",
            "mpnn_score_consensus", "mpnn_score_medoid",
        ]
        if rmsf_profiles:
            header.extend([
                "Q1_stable_core_frac", "Q2_flexible_valid_frac",
                "Q3_suspicious_fixed_frac", "Q4_poorly_predicted_frac",
                "recovery_rmsf_correlation",
            ])
        writer.writerow(header)

        for cid, conf in sorted(cluster_scores.items()):
            row = [
                cid,
                conf.consensus_result.n_residues if conf.consensus_result else "N/A",
                f"{conf.consensus_result.overall_recovery:.4f}" if conf.consensus_result else "N/A",
                f"{conf.medoid_result.overall_recovery:.4f}" if conf.medoid_result else "N/A",
                f"{conf.mean_recovery:.4f}",
                f"{conf.std_recovery:.4f}",
                f"{conf.consensus_result.mpnn_score:.4f}" if conf.consensus_result else "N/A",
                f"{conf.medoid_result.mpnn_score:.4f}" if conf.medoid_result else "N/A",
            ]

            if rmsf_profiles and cid in rmsf_profiles:
                recovery = (
                    conf.consensus_result.per_residue_recovery
                    if conf.consensus_result
                    else conf.medoid_result.per_residue_recovery
                    if conf.medoid_result
                    else None
                )
                if recovery is not None:
                    comparison = compare_recovery_with_rmsf(
                        recovery, rmsf_profiles[cid]
                    )
                    q = comparison["quadrants"]
                    row.extend([
                        f"{q['Q1_stable_core']['fraction']:.4f}",
                        f"{q['Q2_flexible_valid']['fraction']:.4f}",
                        f"{q['Q3_suspicious_fixed']['fraction']:.4f}",
                        f"{q['Q4_poorly_predicted']['fraction']:.4f}",
                        f"{comparison['correlation']['pearson']:.4f}",
                    ])
                else:
                    row.extend(["N/A"] * 5)
            elif rmsf_profiles:
                row.extend(["N/A"] * 5)

            writer.writerow(row)

    print(f"  Confidence report saved: {summary_path}")

    # Per-residue comparison plots (if matplotlib available)
    if HAS_MPL and rmsf_profiles:
        _plot_confidence_heatmap(
            cluster_scores, rmsf_profiles, output_dir, target_name
        )

    return str(summary_path)


def _plot_confidence_heatmap(
    cluster_scores: Dict,
    rmsf_profiles: Dict[int, np.ndarray],
    output_dir: Path,
    target_name: str,
) -> None:
    """Plot per-residue recovery vs RMSF for each cluster."""
    if not HAS_MPL:
        return

    n_clusters = len(cluster_scores)
    if n_clusters == 0:
        return

    fig, axes = plt.subplots(
        n_clusters, 2,
        figsize=(14, 3 * n_clusters),
        squeeze=False,
    )

    for idx, (cid, conf) in enumerate(sorted(cluster_scores.items())):
        # Get recovery data
        recovery = None
        if conf.consensus_result:
            recovery = conf.consensus_result.per_residue_recovery
        elif conf.medoid_result:
            recovery = conf.medoid_result.per_residue_recovery

        rmsf = rmsf_profiles.get(cid)

        if recovery is None or rmsf is None:
            axes[idx, 0].text(0.5, 0.5, "No data", ha="center", va="center")
            axes[idx, 1].text(0.5, 0.5, "No data", ha="center", va="center")
            continue

        n = min(len(recovery), len(rmsf))
        residues = np.arange(1, n + 1)

        # Left: Per-residue profiles
        ax1 = axes[idx, 0]
        ax1_twin = ax1.twinx()
        ax1.bar(residues, recovery[:n], alpha=0.6, color="steelblue", label="Recovery")
        ax1_twin.plot(residues, rmsf[:n], color="red", linewidth=1.2, label="RMSF")
        ax1.set_ylabel("Recovery", color="steelblue")
        ax1_twin.set_ylabel("RMSF (Å)", color="red")
        ax1.set_xlabel("Residue")
        ax1.set_title(f"Cluster {cid}: Per-residue profile")
        ax1.set_ylim(0, 1.05)

        # Right: Scatter (Recovery vs RMSF)
        ax2 = axes[idx, 1]
        ax2.scatter(rmsf[:n], recovery[:n], s=8, alpha=0.6, c="steelblue")
        ax2.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
        ax2.axvline(2.0, color="gray", linestyle="--", alpha=0.5)
        ax2.set_xlabel("RMSF (Å)")
        ax2.set_ylabel("Recovery")
        ax2.set_title(f"Cluster {cid}: Recovery vs RMSF")

        # Label quadrants
        ax2.text(0.5, 0.85, "Q1: Stable core", fontsize=7, color="green",
                 transform=ax2.transAxes, ha="left")
        ax2.text(0.5, 0.15, "Q3: Suspicious", fontsize=7, color="orange",
                 transform=ax2.transAxes, ha="left")
        ax2.text(0.75, 0.85, "Q2: Flexible valid", fontsize=7, color="blue",
                 transform=ax2.transAxes, ha="left")
        ax2.text(0.75, 0.15, "Q4: Poorly predicted", fontsize=7, color="red",
                 transform=ax2.transAxes, ha="left")

        corr = np.corrcoef(rmsf[:n], recovery[:n])[0, 1] if n > 1 else 0
        ax2.set_title(f"Cluster {cid}: Recovery vs RMSF (r={corr:.3f})")

    plt.tight_layout()
    plot_path = output_dir / f"{target_name}_confidence_analysis.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Confidence plot saved: {plot_path}")
