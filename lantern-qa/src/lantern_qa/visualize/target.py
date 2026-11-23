#!/usr/bin/env python3
"""Analyze structure prediction results for a target directory.

Integrates all functionality from analyze_msa.py.

Three independently skippable phases (all run by default):
  1. plots  — contact, distogram, pLDDT, PAE, PDE, Ramachandran per run
  2. msa    — per-a3m PNGs in visualization/{Stoich}/msas/{variant}/
               + msa_summary.png overview grid (chains × MSA sources)
  3. csv    — all_rankings.csv, summary.csv, per_seed_summary.csv

Discovers every MSA variant (af3_msa, mmseqs2_gpu, plmmsa_otalign, …) and model
(af3, bt2, of3, pt2, …) under each stoichiometry. Rankings are keyed by
(stoichiometry, msa_variant, model, seed, sample).

Output mirrors the input path structure:
  target/{Chain}/results/{Stoich}/{msa_variant}/{model}/  →  target/visualization/{Stoich}/{msa_variant}/{model}/
  target/results/{stoich}/{msa_variant}/{model}/          →  target/visualization/{stoich}/{msa_variant}/{model}/

Usage:
    python analyze.py <target_dir>
    python analyze.py <target_dir> --no-plots
    python analyze.py <target_dir> --no-msa
    python analyze.py <target_dir> --no-csv
    python analyze.py <target_dir> --dpi 200 -v
"""
from __future__ import annotations

import logging
import re
import sys
from itertools import groupby
from pathlib import Path

import click
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# lantern-qa: part of the lantern_qa.visualize package (was /viz inside the af3-viz container)

from lantern_qa.visualize.reader import (
    discover_prediction_types,
    find_best_sample,
    find_chain_msa_a3ms,
    get_msa_paths,
    load_confidences,
    load_summary_confidences,
)
from lantern_qa.visualize.metrics import (
    compute_distogram,
    compute_msa_visualization,
    compute_pdockq_for_complex,
    compute_plddt_stats,
    compute_pde_stats,
    compute_ramachandran,
)
from lantern_qa.visualize.ranking import (
    build_per_seed_summary,
    build_ranking_dataframe,
    find_global_best,
)
from lantern_qa.visualize.plots import (
    plot_contact_heatmap,
    plot_distogram,
    plot_msa,
    plot_pae_heatmap,
    plot_pde_heatmap,
    plot_plddt,
    plot_ramachandran,
    save_figure,
)

log = logging.getLogger("analyze")

# --------------------------------------------------------------------------- #
# CSV column ordering                                                          #
# Each ranking row is identified by (stoich, msa_variant, model, seed, sample).#
# --------------------------------------------------------------------------- #
ID_COLS = ["stoich", "msa_variant", "model"]
RANKING_COLS = ID_COLS + [
    "seed", "sample", "ranking_score", "ptm", "iptm",
    "fraction_disordered", "has_clash", "inter_pde_mean",
]
SUMMARY_COLS = RANKING_COLS + ["rank"]
PER_SEED_COLS = ID_COLS + ["seed", "iptm", "ptm", "ranking_score", "fraction_disordered"]


def _ordered(df: "pd.DataFrame", cols: list[str]) -> "pd.DataFrame":
    """Return df with `cols` first (those present), then any remaining columns.

    Drops the internal `label` column from CSV output — the explicit
    stoich/msa_variant/model columns identify each row instead.
    """
    present = [c for c in cols if c in df.columns]
    extra = [c for c in df.columns if c not in present and c != "label"]
    return df[present + extra]


# =========================================================================== #
# Path helpers                                                                 #
# =========================================================================== #

def _run_output_dir(run: "PredictionRun", target_dir: Path) -> Path:
    """Mirror the results path under visualization/.

      target/{Chain}/results/{Stoich}/{msa_variant}/{model}  →  target/visualization/{Stoich}/{msa_variant}/{model}
      target/results/{stoich}/{msa_variant}/{model}          →  target/visualization/{stoich}/{msa_variant}/{model}
    """
    rel = run.af3_dir.relative_to(target_dir)
    parts = list(rel.parts)
    if parts and re.match(r"^[A-Z]$", parts[0]) and len(parts) > 1 and parts[1] == "results":
        parts = parts[1:]
    try:
        idx = parts.index("results")
        parts[idx] = "visualization"
    except ValueError:
        return target_dir / "visualization" / run.label
    return target_dir.joinpath(*parts)


def _stoich_viz_dir(run_dir: Path, target_dir: Path) -> Path:
    """Return the stoich-level visualization dir.

    e.g. target/visualization/A1/af3_msa/af3  →  target/visualization/A1
    """
    rel = run_dir.relative_to(target_dir / "visualization")
    return target_dir / "visualization" / rel.parts[0]


# =========================================================================== #
# MSA summary grid  (ported from analyze_msa.py)                              #
# =========================================================================== #

def _find_chain_msa_layout(target_dir: Path) -> list[tuple[str, Path]]:
    """Return [(chain_letter, chain_msas_dir), ...] for all single-letter chain dirs."""
    layout: list[tuple[str, Path]] = []
    for sub in sorted(target_dir.iterdir()):
        if sub.is_dir() and len(sub.name) == 1 and sub.name.isupper():
            msas = sub / "msas"
            if msas.is_dir():
                layout.append((sub.name, msas))
    if not layout:
        msas = target_dir / "msas"
        if msas.is_dir():
            layout.append(("_", msas))
    return layout


def _find_variants(chain_msas_dir: Path) -> list[tuple[str, Path]]:
    """Return [(variant_rel_path, variant_dir), ...] for dirs containing .a3m files.

    Handles nested layouts such as 'plmmsa/plmalign'.
    """
    seen: dict[str, Path] = {}
    for a3m in sorted(chain_msas_dir.rglob("*.a3m")):
        if not a3m.is_file():
            continue
        label = str(a3m.parent.relative_to(chain_msas_dir))
        if label not in seen:
            seen[label] = a3m.parent
    return [(lbl, d) for lbl, d in sorted(seen.items())]


def _draw_msa_into_ax(ax: plt.Axes, msa_data: dict, title: str) -> "plt.cm.ScalarMappable | None":
    """Draw MSA coverage heatmap into ax. Returns mappable for colorbar."""
    lines_raw = msa_data.get("lines", [])
    if not lines_raw:
        ax.text(0.5, 0.5, "no MSA", ha="center", va="center",
                transform=ax.transAxes, fontsize=8, color="gray")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=8)
        return None

    mat = np.array(lines_raw, dtype=np.float32)
    n_seqs, seq_len = mat.shape

    im = ax.imshow(mat, cmap="rainbow", vmin=0, vmax=1,
                   aspect="auto", origin="lower", interpolation="nearest")

    coverage = msa_data.get("coverage")
    cov_arr = np.array(coverage, dtype=float) if coverage is not None \
        else np.sum(~np.isnan(mat), axis=0).astype(float)
    ax.plot(np.arange(seq_len), cov_arr, color="black", linewidth=0.6)

    for pos in msa_data.get("ln", [])[1:-1]:
        ax.axvline(pos - 0.5, color="black", linewidth=1.0)

    ax.set_xlim(-0.5, seq_len - 0.5)
    ax.set_ylim(-0.5, n_seqs - 0.5)
    ax.tick_params(labelsize=7)
    ax.set_title(title, fontsize=8)
    return im


def _render_msa_summary(
    target_dir: Path,
    layout: list[tuple[str, Path]],
    out_path: Path,
    dpi: int,
) -> int:
    """Build a grid image: rows = (variant, msa_name), cols = chains.

    Output: visualization/msa_summary.png
    Returns number of cells rendered.
    """
    sources: set[tuple[str, str]] = set()
    chain_a3ms: dict[str, dict[tuple[str, str], Path]] = {}
    for chain_label, chain_msas_dir in layout:
        m: dict[tuple[str, str], Path] = {}
        for variant_label, variant_dir in _find_variants(chain_msas_dir):
            for a3m in sorted(variant_dir.glob("*.a3m")):
                key = (variant_label, a3m.stem)
                m[key] = a3m
                sources.add(key)
        chain_a3ms[chain_label] = m

    if not sources:
        log.warning("  [msa] summary: no a3m sources found, skipping")
        return 0

    sorted_sources = sorted(sources)
    chain_labels = [c for c, _ in layout]
    n_rows, n_cols = len(sorted_sources), len(chain_labels)

    fig_w = max(3.5 * n_cols, 8.0)
    fig_h = max(2.2 * n_rows, 4.0)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), squeeze=False)

    last_im = None
    for j, chain_label in enumerate(chain_labels):
        for i, (variant, msa_name) in enumerate(sorted_sources):
            ax = axes[i][j]
            a3m = chain_a3ms[chain_label].get((variant, msa_name))
            if a3m is None:
                ax.text(0.5, 0.5, "—", ha="center", va="center",
                        transform=ax.transAxes, fontsize=10, color="gray")
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                msa_data = compute_msa_visualization([a3m], chain_boundary_positions=None)
                n_seqs = len(msa_data["lines"])
                seq_len = len(msa_data["lines"][0]) if n_seqs else 0
                neff = msa_data["neff"]
                im = _draw_msa_into_ax(ax, msa_data,
                                       f"N={n_seqs} L={seq_len} Neff={neff:.2f}")
                if im is not None:
                    last_im = im
            if i == 0:
                ax.text(0.5, 1.18, f"chain {chain_label}",
                        transform=ax.transAxes, ha="center", va="bottom",
                        fontsize=11, fontweight="bold")
            if j == 0:
                ax.set_ylabel(f"{variant}/{msa_name}", fontsize=9)

    fig.suptitle(f"{target_dir.name} — MSA & template coverage summary", fontsize=13)
    if last_im is not None:
        fig.tight_layout(rect=[0.0, 0.0, 0.94, 0.96])
        cax = fig.add_axes([0.955, 0.10, 0.012, 0.78])
        cbar = fig.colorbar(last_im, cax=cax)
        cbar.set_label("Sequence identity to query", fontsize=9)
        cbar.ax.tick_params(labelsize=8)
    else:
        fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])

    save_figure(fig, out_path, dpi=dpi)
    return n_rows * n_cols


# =========================================================================== #
# Main command                                                                 #
# =========================================================================== #

@click.command()
@click.argument("target_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--dpi", default=150, show_default=True, help="PNG resolution in DPI.")
@click.option("--no-plots", is_flag=True, help="Skip result plots (contact/distogram/pLDDT/PAE/PDE/Ramachandran).")
@click.option("--no-msa",   is_flag=True, help="Skip MSA image generation.")
@click.option("--no-csv",   is_flag=True, help="Skip CSV generation.")
@click.option("-v", "--verbose", is_flag=True, help="Enable debug logging.")
def analyze(
    target_dir: Path,
    dpi: int,
    no_plots: bool,
    no_msa: bool,
    no_csv: bool,
    verbose: bool,
) -> None:
    """Analyze structure prediction results in TARGET_DIR."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    summary_dir = target_dir / "visualization"
    summary_dir.mkdir(parents=True, exist_ok=True)
    log.info("Summary output: %s", summary_dir)

    # ------------------------------------------------------------------ #
    # Discover runs                                                        #
    # ------------------------------------------------------------------ #
    log.info("Scanning %s for prediction types...", target_dir)
    runs = discover_prediction_types(target_dir)
    log.info(
        "Found %d prediction types: %s",
        len(runs),
        ", ".join(f"{r.label} ({'complex' if r.is_complex else 'monomer'})" for r in runs),
    )

    all_ranking_dfs: list[pd.DataFrame] = []
    all_per_seed_dfs: list[pd.DataFrame] = []

    # ------------------------------------------------------------------ #
    # Phase 1 (plots) + MSA combined — per-run loop                       #
    # ------------------------------------------------------------------ #
    for run in runs:
        label = run.label
        run_dir = _run_output_dir(run, target_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        log.info("=== %s  →  %s ===", label, run_dir.relative_to(target_dir))

        best_sf = find_best_sample(run)
        log.info("  Best sample: seed=%d sample=%d", best_sf.seed, best_sf.sample)

        conf = load_confidences(best_sf)
        summary = load_summary_confidences(best_sf.summary_path)

        plddt_stats = compute_plddt_stats(conf)
        log.info("  mean pLDDT: %.1f | fraction_disordered: %.2f",
                 plddt_stats["mean_plddt"], summary.get("fraction_disordered", float("nan")))

        pde_stats = compute_pde_stats(conf)
        log.info("  mean PDE: %.2f Å | chain_pair_pde_mean: %s",
                 pde_stats["mean_pde"],
                 {f"{ci}↔{cj}": f"{v:.2f}"
                  for (ci, cj), v in pde_stats["per_chain_pair_mean"].items() if ci <= cj})

        pdockq_val = compute_pdockq_for_complex(best_sf, summary)
        if pdockq_val is not None:
            log.info("  pDockQ: %.3f", pdockq_val)

        ranking_df = build_ranking_dataframe(run)
        all_ranking_dfs.append(ranking_df)
        all_per_seed_dfs.append(build_per_seed_summary(run))

        best_seed, best_sample_idx, best_score = find_global_best(ranking_df)
        log.info("  Global best: seed=%d sample=%d score=%.4f",
                 best_seed, best_sample_idx, best_score)

        # ---------- Phase 1: result plots ----------
        if not no_plots:
            log.info("  [plots] Contact probability...")
            fig = plot_contact_heatmap(conf, title=f"{label} — Contact Probability")
            save_figure(fig, run_dir / "contact_prob.png", dpi)

            log.info("  [plots] Distogram (all seeds)...")
            disto_list: list[tuple[int, np.ndarray, np.ndarray]] = []
            for seed_dir in sorted(run.seed_dirs, key=lambda d: int(d.name.split("_")[1])):
                s = int(seed_dir.name.split("_")[1])
                npy_path = seed_dir / "distogram_probs.npy"
                if npy_path.exists():
                    try:
                        dd = compute_distogram(npy_path)
                        disto_list.append((s, dd["distogram_expectation"], dd["distogram_argmax"]))
                    except Exception as e:
                        log.warning("    Distogram failed seed=%d: %s", s, e)
            if disto_list:
                fig = plot_distogram(
                    disto_list, confidences=conf,
                    title=f"{label} — Distogram (all seeds, 2 rows: expected / argmax)",
                )
                save_figure(fig, run_dir / "distogram.png", dpi)
            else:
                log.warning("  No distogram .npy found for %s", label)

            for seed_num, seed_iter in groupby(run.samples, key=lambda sf: sf.seed):
                seed_samples = list(seed_iter)
                log.info("  [plots] Seed %d (%d samples): pLDDT / PAE / PDE / Ramachandran...",
                         seed_num, len(seed_samples))

                sample_confs: list[tuple[int, dict]] = []
                for sf in seed_samples:
                    try:
                        sample_confs.append((sf.sample, load_confidences(sf)))
                    except Exception as e:
                        log.warning("    conf load failed sample=%d: %s", sf.sample, e)

                if sample_confs:
                    fig = plot_plddt(sample_confs, title=f"{label} Seed {seed_num} — pLDDT")
                    save_figure(fig, run_dir / f"plddt_seed{seed_num}.png", dpi)
                if sample_confs:
                    fig = plot_pae_heatmap(sample_confs, title=f"{label} Seed {seed_num} — PAE")
                    save_figure(fig, run_dir / f"pae_seed{seed_num}.png", dpi)
                if sample_confs:
                    fig = plot_pde_heatmap(sample_confs, title=f"{label} Seed {seed_num} — PDE")
                    save_figure(fig, run_dir / f"pde_seed{seed_num}.png", dpi)

                rama_panels: list[tuple[int, np.ndarray, np.ndarray, list[str]]] = []
                for sf in seed_samples:
                    try:
                        phi, psi, cids = compute_ramachandran(sf.model_cif_path)
                        if len(phi) > 0:
                            rama_panels.append((sf.sample, phi, psi, cids))
                    except Exception as e:
                        log.warning("    Ramachandran failed sample=%d: %s", sf.sample, e)
                if rama_panels:
                    fig = plot_ramachandran(
                        rama_panels, title=f"{label} Seed {seed_num} — Ramachandran")
                    save_figure(fig, run_dir / f"ramachandran_seed{seed_num}.png", dpi)

        # ---------- MSA combined per run (run_dir/msa.png) ----------
        if not no_msa:
            try:
                msa_paths, chain_bounds = get_msa_paths(run, target_dir)
                if msa_paths:
                    msa_data = compute_msa_visualization(msa_paths, chain_bounds)
                    fig = plot_msa(msa_data, title=f"{label} — MSA (Neff={msa_data['neff']:.1f})")
                    save_figure(fig, run_dir / "msa.png", dpi)
                    log.info("  [msa] combined → %s/msa.png", run_dir.relative_to(target_dir))
                else:
                    log.warning("  [msa] no a3m files found for %s", label)
            except Exception as e:
                log.warning("  [msa] combined plot failed: %s", e)

    # ------------------------------------------------------------------ #
    # Phase 2: MSA — per-a3m PNGs + summary grid                         #
    # ------------------------------------------------------------------ #
    if not no_msa:
        # Per-a3m: visualization/{Stoich}/msas/{variant}/{stem}.png
        processed_stoichs: set[str] = set()
        for run in runs:
            if run.is_complex or not run.chain_letter:
                continue
            run_dir = _run_output_dir(run, target_dir)
            stoich_viz = _stoich_viz_dir(run_dir, target_dir)
            stoich_key = stoich_viz.name
            if stoich_key in processed_stoichs:
                continue
            processed_stoichs.add(stoich_key)

            msa_entries = find_chain_msa_a3ms(target_dir, run.chain_letter)
            if not msa_entries:
                continue
            log.info("=== [msa] %s (chain %s): %d a3m files ===",
                     stoich_key, run.chain_letter, len(msa_entries))
            for variant, stem, a3m_path in msa_entries:
                out_dir = stoich_viz / "msas" / variant
                out_dir.mkdir(parents=True, exist_ok=True)
                out_png = out_dir / f"{stem}.png"
                if out_png.exists():
                    log.debug("  skip (exists): %s", out_png.relative_to(target_dir))
                    continue
                try:
                    msa_data = compute_msa_visualization([a3m_path], chain_boundary_positions=None)
                    n = len(msa_data["lines"])
                    neff = msa_data["neff"]
                    fig = plot_msa(
                        msa_data,
                        title=f"{run.chain_letter} {variant}/{stem}  N={n} Neff={neff:.1f}",
                    )
                    save_figure(fig, out_png, dpi)
                    log.info("  %s/%s → %s", variant, stem, out_png.relative_to(target_dir))
                except Exception as e:
                    log.warning("  MSA failed %s/%s: %s", variant, stem, e)

        # Summary grid: visualization/msa_summary.png
        layout = _find_chain_msa_layout(target_dir)
        if layout:
            summary_png = summary_dir / "msa_summary.png"
            log.info("=== [msa] summary grid: %d chains ===", len(layout))
            try:
                n_cells = _render_msa_summary(target_dir, layout, summary_png, dpi)
                log.info("  → visualization/msa_summary.png  (%d cells)", n_cells)
            except Exception as e:
                log.warning("  MSA summary grid failed: %s", e)

    # ------------------------------------------------------------------ #
    # Phase 3: CSV                                                         #
    # ------------------------------------------------------------------ #
    top5_dfs = []
    for df in all_ranking_dfs:
        top5 = df.nlargest(5, "ranking_score").reset_index(drop=True)
        top5["rank"] = range(1, len(top5) + 1)
        top5_dfs.append(top5)
    summary_df = pd.concat(top5_dfs, ignore_index=True)

    if not no_csv:
        log.info("=== [csv] Writing ranking files ===")
        for run in runs:
            label = run.label
            run_dir = _run_output_dir(run, target_dir)
            run_dir.mkdir(parents=True, exist_ok=True)
            ranking_df = next(df for df in all_ranking_dfs if df["label"].iloc[0] == label)
            ranking_df.sort_values("ranking_score", ascending=False, inplace=True)
            ranking_df.reset_index(drop=True, inplace=True)
            _ordered(ranking_df, RANKING_COLS).to_csv(run_dir / "all_rankings.csv", index=False)
            log.info("  %s/all_rankings.csv", run_dir.relative_to(target_dir))

        per_seed_df = pd.concat(all_per_seed_dfs, ignore_index=True)
        _ordered(per_seed_df, PER_SEED_COLS).to_csv(
            summary_dir / "per_seed_summary.csv", index=False)
        _ordered(summary_df, SUMMARY_COLS).to_csv(
            summary_dir / "summary.csv", index=False)
        log.info("  visualization/per_seed_summary.csv")
        log.info("  visualization/summary.csv")

    # ------------------------------------------------------------------ #
    # Terminal summary                                                     #
    # ------------------------------------------------------------------ #
    _print_summary(target_dir, runs, summary_df, summary_dir)


def _print_summary(
    target_dir: Path,
    runs: list,
    summary_df: pd.DataFrame,
    summary_dir: Path,
) -> None:
    print()
    print("=" * 72)
    print(f"  Target: {target_dir.name}  — Top 5 per prediction type")
    print("=" * 72)
    cols = ["stoich", "msa_variant", "model", "rank", "seed", "sample", "ranking_score",
            "ptm", "iptm", "fraction_disordered", "has_clash", "inter_pde_mean"]
    display_cols = [c for c in cols if c in summary_df.columns]
    fmt_df = summary_df[display_cols].copy()
    for col in ["ranking_score", "ptm", "iptm", "fraction_disordered", "inter_pde_mean"]:
        if col in fmt_df.columns:
            fmt_df[col] = fmt_df[col].map(lambda v: f"{v:.3f}" if pd.notna(v) else "—")
    print(fmt_df.to_string(index=False))
    print()
    print("  Per-run output directories:")
    for run in runs:
        print(f"    {run.label:25s}  {_run_output_dir(run, target_dir).relative_to(target_dir)}")
    print(f"  Summary: {summary_dir}")
    print("=" * 72)


if __name__ == "__main__":
    analyze()
