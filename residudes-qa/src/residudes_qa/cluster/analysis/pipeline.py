"""End-to-end cluster contact-map analysis pipeline.

Runs Phase 1 → 3b for a single set of cluster assignments in one call.
Used by the ``massivecluster cluster-analysis`` CLI subcommand.

Phases
------
1. **contact_map**   — epitope-side frequency / enrichment from ``contacts_long``
2. **signatures**    — per-cluster fingerprint MD reports + overview
3a. **1d_viz**       — top-residue bars, cluster×epitope heatmaps, enrichment grid,
                       physchem bars, freq_by_method (optional)
3b. **2d_viz**       — CDR × epitope 2D heatmaps + channel distribution
                       (requires pair-level extraction from CIFs + ANARCI)

Folder layout
-------------

::

    output/
      manifest.json
      phase1_contact_map/
        global/, per_cluster/cluster_*/data+figures/, cross_cluster/, manifest.json
      phase2_signatures/
        overview.md, cluster_*_fingerprint.md, signatures.json
      phase3a_1d_viz/
        cluster_*_top_epitope_bars.png, cluster_vs_epitope_*_heatmap.png,
        enrichment_grid.png, physchem_distribution.png, by_method/*.png (optional)
      phase3b_2d_viz/
        figures/cluster_*_cdr_epitope_2d.png, channel_distribution_per_cluster.png,
        data/pair_contacts.parquet, anarci_cdr_map.csv
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from .contact_map import (
    build_cluster_contact_maps,
    save_cluster_contact_maps,
)
from .cluster_signature import (
    extract_all_signatures,
    render_cluster_fingerprint_md,
    render_all_clusters_overview_md,
)


def run_cluster_analysis(
    *,
    cluster_assignments: pd.DataFrame,
    contacts_long: Optional[pd.DataFrame] = None,
    master_df: pd.DataFrame,
    output_dir: str | Path,
    target_name: str = "target",
    # cluster + model column names
    cluster_col: str = "cluster",
    model_col: str = "model_id",
    score_col: str = "xgb_pred",
    # phase 3a viz
    run_1d_viz: bool = True,
    run_freq_by_method: bool = True,
    # phase 3b viz (pair-level + ANARCI)
    run_2d_viz: bool = True,
    paratope_chains: Optional[Iterable[str]] = None,
    epitope_chains: Optional[Iterable[str]] = None,
    heavy_chain: Optional[str] = None,
    light_chain: Optional[str] = None,
    cif_col: str = "src_cif",
    pair_cutoff: float = 5.0,
    pair_sanitize_cutoff: float = 10.0,
    pair_workers: int = 8,
    top_n_para: int = 30,
    top_n_epi: int = 30,
    anarci_bin: str = "ANARCI",
    # phase 3a control
    top_n_residues_heatmap: int = 50,
) -> dict:
    """Run Phase 1-3b on one cluster set; return top-level manifest dict.

    Parameters
    ----------
    cluster_assignments
        Long table with ``model_col`` and ``cluster_col`` columns at minimum.
        For Phase 3b, must also include ``cif_col`` (path to model CIF).
    contacts_long
        Per-(model, epitope-residue) contact table. Must include
        ``abc_chain/abc_resnum`` OR ``ag_chain/ag_resnum`` columns. **Optional** —
        if omitted, it is generated from the cluster CIFs (requires
        ``paratope_chains``/``epitope_chains`` + a ``cif_col`` column); the same
        pair extraction is then reused for Phase 3b (no double work).
    master_df
        Master metrics parquet for per-cluster quality summary.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    top_manifest: dict = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "target": target_name,
        "n_models": int(cluster_assignments[model_col].nunique()),
        "n_clusters": int(cluster_assignments[cluster_col].nunique()),
        "phases": {},
    }

    # ------------------------------------------------------------------
    # PHASE 0 (auto) — derive contacts_long from CIFs if not supplied
    # ------------------------------------------------------------------
    precomputed_pair_df = None
    if contacts_long is None:
        if paratope_chains is None or epitope_chains is None:
            raise ValueError(
                "contacts_long not provided and cannot be derived: pass "
                "paratope_chains + epitope_chains (and a cif_col) to auto-generate it."
            )
        if cif_col not in cluster_assignments.columns:
            raise ValueError(
                f"contacts_long not provided and cluster_assignments lacks '{cif_col}' "
                "— cannot extract contacts from CIFs."
            )
        t0 = time.time()
        from .pair_contacts import extract_pair_contacts_batch
        print(f"  [phase0] no contacts_long given — extracting from "
              f"{cluster_assignments[model_col].nunique()} CIFs "
              f"({pair_workers} workers)…")
        precomputed_pair_df = extract_pair_contacts_batch(
            cluster_assignments[[cif_col, model_col]],
            paratope_chains=paratope_chains, epitope_chains=epitope_chains,
            cif_col=cif_col, model_id_col=model_col,
            cutoff=pair_cutoff, sanitize_cutoff=pair_sanitize_cutoff,
            workers=pair_workers,
        )
        if "_err" in precomputed_pair_df.columns:
            precomputed_pair_df = precomputed_pair_df[precomputed_pair_df["_err"].isna()]
        precomputed_pair_df = precomputed_pair_df.dropna(subset=["paratope_chain"])
        contacts_long = _pair_to_epitope_contacts(precomputed_pair_df, model_col=model_col)
        top_manifest["phases"]["phase0_contacts_from_cif"] = {
            "n_pair_rows": int(len(precomputed_pair_df)),
            "n_epitope_rows": int(len(contacts_long)),
            "elapsed_sec": round(time.time() - t0, 2),
        }

    # ------------------------------------------------------------------
    # PHASE 1 — epitope-side contact map
    # ------------------------------------------------------------------
    t0 = time.time()
    phase1_dir = output_dir / "phase1_contact_map"
    contact_maps = build_cluster_contact_maps(
        contacts_long, cluster_assignments,
        cluster_col=cluster_col, model_col=model_col,
    )
    save_cluster_contact_maps(contact_maps, phase1_dir)
    top_manifest["phases"]["phase1_contact_map"] = {
        "dir": str(phase1_dir),
        "manifest": str(phase1_dir / "manifest.json"),
        "elapsed_sec": round(time.time() - t0, 2),
        "n_clusters": len(contact_maps["clusters"]),
    }

    # ------------------------------------------------------------------
    # PHASE 2 — signatures (epitope + physchem from contacts_long)
    # ------------------------------------------------------------------
    t0 = time.time()
    phase2_dir = output_dir / "phase2_signatures"
    phase2_dir.mkdir(exist_ok=True)

    # Channel distribution (epitope-only, residue-type heuristic on antigen side)
    # — the *true* interaction channels come from pair-level in phase 3b.
    epi_channel_dist = None
    try:
        if any(c in contacts_long.columns for c in ("abc_resname", "ag_resname")):
            from .physchem_channels import epitope_residue_type_profile
            epi_channel_dist = epitope_residue_type_profile(
                contacts_long, cluster_assignments,
                cluster_col=cluster_col, model_col=model_col,
            )
            epi_channel_dist.to_csv(phase2_dir / "epitope_residue_type_profile.csv")
    except Exception as e:
        print(f"  [phase2] epitope residue-type profile skipped: {e}")

    signatures = extract_all_signatures(
        cluster_assignments, master_df, contact_maps,
        contacts_long=contacts_long,
        channel_distribution=epi_channel_dist,
        cluster_col=cluster_col, model_col=model_col, score_col=score_col,
    )
    sig_json = {int(k): _json_clean(v) for k, v in signatures.items()}
    (phase2_dir / "signatures.json").write_text(json.dumps(sig_json, indent=2))

    overview_path = phase2_dir / "overview.md"
    render_all_clusters_overview_md(signatures, overview_path, target_name=target_name)
    for cid, sig in signatures.items():
        render_cluster_fingerprint_md(sig, phase2_dir / f"cluster_{cid}_fingerprint.md")

    top_manifest["phases"]["phase2_signatures"] = {
        "dir": str(phase2_dir),
        "overview": str(overview_path),
        "elapsed_sec": round(time.time() - t0, 2),
    }

    # ------------------------------------------------------------------
    # PHASE 3a — 1D viz (epitope-side)
    # ------------------------------------------------------------------
    if run_1d_viz:
        t0 = time.time()
        phase3a_dir = output_dir / "phase3a_1d_viz"
        phase3a_dir.mkdir(exist_ok=True)
        _render_phase3a_viz(
            contact_maps=contact_maps,
            contacts_long=contacts_long,
            cluster_assignments=cluster_assignments,
            master_df=master_df,
            epi_channel_dist=epi_channel_dist,
            output_dir=phase3a_dir,
            cluster_col=cluster_col, model_col=model_col,
            top_n_residues_heatmap=top_n_residues_heatmap,
            run_freq_by_method=run_freq_by_method,
        )
        top_manifest["phases"]["phase3a_1d_viz"] = {
            "dir": str(phase3a_dir),
            "elapsed_sec": round(time.time() - t0, 2),
        }

    # ------------------------------------------------------------------
    # PHASE 3b — pair-level CDR × epitope 2D viz
    # ------------------------------------------------------------------
    if run_2d_viz:
        if paratope_chains is None or epitope_chains is None:
            print("  [phase3b] skipped: paratope_chains/epitope_chains required")
        elif cif_col not in cluster_assignments.columns:
            print(f"  [phase3b] skipped: cluster_assignments lacks '{cif_col}' column")
        else:
            t0 = time.time()
            phase3b_dir = output_dir / "phase3b_2d_viz"
            phase3b_dir.mkdir(exist_ok=True)
            phase3b_manifest = _run_phase3b(
                cluster_assignments=cluster_assignments,
                output_dir=phase3b_dir,
                paratope_chains=paratope_chains,
                epitope_chains=epitope_chains,
                heavy_chain=heavy_chain,
                light_chain=light_chain,
                cif_col=cif_col,
                model_col=model_col,
                cluster_col=cluster_col,
                pair_cutoff=pair_cutoff,
                pair_sanitize_cutoff=pair_sanitize_cutoff,
                pair_workers=pair_workers,
                top_n_para=top_n_para,
                top_n_epi=top_n_epi,
                anarci_bin=anarci_bin,
                precomputed_pair_df=precomputed_pair_df,
            )
            phase3b_manifest["elapsed_sec"] = round(time.time() - t0, 2)
            top_manifest["phases"]["phase3b_2d_viz"] = phase3b_manifest

    # ------------------------------------------------------------------
    (output_dir / "manifest.json").write_text(json.dumps(top_manifest, indent=2))
    return top_manifest


def _pair_to_epitope_contacts(pair_df: pd.DataFrame, *, model_col: str = "model_id") -> pd.DataFrame:
    """Aggregate pair-level contacts to the epitope side → contacts_long schema.

    Phase 1 needs a per-(model, epitope-residue) table with columns
    ``model_id, abc_chain, abc_resnum, abc_resname, n_atom_contacts, min_dist``.
    We collapse the pair table over paratope residues: a model contacts an
    epitope residue if any paratope residue does (sum atom counts, min distance).
    """
    g = (pair_df.groupby([model_col, "epitope_chain", "epitope_resnum"])
         .agg(abc_resname=("epitope_resname", "first"),
              n_atom_contacts=("n_atom_contacts", "sum"),
              min_dist=("min_dist", "min"))
         .reset_index())
    return g.rename(columns={"epitope_chain": "abc_chain",
                             "epitope_resnum": "abc_resnum"})


def _json_clean(obj):
    """Recursively convert numpy scalars / np.nan for JSON serialization."""
    if isinstance(obj, dict):
        return {str(k): _json_clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_clean(v) for v in obj]
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if np.isnan(v) else v
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


def _render_phase3a_viz(
    *,
    contact_maps: dict,
    contacts_long: pd.DataFrame,
    cluster_assignments: pd.DataFrame,
    master_df: pd.DataFrame,
    epi_channel_dist,
    output_dir: Path,
    cluster_col: str,
    model_col: str,
    top_n_residues_heatmap: int,
    run_freq_by_method: bool,
) -> None:
    """Render the Phase 3a 1D viz layer (epitope-side)."""
    from ..visualization.contact_map_1d import (
        plot_cluster_epitope_bars,
        plot_cluster_epitope_heatmap,
        plot_enrichment_grid,
        plot_physchem_bars,
        plot_all_clusters_freq_by_method,
    )

    F_global_by_res = dict(zip(
        contact_maps["global"]["epitope_residues"],
        contact_maps["global"]["F_binary"],
    ))

    # per-cluster top epitope bars
    for cid, cmap in contact_maps["clusters"].items():
        epi = cmap["epitope_residues"]
        F = cmap["F_binary"]
        F_g = np.array([F_global_by_res.get(r, 0.0) for r in epi], dtype=np.float32)
        enrichment = F - F_g
        plot_cluster_epitope_bars(
            cluster_id=cid,
            epitope_residues=epi,
            F_binary=F,
            enrichment=enrichment,
            output_path=output_dir / f"cluster_{cid}_top_epitope_bars.png",
        )

    # cluster × epitope heatmap (frequency + enrichment)
    F_matrix = contact_maps["F_binary_matrix"]
    cluster_ids = contact_maps["cluster_ids"]
    universe = contact_maps["epitope_universe"]
    F_global_aligned = np.array(
        [F_global_by_res.get(r, 0.0) for r in universe], dtype=np.float32,
    )
    plot_cluster_epitope_heatmap(
        F_matrix, cluster_ids, universe,
        output_path=output_dir / "cluster_vs_epitope_frequency_heatmap.png",
        mode="frequency", top_n_residues=top_n_residues_heatmap,
    )
    plot_cluster_epitope_heatmap(
        F_matrix, cluster_ids, universe,
        output_path=output_dir / "cluster_vs_epitope_enrichment_heatmap.png",
        mode="enrichment", F_global=F_global_aligned,
        top_n_residues=top_n_residues_heatmap,
    )

    # enrichment grid (per-cluster small-multiples)
    plot_enrichment_grid(
        contact_maps["clusters"],
        global_F=contact_maps["global"]["F_binary"],
        global_residues=contact_maps["global"]["epitope_residues"],
        output_path=output_dir / "enrichment_grid.png",
    )

    # physchem epitope-side bars (residue-type profile from phase 2)
    if epi_channel_dist is not None:
        try:
            plot_physchem_bars(
                epi_channel_dist,
                output_path=output_dir / "physchem_distribution.png",
            )
        except Exception as e:
            print(f"  [phase3a] physchem bars skipped: {e}")

    # freq_by_method (per-cluster, one method-colored line per cluster)
    if run_freq_by_method and "method" in master_df.columns:
        by_method_dir = output_dir / "by_method"
        by_method_dir.mkdir(exist_ok=True)
        try:
            plot_all_clusters_freq_by_method(
                contacts_long=contacts_long,
                cluster_assignments=cluster_assignments,
                master_df=master_df,
                output_dir=by_method_dir,
                cluster_col=cluster_col, model_col=model_col,
            )
        except Exception as e:
            print(f"  [phase3a] freq_by_method skipped: {e}")


def _run_phase3b(
    *,
    cluster_assignments: pd.DataFrame,
    output_dir: Path,
    paratope_chains: Iterable[str],
    epitope_chains: Iterable[str],
    heavy_chain: Optional[str],
    light_chain: Optional[str],
    cif_col: str,
    model_col: str,
    cluster_col: str,
    pair_cutoff: float,
    pair_sanitize_cutoff: float,
    pair_workers: int,
    top_n_para: int,
    top_n_epi: int,
    anarci_bin: str,
    precomputed_pair_df=None,
) -> dict:
    """Pair-level extraction + 2D viz.

    ``precomputed_pair_df`` — if Phase 0 already extracted pair contacts (because
    no contacts_long was supplied), reuse it instead of re-parsing every CIF.
    """
    from .pair_contacts import (
        extract_pair_contacts_batch, build_cluster_pair_contact_map,
    )
    from .anarci_cdr import build_residue_cdr_map
    from .physchem_channels import annotate_contacts
    from ..visualization.contact_map_2d import (
        plot_all_clusters_2d, plot_cluster_channel_distribution,
    )

    data_dir = output_dir / "data"
    data_dir.mkdir(exist_ok=True)
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    # 1. ANARCI on first representative model — Ab chains share numbering
    cdr_map: dict = {}
    if heavy_chain or light_chain:
        try:
            ref_cif = cluster_assignments[cif_col].iloc[0]
            cdr_map = build_residue_cdr_map(
                ref_cif,
                heavy_chain=heavy_chain, light_chain=light_chain,
                anarci_bin=anarci_bin,
            )
            cdr_df = pd.DataFrame([
                {"chain": c, "resnum": r, "cdr_loop": loop}
                for (c, r), loop in cdr_map.items()
            ]).sort_values(["chain", "resnum"])
            cdr_df.to_csv(data_dir / "anarci_cdr_map.csv", index=False)
        except Exception as e:
            print(f"  [phase3b] ANARCI failed: {e} (continuing without CDR strip)")
            cdr_map = {}

    # 2. Pair contacts (reuse Phase 0 extraction if available, else parse CIFs)
    if precomputed_pair_df is not None:
        print(f"  [phase3b] reusing {len(precomputed_pair_df)} pair rows from phase 0 "
              "(no re-extraction)")
        pair_df = precomputed_pair_df
    else:
        print(f"  [phase3b] extracting pair contacts for "
              f"{cluster_assignments[model_col].nunique()} models with "
              f"{pair_workers} workers…")
        pair_df = extract_pair_contacts_batch(
            cluster_assignments[[cif_col, model_col]],
            paratope_chains=paratope_chains,
            epitope_chains=epitope_chains,
            cif_col=cif_col, model_id_col=model_col,
            cutoff=pair_cutoff, sanitize_cutoff=pair_sanitize_cutoff,
            workers=pair_workers,
        )
        if "_err" in pair_df.columns:
            n_err = pair_df["_err"].notna().sum()
            if n_err:
                print(f"  [phase3b] {n_err} models errored out")
                pair_df = pair_df[pair_df["_err"].isna()]
        pair_df = pair_df.dropna(subset=["paratope_chain"])
    pair_df.to_parquet(data_dir / "pair_contacts.parquet")

    # 3. Cluster pair contact map
    cluster_pair_maps = build_cluster_pair_contact_map(
        pair_df,
        cluster_assignments[[model_col, cluster_col]],
        cluster_col=cluster_col, model_col=model_col,
    )

    # 4. 2D heatmaps
    paths = plot_all_clusters_2d(
        cluster_pair_maps, fig_dir, cdr_map=cdr_map or None,
        top_n_para=top_n_para, top_n_epi=top_n_epi,
    )

    # 4b. Per-cluster top-pairs table: freq + geometric consistency
    #     (mean ± std min-dist) + CDR loop + chemistry channel.
    _write_top_pairs_tables(
        cluster_pair_maps, pair_df, cdr_map, data_dir,
    )

    # 5. Pair-level physchem channel distribution
    pair_annot = annotate_contacts(
        pair_df,
        paratope_aa_col="paratope_resname",
        epitope_aa_col="epitope_resname",
    )
    ch_path = plot_cluster_channel_distribution(
        pair_annot,
        cluster_assignments[[model_col, cluster_col]],
        output_path=fig_dir / "channel_distribution_per_cluster.png",
        cluster_col=cluster_col, model_col=model_col,
    )

    return {
        "dir": str(output_dir),
        "data_dir": str(data_dir),
        "fig_dir": str(fig_dir),
        "n_models_with_pairs": int(pair_df[model_col].nunique()),
        "n_pair_rows": int(len(pair_df)),
        "n_paratope_residues": len(cluster_pair_maps["paratope_universe"]),
        "n_epitope_residues": len(cluster_pair_maps["epitope_universe"]),
        "anarci_n_residues": len(cdr_map),
        "anarci_loop_counts": (
            pd.Series(list(cdr_map.values())).value_counts().to_dict()
            if cdr_map else {}
        ),
        "heatmaps": {int(k): str(v) for k, v in paths.items()},
        "channel_distribution": str(ch_path),
    }


def _write_top_pairs_tables(cluster_pair_maps, pair_df, cdr_map, data_dir):
    """Write per-cluster top-contact tables with geometry + chemistry.

    Columns: paratope, epitope, cdr_loop, channels, freq, n_models,
    mean_min_dist, std_min_dist. Sorted by frequency. This is the concrete
    "is the contact real / how tight / what chemistry" table.
    """
    from .physchem_channels import classify_pair, to_one_letter

    # paratope (chain, resnum) -> resname, epitope likewise, for chemistry.
    para_name = (pair_df.drop_duplicates(["paratope_chain", "paratope_resnum"])
                 .set_index(["paratope_chain", "paratope_resnum"])["paratope_resname"]
                 .to_dict())
    epi_name = (pair_df.drop_duplicates(["epitope_chain", "epitope_resnum"])
                .set_index(["epitope_chain", "epitope_resnum"])["epitope_resname"]
                .to_dict())

    def _parse(label):
        chain, num = label.split(".")
        return chain, int(num)

    for cid, cm in cluster_pair_maps["clusters"].items():
        rows = []
        for tp in cm["top_pairs"]:
            pc, pr = _parse(tp["paratope"])
            ec, er = _parse(tp["epitope"])
            loop = cdr_map.get((pc, pr), "") if cdr_map else ""
            p_aa = to_one_letter(para_name.get((pc, pr), ""))
            e_aa = to_one_letter(epi_name.get((ec, er), ""))
            channels = classify_pair(p_aa, e_aa) if (p_aa and e_aa) else []
            rows.append({
                "paratope": tp["paratope"],
                "paratope_aa": p_aa,
                "cdr_loop": loop,
                "epitope": tp["epitope"],
                "epitope_aa": e_aa,
                "channels": ";".join(channels),
                "freq": round(tp["freq"], 3),
                "n_models": tp["n_models"],
                "mean_min_dist": (round(tp["mean_min_dist"], 2)
                                  if tp["mean_min_dist"] == tp["mean_min_dist"] else None),
                "std_min_dist": (round(tp["std_min_dist"], 2)
                                 if tp["std_min_dist"] == tp["std_min_dist"] else None),
            })
        pd.DataFrame(rows).to_csv(data_dir / f"cluster_{cid}_top_pairs.csv", index=False)


__all__ = ["run_cluster_analysis"]
