"""Per-cluster signature: rich metadata bundle for reporting / decision support.

Combines:
* contact map enrichment (from :mod:`contact_map`)
* physicochemical channel distribution (from :mod:`physchem_channels`)
* xgb / iptm / clash quality aggregates (from master DataFrame)
* method dominance / variant breakdown
* COM geometry (cluster center, spread)
* paratope CDR loop dominance (if ANARCI / pair-contact data available)

The resulting dict drives:
* :func:`render_cluster_fingerprint_md` — single-cluster MD report
* :func:`render_all_clusters_overview_md` — multi-cluster overview
* downstream submission picker (uses ``recommended_pick_model_id``)
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def _safe_mode(series: pd.Series, default=None):
    if series is None or len(series.dropna()) == 0:
        return default
    return series.mode().iloc[0]


def extract_cluster_signature(
    cluster_id: int,
    cluster_assignments: pd.DataFrame,
    master_df: pd.DataFrame,
    contact_maps: dict,
    *,
    contacts_long: Optional[pd.DataFrame] = None,
    channel_distribution: Optional[pd.DataFrame] = None,
    cluster_col: str = "cluster",
    model_col: str = "model_id",
    score_col: str = "xgb_pred",
) -> dict:
    """Build a rich signature dict for one cluster.

    Parameters
    ----------
    contact_maps
        Output of :func:`contact_map.build_cluster_contact_maps`.
    contacts_long
        Optional — needed for top epitope residue list with enrichment.
    channel_distribution
        Optional DataFrame from :func:`physchem_channels.cluster_channel_distribution`.

    Returns
    -------
    dict — all keys safe for json serialization.
    """
    members = cluster_assignments[cluster_assignments[cluster_col] == cluster_id]
    member_ids = members[model_col].tolist()
    master_sub = master_df[master_df[model_col].isin(member_ids)] if model_col in master_df.columns else pd.DataFrame()

    sig: dict = {
        "cluster_id": int(cluster_id),
        "n_models": len(member_ids),
    }

    # Method / variant dominance
    if "method" in master_sub.columns:
        sig["method_dominance"] = master_sub["method"].value_counts(normalize=True).to_dict()
    if "variant" in master_sub.columns:
        sig["variant_dominance"] = master_sub["variant"].value_counts(normalize=True).to_dict()

    # Quality aggregates
    for col in ("xgb_v8_pred", "xgb_pred", "json_iptm", "json_ranking_score",
                "iface_pdockq", "mpnn_score", "n_inter_clash_17",
                "aadars_frac_framework", "aadars_frac_cdr_h3"):
        if col in master_sub.columns:
            s = master_sub[col].dropna()
            if len(s):
                sig.setdefault("quality", {})[col] = {
                    "mean": float(s.mean()),
                    "median": float(s.median()),
                    "max": float(s.max()),
                    "min": float(s.min()),
                    "top5_mean": float(s.nlargest(5).mean()) if len(s) >= 5 else float(s.mean()),
                }

    # COM geometry (if cluster_assignments has fab_com cols)
    for col_set in [("fab_com_B_x", "fab_com_B_y", "fab_com_B_z"),
                    ("com_x", "com_y", "com_z")]:
        if all(c in master_sub.columns for c in col_set):
            xyz = master_sub[list(col_set)].to_numpy(dtype=np.float32)
            if len(xyz):
                sig["fab_com_center"] = [float(xyz[:, 0].mean()),
                                         float(xyz[:, 1].mean()),
                                         float(xyz[:, 2].mean())]
                sig["fab_com_spread_med_A"] = float(np.median(np.linalg.norm(
                    xyz - xyz.mean(0), axis=1)))
            break

    # Recommended pick — top score model among CDR-clean / atom-OK if possible
    if score_col in master_sub.columns:
        candidates = master_sub.dropna(subset=[score_col])
        # prefer CDR-clean
        if "aadars_frac_framework" in candidates.columns:
            clean = candidates[
                (candidates["aadars_frac_framework"] <= 0.30)
                & (candidates.get("aadars_frac_cdr_h3", pd.Series([0.3]*len(candidates))) >= 0.20)
            ]
            if len(clean):
                candidates = clean
        if len(candidates):
            top = candidates.nlargest(1, score_col).iloc[0]
            sig["recommended_pick"] = {
                "model_id": str(top[model_col]),
                **{k: (float(top[k]) if pd.api.types.is_numeric_dtype(type(top[k])) else str(top[k]))
                   for k in ("variant", "method", "seed", "sample", score_col,
                             "json_iptm", "json_ranking_score") if k in top.index},
            }

    # Top epitope residues (from contact maps)
    if cluster_id in contact_maps.get("clusters", {}):
        cmap = contact_maps["clusters"][cluster_id]
        F_global_by_res = dict(zip(contact_maps["global"]["epitope_residues"],
                                     contact_maps["global"]["F_binary"]))
        F_global_aligned = np.array(
            [F_global_by_res.get(r, 0.0) for r in cmap["epitope_residues"]],
            dtype=np.float32,
        )
        delta = cmap["F_binary"] - F_global_aligned
        order = np.argsort(-delta)
        top_eps = []
        for i in order[:15]:
            res = cmap["epitope_residues"][i]
            top_eps.append({
                "chain": str(res[0]),
                "resnum": int(res[1]),
                "label": f"{res[0]}.{res[1]}",
                "freq": float(cmap["F_binary"][i]),
                "enrichment": float(delta[i]),
                "mean_min_dist": float(cmap["D_min"][i]) if not np.isnan(cmap["D_min"][i]) else None,
            })
        sig["top_epitope_residues"] = top_eps
        # primary epitope chain — enrichment-weighted (not majority count).
        # Sum of enrichment per chain across top-10 residues; chain with
        # largest summed enrichment wins. This correctly captures the case
        # where one chain has fewer top residues but each has very high
        # specificity (H1335 cluster 4: A.423 +0.36 dominates over C.130-C.48).
        chain_weights: dict[str, float] = {}
        for e in top_eps[:10]:
            chain_weights[e["chain"]] = chain_weights.get(e["chain"], 0.0) + max(e["enrichment"], 0.0)
        if chain_weights:
            sig["primary_epitope_chain"] = max(chain_weights, key=chain_weights.get)
            sig["epitope_chain_weights"] = {k: round(v, 3) for k, v in chain_weights.items()}

    # Physicochemical channels
    if channel_distribution is not None and cluster_id in channel_distribution.index:
        sig["physchem"] = channel_distribution.loc[cluster_id].to_dict()

    # primary_epi_chain from master (H1335 has this)
    if "primary_epi_chain" in master_sub.columns:
        sig["primary_epi_chain_master"] = _safe_mode(master_sub["primary_epi_chain"], default=None)

    return sig


def extract_all_signatures(
    cluster_assignments: pd.DataFrame,
    master_df: pd.DataFrame,
    contact_maps: dict,
    *,
    contacts_long: Optional[pd.DataFrame] = None,
    channel_distribution: Optional[pd.DataFrame] = None,
    cluster_col: str = "cluster",
    model_col: str = "model_id",
    score_col: str = "xgb_pred",
) -> dict[int, dict]:
    """Build signatures for all clusters."""
    out = {}
    for cid in sorted(cluster_assignments[cluster_col].unique()):
        out[int(cid)] = extract_cluster_signature(
            cid, cluster_assignments, master_df, contact_maps,
            contacts_long=contacts_long,
            channel_distribution=channel_distribution,
            cluster_col=cluster_col, model_col=model_col, score_col=score_col,
        )
    return out


def render_cluster_fingerprint_md(signature: dict, output_path: str | Path) -> Path:
    """Auto-generate a single-cluster fingerprint MD report."""
    lines: list[str] = []
    cid = signature.get("cluster_id", "?")
    n = signature.get("n_models", 0)
    lines.append(f"# Cluster {cid} fingerprint")
    lines.append("")
    lines.append(f"- n_models: **{n}**")

    if "fab_com_center" in signature:
        x, y, z = signature["fab_com_center"]
        sp = signature.get("fab_com_spread_med_A", float("nan"))
        lines.append(f"- Fab COM (gH frame): ({x:.1f}, {y:.1f}, {z:.1f}), spread {sp:.1f} Å")

    if "primary_epitope_chain" in signature:
        lines.append(f"- Primary epitope chain: **{signature['primary_epitope_chain']}**")

    if "method_dominance" in signature:
        items = sorted(signature["method_dominance"].items(), key=lambda kv: -kv[1])
        lines.append("- Method dominance: " + ", ".join(f"{k} ({v*100:.0f}%)" for k, v in items[:4]))
    if "variant_dominance" in signature:
        items = sorted(signature["variant_dominance"].items(), key=lambda kv: -kv[1])
        lines.append("- Variant dominance: " + ", ".join(f"{k} ({v*100:.0f}%)" for k, v in items[:4]))

    if "recommended_pick" in signature:
        rp = signature["recommended_pick"]
        lines.append("")
        lines.append("## Recommended pick")
        lines.append(f"- `{rp.get('variant')}/{rp.get('method')}` "
                     f"seed={int(rp.get('seed', 0))} sample={int(rp.get('sample', 0))}")
        for k in ("xgb_pred", "xgb_v8_pred", "json_iptm", "json_ranking_score"):
            if k in rp:
                lines.append(f"  - {k}: {rp[k]:.3f}")

    if "quality" in signature:
        lines.append("")
        lines.append("## Quality metrics")
        lines.append("| Metric | top1 | top5 mean | median | n |")
        lines.append("|---|---|---|---|---|")
        for k, v in signature["quality"].items():
            lines.append(f"| {k} | {v['max']:.3f} | {v['top5_mean']:.3f} | {v['median']:.3f} | — |")

    if "top_epitope_residues" in signature:
        lines.append("")
        lines.append("## Top epitope residues (by enrichment)")
        lines.append("| Residue | Freq in cluster | ΔF vs all | mean min-dist (Å) |")
        lines.append("|---|---|---|---|")
        for e in signature["top_epitope_residues"][:10]:
            md = e["mean_min_dist"]
            md_s = f"{md:.2f}" if md is not None else "—"
            lines.append(
                f"| {e['label']} | {e['freq']:.2f} | {e['enrichment']:+.2f} | {md_s} |"
            )

    if "physchem" in signature:
        lines.append("")
        lines.append("## Physicochemical channel distribution")
        ch = signature["physchem"]
        for k in ("hydrophobic", "salt_bridge", "polar_hbond", "aromatic", "cation_pi"):
            if k in ch:
                lines.append(f"- {k}: {ch[k]*100:.1f}%")

    output_path = Path(output_path)
    output_path.write_text("\n".join(lines))
    return output_path


def render_all_clusters_overview_md(
    signatures: dict[int, dict],
    output_path: str | Path,
    target_name: Optional[str] = None,
) -> Path:
    """Multi-cluster overview report."""
    lines: list[str] = []
    title = f"# {target_name or 'Target'} — cluster overview"
    lines.append(title)
    lines.append("")
    n_total = sum(s.get("n_models", 0) for s in signatures.values())
    lines.append(f"- Total clusters: **{len(signatures)}**")
    lines.append(f"- Total models: **{n_total}**")
    lines.append("")

    lines.append("## Cluster summary")
    lines.append("| Cluster | n | Primary epitope | Top method | Top xgb | iptm top | Pick |")
    lines.append("|---|---|---|---|---|---|---|")
    for cid in sorted(signatures.keys()):
        sig = signatures[cid]
        method_top = "—"
        if "method_dominance" in sig:
            it = sorted(sig["method_dominance"].items(), key=lambda kv: -kv[1])[:1]
            if it:
                method_top = f"{it[0][0]} ({it[0][1]*100:.0f}%)"
        xgb_top = sig.get("quality", {}).get("xgb_pred", {}).get("max", float("nan"))
        if np.isnan(xgb_top):
            xgb_top = sig.get("quality", {}).get("xgb_v8_pred", {}).get("max", float("nan"))
        iptm_top = sig.get("quality", {}).get("json_iptm", {}).get("max", float("nan"))
        pick = sig.get("recommended_pick", {})
        pick_s = (f"{pick.get('variant', '?')}/{pick.get('method', '?')}"
                  f" s{int(pick.get('seed', 0))}-{int(pick.get('sample', 0))}"
                  if pick else "—")
        lines.append(
            f"| {cid} | {sig.get('n_models')} | "
            f"{sig.get('primary_epitope_chain', '—')} | {method_top} | "
            f"{xgb_top:.3f} | {iptm_top:.3f} | {pick_s} |"
        )

    lines.append("")
    lines.append("## Per-cluster details")
    for cid in sorted(signatures.keys()):
        sig = signatures[cid]
        top = sig.get("top_epitope_residues", [])
        top_labels = ", ".join(
            f"{e['label']}({e['enrichment']:+.2f})" for e in top[:5]
        )
        lines.append(f"- **Cluster {cid}** (n={sig.get('n_models')}): top epitope → {top_labels}")

    output_path = Path(output_path)
    output_path.write_text("\n".join(lines))
    return output_path


__all__ = [
    "extract_cluster_signature",
    "extract_all_signatures",
    "render_cluster_fingerprint_md",
    "render_all_clusters_overview_md",
]
