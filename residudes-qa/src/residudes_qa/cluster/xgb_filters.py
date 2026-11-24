"""MassiveClustering — XGBDimerScorer filter/score bridge.

Adds 6 new ``massivecluster`` subcommands that wrap XGBDimerScorer:

* ``massivecluster filter-clash``           — 1.7 Å heavy-atom clash audit (disulfide-aware)
* ``massivecluster filter-cdr-clean``       — Ab paratope sanity (Fab framework/H3 fractions)
* ``massivecluster filter-strict``          — full STRICT cascade (CDR-clean + clash + atom + gO)
* ``massivecluster filter-regional-rmsd``   — anchor-aligned probe RMSD vs reference (template check)
* ``massivecluster filter-atom-integrity``  — drop pt2-style terminal-O outliers
* ``massivecluster join-xgb``               — join xgb_v8 / xgb_ppi scores onto clustering output

Each subcommand reads/writes parquet on a ``--master`` (or ``--clustering``)
file and is non-destructive (output to ``-o``).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional

# Bring XGBDimer in via PYTHONPATH (set by wrapper) or analysis/ sibling layout.
# residudes-qa: XGBDimerScorer is residudes_qa.interface in the same package; no sys.path needed.


# ---------- argparse setup ---------------------------------------------------


def register_filter_subcommands(subparsers: Any) -> None:
    """Attach all xgb-aware filter/score subcommands to a subparser group."""

    # clash --------------------------------------------------------------
    p = subparsers.add_parser(
        "filter-clash",
        help="Per-model 1.7 Å heavy-atom clash audit (disulfide-aware)",
    )
    p.add_argument("--master", required=True, help="parquet with model_id + src_cif columns")
    p.add_argument("--cif-col", default="src_cif")
    p.add_argument("--threshold", type=float, default=1.7, help="clash distance (Å)")
    p.add_argument("--disulfide-max", type=float, default=2.5, help="exclude CYS-SG pairs ≤ this (Å)")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--max-inter", type=int, default=None,
                   help="if set, also write filtered subset with inter-chain clash ≤ N")
    p.add_argument("-o", "--output", required=True, help="output parquet path")
    p.set_defaults(func=cmd_filter_clash)

    # cdr-clean -----------------------------------------------------------
    p = subparsers.add_parser(
        "filter-cdr-clean",
        help="Apply CDR-clean filter (Ab paratope: framework ≤30%%, H3 ≥20%%, ...)",
    )
    p.add_argument("--master", required=True, help="parquet with aADARS columns")
    p.add_argument("--frac-fr-max", type=float, default=0.30)
    p.add_argument("--frac-h3-min", type=float, default=0.20)
    p.add_argument("--iface-contacts-min", type=int, default=100)
    p.add_argument("--mpnn-iface-min", type=int, default=20)
    p.add_argument("-o", "--output", required=True)
    p.set_defaults(func=cmd_filter_cdr_clean)

    # strict --------------------------------------------------------------
    p = subparsers.add_parser(
        "filter-strict",
        help="STRICT cascade: CDR-clean + atom integrity + no-of3 + gO core ≤6 Å + clash",
    )
    p.add_argument("--master", required=True)
    p.add_argument("--mode", default="ab", choices=["ab", "ppi"])
    p.add_argument("--exclude-methods", default="of3", help="comma-separated, default: of3")
    p.add_argument("--go-core-max", type=float, default=6.0,
                   help="(ab mode) gO core RMSD threshold if column present")
    p.add_argument("-o", "--output", required=True)
    p.set_defaults(func=cmd_filter_strict)

    # regional-rmsd -------------------------------------------------------
    p = subparsers.add_parser(
        "filter-regional-rmsd",
        help="Per-model anchor-aligned probe RMSD vs reference (template check)",
    )
    p.add_argument("--master", required=True, help="parquet with src_cif column")
    p.add_argument("--cif-col", default="src_cif")
    p.add_argument("--ref-cif", required=True, help="reference structure (e.g. 7LBE.cif)")
    p.add_argument("--anchor-chain-pred", required=True, help="anchor chain in predicted models (e.g. B for gH)")
    p.add_argument("--anchor-chain-ref",  required=True, help="anchor chain in reference (e.g. A for 7LBE gH)")
    p.add_argument("--probe-chain-pred",  required=True, help="probe chain in predicted models (e.g. A for gO)")
    p.add_argument("--probe-chain-ref",   required=True, help="probe chain in reference (e.g. C for 7LBE gO)")
    p.add_argument("--regions", default=None,
                   help="comma-separated 'name:start-end' (e.g. 'Nterm:1-80,core:200-440')")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("-o", "--output", required=True)
    p.set_defaults(func=cmd_filter_regional_rmsd)

    # atom-integrity ------------------------------------------------------
    p = subparsers.add_parser(
        "filter-atom-integrity",
        help="Drop models with terminal-O / similar atom outliers (pt2-style bug)",
    )
    p.add_argument("--master", required=True)
    p.add_argument("--cif-col", default="src_cif")
    p.add_argument("--max-outlier-dist", type=float, default=10.0,
                   help="any atom > this Å from its residue CA flags the model")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("-o", "--output", required=True)
    p.set_defaults(func=cmd_filter_atom_integrity)

    # join-xgb ------------------------------------------------------------
    p = subparsers.add_parser(
        "join-xgb",
        help="Join xgb_v8 or xgb_ppi_v3 scores onto an existing clustering parquet",
    )
    p.add_argument("--clustering", required=True, help="parquet with model_id (clustering output)")
    p.add_argument("--xgb-scores", required=True, help="parquet with model_id + xgb_pred")
    p.add_argument("--on", default="model_id",
                   help="preferred join key (default: model_id). If absent on either "
                        "side, falls back to known id columns: clustering "
                        "model_id/model/structure, xgb model_id/candidate_uid")
    p.add_argument("-o", "--output", required=True)
    p.set_defaults(func=cmd_join_xgb)


# ---------- command implementations ------------------------------------------


def _load_parquet(path: str):
    """Load a tabular artifact as a DataFrame.

    Accepts parquet or CSV. This keeps the bridge working with older
    MassiveClustering runs that only emitted ``*_clustering.csv`` (no parquet):
    if the given path is a ``.csv`` — or a ``.parquet`` that is missing but has a
    ``.csv`` sibling — we read the CSV and materialise the parquet next to it so
    the conversion happens once, on demand.
    """
    import pandas as pd
    from pathlib import Path

    p = Path(path)

    if p.suffix == ".csv" or (not p.exists() and p.with_suffix(".csv").exists()):
        csv_path = p if p.suffix == ".csv" else p.with_suffix(".csv")
        df = pd.read_csv(csv_path)
        # Backfill the parquet sibling so subsequent runs hit the fast path.
        try:
            df.to_parquet(csv_path.with_suffix(".parquet"), index=False)
        except Exception:
            pass
        return df

    return pd.read_parquet(p)


def cmd_filter_clash(args) -> None:
    import pandas as pd
    from residudes_qa.interface.filter.clash import run_clash_audit_parallel

    df = _load_parquet(args.master)
    print(f"  loaded master: {df.shape}", flush=True)
    if args.cif_col not in df.columns:
        raise KeyError(f"missing cif column {args.cif_col!r}")
    clash_df = run_clash_audit_parallel(
        df.rename(columns={args.cif_col: "src_cif"}),
        threshold=args.threshold,
        disulfide_max=args.disulfide_max,
        workers=args.workers,
    )
    merged = df.merge(clash_df, on="model_id", how="left")
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out, index=False)
    print(f"  wrote: {out}  shape={merged.shape}", flush=True)
    if args.max_inter is not None:
        sub = merged[merged["n_inter_clash_17"].fillna(99) <= args.max_inter]
        sub_path = out.with_name(out.stem + f"_inter_le{args.max_inter}.parquet")
        sub.to_parquet(sub_path, index=False)
        print(f"  inter ≤ {args.max_inter}: {len(sub)}/{len(merged)} → {sub_path}", flush=True)


def cmd_filter_cdr_clean(args) -> None:
    from residudes_qa.interface.filter.ab_clean import apply_cdr_clean

    df = _load_parquet(args.master)
    print(f"  loaded master: {df.shape}", flush=True)
    out_df, stats = apply_cdr_clean(
        df,
        thresholds={
            "aadars_frac_framework_max": args.frac_fr_max,
            "aadars_frac_cdr_h3_min": args.frac_h3_min,
            "iface_n_contacts_min": args.iface_contacts_min,
            "mpnn_iface_n_residues_min": args.mpnn_iface_min,
        },
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out, index=False)
    print(f"  CDR-clean passed: {stats['n_output']}/{stats['n_input']}  ({stats['fraction_passed']*100:.1f}%)", flush=True)
    print(f"  wrote: {out}", flush=True)
    for step in stats.get("by_filter_step", []):
        print(f"    {step['step']:30s} {step['threshold']:50s} n={step['n_after']}", flush=True)


def cmd_filter_strict(args) -> None:
    from residudes_qa.interface.filter.ab_clean import apply_strict_filter

    df = _load_parquet(args.master)
    print(f"  loaded master: {df.shape}", flush=True)
    additional = {
        "no_methods": args.exclude_methods.split(",") if args.exclude_methods else [],
    }
    if args.mode == "ab":
        additional["gO_core_max"] = args.go_core_max
    out_df, stats = apply_strict_filter(df, mode=args.mode, additional=additional)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out, index=False)
    print(f"  STRICT passed: {len(out_df)}/{len(df)}  ({len(out_df)/max(len(df),1)*100:.1f}%)", flush=True)
    print(f"  wrote: {out}", flush=True)


def cmd_filter_regional_rmsd(args) -> None:
    import pandas as pd
    from residudes_qa.interface.filter.template_rmsd import run_regional_rmsd_parallel

    df = _load_parquet(args.master)
    print(f"  loaded master: {df.shape}", flush=True)
    if args.cif_col not in df.columns:
        raise KeyError(f"missing cif column {args.cif_col!r}")
    regions = None
    if args.regions:
        regions = {}
        for entry in args.regions.split(","):
            name, span = entry.split(":")
            lo, hi = span.split("-")
            regions[name.strip()] = (int(lo), int(hi))
    rmsd_df = run_regional_rmsd_parallel(
        df.rename(columns={args.cif_col: "src_cif"}),
        ref_cif=args.ref_cif,
        anchor_chain_pred=args.anchor_chain_pred,
        anchor_chain_ref=args.anchor_chain_ref,
        probe_chain_pred=args.probe_chain_pred,
        probe_chain_ref=args.probe_chain_ref,
        regions=regions,
        workers=args.workers,
    )
    merged = df.merge(rmsd_df, on="model_id", how="left")
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out, index=False)
    print(f"  wrote: {out}  shape={merged.shape}", flush=True)


def _atom_integrity_worker(arg):
    """Module-level so it is picklable for multiprocessing.Pool.

    arg = (idx, cif_path, max_outlier_dist). Flags a model if any heavy atom
    sits farther than ``max_outlier_dist`` Å from its residue CA (the pt2
    terminal-O blow-up signature).
    """
    import gemmi
    import numpy as np
    idx, path, max_outlier_dist = arg
    try:
        st = gemmi.read_structure(str(path))
        for model in st:
            for ch in model:
                for r in ch:
                    ca = r.find_atom("CA", "*")
                    if not ca:
                        continue
                    ca_p = np.array([ca.pos.x, ca.pos.y, ca.pos.z])
                    for a in r:
                        p = np.array([a.pos.x, a.pos.y, a.pos.z])
                        if float(np.linalg.norm(p - ca_p)) > max_outlier_dist:
                            return (idx, True, f"{ch.name}.{r.seqid.num}.{a.name}")
            break
    except Exception:
        return (idx, True, "READ_ERR")
    return (idx, False, "")


def cmd_filter_atom_integrity(args) -> None:
    import multiprocessing as mp
    import pandas as pd

    df = _load_parquet(args.master)
    print(f"  loaded master: {df.shape}", flush=True)
    items = [(i, p, args.max_outlier_dist) for i, p in zip(df.index, df[args.cif_col])]
    with mp.Pool(args.workers) as pool:
        results = pool.map(_atom_integrity_worker, items)
    flags = pd.DataFrame(results, columns=["idx", "has_atom_outlier", "first_outlier"]).set_index("idx")
    merged = df.join(flags)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out, index=False)
    n_bad = int(merged["has_atom_outlier"].sum())
    print(f"  flagged {n_bad}/{len(merged)} models with atom > {args.max_outlier_dist}Å from CA", flush=True)
    print(f"  wrote: {out}", flush=True)


# Candidate model-id columns, in priority order, for each side of the join.
# MassiveClustering emits `structure` (main clustering) or `model` (model-level
# confidence summary); XGBDimerScorer emits `model_id`. We resolve a usable key
# on each side so the same command works regardless of which file is passed.
_CL_ID_COLS = ("model_id", "model", "structure", "structure_name", "file")
_XB_ID_COLS = ("model_id", "candidate_uid", "model")


def _resolve_id_col(df, preferred, candidates, side):
    """Pick the id column to join on: honour `preferred` if present, else the
    first known candidate column that exists in `df`."""
    if preferred and preferred in df.columns:
        return preferred
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(
        f"{side}: no join key found (looked for {preferred!r} then {candidates}); "
        f"columns present: {list(df.columns)}"
    )


def cmd_join_xgb(args) -> None:
    import pandas as pd

    cl = _load_parquet(args.clustering)
    xb = _load_parquet(args.xgb_scores)
    score_cols = [c for c in xb.columns if c.startswith("xgb_") or c in ("xgb_pred",)]
    if not score_cols:
        raise KeyError(f"no xgb_pred / xgb_* columns in {args.xgb_scores}")

    # `--on` (default "model_id") is treated as a preference, not a hard
    # requirement: if it isn't on a given side we fall back to that side's known
    # id columns and merge with left_on/right_on. Both sides use `*.pdb`/`*.cif`
    # filenames, so the values align across the differently-named columns.
    pref = None if args.on == "model_id" else args.on
    cl_key = _resolve_id_col(cl, pref, _CL_ID_COLS, "clustering")
    xb_key = _resolve_id_col(xb, pref, _XB_ID_COLS, "xgb-scores")

    keep = [xb_key] + score_cols
    merged = cl.merge(xb[keep], left_on=cl_key, right_on=xb_key, how="left")
    if xb_key != cl_key and xb_key in merged.columns:
        merged = merged.drop(columns=[xb_key])
    print(f"  join key: clustering.{cl_key} ← xgb.{xb_key}", flush=True)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out, index=False)
    print(
        f"  joined {len(merged)} cluster rows × {score_cols}  "
        f"→ {out}  (matched: {merged[score_cols[0]].notna().sum()})",
        flush=True,
    )
