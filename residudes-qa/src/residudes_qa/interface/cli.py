"""residudes-qa interface — command-line interface (ex-XGBDimerScorer / xgbdimer).

Subcommands::

    residudes-qa interface discover  --target-dir P --sto S [--variants V] [--methods M] -o OUT
    residudes-qa interface features  --index I --feature {base,cpu,dfire,aadars,mpnn,clash} ... -o OUT
    residudes-qa interface master    --stage-dir S --index I -o OUT
    residudes-qa interface score     --master M --mode {ab,ppi} -o OUT
    residudes-qa interface filter    --master M --mode {ab,ppi} [--strict] -o OUT
    residudes-qa interface cluster   --master M --probe-chains C --anchor-chains C [--ref-cif F] -o OUT
    residudes-qa interface pick      --master M [--n-final 5] -o OUT
    residudes-qa interface run-all   --target-dir P --sto S --mode {ab,ppi} [--ab-chains C] ... -o OUT

Each subcommand prints progress to stdout and writes one or more files under
``<out>``.  The full pipeline (``run-all``) chains them in order:
discover → features → master → score → filter → cluster → pick → viz.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import pandas as pd


# ---------- helpers ----------------------------------------------------------


def _split_csv(s: Optional[str]) -> list[str]:
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def _split_chars(s: Optional[str]) -> list[str]:
    """Split a string of single-character chain ids.

    ``'DE'`` → ``['D','E']``.  ``'A,B'`` → ``['A','B']``.
    """
    if not s:
        return []
    if "," in s:
        return _split_csv(s)
    return [c for c in s if not c.isspace()]


def _load_index(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        sys.exit(f"index file not found: {p}")
    return pd.read_parquet(p)


def _resolve_run_device() -> dict:
    """Resolve the device the GPU-backed scorers (dfire/mpnn/ppi_v4) will use.

    They all pick ``cuda`` iff ``torch.cuda.is_available()`` else ``cpu`` (see
    ppi_v4.Scorers / dfire / mpnn), so a single check here faithfully reflects
    what the feature stages will actually run on.  Returns a small dict suitable
    for ``run_meta.json`` and a ``device`` column — so a result set is never
    ambiguous about whether it came off a GPU or a slow CPU fallback.

    IMPORTANT: the torch/CUDA probe runs in a SUBPROCESS, never in this process.
    ``torch.cuda.is_available()`` initialises a CUDA context, and dfire scores
    with a fork-based multiprocessing Pool — a CUDA-initialised parent then
    forks children that crash with "Cannot re-initialize CUDA in forked
    subprocess". Probing out-of-process keeps the parent CUDA-clean so the fork
    pool works.
    """
    import json
    import os
    import socket
    import subprocess

    info = {
        "device": "cpu",
        "gpu_name": None,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "node": os.environ.get("SLURMD_NODENAME") or socket.gethostname(),
    }
    probe = (
        "import json,torch\n"
        "a=torch.cuda.is_available()\n"
        "print(json.dumps([a, torch.cuda.get_device_name(0) if a else None]))\n"
    )
    try:
        out = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True, text=True, timeout=120,
        )
        avail, name = json.loads(out.stdout.strip().splitlines()[-1])
        if avail:
            info["device"] = "cuda"
            info["gpu_name"] = name
    except Exception:  # noqa: BLE001 — torch missing/broken/timeout → treat as cpu
        pass
    return info


def _write_run_meta(out_dir: Path, dev: dict) -> None:
    """Persist the resolved run device alongside the result tables."""
    import json

    (out_dir / "run_meta.json").write_text(json.dumps(dev, indent=2) + "\n")


# Feature stages that run on a GPU when one is visible (the rest are CPU-only:
# base/cpu/aadars/clash do no torch work).  Used to stamp each stage's output
# with the device IT actually ran on, rather than one global value.
_GPU_STAGES = {"dfire", "mpnn"}


def _stamp_device(path: Path, device: str) -> None:
    """Add/overwrite a ``device`` column on a single parquet/csv table."""
    if not path.exists():
        return
    try:
        df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
        df["device"] = device
        if path.suffix == ".parquet":
            df.to_parquet(path, index=False)
        else:
            df.to_csv(path, index=False)
    except Exception as exc:  # noqa: BLE001 — never fail the run over a stamp
        print(f"  WARN: could not stamp device on {path.name}: {exc}", flush=True)


def _stamp_score_device(out_dir: Path, device: str) -> None:
    """Stamp the final xgb score tables with the GPU-stage device."""
    _stamp_device(out_dir / "xgb_scores.parquet", device)
    _stamp_device(out_dir / "xgb_scores.csv", device)


def _stage_device_info(feat: str) -> dict:
    """Device info for one feature stage: resolve the live GPU for GPU-capable
    stages (dfire/mpnn), else a plain cpu record (base/cpu/clash/aadars do no
    torch work)."""
    import os
    import socket

    if feat in _GPU_STAGES:
        return _resolve_run_device()
    return {
        "device": "cpu",
        "gpu_name": None,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "node": os.environ.get("SLURMD_NODENAME") or socket.gethostname(),
    }


def _record_stage_device(out_dir, feat: str, parquet_path=None,
                         shard: int = 0, nshards: int = 1) -> None:
    """Stamp a feature stage's parquet with the device it ran on, and drop a
    per-stage sidecar (``stage_outputs/.<feat>[.shardK].device.json``).  Each
    parallel feature job (and each shard) records its OWN hardware this way, so a
    separate-job fan-out can be aggregated later (see :func:`_finalize_run_meta`)
    without jobs racing on one shared file."""
    import json

    so = Path(out_dir) / "stage_outputs"
    info = _stage_device_info(feat)
    _stamp_device(parquet_path or (so / f"{feat}_features.parquet"), info["device"])
    sidecar = f".{feat}.shard{shard}.device.json" if nshards > 1 else f".{feat}.device.json"
    try:
        (so / sidecar).write_text(
            json.dumps({"feature": feat, "shard": shard, "nshards": nshards,
                        **info}) + "\n")
    except Exception as exc:  # noqa: BLE001
        print(f"  WARN: could not write device sidecar for {feat}: {exc}",
              flush=True)


def _finalize_run_meta(out_dir) -> str:
    """Aggregate the per-stage device sidecars into ``run_meta.json`` and stamp
    the xgb score tables.  Returns the resolved score device (``cuda`` iff a GPU
    stage actually ran on a GPU, else ``cpu``).  Safe with no sidecars present
    (standalone ``score`` call) — then everything reads as cpu."""
    import json
    import os
    import socket

    out_dir = Path(out_dir)
    so = out_dir / "stage_outputs"
    stages: dict[str, dict] = {}
    if so.exists():
        for p in sorted(so.glob(".*.device.json")):
            try:
                d = json.loads(p.read_text())
                stages[d.get("feature", p.stem.lstrip("."))] = d
            except Exception:  # noqa: BLE001
                pass
    score_device = "cpu"
    for s in _GPU_STAGES:
        if stages.get(s, {}).get("device") == "cuda":
            score_device = "cuda"
            break
    meta = {
        "score_device": score_device,
        "stage_devices": {k: v.get("device") for k, v in stages.items()},
        "stages": stages,
        "node": os.environ.get("SLURMD_NODENAME") or socket.gethostname(),
    }
    _write_run_meta(out_dir, meta)
    _stamp_score_device(out_dir, score_device)
    return score_device


def _load_master(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        sys.exit(f"master parquet not found: {p}")
    return pd.read_parquet(p)


# ---------- subcommand handlers ---------------------------------------------


def cmd_discover(args: argparse.Namespace) -> None:
    from residudes_qa.interface.core.discover import discover_and_save

    discover_and_save(
        target_dir=args.target_dir,
        sto=args.sto,
        output_dir=args.output,
        variants=_split_csv(args.variants) or None,
        methods=_split_csv(args.methods) or None,
        target=args.target,
        pdb=args.pdb,
    )


def cmd_features(args: argparse.Namespace) -> None:
    index = _load_index(args.index)
    feat = args.feature
    # Sharding: split the model set so N jobs each score a disjoint slice in
    # parallel. Strided (iloc[k::n]) keeps every shard balanced across methods
    # /variants. master globs `<feat>_features_*.parquet` and concats, so shards
    # need no separate merge step. nshards=1 (default) = whole set, canonical
    # filename, unchanged behaviour.
    nsh = int(getattr(args, "nshards", 1) or 1)
    k = int(getattr(args, "shard", 0) or 0)
    if nsh > 1:
        index = index.iloc[k::nsh].reset_index(drop=True)
        print(f"  [{feat} shard {k + 1}/{nsh}] scoring {len(index)} models "
              f"(strided slice)", flush=True)

    if feat == "base":
        from residudes_qa.interface.features.base import ChainConfig, run_base_features_parallel
        cfg = ChainConfig(
            ab_chains=_split_chars(args.ab_chains),
            ag_chains=_split_chars(args.ag_chains),
            heavy_chain=args.heavy,
            light_chain=args.light,
            scfv_chain=args.scfv,
        )
        df = run_base_features_parallel(index, cfg, workers=args.workers)
    elif feat == "cpu":
        from residudes_qa.interface.features.cpu import run_cpu_features_parallel
        df = run_cpu_features_parallel(index, workers=args.workers)
    elif feat == "dfire":
        from residudes_qa.interface.features.dfire import run_dfire_parallel
        # DFIRE is GPU-resident and each fork worker opens its own CUDA context;
        # 8 contexts on one card risks OOM (module default is 2). Cap on GPU.
        w = args.workers
        if _stage_device_info("dfire")["device"] == "cuda":
            w = min(w, 2)
            print(f"  dfire: GPU detected → capping workers {args.workers}→{w} "
                  f"(avoid multi-context OOM)", flush=True)
        df = run_dfire_parallel(index, workers=w)
    elif feat == "aadars":
        if not args.heavy:
            sys.exit("aadars requires --heavy")
        from residudes_qa.interface.features.aadars import run_aadars_parallel
        df = run_aadars_parallel(
            index,
            heavy=args.heavy,
            light=args.light,
            ag_chains=_split_chars(args.ag_chains),
            workers=args.workers,
        )
    elif feat == "mpnn":
        from residudes_qa.interface.features.mpnn import run_mpnn_sequential
        df = run_mpnn_sequential(
            index,
            ab_chains=_split_chars(args.ab_chains),
            ag_chains=_split_chars(args.ag_chains),
        )
    elif feat == "clash":
        from residudes_qa.interface.filter.clash import run_clash_audit_parallel
        df = run_clash_audit_parallel(index, workers=args.workers)
    else:
        sys.exit(f"unknown --feature {feat!r}")

    # Central save — shard-aware filename. Shards write distinct part files
    # (matching master's `<feat>_features_*.parquet` glob) so concurrent jobs
    # never collide on one path.
    so = Path(args.output) / "stage_outputs"
    so.mkdir(parents=True, exist_ok=True)
    if nsh > 1:
        out_path = so / f"{feat}_features_shard{k}of{nsh}.parquet"
    else:
        out_path = so / f"{feat}_features.parquet"
    df.to_parquet(out_path, index=False)

    # Record the device THIS stage ran on (stamps the parquet + drops a sidecar
    # for fan-out aggregation). GPU-capable stages report their live GPU; rest cpu.
    _record_stage_device(args.output, feat, out_path, shard=k, nshards=nsh)


def cmd_master(args: argparse.Namespace) -> None:
    from residudes_qa.interface.core.master import join_and_save

    index = _load_index(args.index)
    join_and_save(index, args.stage_dir, args.output, mode=args.mode)


def _parse_targets(raw: Optional[str]) -> list[str]:
    """Parse ``--targets dockq,ics,ips`` → validated list (default ['dockq'])."""
    targets = _split_csv(raw) or ["dockq"]
    bad = [t for t in targets if t not in ("dockq", "ics", "ips")]
    if bad:
        sys.exit(f"unknown --targets {bad} (choose from dockq,ics,ips)")
    # de-dup, keep order; ensure dockq first if present (drives sort / xgb_pred)
    seen: list[str] = []
    for t in (["dockq"] if "dockq" in targets else []) + targets:
        if t not in seen:
            seen.append(t)
    return seen


def cmd_score(args: argparse.Namespace) -> None:
    master = _load_master(args.master)
    targets = _parse_targets(getattr(args, "targets", None))
    if args.mode == "ab":
        from residudes_qa.interface.score.xgb_ab import score_ab
        score_ab(master, args.output, targets=targets,
                 ab_version=getattr(args, "ab_version", "v9"))
    elif args.mode == "ppi":
        from residudes_qa.interface.score.xgb_ppi import score_ppi
        score_ppi(master, args.output, targets=targets)
    else:
        sys.exit(f"unknown --mode {args.mode!r}")
    # Aggregate per-stage device sidecars → run_meta.json, and stamp xgb_scores
    # with the device the GPU feature stages ran on (cuda vs cpu fallback).
    _finalize_run_meta(args.output)


def cmd_filter(args: argparse.Namespace) -> None:
    master = _load_master(args.master)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "ab":
        from residudes_qa.interface.filter.ab_clean import (
            apply_cdr_clean,
            apply_strict_filter,
        )
        if args.strict:
            filtered, stats = apply_strict_filter(master, mode="ab")
            out_name = "filter_strict.parquet"
        else:
            filtered, stats = apply_cdr_clean(master)
            out_name = "filter_cdr_clean.parquet"
    elif args.mode == "ppi":
        from residudes_qa.interface.filter.ppi_clean import apply_ppi_clean
        filtered, stats = apply_ppi_clean(master)
        out_name = "filter_ppi_clean.parquet"
    else:
        sys.exit(f"unknown --mode {args.mode!r}")
    filtered.to_parquet(out_dir / out_name, index=False)
    print(f"  saved: {out_dir/out_name}  shape={filtered.shape}", flush=True)
    # also dump stats as JSON for the funnel plot
    import json
    (out_dir / out_name.replace(".parquet", "_stats.json")).write_text(
        json.dumps(stats, indent=2, default=str)
    )


def cmd_cluster(args: argparse.Namespace) -> None:
    from residudes_qa.interface.cluster.com_kmeans import run_clustering
    master = _load_master(args.master)
    run_clustering(
        master,
        probe_chains=_split_chars(args.probe_chains),
        anchor_chains=_split_chars(args.anchor_chains) or None,
        ref_cif=args.ref_cif,
        output_dir=args.output,
        workers=args.workers,
    )


def cmd_pick(args: argparse.Namespace) -> None:
    from residudes_qa.interface.pick.collect import collect_final_picks
    from residudes_qa.interface.pick.per_cluster import pick_per_cluster

    master = _load_master(args.master)
    picks = pick_per_cluster(master, n_top_xgb=args.n_top_xgb)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    picks.to_csv(out_dir / "per_cluster_summary.csv", index=False)
    collect_final_picks(picks, out_dir, max_picks=args.n_final)


def cmd_run_all(args: argparse.Namespace) -> None:
    """End-to-end pipeline.

    Default (--apply-filter NOT set):
        discover → features (all) → master → score → STOP
        Result: full feature table with xgb scores, NO filtering applied.
        Hand off to MassiveClustering for cluster/pick (or use --apply-filter
        to run the in-tool filter cascade).

    With --apply-filter:
        ... → filter → cluster → pick → viz
        Runs the H1335-style CDR-clean + STRICT + COM clustering + per-cluster
        picks inside XGBDimer (legacy path; MassiveClustering is preferred for
        production).
    """
    from residudes_qa.interface.cli import (
        cmd_cluster as _c,
        cmd_discover as _d,
        cmd_features as _f,
        cmd_filter as _flt,
        cmd_master as _m,
        cmd_pick as _p,
        cmd_score as _s,
    )

    # PPI uses a DIFFERENT engine (interface auto-detect → ppi_v4 features →
    # xgb_ppi), not the staged base/cpu/dfire/mpnn pipeline. Dispatch to it so
    # `run-all --mode ppi` gives the proper per-interface scores (not the old
    # mostly-imputed path). run-ppi-v4 is now just a thin alias for this.
    if args.mode == "ppi":
        # Fan-out support: `--stages discover` does discovery only (writes
        # parquet/index.parquet) so N shard jobs can reuse one index.
        if (getattr(args, "stages", None) or "all").lower() == "discover":
            from residudes_qa.interface.core.discover import discover_and_save
            pq = Path(args.output) / "parquet"
            pq.mkdir(parents=True, exist_ok=True)
            discover_and_save(
                target_dir=args.target_dir, sto=args.sto, output_dir=pq,
                variants=_split_csv(args.variants) or None,
                methods=_split_csv(args.methods) or None,
                target=args.target, pdb=args.pdb)
            return
        print("[run-all] mode=ppi → dispatching to ppi_v4 interface engine "
              f"(model {getattr(args, 'ppi_model', 'v6')})", flush=True)
        cmd_run_ppi_v4(argparse.Namespace(
            target_dir=args.target_dir, sto=args.sto,
            variants=args.variants, methods=args.methods,
            target=args.target, pdb=args.pdb,
            index=getattr(args, "index", None),
            pairs=getattr(args, "pairs", None),
            chain_a=getattr(args, "chain_a", None),
            chain_b=getattr(args, "chain_b", None),
            hub=getattr(args, "hub", None),
            min_contacts=getattr(args, "min_contacts", 10),
            targets=getattr(args, "targets", None),
            device=getattr(args, "device", "cuda"),
            ppi_model=getattr(args, "ppi_model", "v6"),
            shard=getattr(args, "shard", 0),
            nshards=getattr(args, "nshards", 1),
            output=args.output,
        ))
        return

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    stage_outputs = out_dir / "stage_outputs"
    stage_outputs.mkdir(exist_ok=True)

    # --stages lets the parallel fan-out (run_scoring.sh) run this same command
    # one stage at a time across separate Slurm jobs. Default "all" = the full
    # sequential pipeline (unchanged single-job behaviour).
    ALL_FEATS = ["base", "cpu", "dfire", "mpnn", "clash"]
    raw_stages = (getattr(args, "stages", None) or "all").lower()
    sel = {s.strip() for s in raw_stages.split(",") if s.strip()}
    all_stages = "all" in sel
    want_all_feats = all_stages or "features" in sel

    def _want(name: str) -> bool:
        return all_stages or name in sel

    def _want_feat(f: str) -> bool:
        return want_all_feats or f"feat:{f}" in sel

    # Device is resolved + recorded per-stage inside `features`/`score` (so the
    # parallel fan-out path records the same way). One-line heads-up here if the
    # GPU-backed stages will fall back to CPU.
    _dev = _resolve_run_device()
    if _dev["device"] != "cuda" and (_want_feat("dfire") or _want_feat("mpnn")):
        print(f"\n[run-all] WARNING: no GPU visible on {_dev['node']} — "
              f"dfire/mpnn will run on CPU FALLBACK (much slower). If you "
              f"requested a GPU, the job was likely scheduled without "
              f"--gres=gpu.", flush=True)

    index_path = out_dir / "index.parquet"

    # 1. discover
    if _want("discover"):
        print("\n=== STAGE 1: discover ===", flush=True)
        _d(argparse.Namespace(
            target_dir=args.target_dir,
            sto=args.sto,
            variants=args.variants,
            methods=args.methods,
            target=args.target,
            pdb=args.pdb,
            output=out_dir,
        ))

    # 2. features  (base + cpu always; dfire / mpnn / aadars / clash optional)
    sel_feats = [f for f in ALL_FEATS if _want_feat(f)]
    if sel_feats or (_want_feat("aadars") and args.mode == "ab" and args.heavy):
        print("\n=== STAGE 2: features ===", flush=True)
        base_args = dict(
            index=index_path,
            ab_chains=args.ab_chains,
            ag_chains=args.ag_chains,
            heavy=args.heavy,
            light=args.light,
            scfv=args.scfv,
            workers=args.workers,
            output=out_dir,
            shard=getattr(args, "shard", 0),
            nshards=getattr(args, "nshards", 1),
        )
        for feat in sel_feats:
            ns = argparse.Namespace(feature=feat, **base_args)
            try:
                _f(ns)  # cmd_features stamps the stage's device + drops a sidecar
            except SystemExit:
                raise
            except Exception as exc:  # noqa: BLE001
                print(f"  WARN: feature '{feat}' failed: {exc}", flush=True)
        if _want_feat("aadars") and args.mode == "ab" and args.heavy:
            try:
                _f(argparse.Namespace(feature="aadars", **base_args))
            except Exception as exc:  # noqa: BLE001
                print(f"  WARN: aadars failed: {exc}", flush=True)

    # 3. master
    master_path = out_dir / "master.parquet"
    if _want("master"):
        print("\n=== STAGE 3: master ===", flush=True)
        _m(argparse.Namespace(
            index=index_path,
            stage_dir=stage_outputs,
            mode=args.mode,
            output=out_dir,
        ))

    # 4. score
    master_with_xgb_path = out_dir / "master_with_xgb.parquet"
    if _want("score"):
        print("\n=== STAGE 4: score ===", flush=True)
        _s(argparse.Namespace(master=master_path, mode=args.mode, output=out_dir,
                              ab_version=getattr(args, "ab_model", "v9"),
                              targets=getattr(args, "targets", None)))
        # (cmd_score has already aggregated run_meta.json + stamped xgb_scores.)
    elif not _want("score") and not all_stages:
        # Partial (fan-out) job that didn't run scoring — nothing more to do.
        print(f"\n=== run-all done (stages: {sorted(sel)}) ===", flush=True)
        return

    # Default exit point — feature table + xgb scores ready, hand off to
    # MassiveClustering for filter/cluster/pick.
    if not getattr(args, "apply_filter", False):
        print("\n=== run-all done (table-only mode) ===", flush=True)
        print(f"  output root: {out_dir}", flush=True)
        print(f"  ▸ master:           {master_path}", flush=True)
        print(f"  ▸ master_with_xgb:  {master_with_xgb_path}", flush=True)
        print(f"  ▸ xgb_scores:       {out_dir / 'xgb_scores.parquet'}", flush=True)
        print(f"  ▸ xgb_scores (csv): {out_dir / 'xgb_scores.csv'}", flush=True)
        print(
            "\n  Next: hand off to MassiveClustering for filtering+clustering,",
            flush=True,
        )
        print(
            f"    massivecluster filter cdr-clean --master {master_with_xgb_path} ...",
            flush=True,
        )
        print(
            f"    or re-run with `--apply-filter` to use the in-tool legacy path.",
            flush=True,
        )
        return

    # 5. filter (opt-in)
    print("\n=== STAGE 5: filter (legacy in-tool) ===", flush=True)
    _flt(argparse.Namespace(
        master=master_with_xgb_path,
        mode=args.mode,
        strict=(args.mode == "ab"),
        output=out_dir,
    ))
    if args.mode == "ab":
        filt_path = out_dir / "filter_strict.parquet"
    else:
        filt_path = out_dir / "filter_ppi_clean.parquet"

    # 6. cluster
    print("\n=== STAGE 6: cluster ===", flush=True)
    _c(argparse.Namespace(
        master=filt_path,
        probe_chains=args.ab_chains or args.probe_chains,
        anchor_chains=args.anchor_chains,
        ref_cif=args.ref_cif,
        workers=args.workers,
        output=out_dir,
    ))
    clusters_path = out_dir / "master_with_clusters.parquet"

    # 7. pick
    print("\n=== STAGE 7: pick ===", flush=True)
    _p(argparse.Namespace(
        master=clusters_path,
        n_top_xgb=1,
        n_final=args.n_final,
        output=out_dir,
    ))

    print("\n=== run-all done (with filter/cluster/pick) ===", flush=True)
    print(f"  output root: {out_dir}", flush=True)


def _parse_pairs(pairs_arg, chain_a, chain_b, hub=None):
    """Resolve interface pairs from --pairs 'A-B,A-C', --chain-a/--chain-b, or
    --hub A (A vs every other chain).  Returns None to signal 'auto-detect'."""
    if pairs_arg:
        out = []
        for tok in pairs_arg.split(","):
            tok = tok.strip()
            if not tok:
                continue
            a, b = tok.split("-")
            out.append((a.strip(), b.strip()))
        return out
    if chain_a and chain_b:
        return [(chain_a, chain_b)]
    if hub:
        return [("__HUB__", hub)]  # resolved against the structure later
    return None  # auto-detect all contacting interfaces


def cmd_run_ppi_v4(args: argparse.Namespace) -> None:
    """PPI interface scoring: discover -> ppi_v4 features -> p5 engineer -> xgb_ppi (v6).

    NOTE: ``master_ppi_v4`` / ``ppi_v4`` name the FEATURE SCHEMA, not the model —
    the deployed model is ``xgb_ppi_v6`` (override with --ppi-model). Produces the
    exact ``master_ppi_v4`` feature schema (so the model is no longer fed
    mostly-imputed features), one interface ``(chain_a, chain_b)`` per run.
    Layout::

        <output>/
          xgb_dockq.csv          combined summary (meta + confidence + per-iface xgb + mean)
          parquet/               every parquet artifact (master / scores / index)
          csv/                   human-readable CSVs (master + scores per interface)
    """
    import pandas as pd
    from residudes_qa.interface.features.ppi_v4 import (
        run_ppi_v4_multi, engineer_ppi_v4, detect_interface_pairs, parse_structure,
    )
    from residudes_qa.interface.score.xgb_ppi import (
        prep_xgb_ppi_input, predict_xgb_ppi, ppi_model_triple, PPI_TARGET_COL,
    )

    targets = _parse_targets(getattr(args, "targets", None))

    # Sharding: N parallel jobs each score a disjoint model slice; each writes to
    # its own shards/shard{k}of{N}/ subdir (no collision) and run_scoring.sh's
    # merge job concats the per-shard xgb_dockq.csv. Interfaces are auto-detected
    # per shard from the complex topology → identical across shards.
    nsh = int(getattr(args, "nshards", 1) or 1)
    k = int(getattr(args, "shard", 0) or 0)
    out_dir = Path(args.output)
    if nsh > 1:
        out_dir = out_dir / "shards" / f"shard{k}of{nsh}"
    pq_dir = out_dir / "parquet"
    csv_dir = out_dir / "csv"
    for d in (out_dir, pq_dir, csv_dir):
        d.mkdir(parents=True, exist_ok=True)

    # 1. index (reuse existing or discover)
    if args.index:
        index = _load_index(args.index)
    else:
        from residudes_qa.interface.core.discover import discover_and_save
        discover_and_save(
            target_dir=args.target_dir, sto=args.sto, output_dir=pq_dir,
            variants=_split_csv(args.variants) or None,
            methods=_split_csv(args.methods) or None,
            target=args.target, pdb=args.pdb,
        )
        index = _load_index(pq_dir / "index.parquet")
    if args.target and "target" not in index.columns:
        index["target"] = args.target

    # representative cif for interface auto-detection — taken from the FULL index
    # BEFORE sharding, so every shard detects the SAME interface set (a per-shard
    # rep_cif could diverge on borderline contacts → missing columns on merge).
    rep_cif = next((c for c in index["src_cif"]
                    if isinstance(c, str) and Path(c).exists()), None)

    if nsh > 1:
        index = index.iloc[k::nsh].reset_index(drop=True)
        print(f"  [ppi shard {k + 1}/{nsh}] scoring {len(index)} models "
              f"(strided slice)", flush=True)

    pairs = _parse_pairs(args.pairs, args.chain_a, args.chain_b,
                         hub=getattr(args, "hub", None))
    if pairs is None:
        if rep_cif is None:
            sys.exit("run-ppi-v4: no cif found to auto-detect interfaces")
        pairs = detect_interface_pairs(rep_cif, min_contacts=args.min_contacts)

    pairs = _parse_pairs(args.pairs, args.chain_a, args.chain_b,
                         hub=getattr(args, "hub", None))
    if pairs is None:
        if rep_cif is None:
            sys.exit("run-ppi-v4: no cif found to auto-detect interfaces")
        pairs = detect_interface_pairs(rep_cif, min_contacts=args.min_contacts)
        print(f"  auto-detected {len(pairs)} contacting interface(s) "
              f"(>={args.min_contacts} contacts): {pairs}", flush=True)
        if not pairs:
            sys.exit("run-ppi-v4: no contacting interfaces found; pass --pairs explicitly")
    elif pairs and pairs[0][0] == "__HUB__":
        # --hub A  →  A vs every other chain in the structure
        hub = pairs[0][1]
        chains, _ = parse_structure(rep_cif)
        pairs = [(hub, c) for c in chains if c != hub]
        print(f"  hub mode: {hub} vs {[b for _, b in pairs]}", flush=True)

    print(f"\n=== run-ppi-v4: {len(index)} cifs x {len(pairs)} interface(s) "
          f"{pairs} on {args.device} ===", flush=True)

    # extract all interfaces in one pass (whole-complex work shared across pairs)
    feats_by_pair = run_ppi_v4_multi(index, pairs, device=args.device)

    # per-interface column name for a (target, tag): dockq keeps the historical
    # ``xgb_<tag>``; ics/ips get ``xgb_<target>_<tag>``.
    def _pair_col(tgt: str, tag: str) -> str:
        return f"xgb_{tag}" if tgt == "dockq" else f"xgb_{tgt}_{tag}"

    def _mean_col(tgt: str) -> str:
        return "xgb_mean" if tgt == "dockq" else f"xgb_{tgt}_mean"

    combined = None
    base_meta = None
    pair_cols: dict[str, list[str]] = {t: [] for t in targets}
    for ca, cb in pairs:
        tag = f"{ca}-{cb}"
        print(f"\n--- interface {ca}-{cb} ---", flush=True)
        feats = engineer_ppi_v4(feats_by_pair[(ca, cb)], target_col="target")

        # score each requested target (manual prep+predict → avoid score_ppi's fixed filenames)
        pred_cols = []
        for tgt in targets:
            mp, mt, nm = ppi_model_triple(tgt, version=getattr(args, "ppi_model", "v6"))
            print(f"  [target={tgt}]", flush=True)
            X, fcols = prep_xgb_ppi_input(feats, mt, nm)
            col = PPI_TARGET_COL[tgt]
            feats[col] = predict_xgb_ppi(X, fcols, mp)
            pred_cols.append(col)

        # master (full feature table) — parquet + csv
        feats.to_parquet(pq_dir / f"master_{tag}.parquet", index=False)
        feats.to_csv(csv_dir / f"master_{tag}.csv", index=False)
        # per-interface scores (sorted by first target) — parquet + csv
        sc_cols = [c for c in ("model_id", "candidate_uid", "method", "variant",
                               "seed", "sample", *pred_cols) if c in feats.columns]
        scores = feats[sc_cols].sort_values(pred_cols[0], ascending=False)
        scores.to_parquet(pq_dir / f"xgb_scores_{tag}.parquet", index=False)
        scores.to_csv(csv_dir / f"xgb_scores_{tag}.csv", index=False)

        for tgt in targets:
            pcol = _pair_col(tgt, tag)
            pair_cols[tgt].append(pcol)
            col = feats[["model_id", PPI_TARGET_COL[tgt]]].rename(
                columns={PPI_TARGET_COL[tgt]: pcol})
            combined = col if combined is None else combined.merge(col, on="model_id", how="outer")

        # whole-complex meta + confidence are identical across pairs — grab once.
        if base_meta is None:
            want = ["model_id", "target", "method", "variant", "seed", "sample",
                    "json_iptm", "json_ptm", "json_ranking_score", "cif_plddt_mean"]
            have = [c for c in want if c in feats.columns]
            base_meta = feats[have].copy().rename(columns={
                "method": "model", "variant": "msa",
                "json_iptm": "iptm", "json_ptm": "ptm",
                "json_ranking_score": "ranking_score",
                "cif_plddt_mean": "plddt_mean"})

    if combined is not None:
        # mean predicted score across the scored interfaces, per target
        mean_cols = []
        for tgt in targets:
            mcol = _mean_col(tgt)
            combined[mcol] = combined[pair_cols[tgt]].mean(axis=1)
            mean_cols.append(mcol)
        if base_meta is not None:
            combined = base_meta.merge(combined, on="model_id", how="right")
        # column order: meta → confidence → per-interface xgb (grouped by target) → means
        meta_order = [c for c in ["model_id", "target", "model", "msa", "seed",
                                  "sample", "iptm", "ptm", "ranking_score",
                                  "plddt_mean"] if c in combined.columns]
        all_pair_cols = [c for tgt in targets for c in pair_cols[tgt]]
        combined = combined[meta_order + all_pair_cols + mean_cols]
        sort_col = "xgb_mean" if "dockq" in targets else mean_cols[0]
        combined = combined.sort_values(sort_col, ascending=False)
        # round score columns for readable CSV
        for c in all_pair_cols + mean_cols + ["iptm", "ptm", "ranking_score", "plddt_mean"]:
            if c in combined.columns:
                combined[c] = combined[c].astype(float).round(4)
        combined.to_csv(out_dir / "xgb_dockq.csv", index=False)
        print(f"\n=== run-ppi-v4 done (targets={targets}) ===\n"
              f"  combined table: {out_dir/'xgb_dockq.csv'}", flush=True)


# ---------- argparse ---------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="residudes-qa interface",
        description="End-to-end xgb scoring + clustering for CASP17 dimers / multimers",
    )
    sp = p.add_subparsers(dest="command", required=True)

    # discover
    d = sp.add_parser("discover", help="Walk results/<sto>/... and build index.parquet")
    d.add_argument("--target-dir", required=True)
    d.add_argument("--sto", required=True)
    d.add_argument("--variants", default=None, help="comma-separated whitelist")
    d.add_argument("--methods", default=None, help="comma-separated whitelist")
    d.add_argument("--target", default=None)
    d.add_argument("--pdb", default=None)
    d.add_argument("-o", "--output", required=True)
    d.set_defaults(func=cmd_discover)

    # features
    f = sp.add_parser("features", help="Run one feature extractor")
    f.add_argument("--index", required=True)
    f.add_argument(
        "--feature",
        required=True,
        choices=["base", "cpu", "dfire", "aadars", "mpnn", "clash"],
    )
    f.add_argument("--ab-chains", default=None, help="e.g. DE or D,E")
    f.add_argument("--ag-chains", default=None, help="e.g. ABC or A,B,C")
    f.add_argument("--heavy", default=None)
    f.add_argument("--light", default=None)
    f.add_argument("--scfv", default=None)
    f.add_argument("--workers", type=int, default=8)
    f.add_argument("--shard", type=int, default=0,
                   help="this job's shard index K in [0, nshards)")
    f.add_argument("--nshards", type=int, default=1,
                   help="total shards N; this job scores index[K::N]. "
                        "1 = whole set (canonical filename).")
    f.add_argument("-o", "--output", required=True)
    f.set_defaults(func=cmd_features)

    # master
    m = sp.add_parser("master", help="Join per-stage features into master.parquet")
    m.add_argument("--index", required=True)
    m.add_argument("--stage-dir", required=True)
    m.add_argument("--mode", default="ab", choices=["ab", "ppi"])
    m.add_argument("-o", "--output", required=True)
    m.set_defaults(func=cmd_master)

    # score
    s = sp.add_parser("score", help="xgb predict")
    s.add_argument("--master", required=True)
    s.add_argument("--mode", required=True, choices=["ab", "ppi"])
    s.add_argument("--targets", default="dockq",
                   help="comma-separated subset of dockq,ics,ips (default dockq → xgb_pred; "
                        "ics→xgb_ics, ips→xgb_ips). PPI ics/ips=v6, AgAb ics/ips=v10.1")
    s.add_argument("--ab-version", default="v9", choices=["v9", "v10_1"],
                   help="(--mode ab) dockq model family: v9 (default, 12-method LOPO 0.82) "
                        "or v10_1 (deepfold3 7-method, FoldBench labels, LOPO 0.766). "
                        "ICS/IPS always use v10.1.")
    s.add_argument("-o", "--output", required=True)
    s.set_defaults(func=cmd_score)

    # filter
    flt = sp.add_parser("filter", help="Mode-appropriate sanity filter")
    flt.add_argument("--master", required=True)
    flt.add_argument("--mode", required=True, choices=["ab", "ppi"])
    flt.add_argument("--strict", action="store_true", help="Ab mode only — adds gO + clash gates")
    flt.add_argument("-o", "--output", required=True)
    flt.set_defaults(func=cmd_filter)

    # cluster
    c = sp.add_parser("cluster", help="COM KMeans with silhouette sweep")
    c.add_argument("--master", required=True)
    c.add_argument("--probe-chains", required=True, help="e.g. DE or D,E")
    c.add_argument("--anchor-chains", default=None)
    c.add_argument("--ref-cif", default=None)
    c.add_argument("--workers", type=int, default=8)
    c.add_argument("-o", "--output", required=True)
    c.set_defaults(func=cmd_cluster)

    # pick
    pk = sp.add_parser("pick", help="Per-cluster medoid + xgb-top selection")
    pk.add_argument("--master", required=True, help="clustered master parquet")
    pk.add_argument("--n-top-xgb", type=int, default=1)
    pk.add_argument("--n-final", type=int, default=5)
    pk.add_argument("-o", "--output", required=True)
    pk.set_defaults(func=cmd_pick)

    # run-all
    ra = sp.add_parser(
        "run-all",
        help="discover → features → master → score   (filter/cluster/pick OFF by default; use --apply-filter to enable)",
    )
    # --- common (both modes) ---
    ra.add_argument("--target-dir", required=True)
    ra.add_argument("--sto", required=True)
    # ab = antibody–antigen (staged base/cpu/dfire/mpnn → xgb_ab). ppi =
    # protein–protein (interface auto-detect → ppi_v4 features → xgb_ppi). The
    # two modes use DIFFERENT engines and DIFFERENT flags (see the mode groups
    # below in --help).
    ra.add_argument("--mode", required=True, choices=["ab", "ppi"],
                    help="ab = antibody–antigen (use the 'ab-mode' flags); "
                         "ppi = protein–protein interfaces (use the 'ppi-mode' flags)")
    ra.add_argument("--variants", default=None)
    ra.add_argument("--methods", default=None)
    ra.add_argument("--target", default=None)
    ra.add_argument("--pdb", default=None)
    ra.add_argument("--workers", type=int, default=8)
    ra.add_argument("-o", "--output", required=True)

    # --- ab mode (antibody–antigen) ---
    ab_grp = ra.add_argument_group(
        "ab-mode (--mode ab)",
        "antibody–antigen: staged base/cpu/dfire/mpnn → xgb_ab. Ignored in ppi mode.")
    ab_grp.add_argument("--ab-chains", default=None, help="antibody chain ids, e.g. CDEF")
    ab_grp.add_argument("--ag-chains", default=None, help="antigen chain ids, e.g. AB")
    ab_grp.add_argument("--heavy", default=None, help="Fab heavy chain id")
    ab_grp.add_argument("--light", default=None, help="Fab light chain id")
    ab_grp.add_argument("--scfv", default=None, help="scFv chain id (if applicable)")
    ab_grp.add_argument("--ab-model", default="v9",
                        help="xgb_ab model version (default v9; auto-falls back if absent)")

    # --- ppi mode (protein–protein) ---
    ppi_grp = ra.add_argument_group(
        "ppi-mode (--mode ppi)",
        "protein–protein: interface auto-detect → ppi_v4 features → xgb_ppi. "
        "No ab/ag chains. Ignored in ab mode.")
    ppi_grp.add_argument("--pairs", default=None, help="explicit interfaces, e.g. 'A-B,A-C'")
    ppi_grp.add_argument("--chain-a", default=None, help="single interface chain a (with --chain-b)")
    ppi_grp.add_argument("--chain-b", default=None, help="single interface chain b")
    ppi_grp.add_argument("--hub", default=None, help="score this chain vs every other (e.g. A)")
    ppi_grp.add_argument("--min-contacts", type=int, default=10,
                         help="auto-detect: min heavy-atom contacts for an interface (default 10)")
    ppi_grp.add_argument("--targets", default=None,
                         help="subset of dockq,ics,ips (default dockq). Works in "
                              "BOTH modes: ab → xgb_ics/xgb_ips (v10.1); ppi → "
                              "xgb_ics_<iface>/xgb_ips_<iface>")
    ppi_grp.add_argument("--ppi-model", default="v6",
                         help="xgb_ppi model version (default v6; auto-falls back if absent)")
    ppi_grp.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                         help="device for ppi_v4 scorers (default cuda)")
    ppi_grp.add_argument("--index", default=None,
                         help="(fan-out) reuse an existing index.parquet")

    # --- fan-out / sharding (internal; set by run_scoring.sh) ---
    fan_grp = ra.add_argument_group(
        "fan-out (set by run_scoring.sh; rarely passed by hand)")
    fan_grp.add_argument("--shard", type=int, default=0,
                         help="this job's shard index K in [0, nshards)")
    fan_grp.add_argument("--nshards", type=int, default=1,
                         help="total shards N; each scores index[K::N]. 1 = whole set.")
    fan_grp.add_argument(
        "--stages", default="all",
        help="ab mode: subset of discover/features/feat:<name>/master/score. "
             "ppi mode: only 'discover' (discover-only) is special. Default 'all'.")

    # --- legacy in-tool filter (--apply-filter; ab mode) ---
    ra.add_argument(
        "--apply-filter",
        action="store_true",
        help=(
            "Run the legacy in-tool filter (CDR-clean for --mode ab, PPI-clean for --mode ppi) "
            "+ STRICT (if --mode ab) + COM clustering + per-cluster picks. "
            "Default OFF — XGBDimer stops at master+xgb_scores and you hand off to "
            "MassiveClustering for clustering decisions."
        ),
    )
    ra.add_argument("--probe-chains", default=None, help="(PPI mode) probe COM chains; defaults to --ab-chains")
    ra.add_argument("--anchor-chains", default=None)
    ra.add_argument("--ref-cif", default=None)
    ra.add_argument("--n-final", type=int, default=5)
    ra.set_defaults(func=cmd_run_all)

    # run-ppi-v4  (alias of `run-all --mode ppi`; ppi_v4 = feature schema, model = v6)
    rv = sp.add_parser(
        "run-ppi-v4",
        help="[alias of `run-all --mode ppi`] PPI: discover -> ppi_v4 interface "
             "features -> xgb_ppi (v6) score per interface. '_v4' = feature "
             "schema, not model version (model is v6, override with --ppi-model).")
    rv.add_argument("--target-dir", default=None)
    rv.add_argument("--sto", default=None)
    rv.add_argument("--variants", default=None)
    rv.add_argument("--methods", default=None)
    rv.add_argument("--target", default=None)
    rv.add_argument("--pdb", default=None)
    rv.add_argument("--index", default=None, help="reuse an existing index.parquet instead of discovering")
    rv.add_argument("--pairs", default=None, help="explicit interfaces, e.g. 'A-B,A-C,A-D'")
    rv.add_argument("--chain-a", default=None, help="single interface chain a (with --chain-b)")
    rv.add_argument("--chain-b", default=None, help="single interface chain b")
    rv.add_argument("--hub", default=None, help="score this chain vs every other chain (e.g. A)")
    rv.add_argument("--min-contacts", type=int, default=10,
                    help="auto-detect mode: min heavy-atom contacts (5 A) to count as an interface (default 10)")
    rv.add_argument("--targets", default="dockq",
                    help="comma-separated subset of dockq,ics,ips (default dockq). "
                         "Adds per-interface xgb_<tag> (dockq) / xgb_ics_<tag> / xgb_ips_<tag> "
                         "and xgb_mean / xgb_ics_mean / xgb_ips_mean to xgb_dockq.csv")
    rv.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    rv.add_argument("--ppi-model", default="v6",
                    help="ppi xgb model version (default v6; auto-falls back if absent)")
    rv.add_argument("--shard", type=int, default=0, help="(fan-out) shard index K")
    rv.add_argument("--nshards", type=int, default=1,
                    help="(fan-out) total shards N; scores index[K::N], writes shards/shardKofN/")
    rv.add_argument("-o", "--output", required=True)
    rv.set_defaults(func=cmd_run_ppi_v4)

    return p


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
