#!/usr/bin/env python3
"""Collect benchmark results into docs/benchmark/ (CSV + figures) and two documents:

    docs/benchmark.md          the summary: headline numbers, three figures, what changed,
                               how this compares with the literature, how to reproduce
    docs/benchmark_details.md  everything: method, every figure, every run, variants, notes

    bench/collect.py           # reads runs/bench/**/seed_<N>/, rewrites all of the above

Per run it reads .residudes-engine/run.json (exit code, wall time, GPU), timing.json
(forward-pass seconds, written by the model's writer) and the top-ranked
sample's summary.json (pLDDT, pTM). Pending or failed runs are reported, not dropped.

Run directories follow  runs/bench/<target>_<sto>/<model>-<backend>[-v2][-warm|-mem85|-hybrid]/seed_<N>:
  series   ""    = v1, the state of the code when the matrix was first run (2026-11-02 a.m.)
           "-v2" = after the optimisations listed in docs/benchmark_details.md
  variant  "-warm"   = second JAX run with the XLA compile cache hot
           "-mem85"  = JAX rerun with the site's mem_fraction raised 0.65 → 0.85
           "-hybrid" = Boltz-2 JAX pairformers + TensorRT diffusion token transformer
Figures show the current state (v2 where it exists, warm where it exists); the details table lists every run.
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "bench"))

from residudes_engine.verify import check_seed_dir  # noqa: E402
from residudes_engine.runners._common import meta_of  # noqa: E402
import svgchart  # noqa: E402

CFG = json.loads((ROOT / "bench" / "targets.json").read_text())
DOCS = ROOT / "docs" / "benchmark"
BACKENDS = CFG["backends"]
MODELS = CFG["models"]
NAMES = {"pt2": "Protenix v2", "of3": "OpenFold 3", "bt2": "Boltz-2", "af3": "AlphaFold 3"}
V2_ONLY = set(CFG.get("v2_only", []))      # models added after the first matrix: no v1 series to report
BLABEL = {"trt": "TensorRT", "torch": "PyTorch", "jax": "JAX"}
COLORS = {"trt": svgchart.SERIES[0], "torch": svgchart.SERIES[1], "jax": svgchart.SERIES[2]}
SERIES = (("", "v1"), ("-v2", "v2"))
VARIANTS = (("", ""), ("-warm", "warm"), ("-mem85", "mem85"), ("-hybrid", "hybrid"))
# preference for "the current state" of a (target, model, backend): first available wins
PREFERENCE = [("v2", "warm"), ("v2", ""), ("v1", "warm"), ("v1", "")]
PAPER = ("OpenFold-TRT (Didi et al., bioRxiv 10.64898/2026.03.11.711233, March 2026)",
         "https://www.biorxiv.org/content/10.64898/2026.03.11.711233v1")


# ── collection ─────────────────────────────────────────────────────────────

def collect() -> list[dict]:
    rows = []
    for t in CFG["targets"]:
        key = f"{t['name']}_{t['sto']}"
        for m in MODELS:
            for b in BACKENDS:
                for ssuf, series in SERIES:
                    if m in V2_ONLY and series == "v1":
                        continue
                    for vsuf, variant in VARIANTS:
                        seed_dir = ROOT / "runs" / "bench" / key / f"{m}-{b}{ssuf}{vsuf}" / f"seed_{CFG['seed']}"
                        if m in V2_ONLY and not vsuf and not meta_of(seed_dir).exists():
                            pass          # the base v2 run of a late model is listed even before it exists
                        elif (ssuf or vsuf) and not meta_of(seed_dir).exists():
                            continue      # optional passes count only once they exist
                        row = {"target": t["name"], "sto": t["sto"], "tokens": t["tokens"], "model": m,
                               "backend": b, "series": series, "variant": variant,
                               "label": f"{b}{ssuf}{vsuf}", "status": "pending",
                               "wall_s": None, "forward_s": None, "plddt": None, "ptm": None,
                               "gpu": None, "job": None, "engine_max": None, "note": ""}
                        man = meta_of(seed_dir) / "run.json"
                        if man.is_file():
                            d = json.loads(man.read_text())
                            row.update(wall_s=d.get("wall_s"), job=d.get("slurm_job_id"),
                                       gpu=(d.get("gpu") or {}).get("name"), engine_max=d.get("engine_max_seq_len"))
                            if d.get("exit_code") == 0:
                                rep = check_seed_dir(seed_dir)
                                top = rep.top()
                                if rep.ok and top:
                                    row["status"] = "ok"
                                    row["plddt"] = top.summary.get("plddt")
                                    row["ptm"] = top.summary.get("ptm")
                                else:
                                    row["status"] = "incomplete"
                            else:
                                row["status"] = "failed"
                                log = seed_dir / "inference.log"
                                if log.is_file():
                                    txt = log.read_text(errors="replace")
                                    low = txt.lower()
                                    if "out of memory" in low or "OOM" in txt or "CUDA error" in txt:
                                        row["note"] = "OOM"
                                    elif "allocateasync" in low or "allocate_tensor" in low or "runtime execution failed" in low:
                                        row["note"] = "OOM (TRT runtime)"
                                    else:
                                        row["note"] = f"exit {d.get('exit_code')}"
                            tj = seed_dir / "timing.json"
                            if tj.is_file():
                                try:
                                    row["forward_s"] = json.loads(tj.read_text()).get("runtime_s")
                                except ValueError:
                                    pass
                        rows.append(row)
    return rows


def write_csv(rows: list[dict]) -> Path:
    DOCS.mkdir(parents=True, exist_ok=True)
    p = DOCS / "results.csv"
    with p.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return p


# ── derived views ──────────────────────────────────────────────────────────

def _pref(r: dict) -> int:
    try:
        return PREFERENCE.index((r["series"], r["variant"]))
    except ValueError:
        return len(PREFERENCE)      # mem85 / hybrid: table only


def _plot_rows(rows: list[dict]) -> list[dict]:
    """One row per (target, sto, model, backend): the current state, the most
    preferred *ok* run, else the most preferred run that exists."""
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        if _pref(r) >= len(PREFERENCE):
            continue
        groups.setdefault((r["target"], r["sto"], r["model"], r["backend"]), []).append(r)
    out = []
    for g in groups.values():
        g.sort(key=_pref)
        ok = [r for r in g if r["status"] == "ok"]
        out.append(ok[0] if ok else g[0])
    return out


def _torch_base(prow: list[dict]) -> dict[tuple, float]:
    return {(r["target"], r["sto"], r["model"]): r["forward_s"] for r in prow
            if r["backend"] == "torch" and r["status"] == "ok" and r["forward_s"]}


def _speedups(prow: list[dict], base: dict, model: str, backend: str) -> list[float]:
    return [base[(r["target"], r["sto"], r["model"])] / r["forward_s"] for r in prow
            if r["model"] == model and r["backend"] == backend and r["status"] == "ok" and r["forward_s"]
            and (r["target"], r["sto"], r["model"]) in base]


def _improvements(rows: list[dict]) -> list[dict]:
    """v1 → v2 forward-time ratio per (target, model, backend), warm preferred within a series."""
    best: dict[tuple, dict] = {}
    for r in rows:
        if r["status"] != "ok" or not r["forward_s"] or r["variant"] in ("mem85", "hybrid"):
            continue
        k = (r["target"], r["sto"], r["model"], r["backend"], r["series"])
        if k not in best or (r["variant"] == "warm" and best[k]["variant"] != "warm"):
            best[k] = r
    out = []
    for (tg, sto, m, b, s), r1 in best.items():
        if s != "v1":
            continue
        r2 = best.get((tg, sto, m, b, "v2"))
        if r2:
            out.append({"target": tg, "sto": sto, "tokens": r1["tokens"], "model": m, "backend": b,
                        "v1_forward_s": r1["forward_s"], "v2_forward_s": r2["forward_s"],
                        "improvement": r1["forward_s"] / r2["forward_s"]})
    return out


def _largest(prow: list[dict], model: str, backend: str) -> int | None:
    oks = [r["tokens"] for r in prow if r["model"] == model and r["backend"] == backend and r["status"] == "ok"]
    return max(oks) if oks else None


def _agreement() -> dict[tuple, dict]:
    """(model, backend) → {median TM, min TM, n, low: [(target, tokens, tm)]} from agreement.csv, current-state labels."""
    p = DOCS / "agreement.csv"
    if not p.is_file():
        return {}
    rank = lambda lab: (0 if "v2-warm" in lab else 1 if "v2" in lab else 2 if lab.endswith("warm")  # noqa: E731
                        else 3 if ("mem85" not in lab and "hybrid" not in lab) else 9)
    best: dict[tuple, dict] = {}
    with p.open() as fh:
        for r in csv.DictReader(fh):
            if not r.get("tm_score"):
                continue
            lab = r["backend"]
            if rank(lab) >= 9:
                continue
            k = (r["target"], r["sto"], r["model"], lab.split("-")[0])
            if k not in best or rank(lab) < rank(best[k]["backend"]):
                best[k] = r
    out: dict[tuple, dict] = {}
    for (tg, sto, m, b), r in best.items():
        d = out.setdefault((m, b), {"tms": [], "low": []})
        tm = float(r["tm_score"])
        d["tms"].append(tm)
        if tm < 0.9:
            d["low"].append((tg, int(r["tokens"]), tm))
    for d in out.values():
        d["median"] = statistics.median(d["tms"])
        d["min"] = min(d["tms"])
        d["n"] = len(d["tms"])
        d["low"].sort(key=lambda x: x[1])
    return out


# ── figures ────────────────────────────────────────────────────────────────

FIGS = ["forward_time", "wall_time", "speedup", "improvement", "agreement", "plddt"]


def figures_mpl() -> dict[str, Path] | None:
    """bench/plot.py (matplotlib, publication style) in an isolated `uv run` env. None if unavailable."""
    if not shutil.which("uv"):
        return None
    env = dict(os.environ)
    env["UV_CACHE_DIR"] = os.environ.get("RESIDUDES_ENGINE_UV_CACHE_DIR") or str(ROOT / "runs" / "_setup" / "uv-cache")
    env.pop("VIRTUAL_ENV", None)
    res = subprocess.run(["uv", "run", "--quiet", str(ROOT / "bench" / "plot.py")], env=env,
                         capture_output=True, text=True)
    if res.returncode != 0:
        print("plot.py failed, falling back to stdlib SVG:", res.stderr.strip()[-300:], file=sys.stderr)
        return None
    return {n: DOCS / f"{n}.png" for n in FIGS if (DOCS / f"{n}.png").exists()}


def _panels(rows: list[dict], metric: str, backends=BACKENDS) -> list[dict]:
    panels = []
    for m in MODELS:
        pts = {b: [] for b in backends}
        failed = {b: [] for b in backends}
        for r in _plot_rows(rows):
            if r["model"] != m or r["backend"] not in backends:
                continue
            if r["status"] == "ok" and r[metric] is not None:
                pts[r["backend"]].append((r["tokens"], r[metric], f"{NAMES[m]} · {r['label']} · {r['target']}: {r[metric]:.1f}"))
            elif r["status"] == "failed":
                failed[r["backend"]].append((r["tokens"], f"{r['label']} failed at {r['tokens']} tokens: {r['note']}"))
        panels.append({"title": NAMES[m], "points": pts, "failed": {k: v for k, v in failed.items() if v}})
    return panels


def figures_svg(rows: list[dict]) -> dict[str, Path]:
    """Dependency-free fallback (SVG) for the two essential figures."""
    out = {}
    out["forward_time"] = svgchart.small_multiples(
        _panels(rows, "forward_s"), series=BACKENDS, title="Forward-pass time by backend", subtitle="",
        x_label="tokens", y_label="seconds (log)", log_y=True, y_min=1, colors=COLORS,
        out_path=DOCS / "forward_time.svg", note="× = run failed")
    prow = _plot_rows(rows)
    base = _torch_base(prow)
    sp_panels = []
    for m in MODELS:
        pts = {b: [] for b in ("trt", "jax")}
        for r in prow:
            k = (r["target"], r["sto"], r["model"])
            if r["model"] == m and r["backend"] in pts and r["status"] == "ok" and r["forward_s"] and k in base:
                pts[r["backend"]].append((r["tokens"], base[k] / r["forward_s"], f"{r['label']} {r['target']}"))
        sp_panels.append({"title": NAMES[m], "points": pts, "failed": {}})
    out["speedup"] = svgchart.small_multiples(
        sp_panels, series=["trt", "jax"], title="Forward-pass speed-up over PyTorch", subtitle="",
        x_label="tokens", y_label="speed-up (×)", log_y=False, y_min=0, reference_y=1.0, colors=COLORS,
        out_path=DOCS / "speedup.svg")
    return out


# ── documents ──────────────────────────────────────────────────────────────

def _names_list() -> str:
    names = [NAMES[m] for m in MODELS]
    return ", ".join(names[:-1]) + " and " + names[-1] if len(names) > 1 else names[0]


def _fmt_x(v: float | None, nd: int = 2) -> str:
    return "–" if v is None else f"{v:.{nd}f}×"


def _median_max(xs: list[float]) -> tuple[float | None, float | None]:
    return (statistics.median(xs), max(xs)) if xs else (None, None)


def write_summary(rows: list[dict], figs: dict[str, Path]) -> Path:
    prow = _plot_rows(rows)
    base = _torch_base(prow)
    gpu = next((r["gpu"] for r in rows if r["gpu"]), "(GPU unknown)")
    gpu_short = gpu.replace("NVIDIA ", "").replace(" Max-Q Workstation Edition", "")
    n_ok = sum(r["status"] == "ok" for r in rows)
    sizes = sorted({r["tokens"] for r in rows})
    sp = {(m, b): _median_max(_speedups(prow, base, m, b)) for m in MODELS for b in ("trt", "jax")}
    # the README figures (docs/figures/make_figures.py) read the medians from here
    (DOCS / "speedup_summary.json").write_text(json.dumps({
        "gpu": gpu_short, "inputs": len(sizes), "tokens": [sizes[0], sizes[-1]],
        "models": {m: {b: sp[(m, b)] for b in ("trt", "jax")} for m in MODELS}}, indent=2) + "\n")
    hyb = [r for r in rows if r["variant"] == "hybrid" and r["status"] == "ok" and r["forward_s"]
           and (r["target"], r["sto"], r["model"]) in base]
    hyb_sp = _median_max([base[(r["target"], r["sto"], r["model"])] / r["forward_s"] for r in hyb])
    imp = _improvements(rows)
    agr = _agreement()
    largest_all = {m: min(_largest(prow, m, b) or 0 for b in ("trt", "torch")) for m in MODELS}

    def imp_med(m, b):
        xs = [i["improvement"] for i in imp if i["model"] == m and i["backend"] == b]
        return statistics.median(xs) if xs else None

    trt_range = [sp[(m, "trt")][0] for m in MODELS if sp[(m, "trt")][0]]
    trt_max = max((sp[(m, "trt")][1] for m in MODELS if sp[(m, "trt")][1]), default=None)
    jax_range = [sp[(m, "jax")][0] for m in MODELS if sp[(m, "jax")][0]]

    L = [
        "# Benchmark",
        "",
        f"{_names_list()} run to {max(largest_all.values())} tokens on a single {gpu_short}. "
        f"The TensorRT backend finishes the forward pass {min(trt_range):.1f} to {max(trt_range):.1f}× faster than each "
        f"model's own PyTorch path (median over {len(sizes)} inputs from {sizes[0]} to {sizes[-1]} tokens; up to "
        f"{trt_max:.1f}× at 2000 tokens). The JAX backend, a Haiku port of the pairformer and diffusion modules that "
        f"reuses the same weights (for AlphaFold 3, DeepMind's own JAX model), is {min(jax_range):.1f} to "
        f"{max(jax_range):.1f}× faster than PyTorch. "
        f"Same inputs, weights, seed and settings on every backend; {n_ok} completed runs.",
        "",
        "| | " + " | ".join(NAMES[m] for m in MODELS) + " |",
        "|---|" + "---:|" * len(MODELS),
        "| TensorRT vs PyTorch, median (max) | " + " | ".join(
            f"{_fmt_x(sp[(m,'trt')][0])} ({_fmt_x(sp[(m,'trt')][1])})" for m in MODELS) + " |",
        "| JAX vs PyTorch, median (max) | " + " | ".join(
            f"{_fmt_x(sp[(m,'jax')][0])} ({_fmt_x(sp[(m,'jax')][1])})" for m in MODELS) + " |",
    ]
    if hyb:
        L.append("| JAX trunk + TensorRT diffusion, median (max) | " + " | ".join(
            (f"{_fmt_x(hyb_sp[0])} ({_fmt_x(hyb_sp[1])})" if m == "bt2" else "–") for m in MODELS) + " |")
    L.append("| Largest input completed, PyTorch / TensorRT / JAX | " + " | ".join(
        f"{_largest(prow, m, 'torch') or '–'} / {_largest(prow, m, 'trt') or '–'} / {_largest(prow, m, 'jax') or '–'}" for m in MODELS) + " |")
    if imp:
        L.append("| Gain of this release over the first measurement, TensorRT / JAX | " + " | ".join(
            f"{_fmt_x(imp_med(m,'trt'))} / {_fmt_x(imp_med(m,'jax'))}" for m in MODELS) + " |")
    if agr:
        L.append("| Top model vs PyTorch top model, median TM-score, TensorRT / JAX | " + " | ".join(
            f"{agr.get((m,'trt'),{}).get('median', float('nan')):.2f} / {agr.get((m,'jax'),{}).get('median', float('nan')):.2f}" for m in MODELS) + " |")
    L += ["", "Settings are the CASP17 production defaults (1 seed, 5 diffusion samples, 10 recycles); inputs are "
          "CASP17 targets with MSAs and templates. Forward pass = the model writer's `timing.json`; JAX numbers are "
          "with a hot compile cache (the first run at a new shape adds compilation, see the details).", ""]

    n_fig = 0
    if "speedup" in figs:
        n_fig += 1
        L += [f"![speed-up](benchmark/{figs['speedup'].name})", "",
              f"**Figure {n_fig}. Forward-pass speed-up over the PyTorch backend.** Same input, weights and seed; "
              f"dashed line = parity. TensorRT grows with input size as the O(N²) pairformer takes over; Boltz-2's "
              f"TensorRT gain is flatter because its PyTorch path already uses fused cuEquivariance triangle kernels. "
              + ("The dashed diamonds are the Boltz-2 hybrid that keeps the JAX pairformers and runs the diffusion "
                 "token transformer on its TensorRT engine. " if hyb else "")
              + ("AlphaFold 3's PyTorch baseline is the port in this repository and its JAX column is DeepMind's own "
                 "implementation, which the TensorRT path approaches from 1000 tokens up." if "af3" in MODELS else ""), ""]
    if "improvement" in figs and imp:
        n_fig += 1
        L += [f"![gain](benchmark/{figs['improvement'].name})", "",
              f"**Figure {n_fig}. What this release changed.** Forward time of the first measurement divided by "
              f"the current one, same input and backend. The JAX gains come from keeping activations in bf16 so "
              f"the triangle attention runs on cuDNN flash attention instead of XLA's materialised N³ softmax, and "
              f"from compiling the pairformer as one scanned body; Boltz-2's TensorRT gain is the diffusion token "
              f"transformer engine, which the first release left in PyTorch; the TensorRT gains at 1000+ tokens "
              f"come from choosing the best-fitting engine of a size ladder.", ""]
    if "agreement" in figs and agr:
        n_fig += 1
        lows = sorted({(t, tok) for d in agr.values() for (t, tok, _) in d["low"]}, key=lambda x: x[1])
        low_txt = (" Low points are the same targets on every backend: " +
                   ", ".join(f"{t} ({tok} tokens)" for t, tok in lows) +
                   ". These are a low-confidence target (top pLDDT about 50) and multi-chain assemblies whose chain "
                   "placement is ambiguous, where any change in numerics, even a re-run of PyTorch itself, moves the "
                   "top-ranked sample. Per-chain folds and pLDDT/pTM agree throughout.") if lows else ""
        if "af3" in MODELS:
            low_txt += (" AlphaFold 3 draws a different random 1024-row MSA subset at every recycle, so two of its runs "
                        "differ more than two runs of the other models even on one backend; with the subset fixed, "
                        "the port and DeepMind's model agree to bf16 rounding (alphafold3.md).")
        L += [f"![agreement](benchmark/{figs['agreement'].name})", "",
              f"**Figure {n_fig}. The backends predict the same structures.** TM-score of each backend's top-ranked "
              f"model against the PyTorch top model, complex-level (chains concatenated)." + low_txt, ""]

    L += ["## Size ceiling", "",
          "| model | PyTorch | TensorRT | JAX |", "|---|---:|---:|---:|"]
    for m in MODELS:
        L.append(f"| {NAMES[m]} | {_largest(prow, m, 'torch') or '–'} | {_largest(prow, m, 'trt') or '–'} | {_largest(prow, m, 'jax') or '–'} |")
    mem85 = [r for r in rows if r["variant"] == "mem85" and r["series"] == "v2" and r["status"] == "ok"]
    L += ["", "Tokens of the largest input that completed on the 96 GB card with the shipped defaults. "
          + ("JAX reaches " + ", ".join(f"{NAMES[r['model']]} {r['tokens']}" for r in sorted(mem85, key=lambda r: r['tokens'])) +
             " tokens when the XLA memory cap is raised from 0.65 to 0.85 of the card (`XLA_PYTHON_CLIENT_MEM_FRACTION=0.85`)."
             if mem85 else ""), ""]

    L += ["## Compared with the literature", "",
          f"{PAPER[0]} compiles OpenFold, an AlphaFold 2-class model, with TensorRT and reports a 2.51× forward-pass "
          f"speed-up over PyTorch on the same GPU class, with 949 residues as the largest input that fits "
          f"(an 80 GB H100 ran out of memory there), and TM-scores against CASP14 targets as the accuracy check.",
          "",
          f"residudes-engine reaches {min(trt_range):.1f} to {max(trt_range):.1f}× median and up to {trt_max:.1f}× on "
          f"AlphaFold 3-class models, whose pairformer and diffusion stacks are heavier than the Evoformer, and runs "
          f"{max(largest_all.values())} tokens for every model on TensorRT and PyTorch, twice the preprint's ceiling. "
          "The JAX path adds a second, independently implemented fast backend with the same weights; where TensorRT "
          "engines are missing (a new GPU, no build yet) it still beats PyTorch. pLDDT, pTM and TM-score between "
          "backends are published for every run (`benchmark/agreement.csv`), including the inputs where the "
          "top-ranked sample moves, and the matrix, figures and tables are regenerated by three commands from this "
          "repository.",
          "", "## Reproduce", "",
          "```sh", "bench/run.py --suffix v2                      # submit the matrix (Slurm, one job per model × backend × input)",
          "bench/run.py --backends jax --suffix v2-warm  # second JAX pass once the compile cache is hot",
          "bench/collect.py                             # rebuild this page, the details page and the figures",
          "```", "",
          "Every run, variant and reading note: [`benchmark_details.md`](benchmark_details.md). "
          "What changed between the two series, and why: the list at the end of that page."]
    p = ROOT / "docs" / "benchmark.md"
    p.write_text("\n".join(L) + "\n")
    return p


def write_details(rows: list[dict], figs: dict[str, Path]) -> Path:
    prow = _plot_rows(rows)
    base = _torch_base(prow)
    gpu = next((r["gpu"] for r in rows if r["gpu"]), "(GPU unknown)")
    ok = [r for r in rows if r["status"] == "ok"]
    has_v2 = any(r["series"] == "v2" for r in rows)
    L = [
        "# Benchmark details",
        "",
        f"Measured with `bench/run.py` on **{gpu}** (one GPU per job, Slurm, site `ada`), "
        f"{time.strftime('%Y-%m-%d')}. Each run is one seed, 5 diffusion samples, 10 recycles, the CASP17 "
        f"production settings, on prepared AF3-dialect inputs (MSAs + templates). "
        f"{len(ok)}/{len(rows)} runs completed; pending or failed runs are listed, not dropped. "
        f"The summary is [`benchmark.md`](benchmark.md).",
        "",
        "Backends: **trt** = TensorRT engines (pairformer, confidence pairformer and, for Boltz-2 in v2, the diffusion "
        "token transformer, BF16; `cuequiv` RTX PRO 6000 engines; for AlphaFold 3 the trunk pairformer and the "
        "diffusion token transformer, `vanilla` engines, with the rest of the network in the PyTorch port), "
        "**torch** = the model's own PyTorch (for AlphaFold 3, the port in `core/alphafold3/af3torch`), "
        "**jax** = the JAX hot-swap of the same modules (for AlphaFold 3, DeepMind's published JAX model). "
        "Forward time is the writer's `timing.json` "
        "(`runtime_s`, model forward only). Wall time is the whole process as residudes-engine measured it, so it "
        "includes model and engine loading and, for jax, XLA compilation (cached per shape after the first run).",
        "",
    ]
    if has_v2:
        L += ["Two series are listed: **v1** is the first full matrix (2026-11-02, the code as first published) and "
              "**v2** the same matrix after the optimisations listed under *What changed* below: "
              "Boltz-2's diffusion token transformer on its TensorRT engine, per-template chunking on every backend, "
              "JAX activations kept in bf16 (cuDNN flash attention instead of XLA's materialised softmax), "
              "layer-stack scan for the JAX pairformers, equal recycle counts, and best-fit engine selection from an "
              "engine ladder (a 2048-token engine beside the 3072–5120 ones). Figures show the current state (v2 where it exists).",
              ""]
    captions = {
        "forward_time": "Forward-pass time by backend (log scale). × = run out of memory (colour = backend).",
        "wall_time": "End-to-end wall time: model load, engine load or XLA compile, featurisation, inference, writing.",
        "speedup": "Forward-pass speed-up over the PyTorch backend; dashed diamonds = Boltz-2 JAX trunk + TensorRT diffusion.",
        "improvement": "Forward time of the first matrix (v1) divided by the optimised code (v2), same input and backend.",
        "agreement": "TM-score of each backend's top-ranked model against the PyTorch top model (complex level).",
        "plddt": "Top-sample pLDDT by backend: same input, weights and seed.",
    }
    for i, n in enumerate(FIGS, 1):
        if n in figs:
            L += [f"![{n}](benchmark/{figs[n].name})", "", f"**Figure D{i}.** {captions.get(n, n)}", ""]

    L += ["## Summary tables", "",
          "| model | median trt speed-up | median jax speed-up (warm where available) | inputs compared |",
          "|---|---:|---:|---:|"]
    for m in MODELS:
        t = _speedups(prow, base, m, "trt"); j = _speedups(prow, base, m, "jax")
        L.append(f"| {NAMES[m]} | {_fmt_x(statistics.median(t)) if t else '-'} | {_fmt_x(statistics.median(j)) if j else '-'} | {max(len(t), len(j))} |")
    imp = _improvements(rows)
    if imp:
        L += ["", "Forward-time gain of the optimised code (v2) over the first matrix (v1), same input and backend:", "",
              "| model | backend | median v1 → v2 | range | inputs |", "|---|---|---:|---|---:|"]
        for m in MODELS:
            for b in BACKENDS:
                xs = [i["improvement"] for i in imp if i["model"] == m and i["backend"] == b]
                if xs:
                    L.append(f"| {NAMES[m]} | {b} | {statistics.median(xs):.2f}× | {min(xs):.2f}–{max(xs):.2f}× | {len(xs)} |")
    has_mem85 = any(r["variant"] == "mem85" for r in rows)
    L += ["", "Largest input completed and first failure, per model × backend (current code and site defaults"
          + ("; `mem85` = JAX with `mem_fraction = 0.85` instead of the default 0.65" if has_mem85 else "") + "):", "",
          "| model | backend | largest completed | first failure | pending |", "|---|---|---:|---|---:|"]
    for m in MODELS:
        for b in BACKENDS:
            _limits_row(L, NAMES[m], b, [r for r in prow if r["model"] == m and r["backend"] == b])
        if has_mem85:
            rs = [r for r in rows if r["model"] == m and r["variant"] == "mem85"]
            latest = max((r["series"] for r in rs), default=None)
            rs = [r for r in rs if r["series"] == latest]
            if rs:
                _limits_row(L, NAMES[m], f"jax (mem85, {latest})", rs)
    hyb = [r for r in rows if r["variant"] == "hybrid" and r["status"] == "ok" and r["forward_s"]]
    if hyb:
        L += ["", "Boltz-2 hybrid (`RESIDUDES_ENGINE_JAX_TRT_DIFFUSION=1`: JAX pairformers + TensorRT diffusion token transformer):", "",
              "| tokens | hybrid s | PyTorch s | TensorRT s | JAX s | speed-up vs PyTorch |", "|---:|---:|---:|---:|---:|---:|"]
        cur = {(r["target"], r["sto"], r["model"], r["backend"]): r for r in prow}
        ratios = []
        for r in sorted(hyb, key=lambda r: r["tokens"]):
            k = (r["target"], r["sto"], r["model"])
            b = base.get(k)
            t = cur.get(k + ("trt",), {}).get("forward_s"); j = cur.get(k + ("jax",), {}).get("forward_s")
            if b:
                ratios.append(b / r["forward_s"])
                L.append(f"| {r['tokens']} | {r['forward_s']:.1f} | {b:.1f} | {t:.1f} | {j:.1f} | {b / r['forward_s']:.2f}× |"
                         if t and j else f"| {r['tokens']} | {r['forward_s']:.1f} | {b:.1f} | - | - | {b / r['forward_s']:.2f}× |")
        if ratios:
            L.append(f"| median | | | | | {statistics.median(ratios):.2f}× |")

    L += ["", "## All runs", "",
          "| target | tokens | model | run | forward s | wall s | top pLDDT | pTM | status |",
          "|---|---:|---|---|---:|---:|---:|---:|---|"]
    sorder = {s: i for i, (_, s) in enumerate(SERIES)}
    vorder = {v: i for i, (_, v) in enumerate(VARIANTS)}
    for r in sorted(rows, key=lambda r: (r["tokens"], MODELS.index(r["model"]), BACKENDS.index(r["backend"]),
                                         sorder[r["series"]], vorder[r["variant"]])):
        f = lambda v, nd=1: ("-" if v is None else f"{v:.{nd}f}")  # noqa: E731
        st = r["status"] + (f" ({r['note']})" if r["note"] else "")
        L.append(f"| {r['target']} {r['sto']} | {r['tokens']} | {r['model']} | {r['label']} | "
                 f"{f(r['forward_s'])} | {f(r['wall_s'],0)} | {f(r['plddt'])} | {f(r['ptm'],3)} | {st} |")

    if has_v2:
        L += ["", "## What changed between v1 and v2", "",
              "- Boltz-2 diffusion: the token transformer runs on its own TensorRT engine (v1 left it in PyTorch).",
              "- Templates are processed one at a time on every backend; Boltz-2 at 1627 tokens no longer runs out of memory.",
              "- JAX activations stay in bf16 and masks are cast to the compute dtype, so triangle attention runs on cuDNN flash attention instead of XLA's materialised N³ softmax. This is the largest JAX gain.",
              "- The JAX pairformers compile as one scanned layer body instead of 48 unrolled blocks: shorter compiles, smaller executables, same numerics.",
              "- Recycle counts are the same on every backend (10); the first matrix had not guaranteed that.",
              "- An engine ladder: a 2048-token TensorRT engine built next to the large one, and `predict` picks the smallest engine that covers the input.",
              "- The Boltz-2 hybrid: JAX pairformers with the diffusion token transformer on TensorRT (`RESIDUDES_ENGINE_JAX_TRT_DIFFUSION=1`).",
              "- Tried and not adopted: `torch.compile` on the Boltz-2 diffusion module, and a separate JAX environment with tokamax fused attention and cuEquivariance-JAX kernels (`core/jaxenv`, kept as an option).",
              ]
    L += ["", "## Reading the numbers", "",
          "- Forward time is what the backend changes; wall time adds model/engine loading and I/O, which are the same order as the forward pass below ~300 tokens, so end-to-end differences are smaller than forward-pass differences.",
          "- pLDDT/pTM across backends for the same input differ only by BF16 / kernel-order effects (compare rows with the same target and model). `bench/agreement.py` adds TM-score / RMSD of each run's top model against the PyTorch one (`benchmark/agreement.csv`).",
          "- `jax` rows are the first run at each shape: XLA compiles inside the forward pass, so both forward and wall time include compilation. `-warm` rows are a second pass with the compile cache hot; the figures use the warm run where it exists.",
          "- `-mem85` rows rerun a JAX job that OOM'd with the site default `mem_fraction = 0.65` (a 62 GB cap for XLA on this 96 GB card) with the cap raised to 0.85. The default stays at 0.65 because the same site file also serves 24 GB cards; raise it per run with `XLA_PYTHON_CLIENT_MEM_FRACTION=0.85`, or set `mem_fraction` in the site file.",
          "- `OOM` on both `trt` and `torch` for the same input is the model's own PyTorch side running out of memory; the pairformer backend is not involved. In v1 this was Boltz-2's template stack at 1627 tokens; v2 chunks it per template.",
          "- `-hybrid` rows (Boltz-2 only) keep the JAX pairformers and run the diffusion token transformer on its TensorRT engine; they are the fastest Boltz-2 configuration from ~600 tokens up.",
          "- TensorRT rows in v2 use the best-fitting engine of a ladder (`boltz2_cuequiv_{2048,5120}`, `openfold3_p2_cuequiv_{2048,3072}`, `protenix_v2_cuequiv_{2048,4096}`); the 2048-token engines were built with `residudes-engine engines build --max-seqlen 2048 --kernel cuequiv` (6–20 min each).",
          "- AlphaFold 3 joined the matrix after the first series, so it has v2 rows only and no v1 → v2 gain; its PyTorch baseline is the port, its JAX rows are DeepMind's model (`--backend jax`), and its TensorRT rows use `alphafold3_vanilla_2048`.",
          "- Inputs are CASP17 targets on the team's shared filesystem (`bench/targets.json`); re-run with `bench/run.py [--suffix v2]` and rebuild with `bench/collect.py`.",
          "", "Raw data: [`benchmark/results.csv`](benchmark/results.csv), [`benchmark/agreement.csv`](benchmark/agreement.csv). "
          "Figures: `uv run bench/plot.py` (matplotlib; PNG and PDF)."]
    p = ROOT / "docs" / "benchmark_details.md"
    p.write_text("\n".join(L) + "\n")
    return p


def _limits_row(out: list[str], model_name: str, label: str, rs: list[dict]) -> None:
    oks = [r for r in rs if r["status"] == "ok"]
    bad = [r for r in rs if r["status"] == "failed"]
    pend = sum(1 for r in rs if r["status"] in ("pending", "incomplete"))
    top = max(oks, key=lambda r: r["tokens"]) if oks else None
    fb = min(bad, key=lambda r: r["tokens"]) if bad else None
    out.append(f"| {model_name} | {label} | {top['tokens'] if top else '-'} | "
               f"{(str(fb['tokens']) + ' (' + fb['target'] + (', ' + fb['note'] if fb['note'] else '') + ')') if fb else '-'} | "
               f"{pend or ''} |")


def main() -> int:
    rows = collect()
    csv_path = write_csv(rows)
    figs = figures_mpl() or figures_svg(rows)
    summary = write_summary(rows, figs)
    details = write_details(rows, figs)
    n_ok = sum(r["status"] == "ok" for r in rows)
    n_fail = sum(r["status"] == "failed" for r in rows)
    n_pend = len(rows) - n_ok - n_fail
    print(f"{n_ok}/{len(rows)} ok, {n_fail} failed, {n_pend} pending → {summary}, {details}, {csv_path}, {len(figs)} figures")
    if n_fail:
        print("failed:", "; ".join(f"{r['target']} {r['model']}-{r['label']} ({r['note']})" for r in rows if r["status"] == "failed"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
