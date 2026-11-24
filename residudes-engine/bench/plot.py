#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib>=3.8", "pandas>=2.0"]
# ///
"""Benchmark figures in a plain publication style (matplotlib only).

    uv run bench/plot.py        # → docs/benchmark/*.png (300 dpi) and *.pdf

Conventions: one row of panels (one per model) at double-column width,
panel letters a, b, c, ..., the PyTorch baseline in black, TensorRT blue and JAX green
(Okabe–Ito), a distinct marker per backend, legend inside the axes, no grid,
no titles or notes inside the figure — captions live in the documents.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.transforms as transforms  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs" / "benchmark"
CSV = DOCS / "results.csv"

MODELS = json.loads((ROOT / "bench" / "targets.json").read_text())["models"]
NAMES = {"pt2": "Protenix v2", "of3": "OpenFold 3", "bt2": "Boltz-2", "af3": "AlphaFold 3"}
BACKENDS = ["trt", "torch", "jax"]
COLORS = {"trt": "#0072B2", "torch": "#222222", "jax": "#009E73"}
MARKERS = {"trt": "o", "torch": "s", "jax": "^"}
LABELS = {"trt": "TensorRT", "torch": "PyTorch", "jax": "JAX"}
PREFERENCE = [("v2", "warm"), ("v2", ""), ("v1", "warm"), ("v1", "")]

RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "xtick.major.size": 3, "ytick.major.size": 3, "xtick.minor.size": 1.8, "ytick.minor.size": 1.8,
    "xtick.direction": "out", "ytick.direction": "out",
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "legend.handlelength": 1.8, "legend.borderaxespad": 0.3,
    "lines.linewidth": 1.0, "lines.markersize": 3.6, "lines.markeredgewidth": 0.6,
    "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42, "ps.fonttype": 42,
}


def _pref(r) -> int:
    key = (str(r.get("series", "v1")), "" if pd.isna(r.get("variant", "")) else str(r.get("variant", "")))
    return PREFERENCE.index(key) if key in PREFERENCE else len(PREFERENCE)


def load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Current-state rows (one per target/model/backend), v1→v2 gains, hybrid rows."""
    df = pd.read_csv(CSV, keep_default_na=False, na_values=[""])
    if "series" not in df.columns:
        df["series"] = "v1"
        df["variant"] = ""
    df["variant"] = df["variant"].fillna("")
    df["pref"] = df.apply(_pref, axis=1)
    cur = []
    for _, g in df[df.pref < len(PREFERENCE)].groupby(["target", "sto", "model", "backend"], sort=False):
        g = g.sort_values("pref")
        ok = g[g.status == "ok"]
        cur.append((ok if len(ok) else g).iloc[0])
    out = pd.DataFrame(cur)
    okdf = df[(df.status == "ok") & df.forward_s.notna() & (~df.variant.isin(["mem85", "hybrid"]))].sort_values("pref")
    best = okdf.groupby(["target", "sto", "model", "backend", "series"], sort=False).first().reset_index()
    v1 = best[best.series == "v1"].set_index(["target", "sto", "model", "backend"])
    v2 = best[best.series == "v2"].set_index(["target", "sto", "model", "backend"])
    both = v1.join(v2, lsuffix="_v1", rsuffix="_v2", how="inner").reset_index()
    imp = pd.DataFrame({"target": both.target, "sto": both.sto, "tokens": both.tokens_v1, "model": both.model,
                        "backend": both.backend, "improvement": both.forward_s_v1 / both.forward_s_v2,
                        "status": "ok"}) if len(both) else pd.DataFrame()
    hyb = df[(df.variant == "hybrid") & (df.status == "ok") & df.forward_s.notna()].copy()
    return out, imp, hyb


def _panels(ylabel: str, log: bool = False, height: float = 2.25):
    fig, axes = plt.subplots(1, len(MODELS), figsize=(2.4 * len(MODELS), height), sharey=True)
    for ax, m, letter in zip(axes, MODELS, "abcdefgh"):
        ax.set_title(NAMES[m], fontweight="normal", pad=4)
        ax.text(-0.06 if ax is axes[0] else -0.02, 1.08, letter, transform=ax.transAxes,
                fontsize=9.5, fontweight="bold", va="bottom", ha="right")
        ax.set_xlabel("Tokens")
        ax.set_xlim(0, 2150)
        ax.set_xticks([0, 500, 1000, 1500, 2000])
        if log:
            ax.set_yscale("log")
    axes[0].set_ylabel(ylabel)
    return fig, axes


def _series(ax, df: pd.DataFrame, m: str, b: str, ycol: str, label: str | None = None):
    d = df[(df.model == m) & (df.backend == b) & (df.status == "ok") & df[ycol].notna()].sort_values("tokens")
    if len(d):
        ax.plot(d.tokens, d[ycol], color=COLORS[b], marker=MARKERS[b], label=label or LABELS[b],
                markerfacecolor="white" if b == "torch" else COLORS[b], zorder=3 if b == "torch" else 2)
    return len(d)


def _failed(ax, df: pd.DataFrame, m: str, backends):
    """Out-of-memory runs as small crosses on a shelf below the top edge, one shelf per backend."""
    tr = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    for i, b in enumerate(backends):
        f = df[(df.model == m) & (df.backend == b) & (df.status == "failed")]
        if len(f):
            ax.scatter(f.tokens, [0.97 - 0.05 * i] * len(f), transform=tr, marker="x", s=18,
                       linewidths=0.9, color=COLORS[b], zorder=4, clip_on=False)


def _shared_legend(axes, loc: str = "upper left") -> None:
    """One horizontal legend above the panels with every label that appears in any panel
    (keeps the axes free of boxes; `loc` is accepted for API compatibility)."""
    handles, labels = [], []
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h); labels.append(l)
    if handles:
        fig = axes[0].figure
        fig.legend(handles, labels, loc="lower center", ncol=len(handles), bbox_to_anchor=(0.5, 1.0),
                   frameon=False, columnspacing=1.6, handletextpad=0.5)


def _save(fig, name: str) -> list[Path]:
    out = []
    for ext in ("png", "pdf"):
        p = DOCS / f"{name}.{ext}"
        fig.savefig(p)
        out.append(p)
    plt.close(fig)
    return out


def fig_time(df: pd.DataFrame, ycol: str, name: str, ylabel: str) -> list[Path]:
    fig, axes = _panels(ylabel, log=True)
    for ax, m in zip(axes, MODELS):
        for b in BACKENDS:
            _series(ax, df, m, b, ycol)
        _failed(ax, df, m, BACKENDS)
    _shared_legend(axes, "upper left")
    fig.tight_layout(w_pad=1.2)
    return _save(fig, name)


def fig_ratio(df: pd.DataFrame, ycol: str, name: str, ylabel: str, backends, ylim=None, legend_loc="upper left",
              extra: pd.DataFrame | None = None, extra_label: str | None = None) -> list[Path]:
    fig, axes = _panels(ylabel)
    for ax, m in zip(axes, MODELS):
        ax.axhline(1.0, color="#999999", lw=0.7, ls=(0, (4, 3)), zorder=1)
        for b in backends:
            _series(ax, df, m, b, ycol)
        if extra is not None and m in set(extra.model):
            e = extra[extra.model == m].sort_values("tokens")
            ax.plot(e.tokens, e[ycol], color="#CC79A7", marker="D", ls="--", label=extra_label, zorder=2)
    if ylim:
        axes[0].set_ylim(*ylim)
    _shared_legend(axes, legend_loc)
    fig.tight_layout(w_pad=1.2)
    return _save(fig, name)


def main() -> int:
    if not CSV.exists():
        print(f"no {CSV}; run bench/collect.py first", file=sys.stderr)
        return 1
    plt.rcParams.update(RC)
    df, imp, hyb = load()
    out = []
    out += fig_time(df, "forward_s", "forward_time", "Forward pass (s)")
    out += fig_time(df, "wall_s", "wall_time", "Wall time (s)")

    # speed-up over PyTorch, same input
    base = df[(df.backend == "torch") & (df.status == "ok") & df.forward_s.notna()].set_index(["target", "sto", "model"]).forward_s
    sp = df[df.backend.isin(["trt", "jax"]) & (df.status == "ok") & df.forward_s.notna()].copy()
    sp["speedup"] = [base.get((t, s, m), float("nan")) / f for t, s, m, f in zip(sp.target, sp.sto, sp.model, sp.forward_s)]
    sp = sp[sp.speedup.notna()]
    hyb_sp = None
    if len(hyb):
        hyb = hyb.copy()
        hyb["speedup"] = [base.get((t, s, m), float("nan")) / f for t, s, m, f in zip(hyb.target, hyb.sto, hyb.model, hyb.forward_s)]
        hyb_sp = hyb[hyb.speedup.notna()]
    top = max(2.0, float(sp.speedup.max()) * 1.08) if len(sp) else 3.0
    out += fig_ratio(sp, "speedup", "speedup", "Speed-up over PyTorch (×)", ["trt", "jax"], ylim=(0.5, top),
                     legend_loc="lower right", extra=hyb_sp, extra_label="JAX + TensorRT diffusion")
    if len(imp):
        top = max(2.0, float(imp.improvement.max()) * 1.08)
        out += fig_ratio(imp, "improvement", "improvement", "Gain, first → optimised code (×)", BACKENDS, ylim=(0.8, top))

    agr = DOCS / "agreement.csv"
    if agr.exists():
        ag = pd.read_csv(agr)
        ag = ag[ag.tm_score.notna()].copy()
        ag["backend_base"] = ag.backend.str.replace(r"-.*$", "", regex=True)
        ag["rank"] = ag.backend.map(lambda b: 0 if "v2-warm" in b else 1 if "v2" in b else 2 if b.endswith("warm") else 3 if "mem85" not in b and "hybrid" not in b else 9)
        ag = ag[ag["rank"] < 9].sort_values("rank").groupby(["target", "sto", "model", "backend_base"], sort=False).first().reset_index()
        ag["backend"] = ag.backend_base
        ag["status"] = "ok"
        if len(ag):
            fig, axes = _panels("TM-score vs PyTorch model")
            for ax, m in zip(axes, MODELS):
                for b in ("trt", "jax"):
                    _series(ax, ag, m, b, "tm_score")
            axes[0].set_ylim(0, 1.05)
            _shared_legend(axes, "lower left")
            fig.tight_layout(w_pad=1.2)
            out += _save(fig, "agreement")

    fig, axes = _panels("Top-sample pLDDT")
    for ax, m in zip(axes, MODELS):
        for b in BACKENDS:
            _series(ax, df, m, b, "plddt")
    axes[0].set_ylim(0, 100)
    _shared_legend(axes, "lower left")
    fig.tight_layout(w_pad=1.2)
    out += _save(fig, "plddt")

    for p in out:
        print(p.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
