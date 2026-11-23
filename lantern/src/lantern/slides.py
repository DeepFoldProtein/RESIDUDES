"""`lantern slides T` and `lantern render T`: the target report as a Typst slide deck with structure pictures.

The deck (report/report.typ -> report/report.pdf, touying metropolis like the team's target notes) is built from the
dossier, so it can be rebuilt at any stage:
  target and construct · the priors · the n-sweep (pool confidence per stoichiometry x predictor, as a heat table)
  · decisions and the priors they rest on · the checkpoint conversations · the first-results review · structures
  (PyMOL renders of the slate, or of the best model per stoichiometry before there is a slate) · the slate
  · the agent's own slides (report/agent.typ, included as written) · agent cost per stage.

Structure pictures come from report/scenes.pml, written here and rendered by PyMOL through sbatch (never on the
login node): `lantern slides` submits the render when pictures are missing and builds the deck with what exists;
run it again when the job is done. The agent adds its own scenes with `lantern render T --pml report/pml/x.pml`.

  lantern slides T [--no-render] [--with-retro]
  lantern render T --pml FILE
"""
from __future__ import annotations

import csv
import json
import re
import shutil
import statistics
import subprocess
from pathlib import Path

from lantern import report
from lantern.dossier import Dossier, now

# site default; `lantern.toml` [tools] pymol wins, and a pymol on PATH is used before this one
PYMOL_CANDIDATES = ["/gpfs/deepfold/casp/casp17/target_notes/E2450/.venv-pymol/bin/pymol"]
SLOT_COLOURS = ["0x0072B2", "0xD55E00", "0x009E73", "0xCC79A7", "0xE69F00"]


# ----------------------------------------------------------------------------------------------- helpers
def _s(x) -> str:
    """Dossier text as a Typst string literal shown verbatim (no markup interpretation)."""
    return '#"' + str(x).replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ") + '"'


def _cell(x, size: str = "") -> str:
    return f"[#text({size})[{_s(x)}]]" if size else f"[{_s(x)}]"


def _short(x, n: int) -> str:
    x = str(x)
    return x if len(x) <= n else x[: n - 1] + "…"


def _plain(v, width: int = 200) -> str:
    """An evidence value as plain text for a slide: `key: value; ...` (no markdown)."""
    if isinstance(v, dict):
        out = "; ".join(f"{str(k).replace('_', ' ')}: {_plain(x, 90)}" for k, x in v.items())
    elif isinstance(v, list):
        out = ", ".join(_plain(x, 60) for x in v)
    else:
        out = str(v)
    return _short(out, width)


def _copies(sto: str) -> int:
    return sum(int(n) for n in re.findall(r"[A-Z](\d+)", sto.split("@")[0])) or 1


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def tool(cfg, name: str) -> str | None:
    p = cfg.extra.get("tools", {}).get(name)
    if p and Path(p).exists():
        return p
    if name == "pymol":
        return next((c for c in PYMOL_CANDIDATES if Path(c).exists()), shutil.which("pymol"))
    return shutil.which(name)


# ----------------------------------------------------------------------------------------------- data
def pool_table(pool_csv: Path) -> tuple[list[str], list[str], dict]:
    """Best and median-of-top-10 confidence (ipTM for complexes, pTM for monomers) per stoichiometry x runner."""
    by: dict = {}
    with open(pool_csv) as fh:
        for r in csv.DictReader(fh):
            v = _f(r["iptm"] if _copies(r["sto"]) > 1 else r["ptm"])
            if v is not None:
                by.setdefault((r["sto"], r["runner"]), []).append(v)
    stos = sorted({s for s, _ in by}, key=lambda s: (_copies(s), s))
    runs = sorted({r for _, r in by})
    cells = {k: (max(v), statistics.median(sorted(v, reverse=True)[:10]), len(v)) for k, v in by.items()}
    return stos, runs, cells


def figure_models(d: Dossier, ddir: Path) -> list[dict]:
    """The models to draw: the latest slate, or the best model per stoichiometry while there is none."""
    props = d.evidence.get("slate_proposal", [])
    if props and Path(props[-1]["value"]["json"]).exists():
        sl = json.loads(Path(props[-1]["value"]["json"]).read_text())
        return [{"label": f"M{m['slot']}", "name": m["model"], "path": m["path"], "sto": m.get("sto") or m["model"].split("__")[0]}
                for m in sl["models"]]
    pc = ddir / "pool.csv"
    if not pc.exists():
        return []
    best: dict = {}
    with open(pc) as fh:
        for r in csv.DictReader(fh):
            v = _f(r["ranking_score"])
            if v is not None and (r["sto"] not in best or v > best[r["sto"]][0]):
                best[r["sto"]] = (v, r)
    out = []
    for sto in sorted(best, key=lambda s: (_copies(s), s))[:6]:
        r = best[sto][1]
        out.append({"label": sto, "name": f"{sto}__{r['variant']}__{r['runner']}__s{r['seed']}_{r['sample']}", "path": r["path"],
                    "sto": sto})
    return out


def cluster_models(ddir: Path, limit: int = 8) -> list[dict]:
    """The medoid of each big cluster, as a model to render: a cluster says "these are the same shape", and this
    is the model that shows what that shape is."""
    from lantern import figures
    out = []
    for m in figures.cluster_medoids(ddir, limit):
        if Path(m["path"]).exists():
            out.append({"label": m["label"], "name": m["key"], "path": m["path"], "sto": m["sto"], "cluster": m})
    return out


def scenes_pml(models: list[dict], small: tuple[str, ...] = ()) -> str:
    """One PyMOL script for every picture the deck uses (run from report/, writes figs/*.png).

    Two rules the pictures live by. Every model is superposed on the first and drawn from one view, because the
    question these pictures answer is how they differ, and a picture that re-orients each model answers nothing.
    And a chain short enough to vanish at this scale (a peptide, a ligand-sized subunit) is drawn as sticks in the
    accent colour: on H1388 the whole target is where an 11-residue peptide sits on a 233-residue groove, and in
    plain cartoon all eight cluster medoids looked like the same blue blob."""
    lines = ["# lantern scenes: written by `lantern slides`; run with `pymol -cq scenes.pml` from report/",
             "reinitialize", "bg_color white", "set ray_opaque_background, 1", "set antialias, 2", "set ray_shadows, 0",
             "set cartoon_fancy_helices, 1", "set ray_trace_mode, 1", "set ray_trace_color, 0x404040",
             "set_color plddt_vh, [0.000, 0.325, 0.839]", "set_color plddt_h, [0.396, 0.796, 0.953]",
             "set_color plddt_l, [1.000, 0.859, 0.075]", "set_color plddt_vl, [1.000, 0.490, 0.271]"]
    for m in models:
        lines.append(f"load {m['path']}, {m['label']}")
    lines += ["remove solvent", "remove hydro", "hide everything", "show cartoon"]
    first = models[0]["label"] if models else None
    for i, m in enumerate(models):
        lab = m["label"]
        if first and lab != first:
            lines.append(f"super {lab} and name CA, {first} and name CA")
    if first:
        lines += [f"orient {first}", f"zoom {first}, 4"]     # one view for every picture: they are comparable
    for m in models:
        lab = m["label"]
        lines += ["disable all", f"enable {lab}", f"util.cbc {lab}"]
        for ch in small:                                     # the chain that would otherwise disappear
            lines += [f"color 0xD55E00, {lab} and chain {ch}",
                      f"show spheres, {lab} and chain {ch} and not hydro",
                      f"set sphere_scale, 0.45, {lab} and chain {ch}",
                      f"set cartoon_transparency, 0.45, {lab} and not chain {ch}"]
        lines += [f"png figs/{lab}_chains.png, width=1000, height=750, dpi=120, ray=1",
                  f"color plddt_vl, {lab}", f"color plddt_l, {lab} and b > 50", f"color plddt_h, {lab} and b > 70",
                  f"color plddt_vh, {lab} and b > 90",
                  f"png figs/{lab}_plddt.png, width=1000, height=750, dpi=120, ray=1"]
    same = [m for m in models if m.get("sto") == models[0].get("sto")] if models else []
    if len(same) > 1:                                  # superpose only models of one stoichiometry (the first's)
        lines.append("disable all")
        for i, m in enumerate(same):
            lines += [f"enable {m['label']}", f"color {SLOT_COLOURS[i % len(SLOT_COLOURS)]}, {m['label']}"]
        lines += [f"orient {first}", "png figs/overlay.png, width=1200, height=850, dpi=120, ray=1"]
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------------------------- rendering
def render_cmd(cfg, d: Dossier, pml: Path, actor: str) -> str:
    from lantern import ops
    pymol = tool(cfg, "pymol")
    if not pymol:
        raise FileNotFoundError("PyMOL not found: set [tools] pymol = \"/path/to/pymol\" in lantern.toml")
    ddir = Dossier.path(cfg.targets_dir, d.target).parent
    rep = ddir / "report"
    (rep / "figs").mkdir(parents=True, exist_ok=True)
    pml = pml.resolve()
    # PyMOL writes PNG; Typst embeds JPEG as is (PNG gets re-encoded and the deck grows ~4x), so convert in the job
    py = Path(pymol).with_name("python")
    to_jpg = (f" && {py} -c \"import glob,PIL.Image as I; [I.open(f).convert('RGB').save(f[:-4] + '.jpg', quality=88) "
              f"for f in glob.glob('figs/*.png')]\"") if py.exists() else ""
    job = ops._sbatch(f"cd {rep} && {pymol} -cq {pml}{to_jpg}", f"lantern-render-{d.target}", rep / "render.%j.log", gpu=False, hours=1)
    d.runs.append({"time": now(), "by": actor, "job": job, "scheduler": "slurm", "what": f"render: {pml.name}", "state": "submitted",
                   "out": str(rep / "figs")})
    d.note(actor, f"render {pml.name}: job {job}")
    return f"job {job}: PyMOL {pml.name} -> {rep / 'figs'}"


# ----------------------------------------------------------------------------------------------- the deck
def _heat(v: float | None) -> str:
    if v is None:
        return "white"
    v = max(0.0, min(1.0, v))
    r, g, b = (int(255 - (255 - c) * v) for c in (0, 114, 178))              # white -> lantern-engine blue
    return f'rgb({r}, {g}, {b})'


def _pic(figs: Path, stem: str) -> str | None:
    """figs/<stem>.jpg when the render job made one (smaller in the PDF), else the PNG, else None."""
    for ext in (".jpg", ".png"):
        if (figs / f"{stem}{ext}").exists():
            return f"figs/{stem}{ext}"
    return None


def _fit(svg: Path, with_caption: bool) -> str:
    """How to place a figure on a 16:9 slide: a wide one is limited by the page width, a tall one by its height.
    Getting this backwards crops the figure silently, which is how a deck ends up showing half a legend."""
    try:
        head = svg.read_text()[:400]
        w = float(re.search(r'width="([\d.]+)"', head).group(1))
        h = float(re.search(r'height="([\d.]+)"', head).group(1))
    except (OSError, AttributeError, ValueError):
        return "width: 92%"
    usable = 2.0 if not with_caption else 2.2         # the caption line takes a slice of the slide's height
    return "width: 92%" if w / h >= usable else f"height: {76 if with_caption else 80}%"


FIGURE_TITLES = {
    "chain_map": "The chains, and the residues a gate would use",
    "literature": "What the literature settles, and what it does not",
    "sites": "The sites the analysis found, residue by residue",
    "motif": "The motifs, and every place they can sit",
    "clusters": "The pool, clustered: the alternatives a slate hedges across",
}


def figure_summary(d: Dossier, ddir: Path, name: str) -> str:
    """The numbers behind a figure, in one line: a picture in a deck is looked at, not measured off."""
    from lantern import figures
    if name == "chain_map":
        return " · ".join(f"{chr(ord('A') + i)}: {n.split(',')[-1].strip()} {len(s)} aa"
                          for i, (n, s) in enumerate(figures.chains(d)))
    if name == "literature":
        val = figures._latest(d, "literature", "background", "reading")
        cl = figures.claims(val)
        unknown = figures._fields_matching(val, figures._UNKNOWN_F)
        sourced = sum(1 for c in cl if c["source"])
        line = f"{len(cl)} claims, {sourced} with a source"
        return line + (f" · not settled: {_short(unknown[0], 230)}" if unknown else "")
    if name == "sites":
        cs = figures.chains(d)
        letters = {chr(ord("A") + i): len(s) for i, (_, s) in enumerate(cs)}
        seqs = {chr(ord("A") + i): s for i, (_, s) in enumerate(cs)}
        recs = figures.site_records(d, letters)
        mods, _ = figures.modifications(d)
        bits = [f"{r['key']} {r['chain']}: " + ", ".join(f"{seqs[r['chain']][x - 1]}{x}" for x in r["residues"])
                for r in recs]
        return " · ".join(bits + [f"{lab} on {ch}{r}" for ch, r, lab in mods])
    if name == "motif":
        cs = {chr(ord("A") + i): s for i, (_, s) in enumerate(figures.chains(d))}
        bits = []
        for m in figures.motifs(d):
            for ch in ([m["chain"]] if m["chain"] in cs else sorted(cs)):
                h = figures.hits(m["pattern"], cs[ch])
                anchor = next((j for j, (t, _) in enumerate(m["pattern"]) if t[:1] == "p"), 0)
                if h:
                    bits.append(f"{m['token']} on {ch}: " + ", ".join(f"{cs[ch][x + anchor - 1]}{x + anchor}"
                                                                      for x in h[:6]))
        return " · ".join(bits)
    if name == "clusters":
        pts = figures.cluster_points(ddir)
        scores = figures.pool_scores(ddir)
        best: dict[str, float] = {}
        for p in pts:
            x = (scores.get(p["key"]) or {}).get("xgb")
            if x is not None:
                best[p["variant"] + ":" + p["cluster"]] = max(best.get(p["variant"] + ":" + p["cluster"], 0.0), x)
        line = (f"{len({p['key'] for p in pts}):,} models, "
                f"{len({p.get('run', '') + p['cluster'] for p in pts})} clusters in "
                f"{len({p['variant'] for p in pts})} input variants")
        if best:
            top = sorted(best.items(), key=lambda kv: -kv[1])[:3]
            line += " · best predicted DockQ per cluster: " + ", ".join(f"{k} {v:.2f}" for k, v in top)
        return line
    return ""


def figure_pages(d: Dossier, ddir: Path, names: tuple[str, ...]) -> list[str]:
    """One slide per figure that was drawn, with what it was drawn from under it."""
    figs = ddir / "report" / "figs"
    out: list[str] = []
    for name in names:
        if not (figs / f"{name}.svg").exists():
            continue
        try:
            line = figure_summary(d, ddir, name)
        except Exception:                             # the deck is not worth failing over one summary line
            line = ""
        out += [f"== {FIGURE_TITLES.get(name, name)}", "",
                f'#align(center)[#image("figs/{name}.svg", {_fit(figs / f"{name}.svg", bool(line))})]']
        if line:
            out.append(f"#text(size: 11pt)[{_s(_short(line, 330))}]")
        out.append("")
    return out


def build_typ(d: Dossier, ddir: Path, with_retro: bool = False) -> str:
    lk = report.links(d)
    rep = ddir / "report"
    figs = rep / "figs"
    L = ['#import "@preview/touying:0.5.3": *', "#import themes.metropolis: *", "",
         '#set text(font: "DejaVu Sans", size: 17pt)', '#show raw: set text(font: "DejaVu Sans Mono", size: 11pt)',
         "#set figure(numbering: none, gap: 4pt)", "#show figure.caption: set text(size: 10pt)", "",
         "#show: metropolis-theme.with(", '  aspect-ratio: "16-9",',
         f"  config-info(title: [{_s(d.target + ': target report')}], subtitle: [{_s('stage ' + d.stage)}],",
         f"    author: [lantern · DeepFold], date: [{_s(now()[:10])}]),",
         "  config-common(new-section-slide-fn: none),", ")", "", "#title-slide()", ""]

    # in one page: the answer, before the record that produced it (the team's target notes open this way)
    kp = report.key_points(d, ddir)
    if kp:
        L += ["== In one page", "", "#text(size: 14pt)["]
        for label, text in kp:
            plain = re.sub(r"\*\*(.+?)\*\*", r"\1", text)      # the markdown emphasis is for analysis.md
            if label:
                L.append(f"- #text(fill: rgb(\"#D55E00\"), weight: \"bold\")[{_s(label)}] "
                         f"{_s(_short(plain, 330))}")
            else:
                L.append(f"  - {_s(_short(plain, 300))}")
        L += ["]", ""]

    # target
    ev = {k: v for k, v in d.evidence.items() if with_retro or k not in report.HIDDEN}
    L += ["== The target", ""]
    L.append("#text(size: 12pt)[")
    for k in ("construct", "identity", "public_structures", "msa", "templates"):
        if k in ev:
            L.append(f"- *{k.replace('_', ' ')}*: {_s(_plain(ev[k][-1].get('value'), 175))}")
    L.append(f"- *stoichiometry given*: {_s(d.stoichiometry.get('given', 'not given'))}")
    L += ["]", ""]

    # priors
    rest = [k for k in ev if k not in ("construct", "identity", "public_structures", "msa", "templates", "slate_proposal", "pool",
                                        "report", "results_review", "msa_depth", "gates")]
    if rest:
        L += ["== Further priors and checks", "", "#text(size: 12pt)[#table(columns: (auto, auto, 1fr), stroke: 0.4pt + luma(200), inset: 5pt,",
              "  [*prior*], [*found at*], [*what*],"]
        for k in rest[:9]:
            e = ev[k][-1]
            L.append(f"  {_cell(k)}, {_cell(report.stage_at(d, e.get('time', '')))}, {_cell(_plain(e.get('value'), 210))},")
        L += [")]", ""]

    # one page per figure `lantern report` drew from this dossier, each with the numbers behind it underneath
    L += figure_pages(d, ddir, ("chain_map", "literature", "sites", "motif"))

    # MSA coverage (lantern msa --depth draws it)
    if (figs / "msa_coverage.svg").exists() and "msa_depth" in ev:
        rows = report._msa_rows(ev["msa_depth"][-1]["value"])        # per file, or per chain and database
        keep = ("sequences", "≥50% coverage")                        # a slide wants the two that decide the run
        line = " · ".join(f"{_s(label)}: " + ", ".join(f"{counts[k]:,} {k}" for k in keep if isinstance(counts.get(k), int))
                          for label, counts in rows if any(isinstance(counts.get(k), int) for k in keep))
        L += ["== MSA depth and coverage", "", '#align(center)[#image("figs/msa_coverage.svg", width: 92%)]']
        L += (["#text(size: 12pt)[" + line + "]", ""] if line else [""])

    # n-sweep heat table
    pc = ddir / "pool.csv"
    if pc.exists():
        stos, runs, cells = pool_table(pc)
        if stos:
            L += ["== The pool: confidence per stoichiometry and predictor", "",
                  "Best (and median of the top 10) ipTM for complexes, pTM for monomers. Darker is more confident.", "",
                  f"#text(size: 12pt)[#table(columns: {len(stos) + 1}, stroke: 0.4pt + white, inset: 6pt, align: center,",
                  "  [], " + ", ".join(f"[*{_s(s)}*]" for s in stos) + ","]
            for run in runs:
                row = [f"[*{_s(run)}*]"]
                for s in stos:
                    c = cells.get((s, run))
                    if c:
                        txt = f"{c[0]:.2f} ({c[1]:.2f})"
                        col = "white" if c[0] > 0.6 else "black"
                        row.append(f"table.cell(fill: {_heat(c[0])})[#text(fill: {col})[{_s(txt)}]]")
                    else:
                        row.append("[–]")
                L.append("  " + ", ".join(row) + ",")
            L += [")]", ""]

    # the clustering, its own page: the alternatives a slate hedges across
    L += figure_pages(d, ddir, ("clusters",))

    # and what each cluster actually is: the medoid of each, rendered, with the numbers that describe its cluster
    cms = [m for m in cluster_models(ddir) if _pic(figs, f"{m['label']}_chains")]
    if cms:
        L += ["== What each cluster is: the model at the centre of it", "",
              f"#grid(columns: {min(len(cms), 4)}, gutter: 8pt,"]
        for m in cms[:8]:
            c = m["cluster"]
            cap = (f"{m['label']} · {c['n']:,} models"
                   + (f" · conf {c['best_conf']:.2f}" if c.get("best_conf") else "")
                   + (f" · DockQ {c['mean_xgb']:.2f}" if c.get("mean_xgb") is not None else "")
                   + " · " + "/".join(c["runners"]))
            L.append(f'  figure(image("{_pic(figs, m["label"] + "_chains")}", height: 26%), '
                     f"caption: [#text(size: 9pt)[{_s(cap)}]]),")
        L += [")", "#text(size: 11pt)["
              + _s("Each is the model with the highest membership in its cluster, superposed on the first and "
                   "coloured by chain. Clusters are ordered by size; the numbers are the cluster's, not the "
                   "model's.") + "]", ""]

    # decisions and priors
    if d.decisions:
        L += ["== Decisions and the priors they rest on", "",
              "#text(size: 10pt)[#table(columns: (auto, auto, 1fr, auto), stroke: 0.4pt + luma(200), inset: 4pt,",
              "  [*id*], [*type*], [*chose*], [*rests on*],"]
        for dec in d.decisions:
            l = lk[dec["id"]]
            rests = ", ".join(l["evidence"] + l["decisions"] + (["a person"] if l["person"] else [])) or "free text"
            L.append(f"  {_cell(dec['id'])}, {_cell(dec['type'])}, {_cell(_short(dec['chose'], 115))}, {_cell(_short(rests, 70))},")
        L += [")]", ""]

    # checkpoints
    for c in d.checkpoints:
        res = c["resolution"]
        L += [f"== Checkpoint {c['id']} ({c['kind']}) at {c['stage'].replace('_', ' ')}", "", "#text(size: 12pt)["]
        for i, q in enumerate(c["questions"]):
            L.append(f"+ {_s(_short(q, 230))}")
        L.append("]")
        L.append("#text(size: 11pt)[")
        for x in c["thread"][-4:]:
            who = "agent" if x["by"] in ("agent", "lantern") else "person"
            colour = "rgb(\"#2a2a2e\")" if who == "agent" else "rgb(\"#D55E00\")"
            L.append(f"- #text(fill: {colour}, weight: \"bold\")[{who}]: {_s(_short(x['text'], 330))}")
        if res:
            L.append(f"- #text(fill: rgb(\"#D55E00\"), weight: \"bold\")[resolved by a person]: {_s(res['outcome'] + (': ' + res['note'] if res.get('note') else ''))}")
        L += ["]", ""]

    # first-results review
    rr = d.evidence.get("results_review")
    if rr:
        trig = rr[-1]["value"].get("triggers", [])
        L += ["== First results: the review", ""]
        L += [f"- {_s(_short(t, 200))}" for t in trig] or ["- nothing unusual"]
        L.append("")

    # structural gates and the site census (lantern gate)
    from lantern import gate as gatemod
    for g in gatemod.latest_results(d)[-2:]:
        spec = {x.get("name"): x for x in (g.get("spec") or {}).get("gates", [])}
        L += [f"== Structural gates {_s(str(g.get('name')))} at {_s(str(g.get('sto')))}", "",
              f"#text(size: 20pt)[*{g.get('n_pass', 0):,}* of {g.get('n', 0):,} models pass every gate]", "",
              "#text(size: 14pt)[#grid(columns: (1.5fr, 1fr), gutter: 18pt,",
              "  table(columns: (auto, 1fr, auto), stroke: 0.4pt + luma(200), inset: 6pt, [*gate*], [*checks*], [*pass*],"]
        for name, n in (g.get("gates") or {}).items():
            what = "all chains present" if name == "complete" else gatemod.describe(spec.get(name, {}))
            L.append(f"    {_cell(name)}, {_cell(_short(what, 70))}, {_cell(n)},")
        L += ["  ),", "  table(columns: (auto, auto, auto), stroke: 0.4pt + luma(200), inset: 6pt, [*predictor*], [*models*], [*pass*],"]
        for r, v in sorted((g.get("per_runner") or {}).items()):
            L.append(f"    {_cell(r)}, {_cell(v.get('n'))}, {_cell(v.get('pass'))},")
        L += ["  ),", ")]"]
        if g.get("census"):
            L.append("#v(10pt)#text(size: 14pt)[*Site census* (" + _s(str((g.get("spec") or {}).get("census", {}).get("probe", "?"))) + "): "
                     + _s("; ".join(f"{r}: " + ", ".join(f"{k} {n}" for k, n in c.items()) for r, c in sorted(g["census"].items()))) + "]")
        L.append("")

    # structures
    models = figure_models(d, ddir)
    have = [m for m in models if _pic(figs, f"{m['label']}_chains")]
    if models:
        title = "The slate" if d.evidence.get("slate_proposal") else "Best model per stoichiometry"
        if have:
            L += [f"== {title}: structures (by chain)", "",
                  f"#grid(columns: {min(len(have), 3)}, gutter: 6pt,"]
            for m in have[:6]:
                L.append(f'  figure(image("{_pic(figs, m["label"] + "_chains")}", height: 33%), caption: [{_s(m["label"] + ": " + _short(m["name"], 38))}]),')
            L += [")", ""]
            pl = [m for m in have if _pic(figs, f"{m['label']}_plddt")]
            L += [f"== {title}: confidence (pLDDT)", "", f"#grid(columns: {max(1, min(len(pl), 3))}, gutter: 6pt,"]
            for m in pl[:6]:
                L.append(f'  figure(image("{_pic(figs, m["label"] + "_plddt")}", height: 33%), caption: [{_s(m["label"])}]),')
            L += [")", "#text(size: 11pt)[pLDDT: #text(fill: rgb(\"#0053D6\"))[■] > 90 · #text(fill: rgb(\"#65CBF3\"))[■] 70–90 · "
                  "#text(fill: rgb(\"#FFDB13\"))[■] 50–70 · #text(fill: rgb(\"#FF7D45\"))[■] < 50]", ""]
            if _pic(figs, "overlay") and d.evidence.get("slate_proposal"):
                L += [f"== {title}: superposed on {have[0]['label']}", "", f'#align(center)[#image("{_pic(figs, "overlay")}", height: 78%)]', ""]
        else:
            L += [f"== {title}: structures", "", "Pictures are being rendered (`lantern slides` again when the render job is done).", ""]

    # slate table
    props = d.evidence.get("slate_proposal", [])
    if props and Path(props[-1]["value"]["json"]).exists():
        sl = json.loads(Path(props[-1]["value"]["json"]).read_text())
        L += [f"== Slate v{props[-1]['value'].get('version')} ({'chosen by review' if sl.get('chosen') else 'rules'})", "",
              "#text(size: 12pt)[#table(columns: 6, stroke: 0.4pt + luma(200), inset: 5pt,",
              "  [*slot*], [*model*], [*runner*], [*ipTM*], [*z (runner)*], [*interface*],"]
        for m in sl["models"]:
            L.append(f"  {_cell(m['slot'])}, {_cell(m['model'])}, {_cell(m['runner'])}, {_cell(m['iptm'] if m['iptm'] is not None else '–')}, "
                     f"{_cell(round(m['rs_z_runner'], 2))}, {_cell(m['xgb_pred'] if m['xgb_pred'] is not None else '–')},")
        L += [")]", ""]

    # the agent's own slides
    if (rep / "agent.typ").exists():
        L += ['#include "agent.typ"', ""]

    # references: what the analysis read, and what rests on it
    srcs, unsourced = report.sources(d, with_retro)
    if srcs or unsourced:
        L += ["== What the analysis read", "",
              "#text(size: 11pt)[#table(columns: (auto, 1fr, auto), stroke: 0.4pt + luma(200), inset: 5pt,",
              "  [*\\#*], [*source*], [*supports*],"]
        for i, (text, keys) in enumerate(srcs[:14], 1):
            L.append(f"  {_cell(i)}, {_cell(_short(text, 150))}, {_cell(_short(', '.join(keys), 40))},")
        L += [")]"]
        if len(srcs) > 14:
            L.append(f"#text(size: 11pt)[… and {len(srcs) - 14} more, in `analysis.md`.]")
        if unsourced:
            L.append(f"#text(size: 11pt)[*No source recorded*: {_s(', '.join(unsourced))}]")
        L.append("")

    # what it proposes and what it asks: the last page, the way the team's target notes close
    cp = report.closing_points(d, ddir)
    if cp:
        L += ["== What this proposes, and what it is asking", "", "#text(size: 13pt)["]
        for label, text in cp[:9]:
            plain = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
            L.append(f"- #text(fill: rgb(\"#D55E00\"), weight: \"bold\")[{_s(label)}] {_s(_short(plain, 260))}")
        L += ["]", ""]

    # cost
    steps = []
    for x in d.log:
        m = re.search(r"agent step: (\w+), \$([0-9.]+), turns (\d+), log \S*?\d{8}-\d{6}_(\w+)\.log", x["note"])
        if m and x["actor"] == "lantern":
            steps.append((m.group(4), float(m.group(2)), int(m.group(3))))
    if steps:
        by: dict = {}
        for st, c, t in steps:
            a = by.setdefault(st, [0.0, 0, 0])
            a[0] += c; a[1] += t; a[2] += 1
        L += ["== What the agent spent", "", "#text(size: 13pt)[#table(columns: 4, stroke: 0.4pt + luma(200), inset: 5pt,",
              "  [*stage*], [*steps*], [*turns*], [*USD*],"]
        for st, (c, t, n) in by.items():
            L.append(f"  {_cell(st)}, {_cell(n)}, {_cell(t)}, {_cell(f'{c:.2f}')},")
        L.append(f"  [*total*], {_cell(len(steps))}, {_cell(sum(t for _, _, t in steps))}, {_cell(f'{sum(c for _, c, _ in steps):.2f}')},")
        L += [")]", ""]
    return "\n".join(L) + "\n"


def slides_cmd(cfg, d: Dossier, actor: str, render: bool = True, with_retro: bool = False) -> str:
    ddir = Dossier.path(cfg.targets_dir, d.target).parent
    rep = ddir / "report"
    (rep / "figs").mkdir(parents=True, exist_ok=True)
    out = []
    models = figure_models(d, ddir) + cluster_models(ddir)
    if models:
        scenes = rep / "scenes.pml"
        chains = report._chains(d)
        small = tuple(chr(ord("A") + i) for i, (_, seq) in enumerate(chains) if len(seq) <= 40)
        text = scenes_pml(models, small)
        changed = not scenes.exists() or scenes.read_text() != text
        scenes.write_text(text)
        missing = [m for m in models if not _pic(rep / "figs", f"{m['label']}_chains")]
        if render and (changed or missing):
            try:
                out.append(render_cmd(cfg, d, scenes, actor))
            except (FileNotFoundError, ValueError) as exc:
                out.append(f"no render: {exc}")
    typ = rep / "report.typ"
    typ.write_text(build_typ(d, ddir, with_retro))
    typst = tool(cfg, "typst")
    if not typst:
        out.append(f"wrote {typ}; typst not found, so no PDF")
        return "\n".join(out)
    r = subprocess.run([typst, "compile", str(typ), str(rep / "report.pdf")], capture_output=True, text=True, timeout=300)
    if r.returncode != 0:
        raise ValueError(f"typst failed on {typ}:\n{r.stderr[-2000:]}")
    d.evidence.setdefault("report", []).append({"time": now(), "by": actor, "value": {"pdf": str(rep / "report.pdf"),
                                                "with_retro": with_retro,
                                                "figures": sorted(p.name for p in (rep / "figs").glob("*.jpg"))}})
    out.append(f"wrote {rep / 'report.pdf'} ({len(list((rep / 'figs').glob('*.png')))} pictures)")
    return "\n".join(out)
