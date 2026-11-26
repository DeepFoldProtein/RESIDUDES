"""`residudes slides T` and `residudes render T`: the target report as a Typst slide deck with structure pictures.

The deck (report/report.typ -> report/report.pdf, touying metropolis like the team's target notes) is built from the
dossier, so it can be rebuilt at any stage:
  target and construct · the priors · the n-sweep (pool confidence per stoichiometry x predictor, as a heat table)
  · decisions and the priors they rest on · the checkpoint conversations · the first-results review · structures
  (renders of the slate, or of the best model per stoichiometry before there is a slate) · the slate
  · the agent's own slides (report/agent.typ, included as written) · agent cost per stage.

Two renderers, for two moments. While the run goes on - a checkpoint, a conversation, a quick look - the pictures
come from report/scenes.mt, drawn by molterm in the time the command takes. The final report, once the slate is
approved, is ray-traced: `--final` writes report/scenes.pml and has PyMOL render it, with the agent's own scenes in
report/pml/*.pml, through sbatch (never on the login node) and waits for it. molterm rasterises - flat shading, no
anti-aliasing, no shadows - and on H1388's peptide comparison no setting of it came near the ray-traced picture.

  residudes slides T [--final] [--no-render] [--with-retro]
  residudes render T --pml FILE
"""
from __future__ import annotations

import csv
import json
import re
import shlex
import shutil
import statistics
import subprocess
import time
from pathlib import Path

from residudes import report
from residudes.dossier import Dossier, chain_letter, now

# site default; `residudes.toml` [tools] pymol wins, and a pymol on PATH is used before this one
PYMOL_CANDIDATES: list[str] = []              # [tools] pymol in residudes.toml, else PATH
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
    from residudes import figures
    out = []
    for m in figures.cluster_medoids(ddir, limit):
        if Path(m["path"]).exists():
            out.append({"label": m["label"], "name": m["key"], "path": m["path"], "sto": m["sto"], "cluster": m})
    return out


# Report-quality rendering, from molterm's own recipe. Two of these matter more than the rest: the LOD
# thresholds, because the default drops to reduced geometry above 5,000 atoms and every complex we render is
# bigger than that, so a deck built without them is drawn at preview detail; and `sm relative`, which makes a
# view tuned at any size render identically at another, so the same script serves a quick look and the deck.
SCENE_QUALITY = ["set lod_medium 400000", "set lod_low 400000",   # full geometry: our models are 10k+ atoms
                 "set bg white",                                  # molterm defaults to transparent, which the
                 #                                                  JPEG step would flatten onto black
                 "set sm relative",                               # one view, any output size
                 "set csd 24",                                    # cartoon spline subdivisions (default 14)
                 "set ch 1.6", "set csh 1.8", "set cl 0.30",      # helix, sheet, loop widths
                 "set outline on", "set ot 0.2", "set od 0.2", "set fog 0.4"]
FIG_SIZE = "1800 1350 300"                                        # 6x4.5 in at 300 dpi
FIG_WIDE = "2400 1700 300"                                        # the overlay, which carries more structure

# The final report's settings. Ray-traced with shadows and ambient occlusion, anti-aliased, and outlined like the
# molterm pictures before it, so the deck does not change style at the end; the sizes are the molterm ones.
PML_QUALITY = ["bg_color white", "set ray_opaque_background, 1", "set antialias, 2",
               "set ray_trace_mode, 1", "set ray_trace_color, 0x404040",
               "set ray_shadows, 1", "set ambient_occlusion_mode, 1", "set specular, 0.25",
               "set depth_cue, 1", "set fog_start, 0.55",
               "set cartoon_fancy_helices, 1", "set stick_radius, 0.22"]

# Where a small chain (a peptide, a ligand-sized subunit) meets the rest, look at it from outside. `orient` points
# the camera along the chain's own shortest axis, which on H1388 put the 14-3-3 helices between the camera and
# the peptide in the groove. `face` puts the camera on the side of the chain away from the atoms it touches, with
# the chain's long axis across the picture. It runs inside the scene, so it needs nothing but PyMOL.
PML_FACE = """python
import numpy as _np
def face(on, partner, zoom_on="", buffer=4):
    p = cmd.get_coords(on)
    r = cmd.get_coords(f"({partner}) within 10 of ({on})")
    if p is None or r is None:
        cmd.orient(zoom_on or on)
        return
    z = p.mean(0) - r.mean(0)
    z /= _np.linalg.norm(z)
    x = _np.linalg.svd(p - p.mean(0))[2][0]
    x -= x.dot(z) * z
    if _np.linalg.norm(x) < 1e-6:
        x = _np.cross(z, [1.0, 0, 0] if abs(z[0]) < 0.9 else [0, 1.0, 0])
    x /= _np.linalg.norm(x)
    v = list(cmd.get_view())
    v[0:9] = _np.array([x, _np.cross(z, x), z]).T.flatten().tolist()
    cmd.set_view(v)
    cmd.zoom(zoom_on or f"({on}) or ({partner})", float(buffer))
cmd.extend("face", face)
python end"""


def _png(spec: str) -> str:
    """molterm's `W H DPI` as PyMOL's png arguments."""
    w, h, dpi = spec.split()
    return f"width={w}, height={h}, dpi={dpi}, ray=1"


def face_rows(on: list, near: list) -> list[list[float]] | None:
    """The rotation PML_FACE computes, for molterm, which cannot compute one: rows are the screen x, y and z axes
    in model coordinates, z pointing at the viewer (molterm's `camera` convention, checked against a render).
    Plain Python, so it costs no dependency: a handful of atoms per residue is all it reads."""
    def mean(ps):
        return [sum(c) / len(ps) for c in zip(*ps)]

    def dot(a, b):
        return sum(x * y for x, y in zip(a, b))

    def unit(a):
        n = dot(a, a) ** 0.5
        return [x / n for x in a] if n > 1e-9 else None

    def cross(a, b):
        return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]

    if not on or not near:
        return None
    z = unit([a - b for a, b in zip(mean(on), mean(near))])
    if z is None:
        return None
    c = mean(on)
    q = [[a - b for a, b in zip(p, c)] for p in on]
    cov = [[sum(p[i] * p[j] for p in q) for j in range(3)] for i in range(3)]
    x = [1.0, 0.31, 0.17]
    for _ in range(60):                                        # power iteration: the chain's long axis
        x = unit([dot(r, x) for r in cov]) or x
    x = unit([a - dot(x, z) * b for a, b in zip(x, z)]) or unit(cross(z, [1.0, 0, 0] if abs(z[0]) < 0.9 else [0, 1.0, 0]))
    return [x, cross(z, x), z]


def face_camera(path: str, small: tuple[str, ...]) -> str | None:
    """A molterm camera file that looks at the small chains of one model from outside the chains they touch, or
    None when there is nothing to face (no small chain, nothing within 10 A of it, an unreadable file)."""
    try:
        import gemmi
        st = gemmi.read_structure(str(path))
    except (ImportError, OSError, RuntimeError, ValueError):
        return None
    on, rest = [], []
    for ch in st[0]:
        for res in ch:
            a = res.find_atom("CA", "*") or res.find_atom("P", "*") or (res[0] if len(res) else None)
            if a is not None:
                (on if ch.name in small else rest).append([a.pos.x, a.pos.y, a.pos.z])
    near = [r for r in rest if any(sum((u - v) ** 2 for u, v in zip(r, p)) < 100.0 for p in on)]
    rows = face_rows(on, near)
    if rows is None:
        return None
    c = [sum(v) / len(on) for v in zip(*on)]
    return ("# molterm camera v1: written by `residudes slides`, the small chains seen from outside their partners\n"
            f"center = {c[0]:.4f} {c[1]:.4f} {c[2]:.4f}\nzoom = 1.0\npan = 0.0 0.0\n"
            "rot = " + "  ".join(" ".join(f"{v:.6f}" for v in r) for r in rows) + "\n")


def scenes_mt(models: list[dict], small: tuple[str, ...] = (), camera: str | None = None,
              templates: list[dict] = ()) -> str:
    """One molterm script for every picture the deck uses (run from report/, writes figs/*.png).

    The same two rules as the PyMOL scenes it replaces. Every model is superposed on the first and drawn from one
    view, because the question these pictures answer is how they differ, and a picture that re-orients each model
    answers nothing. And a chain short enough to vanish at this scale (a peptide, a ligand-sized subunit) is drawn
    as spheres in the accent colour: on H1388 the whole target is where an 11-residue peptide sits on a 233-residue
    groove, and in plain cartoon all eight cluster medoids looked like the same blue blob.

    What molterm adds over PyMOL here: `align ... automap` pairs chains by sequence, so models whose chains come
    out in a different order (routine across predictors) superpose instead of silently landing apart, and it prints
    the TM-score and RMSD of every superposition, which is evidence the deck can quote."""
    lines = ["# residudes scenes: written by `residudes slides`; run with `molterm --no-tui --script scenes.mt` from report/",
             *SCENE_QUALITY]
    for m in models:
        # molterm names an object after the file stem, and ours are all `model.cif`: the second would come in as
        # `model_1`. Renaming straight after each load frees the stem again, so the next load is `model` once more.
        lines += [f"load {m['path']}", f"rename {Path(m['path']).stem} {m['label']}"]
    first = models[0]["label"] if models else None
    for m in models[1:]:
        lines.append(f"align {m['label']} to {first} automap")     # chain-aware, sequence-paired
    lines.append("show cartoon")
    # One view for every picture: they are comparable. With a small chain, the view that faces it (face_camera);
    # `zoom` after `camera load` refits the frame and keeps the rotation.
    view = [f"camera load {camera}", f"zoom {first}"] if camera else [f"orient {first}"]
    if first:
        lines += view
    for m in models:
        lab = m["label"]
        # A colour scheme takes no selection in molterm (`color chain af3_s1` is refused); it paints every object.
        # That is harmless because only the enabled one is drawn, and it is why the enable comes first.
        lines += ["disable all", f"enable {lab}", "color chain"]
        for ch in small:                                           # the chain that would otherwise disappear
            lines += [f"show spacefill chain {ch}", f"color #D55E00 chain {ch}"]
        lines += [f"screenshot figs/{lab}_chains.png {FIG_SIZE}",
                  "color plddt", f"screenshot figs/{lab}_plddt.png {FIG_SIZE}"]
    same = [m for m in models if m.get("sto") == models[0].get("sto")] if models else []
    if len(same) > 1:                                  # superpose only models of one stoichiometry (the first's)
        lines.append("disable all")
        for i, m in enumerate(same):
            slot = "#" + SLOT_COLOURS[i % len(SLOT_COLOURS)][2:]   # 0x0072B2 (PyMOL) -> #0072B2 (molterm)
            lines += [f"enable {m['label']}", f"color {slot} {m['label']}"]
        lines += [*view, f"screenshot figs/overlay.png {FIG_WIDE}"]
    lines += _template_scene_mt(templates)
    return "\n".join(lines) + "\n"


def _template_scene_mt(templates: list[dict]) -> list[str]:
    """One picture per template (views.template_entries): its chain blue, a bound peptide orange, ligands as balls
    and sticks, each from its own view - facing what it has bound when there is a small chain to face."""
    lines = []
    for x in templates:
        lab, stem = x["label"], Path(x["path"]).stem
        lines += [f"load {x['path']}", f"rename {stem} {lab}", "disable all", f"enable {lab}", "show cartoon",
                  "color #BBBBBB", f"color #5B8DB8 chain {x['chain']}", "show ballstick ligand"]
        for s in x.get("small", []):
            lines += [f"show spacefill chain {s}", f"color #D55E00 chain {s}"]
        lines += ([f"camera load {x['camera']}", f"zoom obj {lab}"] if x.get("camera") else [f"orient obj {lab}"])
        lines.append(f"screenshot figs/{lab}.png {FIG_SIZE}")
    return lines


def _template_scene_pml(templates: list[dict]) -> list[str]:
    lines = []
    for x in templates:
        lab, ch, sm = x["label"], x["chain"], "+".join(x.get("small", []))
        lines += [f"load {x['path']}, {lab}", "disable all", f"enable {lab}", f"hide everything, {lab}",
                  f"show cartoon, {lab}", f"color 0xBBBBBB, {lab}", f"color 0x5B8DB8, {lab} and chain {ch}",
                  f"show sticks, {lab} and organic"]
        if sm:
            lines += [f"color 0xD55E00, {lab} and chain {sm}",
                      f"show sticks, {lab} and chain {sm} and not name N+C+O",
                      f"face {lab} and chain {sm}, {lab} and chain {ch}, {lab}, 0"]
        else:
            lines += [f"orient {lab}", f"zoom {lab}, 0"]
        lines.append(f"png figs/{lab}.png, {_png(FIG_SIZE)}")
    return lines


def scenes_pml(models: list[dict], small: tuple[str, ...] = (), templates: list[dict] = ()) -> str:
    """One PyMOL script for every picture the final deck uses (run from report/, writes figs/*.png), ray-traced.

    The same rules as the molterm scenes before it, at the same sizes. Every model is superposed on the first and
    drawn from one view, because the question these pictures answer is how they differ. A chain short enough to
    vanish at this scale (a peptide, a ligand-sized subunit) is drawn as spheres in the accent colour, and the view
    faces it from outside the chains it touches (PML_FACE): on H1388 the whole target is where an 11-residue peptide
    sits on a 233-residue groove, and in plain cartoon all eight cluster medoids looked like the same blue blob."""
    lines = ["# residudes scenes: written by `residudes slides --final`; ray-traced with `pymol -cq scenes.pml` from report/",
             "reinitialize", *PML_QUALITY,
             "set_color plddt_vh, [0.000, 0.325, 0.839]", "set_color plddt_h, [0.396, 0.796, 0.953]",
             "set_color plddt_l, [1.000, 0.859, 0.075]", "set_color plddt_vl, [1.000, 0.490, 0.271]"]
    for m in models:
        lines.append(f"load {m['path']}, {m['label']}")
    lines += ["remove solvent", "remove hydro", "hide everything", "show cartoon"]
    first = models[0]["label"] if models else None
    for m in models:
        lab = m["label"]
        if first and lab != first:
            lines.append(f"super {lab} and name CA, {first} and name CA")
    sel = "+".join(small)
    if (first and small) or any(x.get("small") for x in templates):
        lines.append(PML_FACE)
    if first and small:
        lines += [f"face {first} and chain {sel}, {first} and not chain {sel}, {first}, 0"]
    elif first:
        # zoom fits a sphere around the model, which already leaves margins on a 4:3 frame: no buffer on top
        lines += [f"orient {first}", f"zoom {first}, 0"]
    for m in models:
        lab = m["label"]
        lines += ["disable all", f"enable {lab}", f"util.cbc {lab}"]
        for ch in small:                                     # the chain that would otherwise disappear
            lines += [f"color 0xD55E00, {lab} and chain {ch}",
                      f"show spheres, {lab} and chain {ch} and not hydro",
                      f"set sphere_scale, 0.45, {lab} and chain {ch}",
                      f"set cartoon_transparency, 0.45, {lab} and not chain {ch}"]
        lines += [f"png figs/{lab}_chains.png, {_png(FIG_SIZE)}",
                  f"color plddt_vl, {lab}", f"color plddt_l, {lab} and b > 50", f"color plddt_h, {lab} and b > 70",
                  f"color plddt_vh, {lab} and b > 90",
                  f"png figs/{lab}_plddt.png, {_png(FIG_SIZE)}"]
    same = [m for m in models if m.get("sto") == models[0].get("sto")] if models else []
    if len(same) > 1:                                  # superpose only models of one stoichiometry (the first's)
        lines.append("disable all")
        for i, m in enumerate(same):
            # `(M1)`, not `M1`: a bare object name sets the object-level value, which the atom-level transparency
            # set above overrides - and thirteen overlapping translucent models, shadowed, did not finish in 25 min.
            lines += [f"enable {m['label']}", f"color {SLOT_COLOURS[i % len(SLOT_COLOURS)]}, {m['label']}",
                      f"hide spheres, {m['label']}", f"set cartoon_transparency, 0, ({m['label']})"]
        lines += [f"png figs/overlay.png, {_png(FIG_WIDE)}"]
    lines += _template_scene_pml(templates)
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------------------------- rendering
def render_mt(cfg, d: Dossier, scene: Path, actor: str) -> str:
    """Render a molterm scene, here and now.

    PyMOL goes through sbatch because ray-tracing a complex costs minutes and a gigabyte; molterm draws the same
    deck in seconds (measured on a T1416 A3 trimer, 10,179 atoms: 0.2 s per 1800x1350 figure, 40 MB, with the
    multimer superposition the slower part at a few seconds each). That fits on a login node, so the pictures are
    there when the command returns instead of a job id to come back to."""
    mt = tool(cfg, "molterm")
    if not mt:
        raise FileNotFoundError('molterm not found: set [tools] molterm = "/path/to/molterm" in residudes.toml')
    rep = scene.parent
    (rep / "figs").mkdir(parents=True, exist_ok=True)
    r = subprocess.run([mt, "--no-tui", "--script", str(scene)], cwd=rep, capture_output=True, text=True, timeout=3600)
    figs = sorted((rep / "figs").glob("*.png"))
    out = (r.stdout or "") + (r.stderr or "")
    # Do not trust the exit code alone: molterm 0.53 exits 1 when a command fails and 0.50 exits 0, while both
    # print the failing line and a tally. A failed `color` or `align` leaves the previous state and the script
    # carries on, so what comes back is a figure of something nobody asked for - an unaligned overlay looks
    # exactly like an aligned one until you read the numbers. Treat either signal as failure.
    failed = [ln for ln in out.splitlines() if f"{scene.name}:" in ln or "command(s) failed" in ln]
    if r.returncode != 0 or failed:
        bad = [ln for ln in failed if f"{scene.name}:" in ln][:3]
        d.note(actor, f"render {scene.name}: molterm failed" + (f" ({bad[0]})" if bad else f" (exit {r.returncode})"))
        return ("molterm failed, so the figures are not trustworthy: "
                + ("; ".join(bad) if bad else f"exit {r.returncode}"))
    _to_jpeg(rep / "figs")
    aligned = [ln for ln in r.stdout.splitlines() if "aligned" in ln.lower()]
    d.runs.append({"time": now(), "by": actor, "job": None, "scheduler": "local", "what": f"render: {scene.name}",
                   "state": "done", "out": str(rep / "figs")})
    d.note(actor, f"render {scene.name}: {len(figs)} figures" + (f"; {aligned[0]}" if aligned else ""))
    return f"molterm {scene.name} -> {len(figs)} figures in {rep / 'figs'}" + (f" | {aligned[0]}" if aligned else "")


def _to_jpeg(figs: Path) -> None:
    """A JPEG beside every PNG, because the deck and AGENTS.md both name `figs/<x>.jpg`.

    It is not the saving it once was: ray-traced PyMOL figures were far heavier as PNG, while molterm's
    flat-shaded ones embed within about 10% either way (measured: 1.0 MB of PDF against 1.1 MB). Kept for the
    convention, and skipped without PIL rather than made a dependency."""
    try:
        from PIL import Image
    except ImportError:
        return
    for f in figs.glob("*.png"):
        jpg = f.with_suffix(".jpg")
        if not jpg.exists() or jpg.stat().st_mtime < f.stat().st_mtime:
            Image.open(f).convert("RGB").save(jpg, quality=88)


def render_cmd(cfg, d: Dossier, pml: Path, actor: str) -> str:
    from residudes import ops
    pymol = tool(cfg, "pymol")
    if not pymol:
        raise FileNotFoundError("PyMOL not found: set [tools] pymol = \"/path/to/pymol\" in residudes.toml")
    ddir = Dossier.path(cfg.targets_dir, d.target).parent
    rep = ddir / "report"
    (rep / "figs").mkdir(parents=True, exist_ok=True)
    pml = pml.resolve()
    # PyMOL writes PNG; Typst embeds JPEG as is (PNG gets re-encoded and the deck grows ~4x), so convert in the job
    py = Path(pymol).with_name("python")
    to_jpg = (f" && {py} -c \"import glob,PIL.Image as I; [I.open(f).convert('RGB').save(f[:-4] + '.jpg', quality=88) "
              f"for f in glob.glob('figs/*.png')]\"") if py.exists() else ""
    job = ops._sbatch(f"cd {rep} && {pymol} -cq {pml}{to_jpg}", f"residudes-render-{d.target}", rep / "render.%j.log", gpu=False, hours=1)
    d.runs.append({"time": now(), "by": actor, "job": job, "scheduler": "slurm", "what": f"render: {pml.name}", "state": "submitted",
                   "out": str(rep / "figs")})
    d.note(actor, f"render {pml.name}: job {job}")
    return f"job {job}: PyMOL {pml.name} -> {rep / 'figs'}"


def render_final(cfg, d: Dossier, rep: Path, actor: str) -> str:
    """Ray-trace the final deck: report/scenes.pml and the agent's own report/pml/*.pml, in one PyMOL job, waited for.

    Through sbatch where there is one - PyMOL never runs on a login node - and in place where there is not (the agent
    image, a workstation). PyMOL carries on past a bad line and exits 0, so success is judged the way render_mt
    judges molterm: every `png` a scene names must have been written by this run."""
    from residudes import ops
    pymol = tool(cfg, "pymol")
    if not pymol:
        raise FileNotFoundError('PyMOL not found: set [tools] pymol = "/path/to/pymol" in residudes.toml')
    scenes = [s for s in [rep / "scenes.pml", *sorted((rep / "pml").glob("*.pml"))] if s.exists()]
    if not scenes:
        return "no PyMOL scenes to render"
    (rep / "figs").mkdir(parents=True, exist_ok=True)
    want = sorted({m for s in scenes for m in re.findall(r"^\s*png\s+(figs/[^,\s]+)", s.read_text(), re.M)})
    run = " && ".join(f"{shlex.quote(pymol)} -cq {shlex.quote(str(s.relative_to(rep)))}" for s in scenes)
    # the JPEGs the deck embeds, made with PyMOL's own interpreter where it has Pillow
    py = Path(pymol).resolve().with_name("python")
    jpg = (f" && ({shlex.quote(str(py))} -c \"import glob,PIL.Image as I; [I.open(f).convert('RGB').save(f[:-4] + '.jpg', "
           f"quality=90) for f in glob.glob('figs/*.png')]\" || true)") if py.exists() else ""
    cmd = f"cd {shlex.quote(str(rep))} && {run}{jpg}"
    t0 = time.time() - 120                       # compute-node clocks and NFS mtimes drift by seconds
    if shutil.which("sbatch"):
        job = ops._sbatch(cmd, f"residudes-render-{d.target}", rep / "render.%j.log", gpu=False, hours=1, wait=True)
        where = f"job {job}"
    else:
        r = subprocess.run(["sh", "-c", cmd], capture_output=True, text=True, timeout=3600)
        if r.returncode != 0:
            raise ValueError(f"PyMOL failed here (no sbatch):\n{(r.stdout + r.stderr)[-1500:]}")
        job, where = None, "here (no sbatch on this host)"
    stale = [w for w in want if not (rep / w).exists() or (rep / w).stat().st_mtime < t0]
    _to_jpeg(rep / "figs")
    d.runs.append({"time": now(), "by": actor, "job": job, "scheduler": "slurm" if job else "local",
                   "what": "render: final deck (PyMOL, ray-traced)", "state": "done", "out": str(rep / "figs")})
    if stale:
        d.note(actor, f"final render: {len(stale)} of {len(want)} figures not written ({', '.join(stale[:3])})")
        return (f"PyMOL {where}: {len(stale)} of {len(want)} figures were not written ({', '.join(stale[:3])}), so those "
                "pictures are not the final ones - read report/render.*.log")
    d.note(actor, f"final render: {len(want)} ray-traced figures ({where})")
    return f"PyMOL {where}: {len(want)} ray-traced figures in {rep / 'figs'}"


# ----------------------------------------------------------------------------------------------- the deck
def _heat(v: float | None) -> str:
    if v is None:
        return "white"
    v = max(0.0, min(1.0, v))
    r, g, b = (int(255 - (255 - c) * v) for c in (0, 114, 178))              # white -> residudes-engine blue
    return f'rgb({r}, {g}, {b})'


def _json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _neff(x: dict) -> str:
    v = x.get("neff") or 0
    return ("≈" if x.get("neff_estimated") else "") + (f"{v:,.0f}" if v >= 10 else f"{v:.1f}")


def _pic(figs: Path, stem: str) -> str | None:
    """figs/<stem>.jpg when it is at least as new as the PNG (smaller in the PDF), else the PNG, else None. A JPEG
    left from an earlier render must not stand in for the picture drawn after it - the final deck's, say."""
    jpg, png = figs / f"{stem}.jpg", figs / f"{stem}.png"
    if jpg.exists() and (not png.exists() or jpg.stat().st_mtime >= png.stat().st_mtime):
        return f"figs/{stem}.jpg"
    return f"figs/{stem}.png" if png.exists() else None


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
    from residudes import figures
    if name == "chain_map":
        return " · ".join(f"{chain_letter(i)}: {n.split(',')[-1].strip()} {len(s)} aa"
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
        letters = {chain_letter(i): len(s) for i, (_, s) in enumerate(cs)}
        seqs = {chain_letter(i): s for i, (_, s) in enumerate(cs)}
        recs = figures.site_records(d, letters)
        mods, _ = figures.modifications(d)
        bits = [f"{r['key']} {r['chain']}: " + ", ".join(f"{seqs[r['chain']][x - 1]}{x}" for x in r["residues"])
                for r in recs]
        return " · ".join(bits + [f"{lab} on {ch}{r}" for ch, r, lab in mods])
    if name == "motif":
        cs = {chain_letter(i): s for i, (_, s) in enumerate(figures.chains(d))}
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
         f"    author: [residudes · DeepFold], date: [{_s(now()[:10])}]),",
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
        if k not in ev:
            continue
        line = report.sweep_summary(ev) if k == "public_structures" else None   # the searches, not the last blob
        L.append(f"- *{k.replace('_', ' ')}*: {_s(line or _plain(ev[k][-1].get('value'), 175))}")
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

    # one page per figure `residudes report` drew from this dossier, each with the numbers behind it underneath
    L += figure_pages(d, ddir, ("chain_map", "literature", "sites", "motif"))

    # MSA depth by source, as ColabFold draws it (views.msa_figure, from the a3m files); the older coverage plot
    # that `residudes msa --depth` draws is the fallback for a dossier drawn before it
    md = _json(figs / "msa_depth.json")
    if (figs / "msa_depth.svg").exists() and md:
        parts = []
        for pn in md.get("panels", []):
            groups = ", ".join(f"{g['name']} {g['n']:,} (Neff {_neff(g)})" for g in pn["groups"])
            parts.append(f"{pn['chain']}: {pn['n']:,} sequences, Neff {_neff(pn)} - {groups}")
        if md.get("query_only"):
            parts.append("query alone: " + ", ".join(md["query_only"]))
        L += ["== MSA depth by source", "", f'#align(center)[#image("figs/msa_depth.svg", {_fit(figs / "msa_depth.svg", True)})]',
              "#text(size: 11pt)[" + _s(" · ".join(parts) + f" (Neff at {md.get('neff_identity', 0.8):.0%} identity; ≈ estimated "
                                         "from a 2,000-sequence sample)") + "]", ""]
    elif (figs / "msa_coverage.svg").exists() and "msa_depth" in ev:
        rows = report._msa_rows(ev["msa_depth"][-1]["value"])        # per file, or per chain and database
        keep = ("sequences", "≥50% coverage")                        # a slide wants the two that decide the run
        line = " · ".join(f"{_s(label)}: " + ", ".join(f"{counts[k]:,} {k}" for k in keep if isinstance(counts.get(k), int))
                          for label, counts in rows if any(isinstance(counts.get(k), int) for k in keep))
        L += ["== MSA depth and coverage", "", '#align(center)[#image("figs/msa_coverage.svg", width: 92%)]']
        L += (["#text(size: 12pt)[" + line + "]", ""] if line else [""])

    # the templates, drawn: the ones the inputs used, then the search's close hits (views.template_entries)
    tp = [x for x in (_json(figs / "templates.json") or {}).get("templates", []) if _pic(figs, x["label"])]
    if tp:
        from residudes.views import template_caption
        L += ["== Templates: the closest structures, and what they have bound", "",
              f"#grid(columns: {min(len(tp), 3)}, gutter: 6pt,"]
        for x in tp[:6]:
            L.append(f'  figure(image("{_pic(figs, x["label"])}", height: {33 if len(tp) > 3 else 52}%), '
                     f'caption: text(size: 10pt)[{_s(template_caption(x))}]),')
        L += [")", "#text(size: 11pt)[Template chain blue, what it has bound orange (peptides) or grey, ligands as sticks.]", ""]

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
            who = "agent" if x["by"] in ("agent", "residudes") else "person"
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

    # structural gates and the site census (residudes gate)
    from residudes import gate as gatemod
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
            L += [f"== {title}: structures", "", "Pictures are being rendered (`residudes slides` again when the render job is done).", ""]

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
        if m and x["actor"] == "residudes":
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


def slides_cmd(cfg, d: Dossier, actor: str, render: bool = True, with_retro: bool = False, final: bool = False) -> str:
    ddir = Dossier.path(cfg.targets_dir, d.target).parent
    rep = ddir / "report"
    (rep / "figs").mkdir(parents=True, exist_ok=True)
    out = []
    models = figure_models(d, ddir) + cluster_models(ddir)
    use_mt, has_pymol = bool(tool(cfg, "molterm")), bool(tool(cfg, "pymol"))
    # The final deck is ray-traced (PyMOL, waited for); every deck before it is molterm's, drawn here in seconds.
    # Without molterm, PyMOL is the only renderer, sent to sbatch without waiting, as before.
    pymol_final = final and has_pymol
    if final and not has_pymol:
        out.append("final deck drawn by molterm: PyMOL not found, so the pictures are not ray-traced ([tools] pymol)")
    # what the deck shows before the models: the MSAs by source, and the templates as structures (views.py)
    from residudes import views
    try:
        views.msa_figure(ddir, d, rep / "figs" / "msa_depth.svg")
    except Exception as exc:                                 # a picture is never worth failing the deck for
        out.append(f"no MSA picture: {type(exc).__name__}: {exc}")
    templates: list[dict] = []
    try:
        templates = views.template_entries(cfg, d, ddir)
    except Exception as exc:
        out.append(f"no template pictures: {type(exc).__name__}: {exc}")
    if models or templates:
        chains = report._chains(d)
        small = tuple(chain_letter(i) for i, (_, seq) in enumerate(chains) if len(seq) <= 40)
        with_pml = pymol_final or not use_mt
        scenes = rep / ("scenes.pml" if with_pml else "scenes.mt")
        camera = None
        if not with_pml and small and models:
            cam = face_camera(models[0]["path"], small)
            if cam:
                (rep / "view.cam").write_text(cam)
                camera = "view.cam"
        for x in templates if not with_pml else []:          # molterm cannot compute a view; PyMOL's `face` can
            cam = face_camera(x["path"], tuple(x["small"])) if x.get("small") else None
            if cam:
                (rep / f"{x['label']}.cam").write_text(cam)
                x["camera"] = f"{x['label']}.cam"
        text = scenes_pml(models, small, templates) if with_pml else scenes_mt(models, small, camera, templates)
        changed = not scenes.exists() or scenes.read_text() != text
        scenes.write_text(text)
        missing = ([m for m in models if not _pic(rep / "figs", f"{m['label']}_chains")]
                   + [x for x in templates if not _pic(rep / "figs", x["label"])])
        if render and not pymol_final and (changed or missing):
            try:
                out.append(render_mt(cfg, d, scenes, actor) if use_mt else render_cmd(cfg, d, scenes, actor))
            except (FileNotFoundError, ValueError, subprocess.TimeoutExpired) as exc:
                out.append(f"no render: {exc}")
    if render and pymol_final:
        try:
            out.append(render_final(cfg, d, rep, actor))
        except (FileNotFoundError, ValueError, subprocess.TimeoutExpired) as exc:
            out.append(f"no final render: {exc}")
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
                                                "with_retro": with_retro, "final": final,
                                                # no renderer when nothing was drawn (--no-render keeps the pictures there)
                                                "renderer": (("pymol" if pymol_final or not use_mt else "molterm")
                                                             if render else None),
                                                "figures": sorted(p.name for p in (rep / "figs").glob("*.jpg"))}})
    out.append(f"wrote {rep / 'report.pdf'} ({len(list((rep / 'figs').glob('*.png')))} pictures)")
    return "\n".join(out)
