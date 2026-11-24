#!/usr/bin/env python3
"""Draw the residudes-engine overview figures as SVG (stdlib only) and rasterise them with rsvg-convert.

    python3 docs/figures/make_figures.py          # writes docs/figures/*.svg and *.png

Figures: the overview (a metro map from a FASTA to a prediction with the measured
speed-up per model and backend), the speed grid, what `predict` does on a
machine, which model consumes which part of the input, and a terminal session.
The speed numbers come from docs/benchmark/speedup_summary.json (bench/collect.py).
Plain style: one sans-serif face, thin strokes, the same three backend colours
as the benchmark figures (TensorRT blue, JAX green, PyTorch black).
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

OUT = Path(__file__).resolve().parent
MASCOT = OUT.parent.parent.parent / "mascot" / "fig"
FONT = "Helvetica, Arial, 'Liberation Sans', 'DejaVu Sans', sans-serif"
MONO = "Menlo, Consolas, 'DejaVu Sans Mono', monospace"
INK = "#222222"
MUTED = "#666666"
LINE = "#8a8a8a"
FILL = "#f3f3f3"
TRT, JAX, TORCH = "#0072B2", "#009E73", "#222222"

HDR, LBL, SUB, CAP = 15, 14, 12, 11.5      # font sizes
IH_SUB, IH = 44, 28                        # item heights with / without a sub-line
SW, GAP, M = 340, 40, 20                   # stage width, gap, margin
W = M * 2 + 3 * SW + 2 * GAP               # three stages per row
XS = [M + i * (SW + GAP) for i in range(3)]
IW = SW - 24


class Svg:
    def __init__(self, w: int, h: int, *, bg="white"):
        self.w, self.h, self.bg = w, h, bg
        self.parts: list[str] = []

    def rect(self, x, y, w, h, *, fill="white", stroke=LINE, r=4, sw=1.0, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>')

    def text(self, x, y, s, *, size=LBL, weight="normal", fill=INK, anchor="start", font=FONT, spacing=None):
        s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        ls = f' letter-spacing="{spacing}"' if spacing else ""
        self.parts.append(f'<text x="{x}" y="{y}" font-family="{font}" font-size="{size}" font-weight="{weight}" '
                          f'fill="{fill}" text-anchor="{anchor}"{ls} xml:space="preserve">{s}</text>')

    def spans(self, x, y, runs, *, size=13, font=MONO):
        """One line of text made of (text, colour[, weight]) runs."""
        out = []
        for run in runs:
            t, col = run[0], run[1]
            wt = run[2] if len(run) > 2 else "normal"
            t = t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            out.append(f'<tspan fill="{col}" font-weight="{wt}">{t}</tspan>')
        self.parts.append(f'<text x="{x}" y="{y}" font-family="{font}" font-size="{size}" xml:space="preserve">{"".join(out)}</text>')

    def line(self, x1, y1, x2, y2, *, stroke=LINE, sw=1.0, arrow=False, dash=None, cap="butt"):
        m = ' marker-end="url(#arrow)"' if arrow else ""
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" stroke-width="{sw}" stroke-linecap="{cap}"{m}{d}/>')

    def path(self, d, *, stroke=LINE, sw=1.0, arrow=False, fill="none", cap="round"):
        m = ' marker-end="url(#arrow)"' if arrow else ""
        self.parts.append(f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" stroke-linecap="{cap}" stroke-linejoin="round"{m}/>')

    def circle(self, cx, cy, r, *, fill=INK, stroke=INK, sw=1.0):
        self.parts.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')

    def chip(self, x, y, label, color, *, w=66, h=19, size=11.5):
        self.rect(x, y, w, h, fill=color, stroke=color, r=3)
        self.text(x + w / 2, y + h / 2 + 4, label, size=size, fill="white", anchor="middle", weight="bold")

    def mascot(self, name, cx, bottom, box_w, box_h):
        """A mascot, scaled to fit the box and standing on `bottom`, embedded so the figure stays one file."""
        import base64, struct
        q = MASCOT / f"{name}.png"
        if not q.exists():
            return
        w0, h0 = struct.unpack(">II", q.read_bytes()[16:24])
        k = min(box_w / w0, box_h / h0)
        w, h = w0 * k, h0 * k
        b64 = base64.b64encode(q.read_bytes()).decode()
        self.parts.append(f'<image x="{cx - w / 2:.1f}" y="{bottom - h:.1f}" width="{w:.1f}" height="{h:.1f}" '
                          f'href="data:image/png;base64,{b64}"/>')

    def save(self, name: str):
        head = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" viewBox="0 0 {self.w} {self.h}">'
                '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse">'
                f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{LINE}"/></marker></defs>'
                f'<rect width="{self.w}" height="{self.h}" fill="{self.bg}"/>')
        svg = head + "".join(self.parts) + "</svg>"
        p = OUT / f"{name}.svg"
        p.write_text(svg)
        if shutil.which("rsvg-convert"):
            subprocess.run(["rsvg-convert", "-w", str(int(self.w * 1.5)), str(p), "-o", str(OUT / f"{name}.png")], check=True)
        return p


def item(s: Svg, x, y, w, label, *, sub=None, kind="proc", h=None):
    """One row inside a stage: a process (grey), a file (white, monospace) or plain text."""
    h = h or (IH_SUB if sub else IH)
    if kind == "proc":
        s.rect(x, y, w, h, fill=FILL, stroke=LINE)
    elif kind == "file":
        s.rect(x, y, w, h, fill="white", stroke=LINE)
    s.text(x + 10, y + 19, label, size=LBL, font=MONO if kind == "file" else FONT)
    if sub:
        s.text(x + 10, y + 36, sub, size=SUB, fill=MUTED)
    return y + h + 8


def stage(s: Svg, x, y, w, h, title, caption=None):
    s.rect(x, y, w, h, fill="white", stroke=INK, sw=1.2, r=7)
    s.rect(x, y, w, 30, fill=INK, stroke=INK, r=7)
    s.rect(x, y + 16, w, 14, fill=INK, stroke=INK, r=0)
    s.text(x + w / 2, y + 20.5, title, size=HDR, weight="bold", fill="white", anchor="middle")
    if caption:
        s.text(x + w / 2, y + h + 21, caption, size=CAP, fill=MUTED, anchor="middle", font=MONO)
    return y + 42


def note(s: Svg, x, y, w, h, lines):
    s.rect(x, y, w, h, fill="white", stroke=LINE, r=7, dash="4 3")
    yy = y + 30
    for ln in lines:
        if ln == "":
            yy += 10
            continue
        s.text(x + 16, yy, ln, size=13, fill=INK)
        yy += 21


def divider(s: Svg, x, y, w):
    s.line(x, y, x + w, y, stroke=LINE, dash="3 3")
    return y + 10


def h_arrow(s: Svg, x1, x2, y):
    s.line(x1 + 4, y, x2 - 4, y, stroke=LINE, sw=1.4, arrow=True)


def elbow_down(s: Svg, x_from, y_from, x_to, y_to):
    """From the bottom of one box to the top of another on the next row."""
    ym = (y_from + y_to) / 2
    s.path(f"M {x_from} {y_from + 4} V {ym} H {x_to} V {y_to - 4}", stroke=LINE, sw=1.4, arrow=True)


# ── Figure 1: the pipeline ────────────────────────────────────────────────

def fig_runtime():
    top, H1, ROWGAP, H2 = 22, 320, 68, 270
    row2 = top + H1 + ROWGAP
    s = Svg(W, row2 + H2 + 34)

    y = stage(s, XS[0], top, SW, H1, "Site", "sites/<name>.toml")
    y = item(s, XS[0] + 12, y, IW, "core", sub="the runtime tree in this checkout", kind="file")
    y = item(s, XS[0] + 12, y, IW, "venv", sub="core/.venv, built from the lockfile", kind="file")
    y = item(s, XS[0] + 12, y, IW, "checkpoints", sub="weights fetched by setup checkpoints", kind="file")
    y = item(s, XS[0] + 12, y, IW, "engines", sub="TensorRT engines, one set per GPU class", kind="file")
    y = item(s, XS[0] + 12, y, IW, "[slurm]  [jax]", sub="partition, memory fraction, JAX knobs", kind="file")

    y = stage(s, XS[1], top, SW, H1, "Machine", "nvidia-smi · tokens in the input")
    y = item(s, XS[1] + 12, y, IW, "GPU class", sub="compute capability → rtx4090, rtx6000ada, b200 …")
    y = item(s, XS[1] + 12, y, IW, "input size", sub="tokens counted from input.json")
    y = item(s, XS[1] + 12, y, IW, "libstdc++", sub="a library with GLIBCXX_3.4.32 is preloaded")

    y = stage(s, XS[2], top, SW, H1, "Backend", "--backend auto | trt | jax | torch")
    x = XS[2] + 12
    s.rect(x, y, IW, 70, fill="white", stroke=TRT, sw=1.3)
    s.chip(x + 10, y + 10, "TensorRT", TRT, w=70, h=20, size=11.5)
    s.text(x + 88, y + 25, "if an engine for this GPU covers", size=13)
    s.text(x + 10, y + 45, "the input; the smallest engine that fits", size=13)
    s.text(x + 10, y + 62, "is used (a 2048 engine before a 5120 one)", size=13)
    y += 78
    s.rect(x, y, IW, 52, fill="white", stroke=TORCH, sw=1.3)
    s.chip(x + 10, y + 10, "PyTorch", TORCH, w=66, h=20, size=11.5)
    s.text(x + 84, y + 25, "otherwise; the same weights, the", size=13)
    s.text(x + 10, y + 45, "same results, no engines needed", size=13)
    y += 60
    s.rect(x, y, IW, 70, fill="white", stroke=JAX, sw=1.3)
    s.chip(x + 10, y + 10, "JAX", JAX, w=42, h=20, size=11.5)
    s.text(x + 60, y + 25, "on request: a Haiku port of the trunk", size=13)
    s.text(x + 10, y + 45, "and diffusion modules; Boltz-2 can run", size=13)
    s.text(x + 10, y + 62, "its diffusion on TensorRT in the same job", size=13)

    mid = top + H1 / 2
    h_arrow(s, XS[0] + SW, XS[1], mid)
    h_arrow(s, XS[1] + SW, XS[2], mid)

    y = stage(s, XS[0], row2, SW, H2, "Run", "core/.venv/bin/python <entry point>")
    y = item(s, XS[0] + 12, y, IW, "stage the input", sub="copy to <out>/.residudes-engine, set the seed")
    y = item(s, XS[0] + 12, y, IW, "build the command", sub="entry point, engines, environment variables")
    y = item(s, XS[0] + 12, y, IW, "subprocess", sub="inside the runtime venv, cwd = core")
    y = item(s, XS[0] + 12, y, IW, "inference.log", sub="everything the model printed", kind="file")

    y = stage(s, XS[1], row2, SW, H2, "Result", "exit 3 if no sample was written")
    y = item(s, XS[1] + 12, y, IW, "post", sub="the model's own nesting → seed/sample layout")
    y = item(s, XS[1] + 12, y, IW, "verify", sub="layout and confidence files checked")
    y = item(s, XS[1] + 12, y, IW, "run.json", sub="command, environment, GPU, commit, wall time", kind="file")

    note(s, XS[2], row2, SW, H2, [
        "The CLI is plain Python 3.11 with no",
        "dependencies. torch, JAX and TensorRT are",
        "imported only inside the subprocess, from",
        "the runtime venv.",
        "",
        "residudes-engine submit does the same through Slurm,",
        "one job per seed; the site file travels with",
        "the job, so nothing is re-detected there.",
    ])

    elbow_down(s, XS[2] + SW / 2, top + H1 + 24, XS[0] + SW / 2, row2)
    h_arrow(s, XS[0] + SW, XS[1], row2 + H2 / 2)
    return s.save("runtime")


# ── Figure 3: which model consumes which part of the input ────────────────

def fig_inputs_matrix():
    rows = [
        ("MSA (unpaired)", "unpairedMsaPath", (1, 1, 1, 1)),
        ("Paired MSA for complexes", "pairedMsaPath", (1, 1, 1, 1)),
        ("Per-chain templates", "templates[]", (1, 1, 1, 1)),
        ("Multi-chain (interface) templates", "multimerTemplates[]", (1, 1, 1, 1)),
        ("Ligands, CCD codes or SMILES", "ligand{ccdCodes|smiles}", (1, 1, 1, 1)),
        ("Glycans, covalent bonds", "bondedAtomPairs", (1, 2, 1, 1)),
        ("Contact constraints", "constraint.contact[]", (1, 0, 0, 0)),
        ("Template weight", "predict --template-weight", (1, 1, 1, 1)),
    ]
    cols = ("Protenix v2", "OpenFold 3", "Boltz-2", "AlphaFold 3")
    Wm, Hm = 920, 66 + 34 * len(rows) + 64
    s = Svg(Wm, Hm)
    x0, xf, xc = 20, 320, [560, 660, 760, 860]
    s.text(x0, 30, "What each model reads from input.json", size=15, weight="bold")
    s.text(x0, 56, "input", size=12, fill=MUTED)
    s.text(xf, 56, "field", size=12, fill=MUTED)
    for x, c in zip(xc, cols):
        s.text(x, 56, c, size=13, weight="bold", anchor="middle")
    s.line(x0, 64, Wm - 20, 64, stroke=LINE)
    y = 86
    for label, field, cells in rows:
        s.text(x0, y + 5, label, size=13.5)
        s.text(xf, y + 5, field, size=12.5, font=MONO)
        for x, v in zip(xc, cells):
            if v == 1:
                s.circle(x, y, 6, fill=INK, stroke=INK)
            elif v == 2:
                s.circle(x, y, 6, fill="white", stroke=INK, sw=1.2)
                s.path(f"M {x} {y - 6} A 6 6 0 0 1 {x} {y + 6} Z", fill=INK, stroke=INK)
            else:
                s.circle(x, y, 6, fill="white", stroke=LINE, sw=1.0)
        s.line(x0, y + 17, Wm - 20, y + 17, stroke="#e3e3e3")
        y += 34
    y += 10
    s.circle(x0 + 6, y, 5.5, fill=INK, stroke=INK); s.text(x0 + 20, y + 4.5, "used", size=12.5, fill=MUTED)
    s.circle(x0 + 76, y, 5.5, fill="white", stroke=INK, sw=1.2); s.path(f"M {x0 + 76} {y - 5.5} A 5.5 5.5 0 0 1 {x0 + 76} {y + 5.5} Z", fill=INK, stroke=INK)
    s.text(x0 + 90, y + 4.5, "partly: OpenFold 3 keeps the glycan as a non-covalent ligand (it has no bond input)", size=12.5, fill=MUTED)
    y += 24
    s.circle(x0 + 6, y, 5.5, fill="white", stroke=LINE); s.text(x0 + 20, y + 4.5, "ignored: the field stays in the file, so one input still drives every model", size=12.5, fill=MUTED)
    return s.save("inputs_matrix")


# ── Figure 4: a terminal session ──────────────────────────────────────────

# Lines are (kind, text): what residudes-engine prints, with absolute paths shortened to
# relative ones (the prepare and verify blocks are copied from a real session).
TERMINAL = [
    ("cmd", "residudes-engine prepare --fasta asna.fasta --sto A2 --source colabfold"),
    ("step", "MSA search: colabfold"),
    ("out", "  colabfold: 1 unique sequence(s), mode env"),
    ("out", "  colabfold A: 3503 homologues, 2 template hits"),
    ("ok", "MSAs in engine_out/asna_inputs/msa"),
    ("out", "  template A: 12as_B (327 aligned, 99% identity)"),
    ("out", "  template A: 4lns_A (302 aligned, 59% identity)"),
    ("ok", "engine_out/asna_inputs/input.json  (1 entities, 2 templates)"),
    ("blank", ""),
    ("cmd", "residudes-engine predict --model bt2 --input engine_out/asna_inputs/input.json"),
    ("step", "residudes-engine 0.1.0  model=bt2 (Boltz-2)  backend=trt  site=ada"),
    ("out", "  (engines found: engines/rtx6000ada/boltz2_cuequiv_2048)"),
    ("step", "gpu : NVIDIA RTX PRO 6000 Blackwell cc=12.0 tag=rtx6000ada"),
    ("step", "out : engine_out/asna_bt2/seed_1"),
    ("step", "engines_dir: engines/rtx6000ada/boltz2_cuequiv_2048"),
    ("step", "diffusion_samples: 5"),
    ("step", "exit 0 after 71s; manifest engine_out/asna_bt2/seed_1/.residudes-engine/run.json"),
    ("blank", ""),
    ("cmd", "residudes-engine verify engine_out/asna_bt2/seed_1"),
    ("out", "seed dir : engine_out/asna_bt2/seed_1"),
    ("out", "layout   : OK  (5 samples)"),
    ("dim", "  sample     atoms   plddt    ptm   iptm   gpde   rank  missing"),
    ("out", "  sample_0    5172    98.1  0.971  0.961  0.293  0.977  -"),
    ("out", "  sample_1    5172    98.1  0.970  0.961  0.298  0.977  -"),
    ("out", "  sample_2    5172    98.1  0.970  0.961  0.298  0.977  -"),
    ("out", "  sample_3    5172    98.0  0.970  0.961  0.299  0.977  -"),
    ("out", "  sample_4    5172    98.0  0.970  0.960  0.296  0.976  -"),
    ("out", "top      : sample_0  model=bt2 seed=1 n_tokens=660 plddt=98.1 ptm=0.971"),
]


def fig_terminal():
    BG, FG, DIM, PROMPT, STEP, OK = "#1b1f27", "#d8dde6", "#7d8590", "#7ee787", "#79c0ff", "#56d364"
    size, lh = 16, 25
    Wt = 1000
    Ht = 44 + lh * len(TERMINAL) + 28
    s = Svg(Wt, Ht + 24, bg="white")
    s.rect(12, 12, Wt - 24, Ht, fill=BG, stroke="#30363d", r=10)
    s.rect(12, 12, Wt - 24, 34, fill="#262b34", stroke="#262b34", r=10)
    s.rect(12, 30, Wt - 24, 16, fill="#262b34", stroke="#262b34", r=0)
    for i, c in enumerate(("#ff5f57", "#febc2e", "#28c840")):
        s.circle(32 + i * 20, 29, 6, fill=c, stroke=c)
    s.text(Wt / 2, 33.5, "residudes-engine", size=12.5, fill="#9aa4b2", anchor="middle", font=MONO)
    y = 46 + 24
    for kind, text in TERMINAL:
        x = 30
        if kind == "cmd":
            s.spans(x, y, [("$ ", PROMPT, "bold"), (text, FG)], size=size)
        elif kind == "step":
            s.spans(x, y, [("▶ ", STEP, "bold"), (text, FG)], size=size)
        elif kind == "ok":
            s.spans(x, y, [("✓ ", OK, "bold"), (text, FG)], size=size)
        elif kind == "dim":
            s.spans(x, y, [(text, DIM)], size=size)
        elif kind == "out":
            s.spans(x, y, [(text, FG)], size=size)
        y += lh
    return s.save("terminal")


# ── README figures ────────────────────────────────────────────────────────

MODELS = ["Protenix v2", "OpenFold 3", "Boltz-2", "AlphaFold 3"]
SUMMARY = OUT.parent / "benchmark" / "speedup_summary.json"    # written by bench/collect.py


def speedups() -> tuple[dict, dict]:
    """{model: {trt: (median, max), jax: (median, max)}} and the run facts, from the benchmark collector."""
    if SUMMARY.is_file():
        d = json.loads(SUMMARY.read_text())
        names = {"pt2": "Protenix v2", "of3": "OpenFold 3", "bt2": "Boltz-2", "af3": "AlphaFold 3"}
        return {names.get(k, k): v for k, v in d["models"].items()}, d
    fallback = {"Protenix v2": {"trt": (1.96, 2.51), "jax": (1.57, 2.17)}, "OpenFold 3": {"trt": (2.11, 2.70), "jax": (1.80, 2.33)},
                "Boltz-2": {"trt": (1.70, 2.07), "jax": (1.16, 1.18)}, "AlphaFold 3": {"trt": (1.33, 1.64), "jax": (1.46, 2.13)}}
    return fallback, {"gpu": "RTX PRO 6000 Blackwell", "inputs": 8, "tokens": [36, 2004]}


def fig_overview():
    """The README hero: one line from the sequence to the structure; the four models as parallel tracks between
    the input.json interchange and the output, each carrying its measured forward-pass speed-up per backend.
    Drawn large: GitHub shows it about 880 px wide, so text is at least 18 px on 1600 px."""
    sp, facts = speedups()
    W, H = 1600, 820
    s = Svg(W, H)
    ink, rail, grey = "#1d1d1f", "#2a2a2e", "#6e6e73"
    s.text(60, 78, "residudes-engine", size=52, weight="bold", fill=ink, spacing="-0.5")
    s.text(60, 120, "from a sequence to a structure, on the backend you have", size=24, fill=grey)
    s.mascot("running", W - 170, 300, 300, 250)

    y = 470
    s.line(80, y, 640, y, stroke=rail, sw=12, cap="round")
    s.line(1340, y, 1560, y, stroke=rail, sw=12, cap="round")
    ys = [y - 165, y - 55, y + 55, y + 165]
    for yy in ys:
        s.path(f"M 640 {y} C 730 {y}, 730 {yy}, 820 {yy} L 1200 {yy} C 1270 {yy}, 1270 {y}, 1340 {y}", stroke=rail, sw=7)
    s.line(390, y - 6, 390, 200, stroke=grey, sw=4, cap="round")
    for yy, lab in ((200, "DeepFold-PLM API"), (262, "ColabFold API"), (324, "local MMseqs2")):
        s.circle(390, yy, 9, fill="white", stroke=grey, sw=4)
        s.text(412, yy + 7, lab, size=20, fill=grey)

    def station(x, yy, label, sub=None, *, big=False, cmd=None):
        s.circle(x, yy, 20 if big else 14, fill="white", stroke=rail, sw=7 if big else 6)
        if big:
            s.circle(x, yy, 6, fill=rail)
        if label:
            s.text(x, yy + 60, label, size=24, weight="bold", fill=ink, anchor="middle")
        if sub:
            s.text(x, yy + 88, sub, size=18, fill=grey, anchor="middle")
        if cmd:
            s.text(x, yy + (116 if sub else 88), cmd, size=18, fill=grey, anchor="middle", font=MONO)

    station(140, y, "Sequence", "FASTA")
    station(390, y, "MSA, templates", "searched or given", cmd="msa")
    station(640, y, "input.json", "ligands · glycans", big=True, cmd="inputs")
    station(1340, y, "", None, big=True)
    station(1460, y, "Prediction", "model.cif · arrays", cmd="verify")
    cw, cg = 92, 12
    for yy, name in zip(ys, MODELS):
        s.circle(840, yy, 12, fill="white", stroke=rail, sw=5)
        s.text(840, yy - 24, name, size=22, weight="bold", fill=ink, anchor="middle")
        badges = [(TRT, f"{sp[name]['trt'][0]:.2f}×"), (JAX, f"{sp[name]['jax'][0]:.2f}×"), (TORCH, "1×")]
        for j, (col, lab) in enumerate(badges):
            bx = 925 + j * (cw + cg)
            s.rect(bx, yy - 18, cw, 36, fill=col, stroke="none", r=18)
            s.text(bx + cw / 2, yy + 8, lab, size=21, weight="bold", fill="white", anchor="middle")
    s.text(925 + (3 * cw + 2 * cg) / 2, ys[0] - 62, "forward pass vs PyTorch", size=19, fill=grey, anchor="middle")
    for j, (col, lab) in enumerate(((TRT, "TensorRT"), (JAX, "JAX"), (TORCH, "PyTorch"))):
        s.text(925 + j * (cw + cg) + cw / 2, ys[-1] + 52, lab, size=20, weight="bold", fill=col, anchor="middle")
    s.text(60, H - 34, f"median over {facts['inputs']} CASP17 inputs of {facts['tokens'][0]} to {facts['tokens'][1]} tokens on one "
           f"{facts['gpu'].replace(' Blackwell', '')}; same input, weights and seed", size=19, fill=grey)
    s.text(60, H - 64, "grey monospace: the residudes-engine subcommand for that step (residudes-engine msa, inputs, verify)",
           size=19, fill=grey)
    return s.save("overview")


def fig_speed():
    """The speed highlight: the model × backend grid with the measured forward-pass speed-up over PyTorch per cell.
    Drawn large (text at least 18 px on 1600 px)."""
    sp, facts = speedups()
    W, H = 1600, 860
    s = Svg(W, H)
    ink, grey, rule = "#141414", "#6d6d6d", "#d9d9d9"
    s.text(60, 78, "residudes-engine", size=48, weight="bold", fill=ink, spacing="-0.5")
    s.text(60, 118, "four models, three backends, one command", size=26, fill=grey)   # own line: fits any title
    gx, gy, cw, rh = 400, 200, 330, 104
    cols = [("PyTorch", TORCH, None), ("TensorRT", TRT, "trt"), ("JAX", JAX, "jax")]

    def mix(c, t):
        r, g, b = int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16)
        r2, g2, b2 = (int(255 + (v - 255) * t) for v in (r, g, b))
        return f"#{r2:02x}{g2:02x}{b2:02x}"

    for j, (lab, col, key) in enumerate(cols):
        s.text(gx + j * cw + cw / 2, gy - 30, lab, size=26, weight="bold", fill=ink, anchor="middle")
        s.rect(gx + j * cw + 10, gy - 16, cw - 20, 5, fill=col, stroke="none", r=0)
    for i, m in enumerate(MODELS):
        yy = gy + i * rh
        s.text(gx - 28, yy + rh / 2 + 9, m, size=26, weight="bold", fill=ink, anchor="end")
        for j, (lab, col, key) in enumerate(cols):
            v = 1.0 if key is None else sp[m][key][0]
            t = max(0.0, min(1.0, (v - 1.0) / 1.2))
            fill = "#f2f2f2" if key is None else mix(col, 0.12 + 0.88 * t)
            s.rect(gx + j * cw + 10, yy + 8, cw - 20, rh - 16, fill=fill, stroke="none", r=10)
            dark = key is not None and t > 0.45
            s.text(gx + j * cw + cw / 2, yy + rh / 2 + 13, "1×" if key is None else f"{v:.2f}×", size=36, weight="bold",
                   fill="white" if dark else ink, anchor="middle")
    s.text(60, gy + 4 * rh + 50, f"forward pass, median over {facts['inputs']} CASP17 inputs of {facts['tokens'][0]} to "
           f"{facts['tokens'][1]} tokens on one {facts['gpu'].replace(' Blackwell', '')}; same input, weights and seed",
           size=20, fill=grey)
    s.text(60, gy + 4 * rh + 80, "AlphaFold 3's PyTorch column is the port in this repository; its JAX column is DeepMind's model",
           size=20, fill=grey)
    s.line(60, H - 70, W - 60, H - 70, stroke=rule, sw=1.5)
    s.text(60, H - 30, "seed_<N>/sample_<i>/model.cif · summary.json · arrays/", size=20, fill=ink, font=MONO)
    s.text(W - 60, H - 30, "the same output for every model and backend", size=20, fill=grey, anchor="end")
    return s.save("speed")


if __name__ == "__main__":
    for f in (fig_overview, fig_speed, fig_runtime, fig_inputs_matrix, fig_terminal):
        print(f())
