#!/usr/bin/env python3
"""Draw the RESIDUDES overview as SVG (stdlib only) and rasterise it with rsvg-convert.

    python3 docs/figures/make_figures.py

One map of the three tools, drawn large (GitHub shows it about 880 px wide, so text is at least 19 px on 1600 px):
lantern's stations on top (the person's two checkpoints in orange, with the loops back), lantern-engine's lane (sequence to
prediction pool) and lantern-qa's lane (pool to ranked pool) below, joined at the prediction pool, with the station
that drives each tool.
"""
from __future__ import annotations

import base64
import shutil
import subprocess
from pathlib import Path

OUT = Path(__file__).resolve().parent
MASCOT = OUT.parent.parent / "mascot" / "fig"           # one picture per stage, keyed onto transparency
FONT = "Helvetica, Arial, 'Liberation Sans', 'DejaVu Sans', sans-serif"
MONO = "Menlo, Consolas, 'DejaVu Sans Mono', monospace"
INK, MUTED, TRACK = "#1d1d1f", "#6e6e73", "#2a2a2e"
BLUE, GREEN, ORANGE = "#0072B2", "#009E73", "#D55E00"          # lantern-engine, lantern-qa, a person


class Svg:
    def __init__(self, w, h):
        self.w, self.h, self.parts = w, h, []

    def text(self, x, y, s, *, size=22, weight="normal", fill=INK, anchor="middle", font=FONT):
        s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        self.parts.append(f'<text x="{x:.1f}" y="{y:.1f}" font-family="{font}" font-size="{size}" font-weight="{weight}" '
                          f'fill="{fill}" text-anchor="{anchor}" xml:space="preserve">{s}</text>')

    def line(self, x1, y1, x2, y2, *, stroke=TRACK, sw=12, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" stroke-width="{sw}" stroke-linecap="round"{d}/>')

    def path(self, d, *, stroke=TRACK, sw=7, dash=None):
        dd = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(f'<path d="{d}" fill="none" stroke="{stroke}" stroke-width="{sw}" stroke-linecap="round"{dd}/>')

    def circle(self, x, y, r, *, fill="white", stroke=TRACK, sw=5):
        self.parts.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')

    def tag(self, x, y, label, colour, *, h=50, size=30):
        w = 34 + 0.6 * size * len(label)
        self.parts.append(f'<rect x="{x}" y="{y - h / 2}" width="{w}" height="{h}" rx="{h / 2}" fill="{colour}"/>')
        self.text(x + w / 2, y + size * 0.36, label, size=size, weight="bold", fill="white")
        return w

    def mascot(self, name, cx, bottom, box_w, box_h):
        """A stage's picture, centred on cx, standing on `bottom`, scaled to fit inside box_w × box_h — the
        pictures differ in shape (a pool is twice as wide as it is tall) and scaling by height alone would let the
        wide ones run into their neighbours. The bytes go in as a data URI so the figure stays one file: a renderer
        that will not follow a relative link still shows it."""
        p = MASCOT / f"{name}.png"
        if not p.exists():
            return
        import struct
        w0, h0 = struct.unpack(">II", p.read_bytes()[16:24])            # PNG header: width, height
        k = min(box_w / w0, box_h / h0)
        w, h = w0 * k, h0 * k
        b64 = base64.b64encode(p.read_bytes()).decode()
        self.parts.append(f'<image x="{cx - w / 2:.1f}" y="{bottom - h:.1f}" width="{w:.1f}" height="{h:.1f}" '
                          f'href="data:image/png;base64,{b64}"/>')

    def arrowhead(self, x, y, colour, up=True):
        b = y + 18 if up else y - 18
        self.parts.append(f'<path d="M {x - 10} {b} L {x} {y} L {x + 10} {b} Z" fill="{colour}"/>')

    def arrow(self, x, y1, y2, colour, label=None, left=False):
        up = y2 < y1
        self.line(x, y1, x, y2 + (14 if up else -14), stroke=colour, sw=4, dash="9 8")
        self.arrowhead(x, y2, colour, up=up)
        if label:
            self.text(x - 14 if left else x + 14, (y1 + y2) / 2 + 7, label, size=19, fill=colour,
                      anchor="end" if left else "start", font=MONO)

    def save(self, name):
        body = f'<rect width="{self.w}" height="{self.h}" fill="white"/>' + "".join(self.parts)
        (OUT / f"{name}.svg").write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" '
                                         f'viewBox="0 0 {self.w} {self.h}">{body}</svg>')
        if shutil.which("rsvg-convert"):
            subprocess.run(["rsvg-convert", "-z", "1.5", "-o", str(OUT / f"{name}.png"), str(OUT / f"{name}.svg")], check=True)
        print("wrote", OUT / f"{name}.svg")


def fig_overview():
    W, H = 1820, 1330
    s = Svg(W, H)
    s.mascot("../logo/logo_512", W - 130, 232, 230, 212)
    s.text(60, 78, "RESIDUDES", size=52, weight="bold", anchor="start")
    s.text(60, 120, "an agent that runs the campaign, a toolkit that predicts, a QA stack that ranks",
           size=24, fill=MUTED, anchor="start")

    # ---- lantern: six stations (the stage machine grouped), checkpoints in orange, loops back above the line ------
    yt = 400
    st = [("ANALYSE", "target · priors", 250, "identified"), ("PLAN REVIEW", "⇄ person", 520, "plan_review"),
          ("PREDICT", "inputs · runs", 790, "running"), ("QA", "gates · pool", 1060, "pool_ready"),
          ("RESULTS REVIEW", "⇄ person", 1370, "results_review"), ("SLATE", "a person approves", 1660, "slate_proposed")]
    X = {n: x for n, _, x, _ in st}
    s.tag(60, 180, "lantern", TRACK)
    s.text(250, 190, "the campaign agent: run in Claude Code and the Agent SDK, laid out for Codex and Antigravity too", size=22, fill=MUTED, anchor="start")
    s.line(80, yt, X["SLATE"], yt)
    s.line(X["SLATE"], yt, W - 50, yt, stroke=ORANGE, dash="3 22")
    s.text(W - 50, yt - 26, "approve · package · submit", size=19, fill=ORANGE, anchor="end")
    for dst, top, label in (("PREDICT", yt - 90, "more sampling"), ("ANALYSE", yt - 140, "new hypotheses")):
        xa, xb = X["RESULTS REVIEW"], X[dst]
        s.path(f"M {xa} {yt - 26} C {xa} {top}, {xa - 30} {top}, {(xa + xb) / 2} {top} "
               f"C {xb + 30} {top}, {xb} {top}, {xb} {yt - 40}", stroke=ORANGE, sw=4)
        s.arrowhead(xb, yt - 24, ORANGE, up=False)
        s.text((xa + xb) / 2, top - 10, label, size=20, fill=ORANGE)
    for name, sub, x, face in st:
        person = "REVIEW" in name or name == "SLATE"
        col = ORANGE if person else INK
        if "REVIEW" in name:
            s.circle(x, yt, 22, stroke=ORANGE, sw=6)
            s.circle(x, yt, 9, fill=ORANGE, stroke=ORANGE, sw=1)
        else:
            s.circle(x, yt, 16, stroke=col, sw=6)
        s.mascot(face, x, yt + 210, 248, 172)
        s.text(x, yt + 252, name, size=24, weight="bold", fill=col)
        s.text(x, yt + 280, sub, size=19, fill=ORANGE if person else MUTED)

    # ---- lantern-engine lane --------------------------------------------------------------------------------------------
    yb = 960
    pool_x, ranked_x = X["QA"], X["RESULTS REVIEW"]
    s.line(80, yb, 500, yb, stroke=BLUE)
    for name, sub, x in (("Sequence", "FASTA", 90), ("MSA", "templates", X["ANALYSE"]), ("input.json", "ligands", 480)):
        s.circle(x, yb, 15, stroke=BLUE, sw=6)
        s.text(x, yb + 52, name, size=22, weight="bold")
        s.text(x, yb + 80, sub, size=18, fill=MUTED)
    mx = X["PREDICT"]
    for name, dy in (("Protenix v2", -96), ("OpenFold 3", -32), ("Boltz-2", 32), ("AlphaFold 3", 96)):
        y = yb + dy
        s.path(f"M 480 {yb} C 560 {yb}, {mx - 110} {y}, {mx - 50} {y} L {mx + 50} {y} C {mx + 120} {y}, {pool_x - 90} {yb}, "
               f"{pool_x} {yb}", stroke=BLUE)
        s.circle(mx, y, 10, stroke=BLUE, sw=4)
        s.text(mx, y - 16, name, size=19)
    s.text(mx, yb + 150, "PyTorch · TensorRT · JAX", size=20, fill=BLUE, weight="bold")

    # ---- lantern-qa lane -----------------------------------------------------------------------------------------
    mid = (pool_x + ranked_x) / 2
    for name, dy in (("interface", -70), ("cluster", 0), ("visualize", 70)):
        y = yb + dy
        s.path(f"M {pool_x} {yb} C {pool_x + 70} {yb}, {mid - 90} {y}, {mid - 45} {y} L {mid + 45} {y} "
               f"C {mid + 90} {y}, {ranked_x - 70} {yb}, {ranked_x} {yb}", stroke=GREEN)
        s.circle(mid, y, 10, stroke=GREEN, sw=4)
        s.text(mid, y - 18, name, size=20, weight="bold")
    s.line(ranked_x, yb, W - 60, yb, stroke=GREEN)
    for x, name, sub in ((pool_x, "prediction pool", "model.cif · summary"), (ranked_x, "ranked pool", "scores · clusters · plots")):
        s.circle(x, yb, 20, stroke=TRACK, sw=7)
        s.circle(x, yb, 6, fill=TRACK, stroke=TRACK, sw=1)
        s.text(x, yb + 128, name, size=22, weight="bold")
        s.text(x, yb + 156, sub, size=18, fill=MUTED)

    # ---- which station drives which tool -------------------------------------------------------------------------
    s.arrow(X["ANALYSE"], yt + 300, yb - 24, BLUE, "lantern-engine msa")
    s.arrow(X["PREDICT"], yt + 300, yb - 146, BLUE, "lantern-engine predict", left=True)
    s.arrow(pool_x, yt + 300, yb - 28, GREEN, "lantern-qa")
    s.arrow(ranked_x, yb - 28, yt + 300, TRACK, "review · slate")

    # ---- lane names ----------------------------------------------------------------------------------------------
    s.tag(60, 1225, "lantern-engine", BLUE)
    s.text(60, 1280, "inputs and predictions: four models, one input file", size=21, fill=MUTED, anchor="start")
    s.tag(pool_x - 30, 1225, "lantern-qa", GREEN)
    s.text(pool_x - 30, 1280, "the CASP17 QA stack: which predictions to trust", size=21, fill=MUTED, anchor="start")
    s.save("overview")


if __name__ == "__main__":
    fig_overview()
