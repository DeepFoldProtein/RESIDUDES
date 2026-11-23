#!/usr/bin/env python3
"""Draw the lantern figures as SVG (stdlib only) and rasterise them with rsvg-convert.

    python3 docs/figures/make_figures.py

Drawn large on purpose: GitHub shows a README figure about 880 px wide, so text is at least 20 px on a 1600 px canvas.
  overview  the stage line: the agent's stations, the person's two checkpoints (with the loops back) and gates,
            and the tools each stretch drives (lantern-engine, lantern-qa)
"""
from __future__ import annotations

import base64
import shutil
import subprocess
from pathlib import Path

OUT = Path(__file__).resolve().parent
LANTERN = OUT.parents[1]
MASCOT = LANTERN.parent / "mascot" / "fig"

FONT = "Helvetica, Arial, 'Liberation Sans', 'DejaVu Sans', sans-serif"
INK, MUTED, TRACK, PERSON = "#1d1d1f", "#6e6e73", "#2a2a2e", "#D55E00"
BLUE, GREEN = "#0072B2", "#009E73"                     # lantern-engine, lantern-qa (as in the RESIDUDES overview)


class Svg:
    def __init__(self, w, h):
        self.w, self.h, self.parts = w, h, []

    def text(self, x, y, s, *, size=22, weight="normal", fill=INK, anchor="middle", font=FONT):
        s = str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        self.parts.append(f'<text x="{x:.1f}" y="{y:.1f}" font-family="{font}" font-size="{size}" font-weight="{weight}" '
                          f'fill="{fill}" text-anchor="{anchor}" xml:space="preserve">{s}</text>')

    def line(self, x1, y1, x2, y2, *, stroke=TRACK, sw=12, dash=None, op=1.0):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" stroke-width="{sw}" '
                          f'stroke-linecap="round" stroke-opacity="{op}"{d}/>')

    def path(self, d, *, stroke=TRACK, sw=5, dash=None, op=1.0):
        dd = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(f'<path d="{d}" fill="none" stroke="{stroke}" stroke-width="{sw}" stroke-linecap="round" stroke-opacity="{op}"{dd}/>')

    def circle(self, x, y, r, *, fill="white", stroke=TRACK, sw=5):
        self.parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')

    def rect(self, x, y, w, h, *, fill, rx=8):
        self.parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" fill="{fill}"/>')

    def diamond(self, x, y, r, colour):
        self.parts.append(f'<path d="M {x:.1f} {y - r:.1f} L {x + r:.1f} {y:.1f} L {x:.1f} {y + r:.1f} L {x - r:.1f} {y:.1f} Z" fill="{colour}"/>')

    def arrowhead(self, x, y, colour, up=True, size=10):
        b = y + 1.6 * size if up else y - 1.6 * size
        self.parts.append(f'<path d="M {x - size} {b} L {x} {y} L {x + size} {b} Z" fill="{colour}"/>')

    def mascot(self, name, cx, bottom, box_w, box_h):
        """A mascot, scaled to fit the box and standing on `bottom`, embedded so the figure stays one file."""
        p = MASCOT / f"{name}.png"
        if not p.exists():
            return
        import struct
        w0, h0 = struct.unpack(">II", p.read_bytes()[16:24])
        k = min(box_w / w0, box_h / h0)
        w, h = w0 * k, h0 * k
        b64 = base64.b64encode(p.read_bytes()).decode()
        self.parts.append(f'<image x="{cx - w / 2:.1f}" y="{bottom - h:.1f}" width="{w:.1f}" height="{h:.1f}" '
                          f'href="data:image/png;base64,{b64}"/>')

    def save(self, name):
        body = f'<rect width="{self.w}" height="{self.h}" fill="white"/>' + "".join(self.parts)
        (OUT / f"{name}.svg").write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" '
                                         f'viewBox="0 0 {self.w} {self.h}">{body}</svg>')
        if shutil.which("rsvg-convert"):
            subprocess.run(["rsvg-convert", "-z", "1.5", "-o", str(OUT / f"{name}.png"), str(OUT / f"{name}.svg")], check=True)
        print("wrote", OUT / f"{name}.svg")


# grouped stations: big labels need room, so neighbouring stages that never involve a person share a station
STATIONS = [("NEW", "analyse"), ("IDENTIFIED", "copy number"), ("HYPOTHESES", "variants, plan"),
            ("PLAN REVIEW", "⇄ person"), ("RUN", "inputs · predictions"), ("POOL READY", "QA · gates"),
            ("RESULTS REVIEW", "⇄ person"), ("SLATE", "approve · person"), ("SUBMIT", "a person"), ("RETRO", "vs native")]
PERSON_ST = {"PLAN REVIEW", "RESULTS REVIEW", "SLATE", "SUBMIT"}
CHECKPOINT = {"PLAN REVIEW", "RESULTS REVIEW"}


def fig_overview():
    W, H = 1600, 820
    s = Svg(W, H)
    s.mascot("../logo/logo_512", W - 120, 160, 180, 158)
    s.text(60, 78, "lantern", size=52, weight="bold", anchor="start")
    s.text(60, 120, "analyses a target, plans with a person, runs the tools, reviews with a person, proposes a slate",
           size=24, fill=MUTED, anchor="start")
    s.text(60, 152, "one instruction set for every harness — Claude Code · Codex · Antigravity — and for the "
                    "Claude Agent SDK runner", size=21, fill=MUTED, anchor="start")
    y, x0, dx = 440, 120, 151
    X = {name: x0 + i * dx for i, (name, _) in enumerate(STATIONS)}

    def band(xa, xb, name, sub, colour, stations):
        top, h = 168, 92
        s.rect(xa, top, xb - xa, h, fill=colour, rx=18)
        s.text((xa + xb) / 2, top + 44, name, size=36, weight="bold", fill="white")
        s.text((xa + xb) / 2, top + 76, sub, size=20, fill="white")
        for st in stations:
            s.line(X[st], top + h + 14, X[st], y - 96, stroke=colour, sw=4, dash="8 7")
            s.arrowhead(X[st], top + h + 4, colour)
    band(X["HYPOTHESES"] - 70, X["RUN"] + 70, "lantern-engine", "MSA · inputs · predictions", BLUE, ["HYPOTHESES", "RUN"])
    band(X["RUN"] + 92, X["RESULTS REVIEW"] + 72, "lantern-qa", "scores · clusters · checks", GREEN,
         ["POOL READY", "RESULTS REVIEW"])

    s.line(60, y, X["SLATE"], y)
    s.line(X["SLATE"], y, X["SUBMIT"], y, stroke=PERSON, dash="3 22")
    s.line(X["SUBMIT"], y, X["RETRO"], y, dash="3 22")
    s.line(X["RETRO"], y, W - 40, y)
    for src, dst, depth, label in (("RESULTS REVIEW", "RUN", 150, "more sampling"),
                                   ("RESULTS REVIEW", "HYPOTHESES", 214, "new hypotheses"),
                                   ("PLAN REVIEW", "HYPOTHESES", 150, "rework")):
        xa, xb = X[src], X[dst]
        start = y + 104 if [n for n, _ in STATIONS].index(src) % 2 else y + 26     # below a label under the line, or at the station
        s.path(f"M {xa} {start} C {xa} {y + depth}, {xa - 30} {y + depth}, {(xa + xb) / 2} {y + depth} "
               f"C {xb + 30} {y + depth}, {xb} {y + depth}, {xb} {y + 40}", stroke=PERSON, sw=4, dash="10 8")
        s.arrowhead(xb, y + 22, PERSON, size=11)
        s.text((xa + xb) / 2, y + depth - 12, label, size=21, fill=PERSON)
    for i, (name, sub) in enumerate(STATIONS):
        x = X[name]
        person = name in PERSON_ST
        if name in CHECKPOINT:
            s.circle(x, y, 22, stroke=PERSON, sw=6)
            s.circle(x, y, 9, fill=PERSON, stroke=PERSON, sw=1)
        else:
            s.circle(x, y, 17 if person else 15, stroke=PERSON if person else TRACK, sw=6)
        above = i % 2 == 0
        col = PERSON if person else INK
        if above:
            s.text(x, y - 66, name, size=23, weight="bold", fill=col)
            s.text(x, y - 38, sub, size=19, fill=col if person else MUTED)
        else:
            s.text(x, y + 62, name, size=23, weight="bold", fill=col)
            s.text(x, y + 90, sub, size=19, fill=col if person else MUTED)
    s.mascot("new", 130, H - 74, 215, 170)                    # the line starts here
    s.mascot("retrospective", X["RETRO"] + 26, y - 30, 168, 150)  # and ends here, scored against the native
    s.mascot("human_only", X["SLATE"] + 20, H - 74, 250, 168)
    s.text(60, H - 30, "every finding and decision is recorded in the target's dossier with its evidence; "
                       "only a person resolves a checkpoint, approves or submits", size=21, fill=MUTED, anchor="start")
    s.save("overview")


if __name__ == "__main__":
    fig_overview()
