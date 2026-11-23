#!/usr/bin/env python3
"""Draw the lantern-qa overview as SVG (stdlib only) and rasterise it with rsvg-convert.

    python3 docs/figures/make_figures.py                 # writes docs/figures/overview.svg and .png

A metro map from a prediction pool to a ranked pool, one line per tool, with the file each line writes.
Drawn large (GitHub shows it about 880 px wide). Same plain style as lantern-engine/docs/figures, Okabe-Ito colours.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

OUT = Path(__file__).resolve().parent
MASCOT = OUT.parent.parent.parent / "mascot" / "fig"
FONT = "Helvetica, Arial, 'Liberation Sans', 'DejaVu Sans', sans-serif"
MONO = "Menlo, Consolas, 'DejaVu Sans Mono', monospace"
INK, MUTED, LINE, GRID, TRACK = "#1d1d1f", "#8e8e93", "#8a8a8a", "#e5e5ea", "#2a2a2e"
BLUE, GREEN, ORANGE, GREY = "#0072B2", "#009E73", "#D55E00", "#b0b0b5"


class Svg:
    def __init__(self, w: int, h: int):
        self.w, self.h, self.parts = w, h, [f'<rect width="{w}" height="{h}" fill="white"/>']

    def add(self, s: str) -> None:
        self.parts.append(s)

    def text(self, x, y, s, *, size=14, weight="normal", fill=INK, anchor="start", font=FONT, rotate=None):
        s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        tr = f' transform="rotate({rotate} {x} {y})"' if rotate is not None else ""
        self.add(f'<text x="{x:.1f}" y="{y:.1f}" font-family="{font}" font-size="{size}" font-weight="{weight}" '
                 f'fill="{fill}" text-anchor="{anchor}"{tr} xml:space="preserve">{s}</text>')

    def line(self, x1, y1, x2, y2, *, stroke=TRACK, sw=1.0, dash=None, cap="round"):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" stroke-width="{sw}" stroke-linecap="{cap}"{d}/>')

    def path(self, d, *, stroke=TRACK, sw=5):
        self.add(f'<path d="{d}" fill="none" stroke="{stroke}" stroke-width="{sw}" stroke-linecap="round" stroke-linejoin="round"/>')

    def circle(self, x, y, r, *, fill="white", stroke=TRACK, sw=4.0, opacity=None):
        o = f' fill-opacity="{opacity}"' if opacity is not None else ""
        self.add(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{o}/>')

    def pill(self, x, y, w, label, colour, *, h=26):
        self.add(f'<rect x="{x - w / 2:.1f}" y="{y - h / 2:.1f}" width="{w}" height="{h}" rx="{h / 2}" fill="{colour}"/>')
        self.text(x, y + 5, label, size=13, weight="bold", fill="white", anchor="middle")

    def mascot(self, name, cx, bottom, box_w, box_h):
        """A mascot, scaled to fit the box and standing on `bottom`, embedded so the figure stays one file."""
        import base64, struct
        p = MASCOT / f"{name}.png"
        if not p.exists():
            return
        w0, h0 = struct.unpack(">II", p.read_bytes()[16:24])
        k = min(box_w / w0, box_h / h0)
        w, h = w0 * k, h0 * k
        b64 = base64.b64encode(p.read_bytes()).decode()
        self.parts.append(f'<image x="{cx - w / 2:.1f}" y="{bottom - h:.1f}" width="{w:.1f}" height="{h:.1f}" '
                          f'href="data:image/png;base64,{b64}"/>')

    def save(self, name: str) -> None:
        svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" '
               f'viewBox="0 0 {self.w} {self.h}">' + "".join(self.parts) + "</svg>")
        (OUT / f"{name}.svg").write_text(svg)
        if shutil.which("rsvg-convert"):
            subprocess.run(["rsvg-convert", "-z", "1.5", "-o", str(OUT / f"{name}.png"), str(OUT / f"{name}.svg")], check=True)
        print("wrote", OUT / f"{name}.svg")


def station(s: Svg, x, y, *, big=False):
    if big:
        s.circle(x, y, 22, sw=7)
        s.circle(x, y, 7, fill=TRACK, sw=0)
    else:
        s.circle(x, y, 13, sw=5)


# Drawn large on purpose: GitHub shows a README figure about 880 px wide, so text is at least 18 px on 1600 px.
def fig_overview(P: dict):
    s = Svg(1600, 900)
    s.text(60, 78, "lantern-qa", size=52, weight="bold")
    s.text(60, 120, "which predictions to trust: score, group and inspect a pool, as the team did in CASP17", size=24, fill=MUTED)
    s.mascot("pool_ready", 1420, 215, 300, 205)

    y0, xj, xm = 470, 330, 1330
    s.line(70, y0, xj, y0, sw=12)
    s.line(xm, y0, 1550, y0, sw=12)
    station(s, 130, y0)
    s.text(130, y0 + 62, "Prediction pool", size=24, weight="bold", anchor="middle")
    s.text(130, y0 + 92, "model.cif · summary", size=19, fill=MUTED, anchor="middle")
    s.text(130, y0 + 122, "from lantern_engine", size=19, fill=MUTED, anchor="middle", font=MONO)
    ys = {"interface": 250, "cluster": 470, "visualize": 690}
    for y in ys.values():
        s.path(f"M {xj} {y0} C {xj + 100} {y0}, {xj + 100} {y}, {xj + 200} {y} L {xm - 200} {y} "
               f"C {xm - 100} {y}, {xm - 100} {y0}, {xm} {y0}", sw=7)
    stops = {
        "interface": [("features", "DFIRE · rotamers · MPNN"), ("XGBoost", "DockQ · ICS · IPS")],
        "cluster":   [("TM · iTM · interface", "GPU distance matrix"), ("clusters", "adaptive K · medoids")],
        "visualize": [("confidence plots", "pLDDT · PAE · PDE · MSA"), ("rankings", "ranking tables")],
    }
    outputs = {"interface": "xgb_dockq.csv", "cluster": "cluster_summary.csv", "visualize": "all_rankings.csv"}
    for line_name, y in ys.items():
        s.text(xj + 210, y - 92, f"lantern-qa {line_name}", size=20, fill=MUTED, font=MONO)
        for (name, sub), x in zip(stops[line_name], (640, 900)):
            station(s, x, y)
            s.text(x, y - 30, name, size=23, weight="bold", anchor="middle")
            s.text(x, y + 44, sub, size=19, fill=MUTED, anchor="middle")
        s.add(f'<rect x="{1035}" y="{y - 21}" width="{250}" height="42" rx="21" fill="{TRACK}"/>')
        s.text(1160, y + 7, outputs[line_name], size=19, weight="bold", fill="white", anchor="middle")
    station(s, xj, y0, big=True)
    station(s, xm, y0, big=True)
    station(s, 1470, y0)
    s.text(1470, y0 + 62, "Ranked pool", size=24, weight="bold", anchor="middle")
    s.text(1470, y0 + 92, "scores · clusters", size=19, fill=MUTED, anchor="middle")
    s.text(1470, y0 + 122, "→ lantern", size=19, fill=MUTED, anchor="middle", font=MONO)
    s.text(60, 862, "interface: DockQ, ICS and IPS predicted for every chain pair · cluster: TM, iTM and interface similarity on GPU",
           size=21, fill=MUTED)
    s.save("overview")


if __name__ == "__main__":
    fig_overview({})
