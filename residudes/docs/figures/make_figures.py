#!/usr/bin/env python3
"""Draw the residudes figures as SVG (stdlib only) and rasterise them with rsvg-convert.

    python3 docs/figures/make_figures.py

Drawn large on purpose: GitHub shows a README figure about 880 px wide, so text is at least 20 px on a 1600 px canvas.
  overview  the stage line: the agent's stations, the person's two checkpoints (with the loops back) and gates,
            and the tools each stretch drives (residudes-engine, residudes-qa)
"""
from __future__ import annotations

import base64
import shutil
import subprocess
from pathlib import Path

OUT = Path(__file__).resolve().parent
RESIDUDES = OUT.parents[1]
MASCOT = RESIDUDES.parent / "mascot" / "fig"

FONT = "Helvetica, Arial, 'Liberation Sans', 'DejaVu Sans', sans-serif"
INK, MUTED, TRACK, PERSON = "#1d1d1f", "#6e6e73", "#2a2a2e", "#D55E00"
BLUE, GREEN = "#0072B2", "#009E73"                     # residudes-engine, residudes-qa (as in the RESIDUDES overview)



# Compatibility marks, kept as path data so the figure stays one file and needs no asset directory.
# Claude and Google Gemini: simple-icons (CC0-1.0, simpleicons.org). ChatGPT: Wikimedia Commons, public domain.
# Each mark belongs to its owner and is used here only to say which harness reads the same instructions.
HARNESSES = [
    (24, "m4.7144 15.9555 4.7174-2.6471.079-.2307-.079-.1275h-.2307l-.7893-.0486-2.6956-.0729-2.3375-.0971-2.2646-.1214-.5707-.1215-.5343-.7042.0546-.3522.4797-.3218.686.0608 1.5179.1032 2.2767.1578 1.6514.0972 2.4468.255h.3886l.0546-.1579-.1336-.0971-.1032-.0972L6.973 9.8356l-2.55-1.6879-1.3356-.9714-.7225-.4918-.3643-.4614-.1578-1.0078.6557-.7225.8803.0607.2246.0607.8925.686 1.9064 1.4754 2.4893 1.8336.3643.3035.1457-.1032.0182-.0728-.164-.2733-1.3539-2.4467-1.445-2.4893-.6435-1.032-.17-.6194c-.0607-.255-.1032-.4674-.1032-.7285L6.287.1335 6.6997 0l.9957.1336.419.3642.6192 1.4147 1.0018 2.2282 1.5543 3.0296.4553.8985.2429.8318.091.255h.1579v-.1457l.1275-1.706.2368-2.0947.2307-2.6957.0789-.7589.3764-.9107.7468-.4918.5828.2793.4797.686-.0668.4433-.2853 1.8517-.5586 2.9021-.3643 1.9429h.2125l.2429-.2429.9835-1.3053 1.6514-2.0643.7286-.8196.85-.9046.5464-.4311h1.0321l.759 1.1293-.34 1.1657-1.0625 1.3478-.8804 1.1414-1.2628 1.7-.7893 1.36.0729.1093.1882-.0183 2.8535-.607 1.5421-.2794 1.8396-.3157.8318.3886.091.3946-.3278.8075-1.967.4857-2.3072.4614-3.4364.8136-.0425.0304.0486.0607 1.5482.1457.6618.0364h1.621l3.0175.2247.7892.522.4736.6376-.079.4857-1.2142.6193-1.6393-.3886-3.825-.9107-1.3113-.3279h-.1822v.1093l1.0929 1.0686 2.0035 1.8092 2.5075 2.3314.1275.5768-.3218.4554-.34-.0486-2.2039-1.6575-.85-.7468-1.9246-1.621h-.1275v.17l.4432.6496 2.3436 3.5214.1214 1.0807-.17.3521-.6071.2125-.6679-.1214-1.3721-1.9246L14.38 17.959l-1.1414-1.9428-.1397.079-.674 7.2552-.3156.3703-.7286.2793-.6071-.4614-.3218-.7468.3218-1.4753.3886-1.9246.3157-1.53.2853-1.9004.17-.6314-.0121-.0425-.1397.0182-1.4328 1.9672-2.1796 2.9446-1.7243 1.8456-.4128.164-.7164-.3704.0667-.6618.4008-.5889 2.386-3.0357 1.4389-1.882.929-1.0868-.0062-.1579h-.0546l-6.3385 4.1164-1.1293.1457-.4857-.4554.0608-.7467.2307-.2429 1.9064-1.3114Z",
     "Claude Code", "#D97757"),
    (320, "m297.06 130.97c7.26-21.79 4.76-45.66-6.85-65.48-17.46-30.4-52.56-46.04-86.84-38.68-15.25-17.18-37.16-26.95-60.13-26.81-35.04-.08-66.13 22.48-76.91 55.82-22.51 4.61-41.94 18.7-53.31 38.67-17.59 30.32-13.58 68.54 9.92 94.54-7.26 21.79-4.76 45.66 6.85 65.48 17.46 30.4 52.56 46.04 86.84 38.68 15.24 17.18 37.16 26.95 60.13 26.8 35.06.09 66.16-22.49 76.94-55.86 22.51-4.61 41.94-18.7 53.31-38.67 17.57-30.32 13.55-68.51-9.94-94.51zm-120.28 168.11c-14.03.02-27.62-4.89-38.39-13.88.49-.26 1.34-.73 1.89-1.07l63.72-36.8c3.26-1.85 5.26-5.32 5.24-9.07v-89.83l26.93 15.55c.29.14.48.42.52.74v74.39c-.04 33.08-26.83 59.9-59.91 59.97zm-128.84-55.03c-7.03-12.14-9.56-26.37-7.15-40.18.47.28 1.3.79 1.89 1.13l63.72 36.8c3.23 1.89 7.23 1.89 10.47 0l77.79-44.92v31.1c.02.32-.13.63-.38.83l-64.41 37.19c-28.69 16.52-65.33 6.7-81.92-21.95zm-16.77-139.09c7-12.16 18.05-21.46 31.21-26.29 0 .55-.03 1.52-.03 2.2v73.61c-.02 3.74 1.98 7.21 5.23 9.06l77.79 44.91-26.93 15.55c-.27.18-.61.21-.91.08l-64.42-37.22c-28.63-16.58-38.45-53.21-21.95-81.89zm221.26 51.49-77.79-44.92 26.93-15.54c.27-.18.61-.21.91-.08l64.42 37.19c28.68 16.57 38.51 53.26 21.94 81.94-7.01 12.14-18.05 21.44-31.2 26.28v-75.81c.03-3.74-1.96-7.2-5.2-9.06zm26.8-40.34c-.47-.29-1.3-.79-1.89-1.13l-63.72-36.8c-3.23-1.89-7.23-1.89-10.47 0l-77.79 44.92v-31.1c-.02-.32.13-.63.38-.83l64.41-37.16c28.69-16.55 65.37-6.7 81.91 22 6.99 12.12 9.52 26.31 7.15 40.1zm-168.51 55.43-26.94-15.55c-.29-.14-.48-.42-.52-.74v-74.39c.02-33.12 26.89-59.96 60.01-59.94 14.01 0 27.57 4.92 38.34 13.88-.49.26-1.33.73-1.89 1.07l-63.72 36.8c-3.26 1.85-5.26 5.31-5.24 9.06l-.04 89.79zm14.63-31.54 34.65-20.01 34.65 20v40.01l-34.65 20-34.65-20z",
     "Codex", "#111111"),
    (24, "M11.04 19.32Q12 21.51 12 24q0-2.49.93-4.68.96-2.19 2.58-3.81t3.81-2.55Q21.51 12 24 12q-2.49 0-4.68-.93a12.3 12.3 0 0 1-3.81-2.58 12.3 12.3 0 0 1-2.58-3.81Q12 2.49 12 0q0 2.49-.96 4.68-.93 2.19-2.55 3.81a12.3 12.3 0 0 1-3.81 2.58Q2.49 12 0 12q2.49 0 4.68.96 2.19.93 3.81 2.55t2.55 3.81",
     "Antigravity", "#4285F4"),
]

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

    def glyph(self, d, box, x, y, size, fill):
        """One icon path, scaled from its own viewBox into a `size` square whose top-left is (x, y)."""
        k = size / box
        self.parts.append(f'<path d="{d}" fill="{fill}" transform="translate({x:.1f} {y:.1f}) scale({k:.4f})"/>')

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
    s.text(60, 78, "residudes", size=52, weight="bold", anchor="start")
    s.text(60, 120, "analyses a target, plans with a person, runs the tools, reviews with a person, proposes a slate",
           size=24, fill=MUTED, anchor="start")
    s.text(60, 152, "one instruction set — the same AGENTS.md and skills — read by every harness:",
           size=21, fill=MUTED, anchor="start")
    for i, (box, d, label, colour) in enumerate(HARNESSES):
        ly = 182 + i * 38
        s.glyph(d, box, 62, ly, 28, colour)
        s.text(100, ly + 21, label, size=22, fill=INK, anchor="start")
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
    band(X["HYPOTHESES"] - 70, X["RUN"] + 70, "residudes-engine", "MSA · inputs · predictions", BLUE, ["HYPOTHESES", "RUN"])
    band(X["RUN"] + 92, X["RESULTS REVIEW"] + 72, "residudes-qa", "scores · clusters · checks", GREEN,
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
