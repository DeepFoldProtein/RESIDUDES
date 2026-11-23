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
        s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        self.parts.append(f'<text x="{x:.1f}" y="{y:.1f}" font-family="{font}" font-size="{size}" font-weight="{weight}" '
                          f'fill="{fill}" text-anchor="{anchor}" xml:space="preserve">{s}</text>')

    def line(self, x1, y1, x2, y2, *, stroke=TRACK, sw=12, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" stroke-width="{sw}" stroke-linecap="round"{d}/>')

    def path(self, d, *, stroke=TRACK, sw=7, dash=None):
        dd = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(f'<path d="{d}" fill="none" stroke="{stroke}" stroke-width="{sw}" stroke-linecap="round"{dd}/>')

    def glyph(self, d, box, x, y, size, fill):
        """One icon path, scaled from its own viewBox into a `size` square whose top-left is (x, y)."""
        self.parts.append(f'<path d="{d}" fill="{fill}" transform="translate({x:.1f} {y:.1f}) scale({size / box:.4f})"/>')

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
    s.text(250, 184, "the campaign agent: one instruction set, read by every harness",
           size=22, fill=MUTED, anchor="start")
    hx = 250                                              # the harnesses that read that instruction set
    for box, d, label, colour in HARNESSES:
        s.glyph(d, box, hx, 198, 28, colour)
        s.text(hx + 36, 220, label, size=22, fill=INK, anchor="start")
        hx += 36 + int(len(label) * 11.6) + 46
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
