"""Find text that collides, spills out of its box or runs off the page in the SVG figures.

Every <text> is measured with the font the figure asks for — Helvetica metrics from Nimbus Sans,
DejaVu Sans Mono for the monospace runs — because the figures are written by hand with no layout
engine to catch a label that grew. Run it over the tracked SVGs after regenerating any figure:

    python3 docs/svg_text_overlap.py $(git ls-files '*.svg')

Needs Pillow and the two font families; it says which files it measured with, since a different
font means different metrics and a different answer.
"""
import re
import sys
import xml.etree.ElementTree as ET

from PIL import ImageFont

NS = "{http://www.w3.org/2000/svg}"
# first that exists wins: distributions disagree about where these live
FONTS = {
    ("sans", "normal"): ("/usr/share/fonts/urw-base35/NimbusSans-Regular.otf",
                         "/usr/share/fonts/opentype/urw-base35/NimbusSans-Regular.otf",
                         "/usr/share/fonts/X11/Type1/n019003l.pfb"),
    ("sans", "bold"): ("/usr/share/fonts/urw-base35/NimbusSans-Bold.otf",
                       "/usr/share/fonts/opentype/urw-base35/NimbusSans-Bold.otf",
                       "/usr/share/fonts/X11/Type1/n019004l.pfb"),
    ("mono", "normal"): ("/usr/share/fonts/dejavu-sans-mono-fonts/DejaVuSansMono.ttf",
                         "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
                         "/usr/share/fonts/dejavu/DejaVuSansMono.ttf"),
    ("mono", "bold"): ("/usr/share/fonts/dejavu-sans-mono-fonts/DejaVuSansMono-Bold.ttf",
                       "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
                       "/usr/share/fonts/dejavu/DejaVuSansMono-Bold.ttf"),
}
_cache = {}
_used = {}


def font_file(kind, weight):
    import os
    for p in FONTS[(kind, weight)]:
        if os.path.exists(p):
            return p
    raise SystemExit(f"no {kind}/{weight} font found; install urw-base35 and dejavu "
                     f"(tried: {', '.join(FONTS[(kind, weight)])})")


def font(kind, weight, size):
    k = (kind, weight, round(size * 4))
    if k not in _cache:
        path = font_file(kind, weight)
        _used[(kind, weight)] = path
        _cache[k] = ImageFont.truetype(path, size=max(1, round(size * 4)))   # 4x for precision
    return _cache[k]


def walk(el, tx=0.0, ty=0.0, rot=False, out=None):
    t = el.get("transform", "")
    m = re.search(r"translate\(\s*([-\d.]+)[ ,]+([-\d.]+)", t)
    if m:
        tx, ty = tx + float(m[1]), ty + float(m[2])
    rot = rot or "rotate" in t or "matrix" in t
    if el.tag == NS + "text":
        out.append((el, tx, ty, rot))
        return
    if el.tag == NS + "rect" and not rot and out is not None:
        try:
            x, y, w, h = (float(el.get(k, "0")) for k in ("x", "y", "width", "height"))
            out.append(("rect", x + tx, y + ty, x + tx + w, y + ty + h))
        except ValueError:
            pass
    for c in el:
        walk(c, tx, ty, rot, out)


def boxes(path):
    root = ET.parse(path).getroot()
    items = []
    walk(root, out=items)
    res = []
    rects = [r[1:] for r in items if r[0] == "rect"]
    items = [r for r in items if r[0] != "rect"]
    for el, tx, ty, rot in items:
        if rot:
            continue
        text = "".join(el.itertext())
        if not text.strip():
            continue
        size = float(el.get("font-size", "16"))
        weight = "bold" if el.get("font-weight") in ("bold", "700", "600") else "normal"
        kind = "mono" if "mono" in el.get("font-family", "").lower() or "menlo" in el.get("font-family", "").lower() else "sans"
        w = font(kind, weight, size).getlength(text) / 4
        w += float(el.get("letter-spacing", "0") or 0) * max(0, len(text) - 1)
        x, y = float(el.get("x", "0")) + tx, float(el.get("y", "0")) + ty
        anchor = el.get("text-anchor", "start")
        x0 = x - (w if anchor == "end" else w / 2 if anchor == "middle" else 0)
        res.append((x0, y - 0.72 * size, x0 + w, y + 0.2 * size, text, x, y - 0.3 * size))
    return res, float(root.get("width", "0") or 0), rects


def main(paths):
    bad = 0
    for p in paths:
        bs, width, rects = boxes(p)
        for i in range(len(bs)):
            a = bs[i]
            if width and a[2] > width + 1:
                print(f"{p}: past the right edge ({a[2]:.0f} > {width:.0f}): {a[4][:60]!r}")
                bad += 1
            # the smallest box drawn around the text's anchor (skip the page background and thin rules)
            inside = [r for r in rects if r[0] <= a[5] <= r[2] and r[1] <= a[6] <= r[3] and (r[3] - r[1]) > 10
                      and not (r[0] <= 0 and r[2] >= width - 1)]
            if inside:
                r = min(inside, key=lambda r: (r[2] - r[0]) * (r[3] - r[1]))
                if a[0] < r[0] - 1 or a[2] > r[2] + 1:
                    print(f"{p}: spills out of its box by {max(r[0] - a[0], a[2] - r[2]):.0f}px: {a[4][:60]!r}")
                    bad += 1
            for b in bs[i + 1:]:
                ox = min(a[2], b[2]) - max(a[0], b[0])
                oy = min(a[3], b[3]) - max(a[1], b[1])
                if ox > 1 and oy > 1:
                    print(f"{p}: overlap {ox:.0f}x{oy:.0f}px: {a[4][:50]!r} / {b[4][:50]!r}")
                    bad += 1
    for (kind, weight), path in sorted(_used.items()):
        print(f"  {kind}/{weight}: {path}")
    print(f"{len(paths)} files, {bad} problem(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
