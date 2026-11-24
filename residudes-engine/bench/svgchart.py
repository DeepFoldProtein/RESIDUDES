"""Minimal, dependency-free SVG charts for the benchmark page.

Follows the data-viz method: thin 2px lines, >=8px markers with a 2px surface
ring, hairline solid gridlines, a legend for >=2 series, selective direct
end-labels, text in ink tokens (never the series colour), native <title>
tooltips on every mark. Light-mode palette (reference instance):
series 1-3 = blue / orange / aqua, which validate all-pairs for CVD.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
FONT = "system-ui, -apple-system, 'Segoe UI', sans-serif"


def _esc(s: str) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;"))


def _nice_ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    if hi <= lo:
        hi = lo + 1
    raw = (hi - lo) / max(n - 1, 1)
    mag = 10 ** math.floor(math.log10(raw))
    for m in (1, 2, 2.5, 5, 10):
        step = m * mag
        if (hi - lo) / step <= n:
            break
    start = math.floor(lo / step) * step
    ticks = []
    v = start
    while v <= hi + 1e-9:
        ticks.append(round(v, 10))
        v += step
    return ticks


def _fmt(v: float) -> str:
    if v == 0:
        return "0"
    if abs(v) >= 1000:
        return f"{v/1000:g}K" if v % 1000 == 0 else f"{v:,.0f}"
    if abs(v) >= 10:
        return f"{v:,.0f}"
    return f"{v:.1f}".rstrip("0").rstrip(".")


def small_multiples(
    panels: Sequence[dict],
    *,
    series: Sequence[str],
    title: str,
    subtitle: str = "",
    x_label: str = "",
    y_label: str = "",
    log_y: bool = False,
    y_min: float | None = None,
    reference_y: float | None = None,
    out_path: Path,
    panel_w: int = 300,
    panel_h: int = 240,
    note: str = "",
    colors: dict[str, str] | None = None,
) -> Path:
    """panels: [{"title": str, "points": {series: [(x, y, tooltip), ...]}, "failed": {series: [(x, tooltip), ...]}}]

    `colors` fixes hue per series name across figures (colour follows the entity);
    unspecified series take the next categorical slot."""
    n = len(panels)
    left, right, top, bottom = 56, 24, 76, 64
    gap = 44
    W = left + n * panel_w + (n - 1) * gap + right
    H = top + panel_h + bottom + (18 if note else 0)
    colors = dict(colors or {})
    for i, s in enumerate(series):
        colors.setdefault(s, SERIES[i % len(SERIES)])

    # shared scales across panels
    xs = [p[0] for pan in panels for pts in pan["points"].values() for p in pts]
    xs += [f[0] for pan in panels for fl in pan.get("failed", {}).values() for f in fl]
    ys = [p[1] for pan in panels for pts in pan["points"].values() for p in pts if p[1] is not None and p[1] > 0]
    if not xs:
        xs = [0, 1]
    if not ys:
        ys = [1, 10]
    x_lo, x_hi = 0, max(xs) * 1.05
    if log_y:
        y_lo = 10 ** math.floor(math.log10(min(ys))) if y_min is None else y_min
        y_hi = 10 ** math.ceil(math.log10(max(ys) * 1.05))
        if y_hi <= y_lo:
            y_hi = y_lo * 10
    else:
        y_lo = 0 if y_min is None else y_min
        y_hi = max(ys) * 1.12
        if reference_y is not None:
            y_hi = max(y_hi, reference_y * 1.3)
    x_ticks = _nice_ticks(x_lo, x_hi, 5)
    if log_y:
        y_ticks = [10 ** k for k in range(int(math.log10(y_lo)), int(math.log10(y_hi)) + 1)]
    else:
        y_ticks = _nice_ticks(y_lo, y_hi, 5)
        y_hi = max(y_hi, y_ticks[-1])

    def sx(x: float, px0: int) -> float:
        return px0 + (x - x_lo) / (x_hi - x_lo) * panel_w

    def sy(y: float) -> float:
        if log_y:
            frac = (math.log10(y) - math.log10(y_lo)) / (math.log10(y_hi) - math.log10(y_lo))
        else:
            frac = (y - y_lo) / (y_hi - y_lo)
        return top + panel_h - frac * panel_h

    o: list[str] = []
    o.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
             f'font-family="{FONT}" role="img" aria-label="{_esc(title)}">')
    o.append(f'<rect width="{W}" height="{H}" fill="{SURFACE}"/>')
    o.append(f'<text x="{left}" y="26" font-size="15" font-weight="600" fill="{INK}">{_esc(title)}</text>')
    if subtitle:
        o.append(f'<text x="{left}" y="45" font-size="12" fill="{INK2}">{_esc(subtitle)}</text>')
    # legend (top right)
    lx = W - right
    for s in reversed(series):
        label_w = 7 * len(s) + 26
        lx -= label_w
        c = colors[s]
        o.append(f'<line x1="{lx}" y1="26" x2="{lx+14}" y2="26" stroke="{c}" stroke-width="2" stroke-linecap="round"/>')
        o.append(f'<circle cx="{lx+7}" cy="26" r="4" fill="{c}" stroke="{SURFACE}" stroke-width="2"/>')
        o.append(f'<text x="{lx+20}" y="30" font-size="12" fill="{INK2}">{_esc(s)}</text>')
        lx -= 10

    for i, pan in enumerate(panels):
        px0 = left + i * (panel_w + gap)
        o.append(f'<text x="{px0}" y="{top-10}" font-size="13" font-weight="600" fill="{INK}">{_esc(pan["title"])}</text>')
        # grid + y ticks
        for t in y_ticks:
            y = sy(t)
            o.append(f'<line x1="{px0}" y1="{y:.1f}" x2="{px0+panel_w}" y2="{y:.1f}" stroke="{GRID}" stroke-width="1"/>')
            if i == 0:
                o.append(f'<text x="{px0-8}" y="{y+4:.1f}" font-size="11" text-anchor="end" fill="{MUTED}" '
                         f'style="font-variant-numeric: tabular-nums">{_esc(_fmt(t))}</text>')
        if reference_y is not None and y_lo <= reference_y <= y_hi:
            y = sy(reference_y)
            o.append(f'<line x1="{px0}" y1="{y:.1f}" x2="{px0+panel_w}" y2="{y:.1f}" stroke="{AXIS}" stroke-width="1"/>')
        # baseline + x ticks
        o.append(f'<line x1="{px0}" y1="{top+panel_h}" x2="{px0+panel_w}" y2="{top+panel_h}" stroke="{AXIS}" stroke-width="1"/>')
        for t in x_ticks:
            if t > x_hi:
                continue
            x = sx(t, px0)
            o.append(f'<text x="{x:.1f}" y="{top+panel_h+16}" font-size="11" text-anchor="middle" fill="{MUTED}" '
                     f'style="font-variant-numeric: tabular-nums">{_esc(_fmt(t))}</text>')
        # failed marks
        for s, fl in (pan.get("failed") or {}).items():
            for x, tip in fl:
                cx = sx(x, px0)
                cy = top + 8
                o.append(f'<g><title>{_esc(tip)}</title>'
                         f'<line x1="{cx-4:.1f}" y1="{cy-4}" x2="{cx+4:.1f}" y2="{cy+4}" stroke="{MUTED}" stroke-width="2"/>'
                         f'<line x1="{cx-4:.1f}" y1="{cy+4}" x2="{cx+4:.1f}" y2="{cy-4}" stroke="{MUTED}" stroke-width="2"/></g>')
        # lines + markers
        ends: list[tuple[float, str, str]] = []
        for s in series:
            pts = sorted((p for p in pan["points"].get(s, []) if p[1] is not None and p[1] > 0), key=lambda p: p[0])
            if not pts:
                continue
            c = colors[s]
            path = " ".join(f"{'M' if k == 0 else 'L'}{sx(x, px0):.1f},{sy(y):.1f}" for k, (x, y, _) in enumerate(pts))
            o.append(f'<path d="{path}" fill="none" stroke="{c}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>')
            for x, y, tip in pts:
                o.append(f'<circle cx="{sx(x, px0):.1f}" cy="{sy(y):.1f}" r="4" fill="{c}" stroke="{SURFACE}" stroke-width="2">'
                         f'<title>{_esc(tip)}</title></circle>')
            ends.append((sy(pts[-1][1]), sx(pts[-1][0], px0), s, _fmt(pts[-1][1])))
        # selective direct labels beside the last point: only when they don't collide,
        # and always inside the panel (anchor flips when there is no room on the right)
        ends.sort()
        if all(b[0] - a[0] >= 13 for a, b in zip(ends, ends[1:])):
            for y, x, s, val in ends:
                est_w = 7 * len(val) + 8
                if x + est_w <= px0 + panel_w:
                    o.append(f'<text x="{x+7:.1f}" y="{y+4:.1f}" font-size="11" fill="{INK2}" '
                             f'style="font-variant-numeric: tabular-nums">{_esc(val)}</text>')
                else:  # no room on the right: sit above the point, clear of the line
                    o.append(f'<text x="{x:.1f}" y="{y-9:.1f}" font-size="11" text-anchor="middle" fill="{INK2}" '
                             f'style="font-variant-numeric: tabular-nums">{_esc(val)}</text>')
    if x_label:
        o.append(f'<text x="{left + (W-left-right)/2:.1f}" y="{top+panel_h+40}" font-size="12" text-anchor="middle" fill="{INK2}">{_esc(x_label)}</text>')
    if y_label:
        o.append(f'<text x="14" y="{top+panel_h/2:.1f}" font-size="12" text-anchor="middle" fill="{INK2}" '
                 f'transform="rotate(-90 14 {top+panel_h/2:.1f})">{_esc(y_label)}</text>')
    if note:
        o.append(f'<text x="{left}" y="{H-10}" font-size="11" fill="{MUTED}">{_esc(note)}</text>')
    o.append("</svg>")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(o) + "\n")
    return out_path
