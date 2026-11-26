"""A tmux window as a PNG, drawn the way a terminal draws it: each pane's cells and colours, the borders, the status
line. tmux cannot take a screenshot, and a terminal's own one is not reproducible; this is how
docs/figures/chat-molterm.png was made, from a `residudes chat` window with the structure pane open.

usage: python3 docs/tmux_shot.py SOCKET_NAME TARGET_WINDOW OUT.png [font_px] [--images]   (needs pillow, fonttools)
       SOCKET_NAME is tmux's -L name ("default" for the usual server); TARGET_WINDOW e.g. residudes:H1388
       --images: also draw the sixel pictures in the panes (tmux 3.4+ built with sixel). capture-pane has only
       text, so a read-only client is attached for a moment at this drawing's cell size in pixels; tmux replays
       its stored pictures to it, scaled to that size, and they are put where tmux placed them.
"""
import colorsys
import fcntl
import os
import pty
import re
import select
import signal
import struct
import subprocess
import sys
import termios
import time
import unicodedata

from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFont

IMAGES = "--images" in sys.argv
ARGS = [a for a in sys.argv[1:] if a != "--images"]
SOCK, WIN, OUT = ARGS[:3]
PX = int(ARGS[3]) if len(ARGS) > 3 else 16


def font_file(pattern: str) -> str | None:
    r = subprocess.run(["fc-match", "-f", "%{file}", pattern], capture_output=True, text=True)
    return r.stdout.strip() or None if r.returncode == 0 else None


MONO = font_file("DejaVu Sans Mono") or "/usr/share/fonts/dejavu-sans-mono-fonts/DejaVuSansMono.ttf"
MONO_B = font_file("DejaVu Sans Mono:bold") or MONO
# glyphs the mono face lacks (Claude Code's symbols, Hangul): the first of these that has them
FALLBACK = [f for f in dict.fromkeys([font_file("DejaVu Sans"), font_file(":lang=ko")]) if f and f != MONO]
BG, FG = (30, 30, 30), (204, 204, 204)                     # VS Code's dark terminal
ANSI = ["#000000", "#cd3131", "#0dbc79", "#e5e510", "#2472c8", "#bc3fbc", "#11a8cd", "#e5e5e5",
        "#666666", "#f14c4c", "#23d18b", "#f5f543", "#3b8eea", "#d670d6", "#29b8db", "#e5e5e5"]
ANSI = [tuple(int(h[i:i + 2], 16) for i in (1, 3, 5)) for h in ANSI]


def tmux(*a):
    return subprocess.run(["tmux", "-L", SOCK, *a], capture_output=True, text=True, check=True).stdout


def c256(n):
    if n < 16:
        return ANSI[n]
    if n < 232:
        n -= 16
        lv = [0, 95, 135, 175, 215, 255]
        return (lv[n // 36], lv[n // 6 % 6], lv[n % 6])
    g = 8 + 10 * (n - 232)
    return (g, g, g)


fonts = {}
def font(path, px=PX):
    if (path, px) not in fonts:
        fonts[path, px] = ImageFont.truetype(path, px)
    return fonts[path, px]


cmaps = {p: set(TTFont(p, fontNumber=0).getBestCmap()) for p in [MONO, *FALLBACK]}
SUBST = {"\u23f5": "\u25b8"}                               # glyphs no installed font has
def font_for(ch, bold):
    o = ord(ch)
    if o in cmaps[MONO]:
        return font(MONO_B if bold else MONO)
    for p in FALLBACK:
        if o in cmaps[p]:
            return font(p, round(PX * 1.2) if ord(ch) >= 0x1100 else PX)   # a wide glyph fills its two cells
    return font(MONO)


f0 = font(MONO)
CW = round(f0.getlength("M"))
ASC, DESC = f0.getmetrics()
CH = ASC + DESC + 1

ww, wh = map(int, tmux("display", "-p", "-t", WIN, "#{window_width} #{window_height}").split())
status = tmux("show", "-gv", "status").strip() != "off"
img = Image.new("RGB", (ww * CW, (wh + status) * CH), BG)
dr = ImageDraw.Draw(img)

SGR = re.compile(r"\x1b\[([0-9;:]*)m|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)|\x1b[^\[\]]")


RESET = dict(fg=None, bg=None, bold=False, dim=False, ul=False, rev=False, it=False)
FLAGS = {1: ("bold", True), 2: ("dim", True), 3: ("it", True), 4: ("ul", True), 7: ("rev", True),
         23: ("it", False), 24: ("ul", False), 27: ("rev", False), 39: ("fg", None), 49: ("bg", None)}


def apply(st, params):
    """One SGR sequence's parameters onto the current cell state."""
    ps = [int(x) if x else 0 for x in re.split("[;:]", params)] if params else [0]
    i = 0
    while i < len(ps):
        p = ps[i]
        if p == 0:
            st.update(RESET)
        elif p == 22:
            st["bold"] = st["dim"] = False
        elif p in FLAGS:
            st[FLAGS[p][0]] = FLAGS[p][1]
        elif 30 <= p <= 37 or 90 <= p <= 97:
            st["fg"] = ANSI[p - 30 if p < 90 else p - 82]
        elif 40 <= p <= 47 or 100 <= p <= 107:
            st["bg"] = ANSI[p - 40 if p < 100 else p - 92]
        elif p in (38, 48) and i + 1 < len(ps):
            key = "fg" if p == 38 else "bg"
            if ps[i + 1] == 5 and i + 2 < len(ps):
                st[key] = c256(ps[i + 2])
                i += 2
            elif ps[i + 1] == 2:
                skip = 1 if len(ps) > i + 5 and params.count(":") >= 5 else 0   # 38:2::r:g:b has a colour-space slot
                st[key] = tuple((ps[i + 2 + skip:i + 5 + skip] + [0, 0, 0])[:3])
                i += 4 + skip
        i += 1


def blend(a, b, t):
    return tuple(round(x * (1 - t) + y * t) for x, y in zip(a, b))


# block elements as (x0, y0, x1, y1) fractions of the cell
BLOCKS = {0x2580: [(0, 0, 1, .5)], 0x2584: [(0, .5, 1, 1)], 0x2588: [(0, 0, 1, 1)], 0x258C: [(0, 0, .5, 1)],
          0x2590: [(.5, 0, 1, 1)], 0x2594: [(0, 0, 1, .125)], 0x2595: [(.875, 0, 1, 1)],
          0x2596: [(0, .5, .5, 1)], 0x2597: [(.5, .5, 1, 1)], 0x2598: [(0, 0, .5, .5)], 0x259D: [(.5, 0, 1, .5)],
          0x2599: [(0, 0, .5, 1), (.5, .5, 1, 1)], 0x259A: [(0, 0, .5, .5), (.5, .5, 1, 1)],
          0x259B: [(0, 0, 1, .5), (0, .5, .5, 1)], 0x259C: [(0, 0, 1, .5), (.5, .5, 1, 1)],
          0x259E: [(.5, 0, 1, .5), (0, .5, .5, 1)], 0x259F: [(.5, 0, 1, .5), (0, .5, 1, 1)]}
for k in range(1, 8):
    BLOCKS[0x2580 + k] = [(0, 1 - k / 8, 1, 1)]           # lower eighths
    BLOCKS[0x2590 - k] = [(0, 0, k / 8, 1)]               # left eighths
BOX = {0x2500: "lr", 0x2501: "LR", 0x2502: "ud", 0x2503: "UD", 0x250C: "rd", 0x2510: "ld", 0x2514: "ru",
       0x2518: "lu", 0x251C: "udr", 0x2524: "udl", 0x252C: "lrd", 0x2534: "lru", 0x253C: "lrud",
       0x256D: "Rd", 0x256E: "Ld", 0x2570: "Ru", 0x256F: "Lu"}   # capital first letter on 256D-2570: a rounded corner


def cell(x, y, ch, st, w=1):
    fg, bg = st["fg"] or FG, st["bg"] or BG
    if st["rev"]:
        fg, bg = bg, fg
    if st["dim"]:
        fg = blend(fg, bg, .45)
    X, Y = x * CW, y * CH
    if bg != BG:
        dr.rectangle([X, Y, X + w * CW - 1, Y + CH - 1], fill=bg)
    o = ord(ch)
    if ch == " ":
        pass
    elif o in BLOCKS:
        for a, b, c, d in BLOCKS[o]:
            dr.rectangle([X + round(a * CW), Y + round(b * CH), X + round(c * CW) - 1, Y + round(d * CH) - 1], fill=fg)
    elif 0x2591 <= o <= 0x2593:
        dr.rectangle([X, Y, X + CW - 1, Y + CH - 1], fill=blend(bg, fg, (o - 0x2590) / 4))
    elif 0x2800 <= o <= 0x28FF:
        bits = o - 0x2800
        pos = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (0, 3), (1, 3)]
        r = max(1, CW // 5)
        for k, (cx, cy) in enumerate(pos):
            if bits >> k & 1:
                px, py = X + CW * (1 + 2 * cx) // 4, Y + CH * (1 + 2 * cy) // 8
                dr.ellipse([px - r, py - r, px + r, py + r], fill=fg)
    elif o in BOX:
        segs = BOX[o]
        lw = 2 if any(s.isupper() for s in segs) and o < 0x256D else 1
        mx, my = X + CW // 2, Y + CH // 2
        if o >= 0x256D:                                    # rounded corner: a quarter arc
            h, v = segs[0].lower(), segs[1]
            r = min(CW, CH) // 2
            cx = mx + (r if h == "r" else -r)
            cy = my + (r if v == "d" else -r)
            start = {("r", "d"): 180, ("l", "d"): 270, ("r", "u"): 90, ("l", "u"): 0}[h, v]
            dr.arc([cx - r, cy - r, cx + r, cy + r], start, start + 90, fill=fg)
            dr.line([cx, my, X + CW, my] if h == "r" else [X, my, cx, my], fill=fg)
            dr.line([mx, cy, mx, Y + CH] if v == "d" else [mx, Y, mx, cy], fill=fg)
        else:
            for s in segs.lower():
                end = {"l": (X, my), "r": (X + CW, my), "u": (mx, Y), "d": (mx, Y + CH)}[s]
                dr.line([(mx, my), end], fill=fg, width=lw)
    else:
        ch = SUBST.get(ch, ch)
        f = font_for(ch, st["bold"])
        dr.text((X + (w * CW - f.getlength(ch)) / 2 if w == 2 else X, Y + ASC), ch, font=f, fill=fg, anchor="ls")
    if st["ul"]:
        dr.line([X, Y + ASC + 2, X + w * CW - 1, Y + ASC + 2], fill=fg)


def unstyled(fmt: str) -> str:
    """A status-line format's text without its #[style] directives."""
    return re.sub(r"#\[[^\]]*\]", "", fmt)


def width(ch):
    if unicodedata.combining(ch) or unicodedata.category(ch) in ("Mn", "Me", "Cf"):
        return 0
    return 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1


def draw_lines(lines, left, top, maxw):
    st = dict(RESET)
    for row, line in enumerate(lines):
        col, pos = 0, 0
        for m in SGR.finditer(line + "\x1b[m"):
            for ch in line[pos:m.start()]:
                w = width(ch)
                if w and col + w <= maxw:
                    cell(left + col, top + row, ch, st, w)
                col += w
            if m.group(1) is not None and m.end() <= len(line):
                apply(st, m.group(1))
            pos = m.end()


panes = [ln.split() for ln in tmux("list-panes", "-t", WIN, "-F",
         "#{pane_id} #{pane_left} #{pane_top} #{pane_width} #{pane_height} #{pane_active}").splitlines()]
for pid, *geom, act in panes:
    left, top, w, h = map(int, geom)
    text = tmux("capture-pane", "-e", "-p", "-N", "-t", pid)
    draw_lines(text.split("\n")[:h], left, top, w)
    edge = (13, 188, 121) if act == "1" else (110, 110, 110)
    if left > 0:                                        # the border column on the pane's left
        X = (left - 1) * CW + CW // 2
        dr.line([X, top * CH, X, (top + h) * CH], fill=edge)
    if top > 0:
        Y = (top - 1) * CH + CH // 2
        dr.line([left * CW, Y, (left + w) * CW, Y], fill=edge)

if status:                                              # tmux's own default bar: black on green
    left = tmux("display", "-p", "-t", WIN, "#{T:status-left}").strip("\n")
    wins = tmux("display", "-p", "-t", WIN,
                "#{W:#{T:window-status-format} ,#{T:window-status-current-format} }").strip("\n")
    right = tmux("display", "-p", "-t", WIN, "#{T:status-right}").strip("\n")
    y = wh
    dr.rectangle([0, y * CH, ww * CW, (y + 1) * CH], fill=(13, 188, 121))
    st = dict(fg=(0, 0, 0), bg=(13, 188, 121), bold=False, dim=False, ul=False, rev=False, it=False)
    bar = unstyled(left) + unstyled(wins)
    for i, ch in enumerate(bar[:ww]):
        cell(i, y, ch, st)
    r = unstyled(right)
    for i, ch in enumerate(r):
        if ww - len(r) + i >= len(bar):
            cell(ww - len(r) + i, y, ch, st)

def replayed(cols: int, rows: int) -> bytes:
    """What tmux sends a read-only client of this drawing's size, in characters and pixels, as it attaches."""
    pid, fd = pty.fork()
    if pid == 0:
        fcntl.ioctl(0, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, cols * CW, rows * CH))
        os.environ["TERM"] = "xterm-256color"
        for k in ("TMUX", "TMUX_PANE"):
            os.environ.pop(k, None)
        os.execvp("tmux", ["tmux", "-L", SOCK, "attach", "-f", "read-only,ignore-size", "-t", WIN])
    raw, end = b"", time.time() + 3
    while time.time() < end:
        if select.select([fd], [], [], 0.2)[0]:
            try:
                raw += os.read(fd, 1 << 16)
            except OSError:
                break
    os.kill(pid, signal.SIGTERM)
    os.waitpid(pid, 0)
    return raw


def sixel(data: str) -> Image.Image:
    """One DECSIXEL body (after `q`) as an RGBA picture; pixels it does not set stay transparent."""
    ra = re.match(r'"(\d+);(\d+);(\d+);(\d+)', data)
    w, h = int(ra.group(3)), int(ra.group(4))
    data = data[ra.end():]
    pic = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    px = pic.load()
    pal, cur, x, y, i = {}, (255, 255, 255, 255), 0, 0, 0
    while i < len(data):
        c = data[i]
        if c == "#":
            m = re.match(r"#(\d+)(?:;(\d+);(\d+);(\d+);(\d+))?", data[i:])
            i += m.end()
            n = int(m.group(1))
            if m.group(2):
                u, a, b, d = map(int, m.group(2, 3, 4, 5))
                rgb = [v * 2.55 for v in (a, b, d)] if u == 2 else \
                    [v * 255 for v in colorsys.hls_to_rgb(((a - 120) % 360) / 360, b / 100, d / 100)]
                pal[n] = (*(round(v) for v in rgb), 255)
            cur = pal.get(n, cur)
            continue
        if c == "!":
            m = re.match(r"!(\d+)(.)", data[i:])
            i += m.end()
            rep, ch = int(m.group(1)), m.group(2)
        elif c in "$-":
            x, y, i = 0, y + 6 * (c == "-"), i + 1
            continue
        elif "?" <= c <= "~":
            rep, ch, i = 1, c, i + 1
        else:
            i += 1
            continue
        bits = ord(ch) - 63
        for _ in range(rep):
            for k in range(6):
                if bits >> k & 1 and x < w and y + k < h:
                    px[x, y + k] = cur
            x += 1
    return pic


if IMAGES:
    raw = replayed(ww, wh + status)
    last = {}
    for m in re.finditer(rb"\x1b\[(\d+);(\d+)H\x1bP[0-9;]*q(.*?)\x1b\\", raw, re.S):
        last[int(m.group(1)), int(m.group(2))] = m.group(3)             # a later picture at one spot replaces it
    for (row, col), data in last.items():
        pic = sixel(data.decode("latin1"))
        img.paste(pic, ((col - 1) * CW, (row - 1) * CH), pic)
    print(f"{len(last)} picture(s) from tmux")

img.save(OUT)
print(OUT, img.size)
