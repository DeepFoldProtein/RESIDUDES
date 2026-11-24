"""Split the residudes mascot sheets into one image per stage, keyed onto transparency.

    python3 split_mascots.py .

  cells/<stage>.png     the 2x2 sheet quartered, exactly as drawn
  cutout/<stage>.png    background removed, trimmed to the art with a small margin

The sheets are drawn on flat white, so the key itself is simple: alpha is how far a pixel is from white. Two
things need care.

*The art is white too.* A page, a clipboard, an envelope, the whites of the eyes are the same 255 as the
background, so only connectivity can tell them apart — and the outlines that should enclose them have hairline
gaps the white leaks through. Closing the art mask first seals those, after which a flood from the border reaches
the background and nothing else.

*A figure can cross the divider between cells* (a tail does), so each cell is cropped past its edges and what
lands in the margin is kept only if it belongs to this cell's own figure.
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

SHEETS = {                   # sheet file -> cells: top-left, top-right, bottom-left, bottom-right
    "sheets/a_new_to_plan_review.png": ["new", "identified", "hypotheses", "plan_review"],
    "sheets/b_inputs_to_results_review.png": ["inputs_ready", "running", "pool_ready", "results_review"],
    "sheets/c_slate_to_submitted.png": ["slate_proposed", "approved", "packaged", "submitted"],
    "sheets/d_retrospective_and_states.png": ["retrospective", "loop_back", "safety_refusal", "human_only"],
}
T0, T1 = 6.0, 26.0           # distance from white: transparent .. opaque
FAINT = 0.14                 # alpha below this is noise, not glow
SEAL = 9                     # how wide a hairline gap in an outline may be and still be sealed
SPECK = 300                  # an isolated mark smaller than this is noise in the white
OVER = 90                    # how far past its own quarter each cell is cropped
PAD = 14


def keyed(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    d = np.linalg.norm(255.0 - rgb, axis=2)
    a = np.clip((d - T0) / (T1 - T0), 0, 1)
    a[a < FAINT] = 0.0            # compression ringing in the white: invisible, but it stretches the crop box

    sealed = ndimage.binary_closing(a > 0.5, np.ones((SEAL, SEAL)))       # bridge the gaps in thin outlines
    lab, _ = ndimage.label(~sealed)
    edge = {int(i) for i in np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]) if i}
    outside = np.isin(lab, sorted(edge))                                  # white the border can reach: background
    a = np.where(outside, a, 1.0)

    # A glow fades into the white it was drawn on, so its colour is a blend. Solving source-over for the source
    # keeps it from washing out on a dark background; the divisor is floored so a nearly clear pixel cannot blow up.
    unmul = 255.0 + (rgb - 255.0) / np.maximum(a, 0.5)[..., None]
    col = np.where((outside & (a > 0.02) & (a < 0.98))[..., None], np.clip(unmul, 0, 255), rgb)
    return col, a


def drop_strays(a: np.ndarray, own: tuple[int, int, int, int]) -> np.ndarray:
    """This cell's picture is the largest shape plus whatever stands with it — a person's hand, a flying envelope,
    the sparks of a run. A shape that is neither near it nor centred inside this cell's own quarter belongs to the
    cell next door; anything tiny and alone is noise."""
    lab, n = ndimage.label(a > 0.15)
    if not n:
        return a
    areas = ndimage.sum(a > 0.15, lab, range(1, n + 1))
    main = int(np.argmax(areas)) + 1
    near = ndimage.binary_dilation(lab == main, np.ones((121, 121)))
    x0, y0, x1, y1 = own
    outside = np.ones_like(a, bool)
    outside[y0:y1, x0:x1] = False
    a = np.where(outside & (lab != main), 0.0, a)
    for i in range(1, n + 1):
        if i == main:
            continue
        comp = lab == i
        if areas[i - 1] < SPECK:
            a = np.where(comp, 0.0, a)
            continue
        if near[comp].any():
            continue
        ys, xs = np.nonzero(comp)
        if not (x0 <= xs.mean() <= x1 and y0 <= ys.mean() <= y1 and areas[i - 1] >= 6000):
            a = np.where(comp, 0.0, a)
    return a


def cut_out(cell: Image.Image, own: tuple[int, int, int, int]) -> Image.Image:
    rgb = np.asarray(cell.convert("RGB")).astype(np.float32)
    h, w, _ = rgb.shape
    col, a = keyed(rgb)
    a = drop_strays(a, own)
    img = Image.fromarray(np.dstack([col, a * 255]).astype(np.uint8), "RGBA")
    box = img.getchannel("A").point(lambda v: 255 if v > 24 else 0).getbbox()
    if box:
        img = img.crop((max(0, box[0] - PAD), max(0, box[1] - PAD), min(w, box[2] + PAD), min(h, box[3] + PAD)))
    return img


def main(root: Path, only: set[str]) -> None:
    (root / "cells").mkdir(exist_ok=True)
    (root / "cutout").mkdir(exist_ok=True)
    for fname, names in SHEETS.items():
        if only and not only & set(names):
            continue
        sheet = Image.open(root / fname)
        w, h = sheet.size
        for i, name in enumerate(names):
            if only and name not in only:
                continue
            left, top = (i % 2) * (w // 2), (i // 2) * (h // 2)
            right, bottom = left + w // 2, top + h // 2
            cl, ct = (max(0, left - OVER) if i % 2 else left), (max(0, top - OVER) if i // 2 else top)
            cr, cb = (right if i % 2 else min(w, right + OVER)), (bottom if i // 2 else min(h, bottom + OVER))
            cell = sheet.crop((cl, ct, cr, cb))
            own = (left - cl, top - ct, right - cl, bottom - ct)
            cell.convert("RGB").save(root / "cells" / f"{name}.png")
            cut = cut_out(cell, own)
            cut.save(root / "cutout" / f"{name}.png")
            al = np.asarray(cut)[:, :, 3]
            print(f"{name:16s} cutout {cut.size[0]:4d}x{cut.size[1]:3d}  opaque {100 * (al > 250).mean():4.1f}%  "
                  f"soft edge {100 * ((al > 10) & (al < 250)).mean():4.1f}%")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "."), set(sys.argv[2:]))
