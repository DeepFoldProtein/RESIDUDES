"""Key the white background out of the logo portrait and write the sizes a README and a deck need.

    python3 make_logo.py .

`logo/source.png` is the drawing as generated: the mascot on flat white. Only white that is *connected to the
border* is removed, so the white inside the picture — the eyes, the lamp's highlight, the glass — stays. The
sticker's white keyline is part of that outer white and goes with it, leaving the navy outline as the edge.
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

T0, T1 = 6.0, 26.0        # distance from white: transparent .. opaque
GAP = 2000                # an enclosed white region this big is background seen through a gap, not a highlight
SPECK = 400               # anything smaller than this, standing on its own, is noise in the white
SIZES = (1024, 512, 256, 128)


def cut(src: Path) -> Image.Image:
    rgb = np.asarray(Image.open(src).convert("RGB")).astype(np.float32)
    d = np.linalg.norm(255.0 - rgb, axis=2)
    a = np.clip((d - T0) / (T1 - T0), 0, 1)

    # White the border can reach is background. White the drawing encloses is either background seen through a gap
    # — under the lantern's handle, between its posts — or part of the picture: the highlights in the eyes. The
    # gaps are large, the highlights are a few hundred pixels, so size tells them apart.
    lab, n = ndimage.label(a < 0.5)
    edge = {int(i) for i in np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]) if i}
    areas = ndimage.sum(a < 0.5, lab, range(1, n + 1))
    background = {i for i in range(1, n + 1) if i in edge or areas[i - 1] >= GAP}
    a = np.where(np.isin(lab, sorted(background)), a, np.maximum(a, 1.0))

    solid = a > 0.15                                                 # a few stray specks of near-white noise sit
    lab, n = ndimage.label(solid)                                    # well below the feet; left in, they pad the
    if n:                                                            # frame and the figure then hangs too high
        areas = ndimage.sum(solid, lab, range(1, n + 1))
        a = np.where(np.isin(lab, [i for i in range(1, n + 1) if areas[i - 1] < SPECK]), 0.0, a)

    img = Image.fromarray(np.dstack([rgb, a * 255]).astype(np.uint8), "RGBA")
    box = img.getchannel("A").point(lambda v: 255 if v > 24 else 0).getbbox()
    return img.crop(box) if box else img


def main(root: Path) -> None:
    out = root / "logo"
    out.mkdir(exist_ok=True)
    src = out / "source.png"
    if not src.exists():
        raise SystemExit(f"put the drawing at {src}")
    img = cut(src)
    img.save(out / "logo.png")
    print(f"logo.png {img.size[0]}x{img.size[1]}")
    for n in SIZES:
        k = n / max(img.size)
        img.resize((round(img.size[0] * k), round(img.size[1] * k)), Image.LANCZOS).save(out / f"logo_{n}.png")
        print(f"logo_{n}.png")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "."))
