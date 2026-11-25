"""Visualize raw MSAs for a target directory (no predictions required).

For each chain X and each MSA variant V (af3, mmseqs2_gpu, plmmsa, ...) under
<target_dir>/X/msas/V/, writes one PNG per .a3m file mirroring the input layout:

  <target_dir>/visualization/<chain>/msas/<variant>/<msa_name>.png

Also writes a single summary grid (one column per chain, one row per
variant/msa source) to:

  <target_dir>/visualization/msa_template_summary.png

If a chain layout is absent (monomer-as-target with msas/ at the top level),
the chain label "_" is used instead.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# When run inside the apptainer wrapper, /viz is bind-mounted to this directory.
# residudes-qa: part of the residudes_qa.visualize package (was /viz inside the af3-viz container)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from residudes_qa.visualize.metrics import compute_msa_visualization  # noqa: E402
from residudes_qa.visualize.plots import plot_msa, save_figure        # noqa: E402


def find_chain_msa_layout(target_dir: Path) -> list[tuple[str, Path]]:
    """Return [(chain_label, chain_msas_dir), ...].

    Multi-chain layout: target/{A,B,...}/msas/<variant>/
    Monomer layout:     target/msas/<variant>/   → label "_"
    """
    layout: list[tuple[str, Path]] = []
    for sub in sorted(target_dir.iterdir()):
        if sub.is_dir() and len(sub.name) == 1 and sub.name.isupper():
            msas = sub / "msas"
            if msas.is_dir():
                layout.append((sub.name, msas))
    if layout:
        return layout
    msas = target_dir / "msas"
    if msas.is_dir():
        return [("_", msas)]
    return []


def find_variants(chain_msas_dir: Path) -> list[tuple[str, Path]]:
    """Return [(variant_label, variant_dir), ...].

    A variant is any directory under chain_msas_dir/ that directly contains
    one or more *.a3m files. The label is the path relative to chain_msas_dir
    (e.g. "af3", "mmseqs2_gpu", "plmmsa/plmalign", "plmmsa/otalign").
    """
    out: list[tuple[str, Path]] = []
    for p in sorted(chain_msas_dir.rglob("*.a3m")):
        if not p.is_file():
            continue
        d = p.parent
        label = str(d.relative_to(chain_msas_dir))
        if not any(lbl == label for lbl, _ in out):
            out.append((label, d))
    return out


def find_a3ms(variant_dir: Path) -> list[Path]:
    return sorted(p for p in variant_dir.glob("*.a3m") if p.is_file())


def render_per_msa(
    a3m_paths: list[Path],
    out_path: Path,
    title: str,
    dpi: int,
) -> tuple[int, int, float]:
    msa_data = compute_msa_visualization(a3m_paths, chain_boundary_positions=None)
    n_seqs = len(msa_data["lines"])
    seq_len = len(msa_data["lines"][0]) if n_seqs else 0
    neff = msa_data["neff"]
    full_title = f"{title}  (N={n_seqs}, L={seq_len}, Neff={neff:.2f})"
    fig = plot_msa(msa_data, title=full_title)
    save_figure(fig, out_path, dpi=dpi)
    return n_seqs, seq_len, neff


def _draw_msa_into_ax(ax: plt.Axes, msa_data: dict, title: str) -> "plt.cm.ScalarMappable | None":
    """Draw a MSA coverage heatmap into a given Axes. Returns the mappable for colorbar."""
    lines_raw = msa_data.get("lines", [])
    if not lines_raw:
        ax.text(0.5, 0.5, "no MSA", ha="center", va="center", transform=ax.transAxes,
                fontsize=8, color="gray")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=8)
        return None

    mat = np.array(lines_raw, dtype=np.float32)
    n_seqs, seq_len = mat.shape

    im = ax.imshow(
        mat,
        cmap="rainbow",
        vmin=0, vmax=1,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
    )

    coverage = msa_data.get("coverage")
    if coverage is not None:
        cov_arr = np.array(coverage, dtype=float)
    else:
        cov_arr = np.sum(~np.isnan(mat), axis=0).astype(float)
    ax.plot(np.arange(seq_len), cov_arr, color="black", linewidth=0.6)

    ln = msa_data.get("ln", [])
    for pos in ln[1:-1]:
        ax.axvline(pos - 0.5, color="black", linewidth=1.0)

    ax.set_xlim(-0.5, seq_len - 0.5)
    ax.set_ylim(-0.5, n_seqs - 0.5)
    ax.tick_params(labelsize=7)
    ax.set_title(title, fontsize=8)
    return im


def render_summary(
    target_dir: Path,
    layout: list[tuple[str, Path]],
    out_path: Path,
    dpi: int,
) -> int:
    """Build a grid: rows = (variant, msa_name), cols = chains.

    Each cell shows the MSA coverage heatmap for that chain × source. Cells
    with no .a3m for that chain are blank.
    """
    # Discover the union of (variant, msa_name) sources across all chains
    sources: set[tuple[str, str]] = set()
    chain_a3ms: dict[str, dict[tuple[str, str], Path]] = {}
    for chain_label, chain_msas_dir in layout:
        m: dict[tuple[str, str], Path] = {}
        for variant_label, variant_dir in find_variants(chain_msas_dir):
            for a3m in find_a3ms(variant_dir):
                key = (variant_label, a3m.stem)
                m[key] = a3m
                sources.add(key)
        chain_a3ms[chain_label] = m

    if not sources:
        print("  summary: no a3m sources found, skipping")
        return 0

    sorted_sources = sorted(sources)
    chain_labels = [c for c, _ in layout]
    n_rows = len(sorted_sources)
    n_cols = len(chain_labels)

    fig_w = max(3.5 * n_cols, 8.0)
    fig_h = max(2.2 * n_rows, 4.0)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), squeeze=False)

    last_im = None
    for j, chain_label in enumerate(chain_labels):
        for i, (variant, msa_name) in enumerate(sorted_sources):
            ax = axes[i][j]
            a3m = chain_a3ms[chain_label].get((variant, msa_name))
            if a3m is None:
                ax.text(0.5, 0.5, "—", ha="center", va="center",
                        transform=ax.transAxes, fontsize=10, color="gray")
                ax.set_xticks([])
                ax.set_yticks([])
                if i == 0:
                    ax.set_title(f"chain {chain_label}", fontsize=10, fontweight="bold")
                if j == 0:
                    ax.set_ylabel(f"{variant}/{msa_name}", fontsize=8)
                continue

            msa_data = compute_msa_visualization([a3m], chain_boundary_positions=None)
            n_seqs = len(msa_data["lines"])
            seq_len = len(msa_data["lines"][0]) if n_seqs else 0
            neff = msa_data["neff"]
            cell_title = f"N={n_seqs} L={seq_len} Neff={neff:.2f}"
            im = _draw_msa_into_ax(ax, msa_data, cell_title)
            if im is not None:
                last_im = im
            if i == 0:
                # column header (top-of-column chain label)
                ax.text(
                    0.5, 1.18, f"chain {chain_label}",
                    transform=ax.transAxes, ha="center", va="bottom",
                    fontsize=11, fontweight="bold",
                )
            if j == 0:
                ax.set_ylabel(f"{variant}/{msa_name}", fontsize=9)

    fig.suptitle(f"{target_dir.name} — MSA & template coverage summary", fontsize=13)

    if last_im is not None:
        fig.tight_layout(rect=[0.0, 0.0, 0.94, 0.96])
        cax = fig.add_axes([0.955, 0.10, 0.012, 0.78])
        cbar = fig.colorbar(last_im, cax=cax)
        cbar.set_label("Sequence identity to query", fontsize=9)
        cbar.ax.tick_params(labelsize=8)
    else:
        fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])

    save_figure(fig, out_path, dpi=dpi)
    return n_rows * n_cols


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("target_dir", type=Path)
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--no-per-msa", action="store_true", help="Skip per-source PNGs.")
    ap.add_argument("--no-summary", action="store_true", help="Skip the summary grid PNG.")
    args = ap.parse_args()

    target_dir = args.target_dir.resolve()
    if not target_dir.is_dir():
        print(f"error: not a directory: {target_dir}", file=sys.stderr)
        return 1

    layout = find_chain_msa_layout(target_dir)
    if not layout:
        print(f"error: no msas/ directory found under {target_dir}", file=sys.stderr)
        return 1

    out_root = target_dir / "visualization"
    chain_labels = [c for c, _ in layout]
    print(f"target={target_dir.name}  chains={chain_labels}")

    n_rendered = 0
    if not args.no_per_msa:
        for chain_label, chain_msas_dir in layout:
            variants = find_variants(chain_msas_dir)
            if not variants:
                print(f"  [{chain_label}] no MSA variants in {chain_msas_dir}; skipping")
                continue

            for variant_label, variant_dir in variants:
                a3ms = find_a3ms(variant_dir)
                tag = f"{chain_label}/msas/{variant_label}"
                if not a3ms:
                    print(f"  [{tag}] no .a3m files; skipping")
                    continue

                print(f"  [{tag}] {len(a3ms)} a3m file(s): {[p.name for p in a3ms]}")
                out_dir = out_root / chain_label / "msas" / variant_label

                for a3m in a3ms:
                    msa_name = a3m.stem
                    title = f"{target_dir.name} chain {chain_label} - {variant_label}/{msa_name}"
                    out_path = out_dir / f"{msa_name}.png"
                    n, length, neff = render_per_msa([a3m], out_path, title, args.dpi)
                    print(f"    -> {out_path.relative_to(target_dir)}  N={n} L={length} Neff={neff:.2f}")
                    n_rendered += 1

    if not args.no_summary:
        summary_path = out_root / "msa_template_summary.png"
        print(f"  summary: {summary_path.relative_to(target_dir)}")
        n_cells = render_summary(target_dir, layout, summary_path, args.dpi)
        print(f"    -> {summary_path.relative_to(target_dir)}  ({n_cells} cells)")

    print(f"done. rendered {n_rendered} per-msa PNGs into {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
