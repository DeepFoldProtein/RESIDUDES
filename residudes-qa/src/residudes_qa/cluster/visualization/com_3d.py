"""3D scatter for `--method com` clustering.

Visualises probe-chain COMs in the anchor frame, with the anchor backbone
drawn as a reference scaffold. Triggered automatically by run_clustering.py
after comprehensive analysis when method == 'com'.

Generates a 4-view PNG (xy / xz / yz / iso) into the results directory:
    <results>/com_3d_clusters.png        — colored by cluster
    <results>/com_3d_by_provenance.png   — colored by generator (af3/bt2/of3/pt2)
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Sequence
import re

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d)


_VIEWS = [('xy', (90, -90)), ('xz', (0, -90)), ('yz', (0, 0)), ('iso', (25, 45))]
_METHOD_COLOR = {
    'af3': '#1f77b4',  # blue
    'bt2': '#2ca02c',  # green
    'of3': '#ffdc00',  # yellow
    'pt2': '#d62728',  # red
}


def _load_chain_ca(cif_path, chain_ids: Sequence[str]) -> Dict[str, np.ndarray]:
    """Return {chain_id: (N, 3) Cα coords} from first model of cif."""
    import gemmi
    st = gemmi.read_structure(str(cif_path))
    out: Dict[str, list] = {c: [] for c in chain_ids}
    for chain in st[0]:
        if chain.name not in out:
            continue
        for res in chain:
            ca = res.find_atom('CA', '*')
            if ca is None:
                for atom in res:
                    if atom.name == 'CA':
                        ca = atom
                        break
            if ca is None:
                continue
            out[chain.name].append((ca.pos.x, ca.pos.y, ca.pos.z))
    return {c: np.array(v, dtype=np.float32) for c, v in out.items()}


def _method_from_name(name: str) -> str:
    m = re.match(r'^([a-z]+\d?)_seed', str(name))
    return m.group(1) if m else 'unknown'


def render_com_3d(
    com3d: np.ndarray,
    names: Sequence[str],
    clustering_csv: Path,
    anchor_cif: Path,
    anchor_chains: Sequence[str],
    out_dir: Path,
    target_tag: str,
) -> None:
    """Generate cluster + provenance 3D scatter with anchor backbone.

    Args:
        com3d: (N, n_probe, 3) — probe-chain Cα COMs in anchor frame.
        names: structure names (length N) matching com3d row order.
        clustering_csv: massivecluster comprehensive's <method>_clustering.csv.
        anchor_cif: any source cif (first model in matrix order is canonical).
        anchor_chains: chain ids used as Kabsch anchor (e.g. ('C', 'D')).
        out_dir: results/ directory (writes <target_tag>_com_3d_*.png).
        target_tag: prefix for output filenames.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Reduce per-model COM list to a single midpoint for cluster scatter
    # (averaging makes the figure readable; per-chain split is also drawn).
    if com3d.ndim != 3 or com3d.shape[2] != 3:
        raise ValueError(f'expected com3d shape (N, n_probe, 3), got {com3d.shape}')
    n_probe = com3d.shape[1]
    mid = com3d.mean(axis=1)

    # Load CD backbone and centre on its centroid (== origin in this frame)
    try:
        anchor_xyz = _load_chain_ca(anchor_cif, anchor_chains)
    except Exception as exc:
        print(f'  [com_3d] could not load anchor backbone: {exc!r}', flush=True)
        anchor_xyz = {}
    if anchor_xyz:
        all_anchor = np.vstack(list(anchor_xyz.values()))
        centroid = all_anchor.mean(0)
        anchor_xyz = {c: v - centroid for c, v in anchor_xyz.items()}

    def _draw_anchor(ax):
        if not anchor_xyz:
            return
        greys = ['#444444', '#888888', '#bbbbbb', '#dddddd']
        for k, (ch, xyz) in enumerate(anchor_xyz.items()):
            col = greys[min(k, len(greys) - 1)]
            ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], color=col, lw=0.9, alpha=0.7)
            ax.scatter(xyz[::5, 0], xyz[::5, 1], xyz[::5, 2], c=col, s=2, alpha=0.5)

    # Load clustering result
    clust = pd.read_csv(clustering_csv)
    if len(clust) != len(names):
        print(f'  [com_3d] WARN length mismatch clust={len(clust)} names={len(names)}', flush=True)
        return
    clusters = clust['cluster'].to_numpy(dtype=int)
    full_names = clust['structure'].astype(str).to_numpy()
    n_clusters = int(clusters.max()) + 1
    cmap = plt.get_cmap('tab20', n_clusters)
    colors = np.array([cmap(c) if c >= 0 else (0.5, 0.5, 0.5, 0.3) for c in clusters])

    # ── cluster-colored 4-view scatter ──────────────────────────────────────
    fig = plt.figure(figsize=(18, 12))
    for i, (label, (elev, azim)) in enumerate(_VIEWS):
        ax = fig.add_subplot(2, 2, i+1, projection='3d')
        _draw_anchor(ax)
        ax.scatter(mid[:, 0], mid[:, 1], mid[:, 2],
                   c=colors, s=4, alpha=0.6, depthshade=False)
        ax.set_xlabel('X (Å)')
        ax.set_ylabel('Y (Å)')
        ax.set_zlabel('Z (Å)')
        ax.set_title(f'probe midpoint ({label})')
        ax.view_init(elev=elev, azim=azim)
    fig.suptitle(
        f'{target_tag}: probe COM in anchor frame — {n_clusters} clusters, N={len(names)}'
        f'  (anchor backbone in grey; probe={n_probe} chain{"s" if n_probe > 1 else ""})',
        fontsize=13,
    )
    plt.tight_layout()
    out1 = out_dir / f'{target_tag}_com_3d_clusters.png'
    plt.savefig(out1, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'  [com_3d] saved {out1.name}', flush=True)

    # ── provenance-colored: af3 blue, bt2 green, of3 yellow, pt2 red ────────
    methods = np.array([_method_from_name(n) for n in full_names])
    fig = plt.figure(figsize=(18, 12))
    for i, (label, (elev, azim)) in enumerate(_VIEWS):
        ax = fig.add_subplot(2, 2, i+1, projection='3d')
        _draw_anchor(ax)
        for meth, col in _METHOD_COLOR.items():
            mask = methods == meth
            if not mask.any():
                continue
            ax.scatter(mid[mask, 0], mid[mask, 1], mid[mask, 2],
                       c=col, s=3, alpha=0.5, depthshade=False,
                       label=f'{meth} (n={mask.sum()})')
        ax.set_xlabel('X (Å)')
        ax.set_ylabel('Y (Å)')
        ax.set_zlabel('Z (Å)')
        ax.set_title(f'by generator ({label})')
        ax.view_init(elev=elev, azim=azim)
        if i == 3:
            ax.legend(fontsize=9, loc='upper left')
    fig.suptitle(
        f'{target_tag}: probe COM by generator (af3 blue, bt2 green, of3 yellow, pt2 red)',
        fontsize=13,
    )
    plt.tight_layout()
    out2 = out_dir / f'{target_tag}_com_3d_by_provenance.png'
    plt.savefig(out2, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'  [com_3d] saved {out2.name}', flush=True)
