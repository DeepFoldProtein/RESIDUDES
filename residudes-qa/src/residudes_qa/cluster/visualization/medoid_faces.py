"""Medoid face rendering — Cα trace overlay panels for cluster medoids.

For each cluster (or community), pick the medoid = structure with minimum
mean distance to all other members. Render its Cα trace, anchor-aligned to
the largest cluster's medoid, as a small panel. Tile all medoids into one
figure so the user can compare poses side-by-side.

Generic: visual emphasis is driven by `anchor_chains` (Kabsch reference,
faded background) and `probe_chains` (what we're comparing, bold) so the
same renderer works for TCR-pMHC, Ab-antigen, homo-oligomer, generic
multimer — not only the original TCR-pMHC convention.

Usage (TCR-pMHC):
    from residudes_qa.cluster.visualization.medoid_faces import render_medoid_faces
    fig = render_medoid_faces(
        distance_matrix=M, labels=cluster_ids,
        structure_names=names, pdb_view_dir=Path('<run>/_pdb_view'),
        anchor_chains=('C','D','E'),         # MHC + β2m + peptide
        probe_chains=('A','B'),              # TCR α/β
        highlight_markers={'E': 'dot'},      # peptide overlay
    )

Usage (generic multimer):
    fig = render_medoid_faces(
        distance_matrix=M, labels=labels,
        structure_names=names, pdb_view_dir=pdb_view,
        anchor_chains=('A','B'),             # any two chains as reference
        probe_chains=('C','D'),              # others as probes
    )
"""
from __future__ import annotations
import re
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union, Dict, List
import numpy as np
import matplotlib.pyplot as plt


# TCR-pMHC default palette (backward compat). Auto-overridden for chain sets
# that diverge significantly from this convention.
_DEFAULT_CHAIN_COLORS = {
    'A': 'tab:blue', 'B': 'tab:orange',
    'C': '#888888', 'D': '#bbbbbb',
    'E': 'tab:red',  'F': '#7c3aed', 'G': '#0d9488',
}


def _generic_palette(chains) -> Dict[str, str]:
    """tab10 cycle keyed by chain label — used when chain set diverges
    from the TCR-pMHC default."""
    palette = plt.get_cmap('tab10')
    return {c: palette(i % 10) for i, c in enumerate(sorted(chains))}


def _pick_palette(all_chains, override: Optional[dict]):
    if override:
        return {**_DEFAULT_CHAIN_COLORS, **override}
    in_default = sum(1 for c in all_chains if c in _DEFAULT_CHAIN_COLORS)
    if in_default >= 0.7 * max(1, len(all_chains)):
        out = dict(_DEFAULT_CHAIN_COLORS)
        for c in all_chains:
            out.setdefault(c, '#666666')
        return out
    return _generic_palette(all_chains)


def parse_chain_spec(spec: str) -> Tuple[str, ...]:
    """Pull single-letter chain IDs from a chain spec string.

    Accepts the notation used by run_clustering's --anchor-chains /
    --probe-chains / --anchor-spec flags:

        'CDE'             → ('C','D','E')
        'AB-framework'    → ('A','B')         (mode suffix ignored)
        'AB:cdr+E'        → ('A','B','E')     ('+'-separated mixed spec)
        ''                → ()
    """
    if not spec:
        return ()
    chains: List[str] = []
    seen = set()
    for seg in spec.split('+'):
        m = re.match(r'^([A-Z]+)', seg)
        if m:
            for c in m.group(1):
                if c not in seen:
                    chains.append(c)
                    seen.add(c)
    return tuple(chains)


def _load_ca_by_chain(path: Path) -> dict:
    """Read first model of cif/pdb, return {chain_name: (N,3) np.array}."""
    import gemmi
    s = gemmi.read_structure(str(path))
    out = {}
    for model in s:
        for chain in model:
            xs = []
            for res in chain:
                for atom in res:
                    if atom.name == 'CA':
                        xs.append([atom.pos.x, atom.pos.y, atom.pos.z])
            if xs:
                out[chain.name] = np.asarray(xs)
        break
    return out


def _kabsch_R_t(A: np.ndarray, B: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return R, t such that A @ R.T + t ≈ B."""
    Ac, Bc = A.mean(0), B.mean(0)
    H = (A - Ac).T @ (B - Bc)
    U, _, Vt = np.linalg.svd(H)
    d = float(np.sign(np.linalg.det(Vt.T @ U.T)))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, Bc - R @ Ac


def _find_medoid_idx(D_sub: np.ndarray) -> int:
    """Index of row with minimum mean distance to others."""
    return int(np.argmin(D_sub.mean(axis=1)))


def render_medoid_faces(
    distance_matrix: np.ndarray,
    labels: np.ndarray,
    structure_names: Sequence[str],
    pdb_view_dir: Union[str, Path],
    anchor_chains: Sequence[str] = (),
    probe_chains: Optional[Sequence[str]] = None,
    highlight_markers: Optional[Dict[str, str]] = None,
    title: str = '',
    top_k: Optional[int] = None,
    ncols: Optional[int] = None,
    two_views: bool = True,
    chain_colors: Optional[dict] = None,
    label_color_by: Optional[Sequence] = None,
    label_palette: Optional[dict] = None,
    output_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Render Cα-trace medoid panel for each cluster.

    Args:
      distance_matrix : NxN pairwise distance matrix (lower = more similar).
      labels          : length-N cluster id per structure (matrix row order).
      structure_names : length-N name per structure (matches files in
                        pdb_view_dir).
      pdb_view_dir    : Directory with <name>.pdb (or .cif) files.
      anchor_chains   : Chain ids used to superpose all medoids onto the
                        largest cluster's medoid. Drawn FADED in the plot
                        (background "stage"). Pass () to skip superposition.
      probe_chains    : Chain ids to emphasise visually (drawn BOLD, on top).
                        If None: probe = all chains NOT in anchor_chains.
                        Pass () for uniform style (no role distinction).
      highlight_markers : Optional {chain_id: 'dot' | 'star'} to overlay
                        residue markers on a chain (e.g. peptide dots).
      title           : Figure-level title.
      top_k           : Show only K largest clusters by size. None = all.
      ncols           : Panels per row. Default: 6 (two views) or 10.
      two_views       : True → side-view (xz) + top-view (xy) per medoid.
      chain_colors    : Override default chain→color map.
      label_color_by  : Optional length-N secondary label used to colour
                        each per-panel title.
      label_palette   : Map secondary_label_value → color hex for titles.
      output_path     : If given, fig.savefig(path) called.

    Returns:
      matplotlib.figure.Figure
    """
    pdb_view_dir = Path(pdb_view_dir)
    labels = np.asarray(labels)
    structure_names = list(structure_names)
    anchor_chains = tuple(anchor_chains)
    highlight_markers = highlight_markers or {}

    # Cluster ids ordered by size descending
    unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True)
    order = np.argsort(-counts)
    ordered = [(int(unique_labels[i]), int(counts[i])) for i in order]
    if top_k:
        ordered = ordered[:top_k]

    # Per-cluster medoid identification
    medoids: List[dict] = []
    for cl, sz in ordered:
        idxs = np.where(labels == cl)[0]
        if len(idxs) == 0:
            continue
        if len(idxs) == 1:
            local_med = 0
        else:
            sub = distance_matrix[np.ix_(idxs, idxs)]
            local_med = _find_medoid_idx(sub)
        mat_idx = int(idxs[local_med])
        med_name = structure_names[mat_idx]
        medoids.append(dict(
            cluster=cl, size=sz, idx=mat_idx, name=med_name,
            label2=(label_color_by[mat_idx] if label_color_by is not None else None),
        ))

    # Load Cα coords for each medoid
    coords: Dict[int, dict] = {}
    for m in medoids:
        p = pdb_view_dir / m['name']
        if not p.is_file():
            alt = pdb_view_dir / Path(m['name']).with_suffix('.cif').name
            if alt.is_file():
                p = alt
        if not p.is_file():
            continue
        try:
            coords[m['cluster']] = _load_ca_by_chain(p)
        except Exception:
            continue
    medoids = [m for m in medoids if m['cluster'] in coords]
    if not medoids:
        raise RuntimeError(f'No medoid PDBs loadable from {pdb_view_dir}')

    # Discover full chain set + resolve probe
    all_chains: set = set()
    for d in coords.values():
        all_chains.update(d.keys())
    if probe_chains is None:
        if anchor_chains:
            probe_chains = tuple(sorted(all_chains - set(anchor_chains)))
        else:
            probe_chains = ()
    probe_chains = tuple(probe_chains)

    anchor_set = set(anchor_chains)
    probe_set = set(probe_chains)
    chain_colors = _pick_palette(all_chains, chain_colors)

    # Anchor-align all medoids to the first (largest) cluster
    do_align = bool(anchor_chains)
    ref_anchor = None
    if do_align:
        ref = coords[medoids[0]['cluster']]
        ref_anchor_parts = [ref[c] for c in anchor_chains if c in ref]
        if ref_anchor_parts:
            ref_anchor = np.vstack(ref_anchor_parts)
        else:
            do_align = False
    aligned: Dict[int, dict] = {}
    for m in medoids:
        d = coords[m['cluster']]
        if do_align:
            parts = [d[c] for c in anchor_chains if c in d]
            if parts:
                anchor = np.vstack(parts)
                n = min(len(anchor), len(ref_anchor))
                R, t = _kabsch_R_t(anchor[:n], ref_anchor[:n])
                aligned[m['cluster']] = {
                    ch: (R @ xyz.T).T + t for ch, xyz in d.items()
                }
                continue
        aligned[m['cluster']] = d

    # Layout
    n_med = len(medoids)
    if ncols is None:
        ncols = 6 if two_views else 10
    rows_per_panel = 2 if two_views else 1
    nrows_top = int(np.ceil(n_med / ncols)) * rows_per_panel
    fig, axes = plt.subplots(
        nrows_top, ncols,
        figsize=(2.7 * ncols, 2.6 * nrows_top),
    )
    if nrows_top == 1:
        axes = np.array([axes])
    if ncols == 1:
        axes = axes.reshape(-1, 1)

    # Plot order: anchor first (back), then "other", then probe (front)
    other_chains = [c for c in sorted(all_chains)
                    if c not in anchor_set and c not in probe_set]
    plot_order = list(anchor_chains) + other_chains + list(probe_chains)

    for plot_i, m in enumerate(medoids):
        d = aligned[m['cluster']]
        title_color = (
            (label_palette or {}).get(m['label2'], 'black')
            if m['label2'] is not None else 'black'
        )
        for view_i, view in enumerate(['xz'] + (['xy'] if two_views else [])):
            row = (plot_i // ncols) * rows_per_panel + view_i
            col = plot_i % ncols
            ax = axes[row, col]
            for ch in plot_order:
                if ch not in d:
                    continue
                xyz = d[ch]
                xx, yy = (
                    (xyz[:, 0], xyz[:, 2]) if view == 'xz'
                    else (xyz[:, 0], xyz[:, 1])
                )
                # role-based style
                if ch in probe_set:
                    lw, alpha = 1.8, 0.95
                elif ch in anchor_set:
                    lw, alpha = 1.0, 0.45
                else:
                    lw, alpha = 1.0, 0.65
                color = chain_colors.get(ch, '#666666')
                ax.plot(xx, yy, '-', color=color, lw=lw, alpha=alpha)
                if ch in highlight_markers:
                    style = highlight_markers[ch]
                    if style == 'dot':
                        ax.scatter(xx, yy, color=color, s=10, zorder=5)
                    elif style == 'star':
                        ax.scatter(xx, yy, color=color, marker='*', s=40, zorder=5)
            ax.set_aspect('equal')
            ax.set_xticks([])
            ax.set_yticks([])
            if view_i == 0:
                lab = f"L{m['cluster']}  n={m['size']:,}"
                if m['label2'] is not None:
                    lab += f"  [{m['label2']}]"
                ax.set_title(lab, fontsize=8, fontweight='bold', color=title_color)
            else:
                ax.set_xlabel('top view (xy)', fontsize=6)

    # Hide unused panels
    for ax_row in axes:
        for ax in ax_row:
            if not ax.lines and not ax.get_xlabel() and not ax.get_title():
                ax.set_axis_off()

    # Suptitle reflects anchor/probe configuration
    parts = []
    if anchor_chains:
        parts.append(f"anchor={''.join(anchor_chains)}")
    if probe_chains:
        parts.append(f"probe={''.join(probe_chains)}")
    suptitle = title or 'Cluster medoids — Cα trace'
    if parts:
        suptitle = f'{suptitle}   ({"  |  ".join(parts)})'
    fig.suptitle(suptitle, fontsize=11, y=1.0)
    fig.tight_layout()

    if output_path:
        fig.savefig(str(output_path), dpi=140, bbox_inches='tight', facecolor='white')

    return fig


__all__ = ['render_medoid_faces', 'parse_chain_spec']
