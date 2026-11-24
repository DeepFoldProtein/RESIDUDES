"""
Network Visualization Module

Provides visualizations for network analysis results:
- Force-directed graph layout with community/cluster coloring
- Community vs clustering comparison (contingency matrix)
- Transition probability heatmap
- Leiden resolution scan plot
- Multi-panel network report
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Optional, Union, List, Dict, Any, Tuple
from pathlib import Path


def plot_network_graph(
    distance_matrix: np.ndarray,
    labels: np.ndarray,
    communities: Optional[np.ndarray] = None,
    structure_names: Optional[List[str]] = None,
    hub_indices: Optional[List[int]] = None,
    degree_centrality: Optional[np.ndarray] = None,
    title: str = "Structure Network",
    color_by: str = "cluster",
    output_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (12, 10),
    max_nodes: int = 500,
    edge_alpha_min: float = 0.05,
    edge_weight_threshold: float = 0.3,
    highlight_indices: Optional[List[int]] = None,
    corner_table: Optional[str] = None,
    corner: str = "upper right",
) -> plt.Figure:
    """
    Plot force-directed network graph.

    Node color = cluster or community assignment.
    Node size = proportional to degree centrality.
    Hub nodes are highlighted with thick black border.

    Args:
        distance_matrix: N*N distance matrix
        labels: Cluster labels
        communities: Community labels (optional, used when color_by="community")
        structure_names: Optional node labels
        hub_indices: Indices of hub structures to highlight
        degree_centrality: Per-node degree centrality for sizing
        title: Plot title
        color_by: "cluster" or "community"
        output_path: Save path (optional)
        figsize: Figure size
        max_nodes: Max nodes to render (subsamples if exceeded)
        edge_alpha_min: Minimum edge alpha
        edge_weight_threshold: Only draw edges with weight above this
    """
    import networkx as nx
    from residudes_qa.cluster.core.network import build_structure_graph

    n = len(distance_matrix)
    G = build_structure_graph(distance_matrix, structure_names=structure_names)

    # Subsample if too large
    keep_mask = np.ones(n, dtype=bool)
    if n > max_nodes:
        rng = np.random.RandomState(42)
        keep_set = set()
        if hub_indices:
            keep_set.update(hub_indices[:min(len(hub_indices), 20)])
        if highlight_indices:
            keep_set.update(highlight_indices)   # never drop highlighted nodes
        remaining = max_nodes - len(keep_set)
        candidates = [i for i in range(n) if i not in keep_set]
        keep_set.update(rng.choice(candidates, size=min(remaining, len(candidates)), replace=False))
        keep_mask = np.zeros(n, dtype=bool)
        keep_mask[list(keep_set)] = True
        G = G.subgraph(list(keep_set)).copy()

    # Filter edges by weight threshold
    edges_to_remove = [
        (u, v) for u, v, d in G.edges(data=True)
        if d.get('weight', 0) < edge_weight_threshold
    ]
    G.remove_edges_from(edges_to_remove)

    fig, ax = plt.subplots(figsize=figsize)

    if G.number_of_nodes() == 0:
        ax.text(0.5, 0.5, 'No nodes to display', ha='center', va='center',
                transform=ax.transAxes, fontsize=12)
        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
        return fig

    # Layout
    pos = nx.spring_layout(G, weight='weight', k=1.5 / np.sqrt(G.number_of_nodes()),
                           iterations=50, seed=42)

    # Color by cluster or community
    node_list = list(G.nodes())
    if color_by == "community" and communities is not None:
        color_values = [communities[i] for i in node_list]
    else:
        color_values = [labels[i] for i in node_list]

    n_colors = max(color_values) + 1 if color_values else 1
    cmap = plt.get_cmap('tab20', max(n_colors, 2))   # plt.cm.get_cmap is gone in Matplotlib 3.11
    node_colors = [cmap(c % 20) for c in color_values]

    # Node sizes from degree centrality
    if degree_centrality is not None:
        dc_vals = np.array([degree_centrality[i] for i in node_list])
        dc_max = dc_vals.max() if dc_vals.max() > 0 else 1.0
        node_sizes = 30 + 200 * dc_vals / dc_max
    else:
        node_sizes = 80

    # Edge widths and alphas from weight
    edge_weights = [G[u][v].get('weight', 0.5) for u, v in G.edges()]
    if edge_weights:
        w_arr = np.array(edge_weights)
        w_max = w_arr.max() if w_arr.max() > 0 else 1.0
        edge_widths = 0.3 + 1.5 * w_arr / w_max
        edge_alphas = edge_alpha_min + (1.0 - edge_alpha_min) * w_arr / w_max
    else:
        edge_widths = []
        edge_alphas = []

    # Draw edges
    for idx, (u, v) in enumerate(G.edges()):
        x = [pos[u][0], pos[v][0]]
        y = [pos[u][1], pos[v][1]]
        ax.plot(x, y, '-', color='#999999', linewidth=edge_widths[idx],
                alpha=float(edge_alphas[idx]), zorder=1)

    # Draw nodes
    hub_set = set(hub_indices) if hub_indices else set()
    for idx, node in enumerate(node_list):
        edgecolor = 'black' if node in hub_set else 'white'
        linewidth = 2.0 if node in hub_set else 0.5
        size = node_sizes[idx] if hasattr(node_sizes, '__len__') else node_sizes
        if node in hub_set:
            size *= 1.5
        ax.scatter(pos[node][0], pos[node][1], c=[node_colors[idx]],
                   s=size, edgecolors=edgecolor, linewidths=linewidth, zorder=3)

    # Highlight specific nodes (e.g. our submissions) as labelled stars on top
    if highlight_indices:
        for node in highlight_indices:
            if node not in pos:
                continue
            ax.scatter(pos[node][0], pos[node][1], s=300, marker='*',
                       facecolor='white', edgecolors='black', linewidths=1.6, zorder=5)
            lbl = None
            if structure_names is not None and node < len(structure_names):
                lbl = (str(structure_names[node]).replace('OURS_relaxed_', '')
                       .replace('OURS_', ''))
            if lbl:
                ax.annotate(lbl, (pos[node][0], pos[node][1]), fontsize=8,
                            fontweight='bold', xytext=(5, 5),
                            textcoords='offset points', zorder=6)

    # Optional monospace summary table pinned to a corner (e.g. per-community
    # confidence). corner: 'upper right'|'upper left'|'lower right'|'lower left'.
    if corner_table:
        vy, hx = corner.split()
        x = 0.99 if hx == "right" else 0.01
        y = 0.99 if vy == "upper" else 0.01
        ax.text(x, y, corner_table, transform=ax.transAxes,
                ha=("right" if hx == "right" else "left"),
                va=("top" if vy == "upper" else "bottom"),
                family="monospace", fontsize=9, zorder=10,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                          edgecolor="#444", alpha=0.95))

    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_axis_off()

    # Legend for clusters/communities
    unique_vals = sorted(set(color_values))
    if len(unique_vals) <= 20:
        legend_label = "Community" if color_by == "community" else "Cluster"
        handles = [plt.scatter([], [], c=[cmap(v % 20)], s=60, edgecolors='white',
                               label=f'{legend_label} {v}') for v in unique_vals]
        ax.legend(handles=handles, loc='lower right', fontsize=7, framealpha=0.8,
                  ncol=max(1, len(unique_vals) // 10))

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
    return fig


def plot_community_comparison(
    labels: np.ndarray,
    communities: np.ndarray,
    title: str = "Cluster vs Community",
    output_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (10, 8),
) -> plt.Figure:
    """
    Plot confusion/contingency matrix: clustering labels vs communities.

    Shows how community detection aligns with clustering assignments.

    Args:
        labels: Cluster labels
        communities: Community labels
        title: Plot title
        output_path: Save path (optional)
        figsize: Figure size
    """
    import pandas as pd
    import seaborn as sns

    if communities is None:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No community data available', ha='center', va='center',
                transform=ax.transAxes, fontsize=12)
        ax.set_title(title)
        ax.set_axis_off()
        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
        return fig

    ct = pd.crosstab(
        pd.Series(labels, name='Cluster'),
        pd.Series(communities, name='Community'),
    )
    # Normalize by row (fraction of each cluster in each community)
    ct_norm = ct.div(ct.sum(axis=1), axis=0)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Raw counts
    sns.heatmap(ct, annot=True, fmt='d', cmap='YlOrRd', ax=axes[0],
                cbar_kws={'label': 'Count'})
    axes[0].set_title('Counts', fontsize=10, fontweight='bold')

    # Normalized
    sns.heatmap(ct_norm, annot=True, fmt='.2f', cmap='YlOrRd', ax=axes[1],
                cbar_kws={'label': 'Fraction'})
    axes[1].set_title('Row-normalized', fontsize=10, fontweight='bold')

    fig.suptitle(title, fontsize=12, fontweight='bold')
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
    return fig


def plot_transition_heatmap(
    transition_matrix: np.ndarray,
    cluster_labels: Optional[List[str]] = None,
    title: str = "Cluster Transition Probabilities",
    output_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (8, 6),
) -> plt.Figure:
    """
    Plot transition probability matrix as annotated heatmap.

    Args:
        transition_matrix: K*K transition probability matrix
        cluster_labels: Labels for each cluster
        title: Plot title
        output_path: Save path (optional)
        figsize: Figure size
    """
    import seaborn as sns

    k = len(transition_matrix)
    if cluster_labels is None:
        cluster_labels = [f'C{i}' for i in range(k)]

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        transition_matrix,
        annot=True, fmt='.2f',
        cmap='YlOrRd',
        xticklabels=cluster_labels,
        yticklabels=cluster_labels,
        ax=ax,
        vmin=0, vmax=1,
        cbar_kws={'label': 'Transition Probability'},
        linewidths=0.5, linecolor='white',
    )
    ax.set_xlabel('To Cluster', fontsize=10)
    ax.set_ylabel('From Cluster', fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.tick_params(labelsize=8)

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
    return fig


def plot_resolution_scan(
    scan_results: Dict[str, Any],
    title: str = "Leiden Resolution Scan",
    output_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (10, 5),
) -> plt.Figure:
    """
    Plot resolution scan results: n_communities and modularity vs resolution.

    Dual y-axis: left = n_communities, right = modularity.

    Args:
        scan_results: Dict from resolution_scan() with resolutions, n_communities, modularity
        title: Plot title
        output_path: Save path (optional)
        figsize: Figure size
    """
    resolutions = scan_results['resolutions']
    n_comms = scan_results['n_communities']
    modularity = scan_results['modularity']

    fig, ax1 = plt.subplots(figsize=figsize)
    ax2 = ax1.twinx()

    # N communities (left axis)
    ax1.plot(resolutions, n_comms, 'o-', color='#2196F3', lw=2, ms=5,
             label='N communities')
    ax1.set_xlabel('Resolution', fontsize=10)
    ax1.set_ylabel('Number of Communities', fontsize=10, color='#2196F3')
    ax1.tick_params(axis='y', labelcolor='#2196F3')
    ax1.set_xscale('log')

    # Modularity (right axis)
    ax2.plot(resolutions, modularity, 's-', color='#FF9800', lw=2, ms=5,
             label='Modularity')
    ax2.set_ylabel('Modularity', fontsize=10, color='#FF9800')
    ax2.tick_params(axis='y', labelcolor='#FF9800')

    # Mark suggested resolution
    suggested = scan_results.get('suggested_resolution')
    if suggested is not None:
        ax1.axvline(x=suggested, color='red', linestyle='--', linewidth=1.5, alpha=0.6)
        ax1.annotate(
            f'Suggested\nres={suggested:.3f}',
            xy=(suggested, max(n_comms) * 0.9),
            fontsize=7, color='red', ha='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='red', alpha=0.85)
        )

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8)

    ax1.set_title(title, fontsize=12, fontweight='bold')
    ax1.grid(alpha=0.2)

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
    return fig


def create_network_report(
    distance_matrix: np.ndarray,
    labels: np.ndarray,
    network_result: Any,
    transition_matrix: Optional[np.ndarray] = None,
    scan_results: Optional[Dict[str, Any]] = None,
    target_name: str = "Target",
    output_dir: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Create multi-panel network analysis figure.

    Layout:
    [Graph topology      ] [Community comparison]
    [Transition heatmap  ] [Degree distribution ]

    Args:
        distance_matrix: N*N distance matrix
        labels: Cluster labels
        network_result: NetworkAnalysisResult
        transition_matrix: K*K transition matrix (optional)
        scan_results: Resolution scan results (optional)
        target_name: Target label
        output_dir: Save directory (optional)
    """
    import seaborn as sns

    n_panels = 4
    fig = plt.figure(figsize=(18, 14))

    # Panel 1: Network graph (top-left)
    ax1 = fig.add_subplot(2, 2, 1)
    _plot_network_graph_on_ax(
        ax1, distance_matrix, labels,
        communities=network_result.communities,
        hub_indices=network_result.hub_indices,
        degree_centrality=network_result.degree_centrality,
        title=f'{target_name} - Network ({network_result.community_method})',
    )

    # Panel 2: Community comparison or resolution scan (top-right)
    ax2 = fig.add_subplot(2, 2, 2)
    if network_result.communities is not None:
        _plot_community_comparison_on_ax(ax2, labels, network_result.communities)
    elif scan_results:
        _plot_resolution_scan_on_ax(ax2, scan_results)
    else:
        ax2.text(0.5, 0.5, 'No community data', ha='center', va='center',
                 transform=ax2.transAxes, fontsize=10)
        ax2.set_axis_off()

    # Panel 3: Transition heatmap (bottom-left)
    ax3 = fig.add_subplot(2, 2, 3)
    if transition_matrix is not None:
        k = len(transition_matrix)
        sns.heatmap(transition_matrix, annot=True, fmt='.2f', cmap='YlOrRd',
                    ax=ax3, vmin=0, vmax=1, linewidths=0.5, linecolor='white',
                    xticklabels=[f'C{i}' for i in range(k)],
                    yticklabels=[f'C{i}' for i in range(k)])
        ax3.set_title('Transition Probabilities', fontsize=10, fontweight='bold')
        ax3.set_xlabel('To Cluster', fontsize=8)
        ax3.set_ylabel('From Cluster', fontsize=8)
    else:
        ax3.text(0.5, 0.5, 'No transition data', ha='center', va='center',
                 transform=ax3.transAxes, fontsize=10)
        ax3.set_axis_off()

    # Panel 4: Degree centrality distribution (bottom-right)
    ax4 = fig.add_subplot(2, 2, 4)
    dc = network_result.degree_centrality
    unique_labels = np.unique(labels)
    dc_max = dc.max() if dc.max() > 0 else 1.0
    bins = np.linspace(0, dc_max * 1.05, 30)
    for cl in unique_labels:
        mask = labels == cl
        ax4.hist(dc[mask], bins=bins, alpha=0.5, label=f'C{cl}')
    ax4.set_xlim(0, dc_max * 1.05)
    ax4.set_xlabel('Degree Centrality', fontsize=9)
    ax4.set_ylabel('Count', fontsize=9)
    ax4.set_title('Degree Centrality Distribution', fontsize=10, fontweight='bold')
    if len(unique_labels) <= 15:
        ax4.legend(fontsize=6, ncol=2)
    ax4.grid(alpha=0.2)

    # Add modularity info
    mod_text = f'Modularity: {network_result.modularity:.3f}' if network_result.modularity else ''
    method_text = f'Method: {network_result.community_method}'
    n_comm = len(np.unique(network_result.communities)) if network_result.communities is not None else '?'
    info = f'{method_text} | Communities: {n_comm} | {mod_text}'
    fig.suptitle(f'{target_name} - Network Analysis\n{info}',
                 fontsize=13, fontweight='bold')

    fig.tight_layout(rect=[0, 0, 1, 0.94])

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir / f'{target_name}_network_report.png',
                    dpi=150, bbox_inches='tight')

    return fig


def create_pretty_community_graph(
    graph: Any,
    communities: np.ndarray,
    structure_names: List[str],
    grouping_metadata: Optional[Dict[str, Any]] = None,
    grouping_palette: Optional[Dict[Any, str]] = None,
    group_centers: Optional[Dict[Any, Tuple[float, float]]] = None,
    top_k: Optional[int] = None,
    title: str = '',
    figsize: Tuple[float, float] = (15, 12),
) -> plt.Figure:
    """Community-level "pretty" graph: nodes = communities, grouped by metadata.

    Each community-node is drawn as outer-ring (dominant grouping label color)
    + inner pie (predictor mix). Edges = number of inter-community kNN edges.

    Args:
      graph: networkx.Graph of structures (integer node ids), e.g. the kNN
        graph produced by build_structure_graph.
      communities: 1-D array of community labels, length N (matching graph nodes
        0..N-1, same order as structure_names).
      structure_names: structure names in the same order as `communities`.
      grouping_metadata: optional dict mapping structure_name -> group_label.
        If provided, communities are arranged on the canvas grouped by their
        dominant group_label. If None, falls back to dominant predictor
        (parsed from structure name).
      grouping_palette: dict mapping group_label -> color hex string.
      group_centers: dict mapping group_label -> (x,y) canvas anchor. If None,
        groups are auto-placed on a circle.
      top_k: if set, show only the top-K largest communities (useful when
        N_communities is large, e.g. dRMSD with 3000+ micro-communities).
      title: figure title.
      figsize: matplotlib figure size.

    Returns:
      matplotlib.figure.Figure.
    """
    import networkx as nx
    import re
    from collections import Counter
    from matplotlib.patches import Wedge, Circle, PathPatch
    from matplotlib.path import Path as MplPath
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    # ---- Parse predictor from structure names (af3/bt2/of3/pt2/...) ----
    _PRED_RE = re.compile(r'^([a-z][a-z0-9]+)_seed\d+_sample\d+\.pdb$')
    def _predictor(name):
        m = _PRED_RE.match(name)
        return m.group(1) if m else 'other'

    pred_pal_default = {
        'af3': '#1d4ed8', 'bt2': '#b91c1c', 'of3': '#ea580c',
        'pt2': '#15803d', 'other': '#6b7280',
    }

    # ---- Build group_label per structure (for grouping communities) ----
    if grouping_metadata is None:
        # Fall back: group by dominant predictor
        group_for_name = {n: _predictor(n) for n in structure_names}
    else:
        group_for_name = {n: grouping_metadata.get(n, 'other')
                          for n in structure_names}

    # ---- Build community → metadata table ----
    name_to_comm = {n: int(c) for n, c in zip(structure_names, communities)}
    comm_sizes = Counter(name_to_comm.values())

    if top_k and len(comm_sizes) > top_k:
        kept = set(sorted(comm_sizes, key=comm_sizes.get, reverse=True)[:top_k])
    else:
        kept = set(comm_sizes)

    comm_meta = {}
    for c in kept:
        members = [n for n in structure_names if name_to_comm[n] == c]
        pred_mix = Counter(_predictor(n) for n in members)
        group_mix = Counter(group_for_name[n] for n in members)
        dom_group = group_mix.most_common(1)[0][0]
        pure = group_mix.most_common(1)[0][1] / len(members) * 100
        comm_meta[c] = dict(
            size=len(members), pred_mix=dict(pred_mix),
            group_mix=dict(group_mix), dom_group=dom_group, pure_pct=pure,
        )

    # ---- Inter-community kNN edges ----
    # graph nodes are integer indices into structure_names ordering
    idx_to_comm = {i: name_to_comm[n] for i, n in enumerate(structure_names)}
    inter = Counter()
    for u, v in graph.edges():
        cu = idx_to_comm.get(int(u))
        cv = idx_to_comm.get(int(v))
        if cu is None or cv is None or cu == cv: continue
        if cu not in kept or cv not in kept: continue
        inter[(min(cu, cv), max(cu, cv))] += 1
    # adaptive threshold: keep top-100 inter-comm pairs
    max_inter_pairs = 100
    top_inter = inter.most_common(max_inter_pairs)
    edges_w = {k: w for k, w in top_inter if w > 0}

    # ---- Layout: groups in a circle, communities around their group center ----
    groups_used = sorted(set(m['dom_group'] for m in comm_meta.values()))
    if not grouping_palette:
        # If grouping is predictor-based, reuse predictor palette
        if all(g in pred_pal_default for g in groups_used):
            grouping_palette = {g: pred_pal_default[g] for g in groups_used}
        else:
            cmap = plt.get_cmap('tab10')
            grouping_palette = {g: cmap(i % 10) for i, g in enumerate(groups_used)}

    if not group_centers:
        n_groups = len(groups_used)
        group_centers = {}
        for i, g in enumerate(groups_used):
            theta = 2 * np.pi * i / n_groups + np.pi / 2
            R = 2.4 if n_groups > 1 else 0.0
            group_centers[g] = (R * np.cos(theta), R * np.sin(theta))

    # Per-group, place communities in a small ring around the anchor
    by_group = {}
    for c, m in comm_meta.items():
        by_group.setdefault(m['dom_group'], []).append((c, m['size']))
    pos = {}
    for g, items in by_group.items():
        items.sort(key=lambda x: -x[1])
        cx, cy = group_centers[g]
        if len(items) == 1:
            pos[items[0][0]] = (cx, cy)
        else:
            R = 0.55 + 0.04 * np.sqrt(len(items))
            R = min(R, 1.0)
            for i, (c, _) in enumerate(items):
                a = 2 * np.pi * i / len(items)
                pos[c] = (cx + R * np.cos(a), cy + R * np.sin(a))

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor('#fafafa')

    # Group region shading
    for g in groups_used:
        gx, gy = group_centers[g]
        ax.add_patch(Circle((gx, gy), 1.3, facecolor=grouping_palette[g],
                            alpha=0.08, edgecolor=grouping_palette[g],
                            lw=2, ls='--', zorder=0))
        ax.text(gx, gy + 1.42, f'{g}  ({len(by_group[g])} comm.)',
                fontsize=11, fontweight='bold', ha='center', va='center',
                color=grouping_palette[g])

    # Edges (curved)
    max_w = max(edges_w.values()) if edges_w else 1
    for (u, v), w in edges_w.items():
        if u not in pos or v not in pos: continue
        alpha = 0.30 + 0.55 * (w / max_w)
        lw = 0.8 + 4.0 * (w / max_w)
        x0, y0 = pos[u]; x1, y1 = pos[v]
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        dx, dy = x1 - x0, y1 - y0
        nrm = (dx**2 + dy**2)**0.5 + 1e-9
        px, py = -dy / nrm, dx / nrm
        cxp, cyp = mx + 0.12 * px, my + 0.12 * py
        g_u = comm_meta[u]['dom_group']; g_v = comm_meta[v]['dom_group']
        color = grouping_palette[g_u] if g_u == g_v else '#64748b'
        path = MplPath([(x0, y0), (cxp, cyp), (x1, y1)],
                       [MplPath.MOVETO, MplPath.CURVE3, MplPath.CURVE3])
        ax.add_patch(PathPatch(path, fill=False, edgecolor=color,
                               lw=lw, alpha=alpha, zorder=2))
        if w >= max_w * 0.35:
            ax.text(cxp, cyp, f'{w:,}', fontsize=7, ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                              edgecolor=color, lw=0.6, alpha=0.92), zorder=4)

    # Nodes
    for c, m in comm_meta.items():
        x, y = pos[c]; sz = m['size']
        r_outer = 0.06 + 0.0035 * np.sqrt(sz / 5)
        r_outer = min(r_outer, 0.20)
        r_inner = r_outer * 0.74
        ax.add_patch(Circle((x, y), r_outer,
                            facecolor=grouping_palette[m['dom_group']],
                            edgecolor='black', lw=1.5, zorder=5))
        # predictor pie
        pm = m['pred_mix']; total = sum(pm.values()) or 1
        a0 = 90
        for p in sorted(pred_pal_default, key=lambda x: -pm.get(x, 0)):
            cnt = pm.get(p, 0)
            if cnt == 0: continue
            a1 = a0 - 360 * cnt / total
            ax.add_patch(Wedge((x, y), r_inner, a1, a0,
                               facecolor=pred_pal_default[p],
                               edgecolor='white', lw=0.5, zorder=6))
            a0 = a1
        fs = 9 if sz > 800 else (8 if sz > 200 else 7)
        ax.text(x, y, f'L{c}', fontsize=fs, fontweight='bold',
                ha='center', va='center', color='white', zorder=10)
        ax.text(x, y - r_outer - 0.07, f'n={sz:,}\n{m["pure_pct"]:.0f}% pure',
                fontsize=6.5, ha='center', va='top', color='#333', zorder=10,
                bbox=dict(boxstyle='round,pad=0.13', facecolor='white',
                          edgecolor='#aaa', lw=0.4, alpha=0.95))

    # Legends
    ang_leg = [Patch(facecolor=grouping_palette[g], edgecolor='black', label=str(g))
               for g in groups_used]
    pred_keys = sorted(set(k for m in comm_meta.values() for k in m['pred_mix']))
    pred_leg = [Patch(facecolor=pred_pal_default.get(p, '#888'),
                      edgecolor='white', label=p) for p in pred_keys]
    edge_leg = [Line2D([0], [0], color=grouping_palette[groups_used[0]], lw=3,
                       label='same group (within)'),
                Line2D([0], [0], color='#64748b', lw=3,
                       label='group mismatch (between)')]
    grouping_kind = 'predictor' if grouping_metadata is None else 'metadata'
    leg1 = ax.legend(handles=ang_leg, title=f'outer ring = dominant {grouping_kind}',
                     loc='upper left', fontsize=9, frameon=True, facecolor='white')
    leg2 = ax.legend(handles=pred_leg, title='inner pie = predictor mix',
                     loc='upper right', fontsize=9, frameon=True, facecolor='white')
    ax.add_artist(leg1)
    leg3 = ax.legend(handles=edge_leg,
                     title=f'edge = inter-community kNN edges (max {max_w:,})',
                     loc='lower right', fontsize=9, frameon=True, facecolor='white')
    ax.add_artist(leg2)

    ax.set_xlim(-4, 4); ax.set_ylim(-4, 4)
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_color('#ddd')
    sub_title = (f'top {top_k} of {len(comm_sizes)} communities by size'
                 if top_k and len(comm_sizes) > top_k
                 else f'{len(comm_sizes)} communities')
    full_title = title or f'{sub_title};  grouped by dominant {grouping_kind}'
    ax.set_title(full_title, fontsize=13, fontweight='bold', pad=12)

    fig.tight_layout()
    return fig


def _plot_network_graph_on_ax(
    ax, distance_matrix, labels, communities=None, hub_indices=None,
    degree_centrality=None, title='Network',
):
    """Render network graph on a given axis (for multi-panel figures)."""
    import networkx as nx
    from residudes_qa.cluster.core.network import build_structure_graph

    n = len(distance_matrix)
    G = build_structure_graph(distance_matrix)

    # Subsample for large graphs
    if n > 300:
        rng = np.random.RandomState(42)
        keep = set()
        if hub_indices:
            keep.update(hub_indices[:15])
        remaining = 300 - len(keep)
        candidates = [i for i in range(n) if i not in keep]
        keep.update(rng.choice(candidates, size=min(remaining, len(candidates)), replace=False))
        G = G.subgraph(list(keep)).copy()

    # Remove weak edges
    edges_to_remove = [
        (u, v) for u, v, d in G.edges(data=True)
        if d.get('weight', 0) < 0.3
    ]
    G.remove_edges_from(edges_to_remove)

    if G.number_of_nodes() == 0:
        ax.text(0.5, 0.5, 'No nodes', ha='center', va='center', transform=ax.transAxes)
        ax.set_axis_off()
        return

    pos = nx.spring_layout(G, weight='weight', k=1.5 / np.sqrt(max(G.number_of_nodes(), 1)),
                           iterations=50, seed=42)

    node_list = list(G.nodes())
    color_values = [labels[i] for i in node_list]
    cmap = plt.get_cmap('tab20', max(max(color_values) + 1, 2))
    node_colors = [cmap(c % 20) for c in color_values]

    # Draw edges
    for u, v, d in G.edges(data=True):
        w = d.get('weight', 0.5)
        ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], '-',
                color='#cccccc', linewidth=0.3 + w, alpha=0.1 + 0.3 * w, zorder=1)

    # Draw nodes
    hub_set = set(hub_indices) if hub_indices else set()
    sizes = []
    for i, node in enumerate(node_list):
        s = 30
        if degree_centrality is not None:
            dc_max = degree_centrality.max() if degree_centrality.max() > 0 else 1
            s = 20 + 100 * degree_centrality[node] / dc_max
        if node in hub_set:
            s *= 1.5
        sizes.append(s)

    xs = [pos[n][0] for n in node_list]
    ys = [pos[n][1] for n in node_list]
    ax.scatter(xs, ys, c=node_colors, s=sizes, edgecolors='white', linewidths=0.3, zorder=3)

    # Highlight hubs
    for node in node_list:
        if node in hub_set:
            ax.scatter(pos[node][0], pos[node][1], facecolors='none',
                       edgecolors='black', s=sizes[node_list.index(node)] * 1.3,
                       linewidths=1.5, zorder=4)

    ax.set_title(title, fontsize=9, fontweight='bold')
    ax.set_axis_off()


def _plot_community_comparison_on_ax(ax, labels, communities):
    """Render community comparison on a given axis."""
    import pandas as pd
    import seaborn as sns

    if communities is None:
        ax.text(0.5, 0.5, 'No community data', ha='center', va='center',
                transform=ax.transAxes, fontsize=10)
        ax.set_axis_off()
        return

    ct = pd.crosstab(
        pd.Series(labels, name='Cluster'),
        pd.Series(communities, name='Community'),
    )
    ct_norm = ct.div(ct.sum(axis=1), axis=0)

    sns.heatmap(ct_norm, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax,
                cbar_kws={'label': 'Fraction'}, linewidths=0.5, linecolor='white')
    ax.set_title('Cluster vs Community (normalized)', fontsize=9, fontweight='bold')
    ax.tick_params(labelsize=7)


def _plot_resolution_scan_on_ax(ax, scan_results):
    """Render resolution scan on a given axis."""
    resolutions = scan_results['resolutions']
    n_comms = scan_results['n_communities']
    modularity = scan_results['modularity']

    ax.plot(resolutions, n_comms, 'o-', color='#2196F3', lw=1.5, ms=4, label='N communities')
    ax.set_xlabel('Resolution', fontsize=8)
    ax.set_ylabel('N Communities', fontsize=8, color='#2196F3')
    ax.set_xscale('log')
    ax.tick_params(axis='y', labelcolor='#2196F3', labelsize=7)

    ax2 = ax.twinx()
    ax2.plot(resolutions, modularity, 's-', color='#FF9800', lw=1.5, ms=4, label='Modularity')
    ax2.set_ylabel('Modularity', fontsize=8, color='#FF9800')
    ax2.tick_params(axis='y', labelcolor='#FF9800', labelsize=7)

    suggested = scan_results.get('suggested_resolution')
    if suggested:
        ax.axvline(x=suggested, color='red', ls='--', lw=1, alpha=0.6)

    ax.set_title('Resolution Scan', fontsize=9, fontweight='bold')
    ax.grid(alpha=0.2)
