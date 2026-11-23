"""Centralized color conventions for MassiveClustering visualizations.

Method naming has two flavours across the codebase:

* **method** (raw, as inferred from variant path): ``af3``, ``bt2_jax``,
  ``of3``, ``pt2``
* **method_master** (xgb-canonical): ``af3``, ``b2``, ``of3``, ``v2``

These are *the same prediction tool* — just two naming conventions. We
normalize to the raw form and assign one color per canonical method.
"""
from __future__ import annotations


# Canonical 4 methods (raw naming convention).
METHOD_COLORS: dict[str, str] = {
    "af3":      "#1f77b4",   # blue        AlphaFold-3
    "bt2_jax":  "#ffd700",   # yellow/gold Boltz-2
    "of3":      "#98df8a",   # light green OpenFold-3
    "pt2":      "#d62728",   # red         Protenix v2
}

# method_master / alias → canonical method
METHOD_ALIASES: dict[str, str] = {
    # method_master form used by xgb pipeline
    "b2":     "bt2_jax",
    "v2":     "pt2",
    # version suffixes
    "af3_v2": "af3",
    "bt2":    "bt2_jax",
    "boltz2": "bt2_jax",
    "protenix":   "pt2",
    "protenix2":  "pt2",
    "openfold3":  "of3",
    "alphafold3": "af3",
}

_FALLBACK = "#888888"


def normalize_method(method) -> str:
    """Map any aliasing of a method to its canonical raw form.

    Examples:
        af3        -> af3
        af3_v2     -> af3
        bt2_jax    -> bt2_jax
        b2         -> bt2_jax
        pt2 / v2   -> pt2
    """
    if method is None:
        return "_unknown"
    m = str(method).lower()
    if m in METHOD_COLORS:
        return m
    return METHOD_ALIASES.get(m, m)


def get_method_color(method) -> str:
    """Canonical color for a method, falling back to gray for unknowns."""
    canonical = normalize_method(method)
    return METHOD_COLORS.get(canonical, _FALLBACK)


# Cluster categorical palette.
CLUSTER_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


def get_cluster_color(cluster_id: int) -> str:
    return CLUSTER_PALETTE[int(cluster_id) % len(CLUSTER_PALETTE)]


__all__ = [
    "METHOD_COLORS",
    "METHOD_ALIASES",
    "CLUSTER_PALETTE",
    "normalize_method",
    "get_method_color",
    "get_cluster_color",
]
