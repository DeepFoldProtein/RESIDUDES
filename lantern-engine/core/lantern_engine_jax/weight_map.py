# SPDX-License-Identifier: Apache-2.0
"""PyTorch state_dict <-> Haiku param tree conversion.

Haiku params are nested dicts: ``params[module_path][param_name]`` where
``module_path`` is the slash-separated Haiku scope (e.g.
``'pair_former_block_0/triangle_multiplication_outgoing'``) and
``param_name`` is one of ``'w'``, ``'b'``, ``'scale'``, ``'offset'``.

PyTorch state_dicts use dot-separated paths and nn.Linear-style naming
(``weight``, ``bias``). Each model (OF3 / Protenix / Boltz) has slightly
different naming, but the per-block layout is identical to AlphaFold3.

This module provides the rename + reshape primitives. Per-model rule tables
live in ``lantern_engine_jax.weights.{of3,protenix,boltz}``.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


@dataclass
class WeightRule:
    """One PyTorch -> Haiku rename rule.

    pt_pattern: regex matched against the PT key. Capture groups are
        substituted into hk_template with ``{0}``, ``{1}`` etc.
    hk_template: target Haiku scope path. End with ``/<param_name>``.
    transform: optional fn that reshapes / transposes the tensor.
        nn.Linear stores [out, in]; Haiku stores [in, out]. Default xform
        for Linear weights is transpose.
    """
    pt_pattern: str
    hk_template: str
    transform: Callable[[np.ndarray], np.ndarray] | None = None


def linear_w_to_haiku(t: np.ndarray) -> np.ndarray:
    """nn.Linear.weight is [out, in]; Haiku Linear.w is [in, out]."""
    return np.ascontiguousarray(t.T)


def identity(t: np.ndarray) -> np.ndarray:
    return t


def apply_rules(
    pt_state: dict[str, Any],
    rules: Iterable[WeightRule],
    *,
    strict: bool = True,
) -> dict[str, dict[str, np.ndarray]]:
    """Apply rule list to a PT state dict, return Haiku-style nested params.

    If ``strict``, raise on unmatched PT keys. Otherwise skip with a list
    returned via ``apply_rules.unmatched`` global... actually return tuple.
    """
    compiled = [(re.compile(r.pt_pattern), r) for r in rules]
    out: dict[str, dict[str, np.ndarray]] = {}
    matched: set[str] = set()

    for pt_key, pt_val in pt_state.items():
        if not hasattr(pt_val, "shape"):
            continue
        arr = np.asarray(pt_val.detach().cpu()) if hasattr(pt_val, "detach") else np.asarray(pt_val)
        for pat, rule in compiled:
            m = pat.fullmatch(pt_key)
            if not m:
                continue
            target = rule.hk_template.format(*m.groups())
            scope, _, leaf = target.rpartition("/")
            if rule.transform is not None:
                arr = rule.transform(arr)
            out.setdefault(scope, {})[leaf] = arr
            matched.add(pt_key)
            break

    if strict:
        unmatched = [k for k in pt_state if k not in matched and hasattr(pt_state[k], "shape")]
        if unmatched:
            sample = unmatched[:5]
            raise KeyError(
                f"Unmatched PT keys ({len(unmatched)} total). First few: {sample}"
            )
    return out


def save_pytree(params: dict[str, dict[str, np.ndarray]], path: str) -> None:
    """Save Haiku params as a flat .npz with slash-encoded keys."""
    flat: dict[str, np.ndarray] = {}
    for scope, leafs in params.items():
        for leaf, arr in leafs.items():
            flat[f"{scope}/{leaf}"] = arr
    np.savez(path, **flat)


def load_pytree(path: str) -> dict[str, dict[str, np.ndarray]]:
    """Inverse of save_pytree()."""
    data = np.load(path)
    out: dict[str, dict[str, np.ndarray]] = {}
    for k in data.files:
        scope, _, leaf = k.rpartition("/")
        out.setdefault(scope, {})[leaf] = data[k]
    return out
