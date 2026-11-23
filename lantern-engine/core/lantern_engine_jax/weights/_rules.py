# SPDX-License-Identifier: Apache-2.0
"""Rule primitives for PT state_dict -> Haiku pytree conversion.

Two rule types:
  - SimpleRule: 1 PT key -> 1 Haiku target, optional reshape/transpose
  - CompositeRule: N PT keys -> 1 Haiku target, with N-arg transform

Both share a common `apply()` interface returning ``(target_path, ndarray)``
or raising KeyError if any source PT key is missing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np


def linear_w_to_haiku(t: np.ndarray) -> np.ndarray:
    """nn.Linear.weight [out, in] -> Haiku Linear.w [in, out]."""
    return np.ascontiguousarray(t.T)


def linear_w_to_haiku_multihead(num_head: int, head_dim: int) -> Callable:
    """nn.Linear.weight [num_head*head_dim, in] -> Haiku Linear.w [in, num_head, head_dim]."""
    def fn(t: np.ndarray) -> np.ndarray:
        in_dim = t.shape[1]
        return np.ascontiguousarray(t.T.reshape(in_dim, num_head, head_dim))
    return fn


def linear_b_to_haiku_multihead(num_head: int, head_dim: int) -> Callable:
    """nn.Linear.bias [num_head*head_dim] -> Haiku Linear.b [num_head, head_dim]."""
    def fn(t: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(t.reshape(num_head, head_dim))
    return fn


def concat_linear_w(t_a: np.ndarray, t_b: np.ndarray) -> np.ndarray:
    """Two PT Linear weights [out, in] -> stacked Haiku weight [in, 2*out].

    For SwiGLU and tri-mul projections that PT splits into two linears.
    Output ordering: [a; b] along the last axis (matches our split direction
    in haiku/transition.py and haiku/triangle.py).
    """
    return np.ascontiguousarray(
        np.concatenate([t_a.T, t_b.T], axis=-1)
    )


def stack_linear_w(t_a: np.ndarray, t_b: np.ndarray) -> np.ndarray:
    """Two PT Linear weights [out, in] -> stacked Haiku weight [in, 2, out].

    Same data as ``concat_linear_w``, but laid out along a new axis=1 — this
    is the layout ``tokamax.gated_linear_unit`` expects for its fused SwiGLU
    kernel. ``weights[:, 0, :]`` is the gated branch (gets activation),
    ``weights[:, 1, :]`` is the value branch.

    PT order: ``linear_no_bias_a`` (or ``_a1``) is the first arg → ends up at
    ``weights[:, 0, :]``, matching ``swish(linear_a(x)) * linear_b(x)`` in
    Protenix / OF3 / Boltz checkpoints.
    """
    return np.ascontiguousarray(
        np.stack([t_a.T, t_b.T], axis=1)
    )


def identity(t: np.ndarray) -> np.ndarray:
    return t


@dataclass
class SimpleRule:
    pt_key: str
    hk_target: str  # "scope/leaf"
    transform: Callable[[np.ndarray], np.ndarray] = identity

    def apply(self, sd: dict) -> tuple[str, np.ndarray]:
        if self.pt_key not in sd:
            raise KeyError(self.pt_key)
        arr = _to_numpy(sd[self.pt_key])
        return self.hk_target, self.transform(arr)


@dataclass
class CompositeRule:
    pt_keys: Sequence[str]
    hk_target: str
    transform: Callable[..., np.ndarray]

    def apply(self, sd: dict) -> tuple[str, np.ndarray]:
        for k in self.pt_keys:
            if k not in sd:
                raise KeyError(k)
        arrs = [_to_numpy(sd[k]) for k in self.pt_keys]
        return self.hk_target, self.transform(*arrs)


Rule = SimpleRule | CompositeRule


def _to_numpy(t) -> np.ndarray:
    if hasattr(t, "detach"):
        return t.detach().cpu().numpy()
    return np.asarray(t)


def apply_rules(rules: Sequence[Rule], pt_state: dict) -> dict[str, dict[str, np.ndarray]]:
    """Apply all rules, return Haiku-style nested params: {scope: {leaf: arr}}.

    Returns a dict suitable for `haiku.transform.init`'s output type.
    Raises if any rule's source key is missing.
    """
    out: dict[str, dict[str, np.ndarray]] = {}
    for rule in rules:
        target, arr = rule.apply(pt_state)
        scope, _, leaf = target.rpartition("/")
        out.setdefault(scope, {})[leaf] = arr
    return out


def consumed_keys(rules: Sequence[Rule]) -> set[str]:
    """Return the set of PT keys that the rule list will consume."""
    out: set[str] = set()
    for rule in rules:
        if isinstance(rule, SimpleRule):
            out.add(rule.pt_key)
        else:
            out.update(rule.pt_keys)
    return out
