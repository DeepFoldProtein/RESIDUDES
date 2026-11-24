# SPDX-License-Identifier: Apache-2.0
"""Boltz-2 specific token (diffusion) transformer.

Differs from OF3/Protenix in two ways:

1. Pair bias is passed in pre-computed (per-layer slice of a [B, N, N, L, D//L]
   tensor projected once in score_model). This module's blocks accept
   ``pair_logits`` directly — they don't compute a Linear(z) internally.

2. ConditionedTransitionBlock has THREE linears, not the standard SwiGLU:
   - swish_gate: 768 -> 3072 (=2*1536), split via SwiGLU into 1536
   - a_to_b: 768 -> 1536  (parallel projection)
   - b_to_a: 1536 -> 768  (output)
   With AdaLN-zero gate ``output_projection_linear`` (bias init -2.0).

Reference: boltz/model/modules/transformersv2.py:DiffusionTransformerLayer.
"""

from __future__ import annotations

from dataclasses import dataclass

import haiku as hk
import jax
import jax.numpy as jnp

from residudes_engine_jax.haiku.adaln import (
    adaptive_layernorm,
    self_attention,
)
from residudes_engine_jax.haiku.primitives import LayerNorm, Linear


@dataclass(frozen=True)
class BoltzTokenTransformerVariant:
    num_layers: int = 24
    c_a: int = 768
    c_s: int = 768            # NOTE: Boltz default — single-cond dim equals token dim
    num_head: int = 16
    transition_factor: int = 2  # dim_inner = c_a * 2 = 1536


def _adaln(x, s, *, name: str):
    """Boltz AdaLN matches OF3's adaptive_layernorm structure."""
    return adaptive_layernorm(x, s, name=name)


def _adaln_zero(s, *, dim: int, name: str):
    """sigmoid(Linear(s, bias_init=-2.0)) — output gate factor."""
    return jax.nn.sigmoid(
        Linear(dim, use_bias=True, bias_init=-2.0, initializer="zeros",
               name=name)(s)
    )


def _conditioned_transition(a, s, *, factor: int, dim: int, name: str):
    """Boltz transition block: swish_gate * a_to_b * (sigmoid(s)) * b_to_a."""
    inner = dim * factor
    a_n = _adaln(a, s, name=f"{name}_adaln_")
    # swish_gate: 1 linear with 2*inner output, swiglu-split.
    # NOTE: Boltz SwiGLU does silu(second_half) * first_half (NOT the more
    # common silu(first) * second). See boltz/model/modules/utils.py.
    sg = Linear(inner * 2, use_bias=False, name=f"{name}_swish_gate")(a_n)
    first, second = jnp.split(sg, 2, axis=-1)
    swish_v = jax.nn.swish(second) * first   # [..., inner]
    # a_to_b: parallel projection
    ab = Linear(inner, use_bias=False, name=f"{name}_a_to_b")(a_n)
    b = swish_v * ab                # [..., inner]
    # b_to_a: output linear
    out = Linear(dim, use_bias=False, name=f"{name}_b_to_a")(b)
    # AdaLN-zero gate from s
    gate = _adaln_zero(s, dim=dim, name=f"{name}_output_projection")
    return gate * out


class BoltzTokenTransformerBlock(hk.Module):
    """One Boltz diffusion transformer layer."""

    def __init__(self, *, variant: BoltzTokenTransformerVariant, name: str | None = None):
        super().__init__(name=name)
        self.variant = variant

    def __call__(self, a, s, pair_logits, mask):
        """
        Args:
          a: [N, c_a] activations
          s: [N, c_s] single conditioning
          pair_logits: [num_head, N, N] per-layer pair-bias logits (already projected)
          mask: [N] token mask
        Returns:
          a: [N, c_a] post-block activation
        """
        v = self.variant
        # Pre-attention: AdaLN on a using s.
        b = _adaln(a, s, name="adaln_")
        # Self-attention with pair_logits (but no internal pair-bias projection
        # — Boltz score_model precomputes it).
        b = self_attention(
            b, mask, pair_logits,
            num_head=v.num_head,
            single_cond=None,
            pre_norm=False,          # Boltz: AdaLN already applied externally
            name="pair_bias_attn_",
        )
        # AdaLN-zero gate (output_projection)
        b = _adaln_zero(s, dim=v.c_a, name="output_projection") * b
        a = a + b
        # Conditioned transition (3-linear gated FFW with AdaLN + AdaLN-zero)
        a = a + _conditioned_transition(
            a, s, factor=v.transition_factor, dim=v.c_a, name="transition",
        )
        return a


class BoltzTokenTransformer(hk.Module):
    """Stack of Boltz token transformer layers.

    Forward signature:
      __call__(a, s, bias, mask) -> a'

    where ``bias`` is the FULL pair-bias tensor pre-projected to
    [N, N, num_layers, num_head_per_layer]. We slice it per-layer.
    """

    def __init__(self, *, variant: BoltzTokenTransformerVariant, name: str = "token_transformer"):
        super().__init__(name=name)
        self.variant = variant

    def __call__(self, a, s, bias, mask):
        # bias: [N, N, num_layers * num_head] OR [N, N, num_layers, num_head]
        # If 3-D, reshape to 4-D split.
        v = self.variant
        L = v.num_layers
        if bias.ndim == 3:
            # [N, N, L*H] -> [N, N, L, H]
            bias = bias.reshape(*bias.shape[:-1], L, bias.shape[-1] // L)
        elif bias.ndim != 4:
            raise ValueError(f"Unexpected bias rank: {bias.ndim}")

        for i in range(L):
            # Per-layer pair_logits: extract slice [N, N, H], permute to [H, N, N]
            pl = jnp.transpose(bias[..., i, :], (2, 0, 1))
            a = BoltzTokenTransformerBlock(
                variant=v, name=f"layer_{i}",
            )(a, s, pl, mask)
        return a
