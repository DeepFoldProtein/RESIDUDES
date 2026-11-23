# SPDX-License-Identifier: Apache-2.0
"""Diffusion token transformer (Haiku).

Reference:
  alphafold3/src/alphafold3/model/network/diffusion_transformer.py
  models/openfold-3/openfold3/core/model/layers/diffusion_transformer.py

Each block:
  a += AttentionPairBias(a, s, z, mask)     # self-attn with pair-bias + AdaLN
  a += ConditionedTransition(a, s, mask)    # SwiGLU FFW with AdaLN-zero

The TRT compat layer handles per-multiplicity microbatching (max=5) — the
JAX core just runs one forward pass over [n_token, c_token] inputs and the
PyTorch wrapper loops + concatenates samples.

We diverge from AF3's ``Transformer`` class in one place: AF3 groups blocks
into super_blocks of 4 to amortize the pair-logits projection. The OF3 /
Protenix / Boltz-2 PyTorch implementations all project per-block, so we do
the same — keeps the weight converter trivial (one set of pair_logits
params per block).
"""

from __future__ import annotations

from dataclasses import dataclass

import haiku as hk
import jax
import jax.numpy as jnp

from lantern_engine_jax.haiku.adaln import (
    adaptive_layernorm,
    adaptive_zero_init,
    self_attention,
    transition_block,
)
from lantern_engine_jax.haiku.primitives import LayerNorm, Linear


@dataclass(frozen=True)
class TokenTransformerVariant:
    """Per-model token-transformer config."""
    num_blocks: int        # OF3 / Protenix v1/v2 / Boltz-2 default: 24
    c_a: int               # token (atom-conditioned) channels (typically 768)
    c_s: int               # single conditioning channels (typically 384)
    c_z: int               # pair conditioning channels (OF3=128, Protenix-v2=256)
    num_head: int = 16
    transition_factor: int = 2

    @classmethod
    def openfold3(cls, num_blocks: int = 24) -> "TokenTransformerVariant":
        return cls(num_blocks=num_blocks, c_a=768, c_s=384, c_z=128)

    @classmethod
    def protenix_v1(cls, num_blocks: int = 24) -> "TokenTransformerVariant":
        return cls(num_blocks=num_blocks, c_a=768, c_s=384, c_z=128)

    @classmethod
    def protenix_v2(cls, num_blocks: int = 24) -> "TokenTransformerVariant":
        return cls(num_blocks=num_blocks, c_a=768, c_s=384, c_z=256)

    @classmethod
    def boltz2(cls, num_blocks: int = 24) -> "TokenTransformerVariant":
        return cls(num_blocks=num_blocks, c_a=768, c_s=384, c_z=128)


def _attention_pair_bias(
    a: jax.Array,
    s: jax.Array,
    z: jax.Array,
    mask: jax.Array,
    *,
    num_head: int,
    name: str,
) -> jax.Array:
    """Self-attention on a, biased by per-head logits projected from z."""
    z_norm = LayerNorm(name=f"{name}pair_norm", create_offset=False)(z)
    pair_logits = Linear(
        num_head, use_bias=False, name=f"{name}pair_bias_projection"
    )(z_norm)
    pair_logits = jnp.transpose(pair_logits, (2, 0, 1))  # [H, N, N]
    return self_attention(
        a, mask, pair_logits, num_head=num_head, single_cond=s,
        name=name,
    )


def _conditioned_transition(
    a: jax.Array,
    s: jax.Array,
    *,
    num_intermediate_factor: int,
    name: str,
) -> jax.Array:
    """SwiGLU FFW with AdaLN modulation from s."""
    return transition_block(
        a, num_intermediate_factor=num_intermediate_factor,
        single_cond=s, name=name,
    )


class TokenTransformerBlock(hk.Module):
    """One block: AttentionPairBias + ConditionedTransition."""

    def __init__(
        self,
        *,
        variant: TokenTransformerVariant,
        name: str | None = None,
    ):
        super().__init__(name=name)
        self.variant = variant

    def __call__(
        self,
        a: jax.Array,
        s: jax.Array,
        z: jax.Array,
        mask: jax.Array,
    ) -> jax.Array:
        v = self.variant
        a = a + _attention_pair_bias(
            a, s, z, mask, num_head=v.num_head, name="attn_",
        )
        a = a + _conditioned_transition(
            a, s, num_intermediate_factor=v.transition_factor, name="trans_",
        )
        return a


class TokenTransformer(hk.Module):
    """Stack of N TokenTransformerBlocks.

    Forward signature matches the TRT engine I/O:
      a: [N_token, c_a]   (single-multiplicity slice; the wrapper loops)
      s: [N_token, c_s]
      z: [N_token, N_token, c_z]
      mask: [N_token]
    """

    def __init__(
        self,
        *,
        variant: TokenTransformerVariant,
        use_scan: bool = False,
        name: str = "token_transformer",
    ):
        super().__init__(name=name)
        self.variant = variant
        self.use_scan = use_scan

    def __call__(
        self,
        a: jax.Array,
        s: jax.Array,
        z: jax.Array,
        mask: jax.Array,
    ) -> jax.Array:
        if self.use_scan:
            return self._scan_call(a, s, z, mask)
        for i in range(self.variant.num_blocks):
            a = TokenTransformerBlock(
                variant=self.variant, name=f"block_{i}",
            )(a, s, z, mask)
        return a

    @hk.transparent
    def _scan_call(
        self,
        a: jax.Array,
        s: jax.Array,
        z: jax.Array,
        mask: jax.Array,
    ) -> jax.Array:
        variant = self.variant
        def block_fn(carry):
            return TokenTransformerBlock(variant=variant, name="block")(
                carry, s, z, mask)
        stack = hk.experimental.layer_stack(self.variant.num_blocks)(block_fn)
        return stack(a)
