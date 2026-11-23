# SPDX-License-Identifier: Apache-2.0
"""Pairformer block + stack (Haiku).

Reference:
  alphafold3/src/alphafold3/model/network/modules.py::PairFormerIteration

OF3 / Protenix / Boltz-2 all share this layout. The differences are:
  - num_blocks
  - channel widths (token_s, token_z)
  - num_heads (pair attn, single attn)

These are passed as ``PairformerVariant`` config rather than baked-in.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import haiku as hk
import jax
import jax.numpy as jnp

from lantern_engine_jax.haiku.adaln import self_attention, transition_block
from lantern_engine_jax.haiku.primitives import LayerNorm, Linear
from lantern_engine_jax.haiku.transition import TransitionBlock
from lantern_engine_jax.haiku.triangle import TriangleAttention, TriangleMultiplication


@dataclass(frozen=True)
class PairformerVariant:
    """Per-model pairformer config (channel widths, head counts, depth)."""
    num_blocks: int
    token_s: int          # single track channels (e.g. OF3=384)
    token_z: int          # pair track channels (OF3=128, Protenix-v2=256)
    num_head_pair: int    # heads for triangle attention (OF3/v1=4, v2=8)
    num_head_single: int  # heads for single self-attention (OF3=16)
    transition_factor: int = 4
    # Explicit per-head dim for pair-track attention. ``None`` = derive from
    # ``token_z // num_head_pair``. Required for boltz template pairformer
    # where head_dim != token_z / num_head (token_z=64, num_head=4, head_dim=32).
    head_dim_pair: int | None = None
    # ``None`` = auto (cudnn on GPU, XLA otherwise). Override per-call.
    triangle_attn_impl: Literal["cudnn", "xla"] | None = None

    @classmethod
    def openfold3(cls, num_blocks: int = 48) -> "PairformerVariant":
        return cls(num_blocks=num_blocks, token_s=384, token_z=128,
                   num_head_pair=4, num_head_single=16)

    @classmethod
    def protenix_v1(cls, num_blocks: int = 48) -> "PairformerVariant":
        return cls(num_blocks=num_blocks, token_s=384, token_z=128,
                   num_head_pair=4, num_head_single=16)

    @classmethod
    def protenix_v2(cls, num_blocks: int = 48) -> "PairformerVariant":
        return cls(num_blocks=num_blocks, token_s=384, token_z=256,
                   num_head_pair=8, num_head_single=16)

    @classmethod
    def boltz2(cls, num_blocks: int = 48) -> "PairformerVariant":
        # Boltz-2 structure pairformer — same widths as OF3.
        return cls(num_blocks=num_blocks, token_s=384, token_z=128,
                   num_head_pair=4, num_head_single=16)

    @classmethod
    def boltz2_confidence(cls, num_blocks: int = 4) -> "PairformerVariant":
        # Shallower confidence pairformer.
        return cls(num_blocks=num_blocks, token_s=384, token_z=128,
                   num_head_pair=4, num_head_single=16)

    @classmethod
    def boltz2_template(cls, num_blocks: int = 2) -> "PairformerVariant":
        # Template module pairformer (no-seq variant). Smaller token_z (64)
        # but same head config as main; head_dim explicit because token_z //
        # num_head = 16 != PT's 32.
        return cls(num_blocks=num_blocks, token_s=384, token_z=64,
                   num_head_pair=4, num_head_single=16, head_dim_pair=32)

    @classmethod
    def boltz2_msa(cls, num_blocks: int = 4) -> "PairformerVariant":
        # MSA module pairformer_layer (no-seq variant), one per MSA layer.
        # Same channel widths as main pairformer (token_z=128).
        return cls(num_blocks=num_blocks, token_s=384, token_z=128,
                   num_head_pair=4, num_head_single=16)


class PairformerBlock(hk.Module):
    """One iteration of pairformer: pair stack + (optional) single track."""

    def __init__(
        self,
        *,
        variant: PairformerVariant,
        with_single: bool = True,
        name: str | None = None,
    ):
        super().__init__(name=name)
        self.variant = variant
        self.with_single = with_single

    def __call__(
        self,
        z: jax.Array,
        pair_mask: jax.Array,
        s: jax.Array | None = None,
        single_mask: jax.Array | None = None,
    ):
        v = self.variant

        # Pair stack
        z = z + TriangleMultiplication(
            equation="ikc,jkc->ijc",
            name="triangle_multiplication_outgoing",
        )(z, pair_mask)
        z = z + TriangleMultiplication(
            equation="kic,kjc->ijc",
            name="triangle_multiplication_incoming",
        )(z, pair_mask)
        z = z + TriangleAttention(
            num_head=v.num_head_pair,
            head_dim=v.head_dim_pair,
            transpose=False,
            impl=v.triangle_attn_impl,
            name="pair_attention1",
        )(z, pair_mask)
        z = z + TriangleAttention(
            num_head=v.num_head_pair,
            head_dim=v.head_dim_pair,
            transpose=True,
            impl=v.triangle_attn_impl,
            name="pair_attention2",
        )(z, pair_mask)
        z = z + TransitionBlock(
            num_intermediate_factor=v.transition_factor,
            name="pair_transition",
        )(z)

        if not self.with_single:
            return z

        assert s is not None and single_mask is not None
        # Single track: pair-biased self-attention + transition.
        # PT name: attn_pair_bias.linear_z (no bias).
        pair_logits = Linear(
            v.num_head_single, use_bias=False,
            name="single_pair_logits_projection",
        )(LayerNorm(name="single_pair_logits_norm")(z))
        pair_logits = jnp.transpose(pair_logits, (2, 0, 1))  # [H, N, N]

        s = s + self_attention(
            s, single_mask, pair_logits,
            num_head=v.num_head_single,
            name="single_attention_",
        )
        s = s + transition_block(
            s,
            num_intermediate_factor=v.transition_factor,
            name="single_transition_",
        )
        return z, s


class PairformerStack(hk.Module):
    """N PairformerBlocks applied in sequence.

    Forward signature:
      __call__(z, pair_mask, s=None, single_mask=None) -> (s', z') if with_single
                                                          else z'

    The stack uses ``hk.remat`` (gradient-checkpointing) on each block. We
    measured this faster than ``remat=False`` even at inference on a 4090 —
    XLA produces tighter per-block fusion and avoids materializing all 48
    blocks' intermediate activations as a single huge graph. Counter-intuitive
    but consistent across N=76 and N=644.
    """

    def __init__(
        self,
        *,
        variant: PairformerVariant,
        with_single: bool = True,
        remat: bool = True,
        use_scan: bool = False,
        name: str = "pairformer",
    ):
        super().__init__(name=name)
        self.variant = variant
        self.with_single = with_single
        self.remat = remat
        self.use_scan = use_scan

    def __call__(
        self,
        z: jax.Array,
        pair_mask: jax.Array,
        s: jax.Array | None = None,
        single_mask: jax.Array | None = None,
    ):
        if self.use_scan:
            return self._scan_call(z, pair_mask, s, single_mask)

        for i in range(self.variant.num_blocks):
            block = PairformerBlock(
                variant=self.variant,
                with_single=self.with_single,
                name=f"pair_former_block_{i}",
            )
            # Wrap with hk.remat: forces XLA to discard intermediate
            # activations between blocks, reducing memory by ~num_blocks×.
            # In inference mode the recompute on backward is irrelevant.
            block_fn = hk.remat(block) if self.remat else block
            if self.with_single:
                z, s = block_fn(z, pair_mask, s, single_mask)
            else:
                z = block_fn(z, pair_mask)

        if self.with_single:
            return s, z
        return z

    @hk.transparent
    def _scan_call(
        self,
        z: jax.Array,
        pair_mask: jax.Array,
        s: jax.Array | None = None,
        single_mask: jax.Array | None = None,
    ):
        """layer_stack version: 48 blocks share one compiled body via
        ``jax.lax.scan``. Compile time drops from ~25 s to ~1-2 s and runtime
        per pass cuts dispatch overhead. Block parameters live under
        ``pair_former_block/...`` with a leading axis of ``num_blocks``.
        """
        variant = self.variant
        with_single = self.with_single

        if with_single:
            def block_fn(carry):
                zz, ss = carry
                block = PairformerBlock(
                    variant=variant, with_single=True,
                    name="pair_former_block",
                )
                zz, ss = block(zz, pair_mask, ss, single_mask)
                return (zz, ss)

            stack = hk.experimental.layer_stack(self.variant.num_blocks)(block_fn)
            z, s = stack((z, s))
            return s, z

        def block_fn_pair(carry):
            zz = carry
            block = PairformerBlock(
                variant=variant, with_single=False,
                name="pair_former_block",
            )
            return block(zz, pair_mask)

        stack = hk.experimental.layer_stack(self.variant.num_blocks)(block_fn_pair)
        return stack(z)
