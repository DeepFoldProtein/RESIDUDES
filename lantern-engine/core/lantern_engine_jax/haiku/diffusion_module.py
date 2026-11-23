# SPDX-License-Identifier: Apache-2.0
"""Full DiffusionModule (Algorithm 20) in JAX.

Reference:
  models/protenix/protenix/model/modules/diffusion.py::DiffusionModule.f_forward

Pipeline:
  1. DiffusionConditioning(t_hat, relp, s_inputs, s_trunk, z_trunk)
       -> single_s [N_sample, N_token, c_s], pair_z [N_token, N_token, c_z]
  2. atom_attention_encoder(...)
       -> a_token [N_sample, N_token, c_token], q_skip, c_skip, p_skip
  3. a_token = a_token + linear_no_bias_s(layernorm_s(single_s))
  4. diffusion_transformer(a_token, single_s, pair_z, mask=ones)
  5. a_token = layernorm_a(a_token)
  6. r_update = atom_attention_decoder(atom_to_token_idx, a_token, q_skip, c_skip, p_skip)

This is the "raw network" F_θ(c_in * x, c_noise(σ)) of EDM eq (7).
The outer denoise function in the sampler computes
    x_denoised = skip * x_noisy + out_scale * r_update
"""

from __future__ import annotations

from dataclasses import dataclass

import haiku as hk
import jax
import jax.numpy as jnp

from lantern_engine_jax.haiku.atom_attention import (
    AtomAttentionDecoder,
    AtomAttentionDecoderConfig,
    AtomAttentionEncoder,
    AtomAttentionEncoderConfig,
)
from lantern_engine_jax.haiku.diffusion_conditioning import (
    DiffusionConditioning,
    DiffusionConditioningConfig,
    RelPEConfig,
)
from lantern_engine_jax.haiku.primitives import LayerNorm, Linear
from lantern_engine_jax.haiku.token_transformer import (
    TokenTransformer,
    TokenTransformerVariant,
)


@dataclass(frozen=True)
class DiffusionModuleConfig:
    sigma_data: float = 16.0
    c_token: int = 768
    c_atom: int = 128
    c_atompair: int = 16
    c_s: int = 384
    c_z: int = 256              # Protenix v2 (128 for v1)
    c_s_inputs: int = 449
    c_noise_embedding: int = 256
    transformer_num_blocks: int = 24
    transformer_num_head: int = 16

    def conditioning(self) -> DiffusionConditioningConfig:
        return DiffusionConditioningConfig(
            sigma_data=self.sigma_data,
            c_z=self.c_z, c_s=self.c_s, c_s_inputs=self.c_s_inputs,
            c_noise_embedding=self.c_noise_embedding,
            relpe=RelPEConfig(c_z=self.c_z),
        )

    def encoder(self) -> AtomAttentionEncoderConfig:
        return AtomAttentionEncoderConfig(
            c_token=self.c_token, c_atom=self.c_atom,
            c_atompair=self.c_atompair, c_s=self.c_s, c_z=self.c_z,
        )

    def decoder(self) -> AtomAttentionDecoderConfig:
        return AtomAttentionDecoderConfig(
            c_token=self.c_token, c_atom=self.c_atom,
            c_atompair=self.c_atompair,
        )

    def transformer(self) -> TokenTransformerVariant:
        # Protenix v2: c_a=768, c_s=384, c_z=256, num_head=16
        return TokenTransformerVariant(
            num_blocks=self.transformer_num_blocks,
            c_a=self.c_token, c_s=self.c_s, c_z=self.c_z,
            num_head=self.transformer_num_head,
        )


class DiffusionModule(hk.Module):
    """Algorithm 20 full forward."""

    def __init__(self, *, config: DiffusionModuleConfig, name: str = "diffusion_module"):
        super().__init__(name=name)
        self.config = config

    def __call__(
        self,
        # Diffusion sample inputs
        r_noisy: jax.Array,            # [N_sample, N_atom, 3]
        t_hat_noise_level: jax.Array,  # [N_sample]
        # Conditioning inputs (precomputed once per inference)
        s_inputs: jax.Array,           # [N_token, c_s_inputs]
        s_trunk: jax.Array,            # [N_token, c_s]
        z_trunk: jax.Array | None,     # [N_token, N_token, c_z] OR None when pair_z_cache provided
        relp_feature: jax.Array,       # [N_token, N_token, in_features]
        # Atom-level inputs
        atom_to_token_idx: jax.Array,  # [N_atom]
        ref_pos: jax.Array,            # [N_atom, 3]
        ref_charge: jax.Array,         # [N_atom]
        ref_mask: jax.Array,           # [N_atom]
        ref_element: jax.Array,        # [N_atom, 128]
        ref_atom_name_chars: jax.Array, # [N_atom, 256]
        d_lm: jax.Array,               # [n_blocks_atom, n_q, n_k, 3]
        v_lm: jax.Array,               # [n_blocks_atom, n_q, n_k, 1]
        mask_trunked: jax.Array,       # [n_blocks_atom, n_q, n_k]
        # Optional pre-computed caches (set across diffusion steps)
        pair_z_cache: jax.Array | None = None,  # [N_token, N_token, c_z]
        p_lm_cache: jax.Array | None = None,
        c_l_cache: jax.Array | None = None,
    ) -> jax.Array:
        cfg = self.config
        n_token = s_trunk.shape[-2]

        # 1. Diffusion conditioning (single_s always recomputed; pair_z cached)
        single_s, pair_z = DiffusionConditioning(
            config=cfg.conditioning(),
        )(t_hat_noise_level, relp_feature, s_inputs, s_trunk,
          z_trunk if z_trunk is not None else jnp.zeros(
              (s_trunk.shape[-2], s_trunk.shape[-2], cfg.c_z), dtype=s_trunk.dtype),
          pair_z=pair_z_cache)
        # single_s: [N_sample, N_token, c_s]
        # pair_z:   [N_token, N_token, c_z]

        # Broadcast trunk to N_sample for encoder inputs
        # PT does: s_trunk = expand_at_dim(s_trunk, -3, 1) -> [..., 1, N_token, c_s]
        # And z_pair = expand_at_dim(z_pair, -4, 1) -> [..., 1, N_token, N_token, c_z]
        # Then encoder broadcasts internally to N_sample via broadcasting.
        s_trunk_b = jnp.broadcast_to(
            s_trunk[None], (single_s.shape[0],) + s_trunk.shape,
        )  # [N_sample, N_token, c_s]
        pair_z_b = jnp.broadcast_to(
            pair_z[None], (single_s.shape[0],) + pair_z.shape,
        )  # [N_sample, N_token, N_token, c_z]

        # 2. Atom attention encoder
        a_token, q_skip, c_skip, p_skip = AtomAttentionEncoder(
            config=cfg.encoder(),
        )(
            atom_to_token_idx, ref_pos, ref_charge, ref_mask, ref_element,
            ref_atom_name_chars, d_lm, v_lm, mask_trunked,
            r_l=r_noisy, s=s_trunk_b, z=pair_z_b, n_token=n_token,
            p_lm_cache=p_lm_cache, c_l_cache=c_l_cache,
        )
        # a_token: [N_sample, N_token, c_token]

        # 3. + linear_no_bias_s(layernorm_s(single_s)) (zero-init)
        a_token = a_token + Linear(
            cfg.c_token, use_bias=False, initializer="zeros", name="linear_no_bias_s",
        )(LayerNorm(create_offset=False, name="layernorm_s")(single_s))

        # 4. DiffusionTransformer (token-level cross-attention)
        # vmap over N_sample
        def call_tt(a, s, z):
            mask = jnp.ones((a.shape[0],), dtype=a.dtype)
            return TokenTransformer(variant=cfg.transformer())(a, s, z, mask)
        # single_s + pair_z share the sample dim from conditioning;
        # but pair_z has no sample dim, so vmap with in_axes=(0, 0, None)
        a_token = jax.vmap(call_tt, in_axes=(0, 0, None))(a_token, single_s, pair_z)

        # 5. layernorm_a
        a_token = LayerNorm(create_offset=False, name="layernorm_a")(a_token)

        # 6. Atom attention decoder -> r_update
        r_update = AtomAttentionDecoder(config=cfg.decoder())(
            atom_to_token_idx, a_token, q_skip, c_skip, p_skip,
        )
        return r_update
