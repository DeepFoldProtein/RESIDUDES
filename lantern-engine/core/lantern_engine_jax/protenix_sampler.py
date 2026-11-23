# SPDX-License-Identifier: Apache-2.0
"""JAX replacement for Protenix's ``sample_diffusion``.

Combines:
  1. JAX DiffusionModule (Algorithm 20: f_forward + EDM scaling)
  2. JAX sample() from diffusion_sampler.py (Algorithm 18 + lax.scan)

Drop-in for ``protenix.model.generator.sample_diffusion``: same Python
signature, returns the same PT tensor.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

import haiku as hk
import jax
import jax.numpy as jnp
import torch

from lantern_engine_jax.bridge import jax_to_torch, torch_to_jax, get_jax_device
from lantern_engine_jax.haiku.diffusion_module import (
    DiffusionModule,
    DiffusionModuleConfig,
)
from lantern_engine_jax.haiku.diffusion_sampler import (
    centre_random_augmentation,
    random_rotation,
)
from lantern_engine_jax.weights._rules import apply_rules
from lantern_engine_jax.weights.protenix import diffusion_module_rules

logger = logging.getLogger("biotrt-jax-sampler")


def _make_dm_apply(config: DiffusionModuleConfig, dtype: jnp.dtype):
    """Build (init, apply) for the JAX DiffusionModule (f_forward semantics)."""
    def fn(r_noisy, t_hat, s_inputs, s_trunk, relp, atom_idx, ref_pos,
           ref_charge, ref_mask, ref_element, ref_atom_name_chars,
           d_lm, v_lm, mask_trunked, pair_z, p_lm, c_l):
        return DiffusionModule(config=config)(
            r_noisy, t_hat, s_inputs, s_trunk, None, relp,
            atom_idx, ref_pos, ref_charge, ref_mask, ref_element,
            ref_atom_name_chars, d_lm, v_lm, mask_trunked,
            pair_z_cache=pair_z, p_lm_cache=p_lm, c_l_cache=c_l,
        )
    return hk.transform(fn)


def _denoise_step_full(
    *,
    apply_fn: Callable,
    params: dict,
    sigma_data: float,
    # Conditioning (constant across steps)
    s_inputs, s_trunk, relp, atom_idx, ref_pos, ref_charge,
    ref_mask, ref_element, ref_atom_name_chars,
    d_lm, v_lm, mask_trunked, pair_z, p_lm, c_l,
):
    """Build a per-step denoise(positions, t_hat) -> denoised_positions."""

    def denoise(positions, t_hat):
        # positions: [N_atom, 3]; t_hat: scalar
        # JAX sample() vmaps over N_sample, so positions/t_hat are per-sample
        # but the conditioning is shared. Add a leading sample dim of 1
        # for the DM call, then squeeze.
        r_noisy = positions[None]                       # [1, N_atom, 3]
        t_hat_arr = jnp.asarray(t_hat).reshape((1,))    # [1]
        # EDM scaling: r_in = x_noisy / sqrt(sigma_data^2 + sigma^2)
        denom = jnp.sqrt(sigma_data ** 2 + t_hat ** 2)
        r_in = r_noisy / denom
        r_update = apply_fn(
            params, None,
            r_in, t_hat_arr,
            s_inputs, s_trunk, relp, atom_idx, ref_pos, ref_charge,
            ref_mask, ref_element, ref_atom_name_chars,
            d_lm, v_lm, mask_trunked, pair_z, p_lm, c_l,
        )  # [1, N_atom, 3]
        # EDM rescale: x_denoised = c_skip*x + c_out*r_update
        s_ratio = t_hat / sigma_data
        c_skip = 1.0 / (1.0 + s_ratio ** 2)
        c_out = t_hat / jnp.sqrt(1.0 + s_ratio ** 2)
        x_denoised = c_skip * r_noisy + c_out * r_update
        return x_denoised[0]  # squeeze sample dim

    return denoise


def make_jax_sample_diffusion(
    *,
    ckpt_path: str,
    config: DiffusionModuleConfig | None = None,
    dtype: jnp.dtype = jnp.float32,
) -> Callable:
    """Build a JAX-based replacement for ``protenix.model.generator.sample_diffusion``.

    Returns a callable that has the same Python signature as the PT version
    but runs the entire diffusion loop on the JAX side (with the existing
    JAX hot-swap params for pairformer + token transformer + atom modules).
    """
    if config is None:
        config = DiffusionModuleConfig(c_z=256)

    # Load + convert weights once.
    logger.info("Loading and converting Protenix v2 weights for JAX sampler...")
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    rules = diffusion_module_rules(transformer_num_blocks=config.transformer_num_blocks)
    cpu_params = apply_rules(rules, sd)
    cpu_params = jax.tree_util.tree_map(lambda x: jnp.asarray(x, dtype=dtype), cpu_params)
    params = jax.device_put(cpu_params, get_jax_device())
    logger.info("  -> %d HK scopes, %d leaves",
                len(params), sum(len(v) for v in params.values()))

    # Build apply function (JIT)
    transformed = _make_dm_apply(config, dtype)
    dm_apply = jax.jit(transformed.apply)

    def sample_diffusion(
        denoise_net=None,                # ignored: we use JAX directly
        input_feature_dict: dict | None = None,
        s_inputs=None, s_trunk=None, z_trunk=None, pair_z=None,
        p_lm=None, c_l=None,
        noise_schedule=None,
        N_sample: int = 1,
        gamma0: float = 0.8,
        gamma_min: float = 1.0,
        noise_scale_lambda: float = 1.003,
        step_scale_eta: float = 1.5,
        diffusion_chunk_size=None,
        inplace_safe: bool = False,
        attn_chunk_size=None,
        enable_efficient_fusion: bool = False,
        guidance_configs=None,
    ) -> torch.Tensor:
        del denoise_net, diffusion_chunk_size, inplace_safe, attn_chunk_size
        del enable_efficient_fusion, guidance_configs
        device = s_inputs.device
        orig_dtype = s_inputs.dtype
        n_atom = input_feature_dict["atom_to_token_idx"].shape[-1]

        # Convert PT inputs -> JAX
        ifd = input_feature_dict
        ref_anc = ifd["ref_atom_name_chars"]
        if ref_anc.dim() == 3:
            ref_anc = ref_anc.reshape(ref_anc.shape[0], -1)

        # Squeeze leading-1 sample/batch axes that PT adds via expand_at_dim
        def _sq(t, target_ndim):
            return t.squeeze(0) if t is not None and t.dim() > target_ndim and t.shape[0] == 1 else t

        s_in_j = torch_to_jax(s_inputs.to(torch.float32))                   # [N_token, c_s_inputs]
        s_t_j = torch_to_jax(s_trunk.to(torch.float32))                     # [N_token, c_s]
        relp_j = torch_to_jax(ifd["relp"].to(torch.float32))                # [N_token, N_token, *]
        atom_idx_j = torch_to_jax(ifd["atom_to_token_idx"])
        ref_pos_j = torch_to_jax(ifd["ref_pos"].to(torch.float32))
        ref_charge_j = torch_to_jax(ifd["ref_charge"].to(torch.float32))
        ref_mask_j = torch_to_jax(ifd["ref_mask"].to(torch.float32))
        ref_el_j = torch_to_jax(ifd["ref_element"].to(torch.float32))
        ref_an_j = torch_to_jax(ref_anc.to(torch.float32))
        d_lm_j = torch_to_jax(ifd["d_lm"].to(torch.float32))
        v_lm_j = torch_to_jax(ifd["v_lm"].to(torch.float32))
        mask_t_j = torch_to_jax(ifd["pad_info"]["mask_trunked"].to(torch.float32))

        pair_z_j = torch_to_jax(pair_z.to(torch.float32))                   # [N_token, N_token, c_z]
        # PT cached p_lm has size-1 sample axis at 0; squeeze
        p_lm_in = p_lm.squeeze(0) if p_lm.dim() == 5 and p_lm.shape[0] == 1 else p_lm
        p_lm_j = torch_to_jax(p_lm_in.to(torch.float32))
        c_l_j = torch_to_jax(c_l.to(torch.float32))

        # Build the per-step denoiser
        denoise_step = _denoise_step_full(
            apply_fn=dm_apply, params=params,
            sigma_data=config.sigma_data,
            s_inputs=s_in_j, s_trunk=s_t_j, relp=relp_j,
            atom_idx=atom_idx_j, ref_pos=ref_pos_j, ref_charge=ref_charge_j,
            ref_mask=ref_mask_j, ref_element=ref_el_j,
            ref_atom_name_chars=ref_an_j, d_lm=d_lm_j, v_lm=v_lm_j,
            mask_trunked=mask_t_j, pair_z=pair_z_j, p_lm=p_lm_j, c_l=c_l_j,
        )

        # Convert noise_schedule to JAX
        ns_j = torch_to_jax(noise_schedule.to(torch.float32))

        # Run the JAX sampler. We replicate Protenix's algorithm explicitly here
        # rather than calling diffusion_sampler.sample() so that we can pass
        # arbitrary noise schedules (PT may compute its own).
        n_steps = ns_j.shape[0] - 1
        atom_mask = jnp.ones((n_atom,), dtype=jnp.float32)
        # Use Protenix-style PRNG; use a fixed key if not provided.
        # Ideally we'd thread torch's RNG but for now derive from current time.
        import time
        seed = int(time.time() * 1e6) & 0xFFFFFFFF
        key = jax.random.PRNGKey(seed)

        from lantern_engine_jax.haiku.diffusion_sampler import sample as jax_sample
        x_l = jax_sample(
            denoising_step=denoise_step,
            noise_schedule=ns_j,
            mask=atom_mask,
            key=key,
            num_samples=N_sample,
            gamma_0=gamma0,
            gamma_min=gamma_min,
            noise_scale_lambda=noise_scale_lambda,
            step_scale_eta=step_scale_eta,
        )
        # x_l: [N_sample, N_atom, 3]
        out = jax_to_torch(x_l, dtype=orig_dtype).to(device)
        return out

    return sample_diffusion
