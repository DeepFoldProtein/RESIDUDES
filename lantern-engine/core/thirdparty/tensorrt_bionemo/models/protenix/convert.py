# SPDX-License-Identifier: Apache-2.0
"""
Convert Protenix checkpoint weights to TRT engine safetensors format.

Protenix weight naming differs from OF3:
  - No `pair_stack.` prefix (flat structure)
  - `attention_pair_bias` (not `attn_pair_bias`)
  - `attention.linear_q` (not `mha.linear_q`)
  - `layernorm_a/z` (not `layer_norm_a/z`)
  - `linear_nobias_z` (not `linear_z`)
  - `layernorm1` (not `layer_norm`) for transitions
  - `linear_no_bias_a/b` + `linear_no_bias` (not `swiglu.linear_a/b` + `linear_out`)
  - `linear` (tri_att, not `linear_z`)

The TRT engine uses boltz1-convention names internally.
"""

from collections import OrderedDict

import torch
from tensorrt_llm_lite import str_dtype_to_torch
from tensorrt_llm_lite.logger import logger

from tensorrt_bionemo.configs import (DiffusionTransformerConfig,
                                      PairformerConfig)
from tensorrt_bionemo.hubs import load_weights
from tensorrt_bionemo.mapping import Mapping
from tensorrt_bionemo.models.boltz1.convert import (
    get_adaln_weights, get_output_projection_weights, get_pairwise_attn_weights,
    get_post_norm_weights, get_transition_weights, get_tri_attn_node_weights,
    get_tri_mul_node_weights)


def split(*args, **kwargs):
    pass


def _load_protenix_state_dict(local_checkpoint: str) -> dict:
    """Load Protenix checkpoint, handling DDP module. prefix."""
    ckpt = torch.load(local_checkpoint, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
    else:
        state = ckpt
    # Strip DDP "module." prefix
    state = {k.replace("module.", "", 1) if k.startswith("module.") else k: v
             for k, v in state.items()}
    return state


def convert_hf_pairformer(config: PairformerConfig,
                          mapping: Mapping,
                          local_checkpoint: str = None,
                          model_name: str = "protenix"):
    """Convert Protenix pairformer_stack weights to TRT engine format.

    Maps Protenix weight names → boltz1 internal convention used by TRT engine.
    """
    state_dict = _load_protenix_state_dict(local_checkpoint)

    pairformer_state_dict = {}
    prefix = "pairformer_stack"

    for name, param in state_dict.items():
        if not name.startswith(prefix + ".blocks."):
            continue

        original_name = name
        block_number = int(name.split(".")[2])
        name = name.replace(prefix, "pairformer_module")
        name = name.replace("blocks.", "layers.")

        # ── attention_pair_bias → attention ────────────────────────────
        if "attention_pair_bias" in name:
            name = name.replace("attention_pair_bias.", "attention.")
            # layernorm_a → norm_s
            name = name.replace("layernorm_a", "norm_s")
            # layernorm_z → proj_z.0
            name = name.replace("layernorm_z", "proj_z.0")
            # linear_nobias_z → proj_z.1
            name = name.replace("linear_nobias_z", "proj_z.1")
            # attention.attention.linear_q → attention.proj_q
            name = name.replace("attention.linear_q", "proj_q")
            name = name.replace("attention.linear_k", "proj_k")
            name = name.replace("attention.linear_v", "proj_v")
            name = name.replace("attention.linear_o", "proj_o")
            name = name.replace("attention.linear_g", "proj_g")
            pairformer_state_dict[name] = param
            continue

        # ── single_transition → transition_s ──────────────────────────
        if "single_transition" in name:
            name = name.replace("single_transition", "transition_s")
            name = name.replace("layernorm1", "norm")
            name = name.replace("linear_no_bias_a", "fc1")
            name = name.replace("linear_no_bias_b", "fc2")
            name = name.replace("linear_no_bias", "fc3")
            pairformer_state_dict[name] = param
            continue

        # ── tri_att_start / tri_att_end (flat, no pair_stack prefix) ──
        if "tri_att_start" in name or "tri_att_end" in name:
            # Protenix: tri_att_start.linear → TRT: tri_att_start.linear (same)
            # Protenix: tri_att_start.mha.linear_q → stays
            pairformer_state_dict[name] = param
            continue

        # ── tri_mul_out / tri_mul_in (flat) ───────────────────────────
        if "tri_mul_out" in name or "tri_mul_in" in name:
            tri_mul_type = "tri_mul_out" if "tri_mul_out" in name else "tri_mul_in"
            # Concatenate linear_a_p + linear_b_p → p_in
            if "linear_a_p" in name or "linear_b_p" in name:
                stripped = name.replace(".linear_a_p.weight", "").replace(".linear_b_p.weight", "")
                p_in_name = stripped + ".p_in.weight"
                if p_in_name not in pairformer_state_dict:
                    p0 = state_dict[f"{prefix}.blocks.{block_number}.{tri_mul_type}.linear_a_p.weight"]
                    p1 = state_dict[f"{prefix}.blocks.{block_number}.{tri_mul_type}.linear_b_p.weight"]
                    pairformer_state_dict[p_in_name] = torch.cat([p0, p1], dim=0)
                continue
            # Concatenate linear_a_g + linear_b_g → g_in
            if "linear_a_g" in name or "linear_b_g" in name:
                stripped = name.replace(".linear_a_g.weight", "").replace(".linear_b_g.weight", "")
                g_in_name = stripped + ".g_in.weight"
                if g_in_name not in pairformer_state_dict:
                    g0 = state_dict[f"{prefix}.blocks.{block_number}.{tri_mul_type}.linear_a_g.weight"]
                    g1 = state_dict[f"{prefix}.blocks.{block_number}.{tri_mul_type}.linear_b_g.weight"]
                    pairformer_state_dict[g_in_name] = torch.cat([g0, g1], dim=0)
                continue
            # linear_z → p_out, linear_g → g_out, layer_norm_in → norm_in
            name = name.replace("linear_z", "p_out")
            name = name.replace("linear_g", "g_out")
            name = name.replace("layer_norm_in", "norm_in")
            name = name.replace("layer_norm_out", "norm_out")
            pairformer_state_dict[name] = param
            continue

        # ── pair_transition → transition_z ────────────────────────────
        if "pair_transition" in name:
            name = name.replace("pair_transition", "transition_z")
            name = name.replace("layernorm1", "norm")
            name = name.replace("linear_no_bias_a", "fc1")
            name = name.replace("linear_no_bias_b", "fc2")
            name = name.replace("linear_no_bias", "fc3")
            pairformer_state_dict[name] = param
            continue

        logger.warning(f"Protenix pairformer: missed weight: {original_name}")

    # ── Now use the same boltz1 weight extraction as OF3 ──────────────
    mapping = mapping if mapping is not None else Mapping()
    prefix = "pairformer_module.layers"
    tbm_prefix = "layers"
    weights = {}
    for i in range(config.num_blocks):
        layer_prefix = f"{prefix}.{i}"
        layer_tbm_prefix = f"{tbm_prefix}.{i}"
        if not config.no_update_s:
            weights.update(
                get_pairwise_attn_weights(mapping, pairformer_state_dict,
                                          f"{layer_prefix}.attention",
                                          f"{layer_tbm_prefix}.attention",
                                          config.max_attention_pairwise_tp_size,
                                          config.num_heads, dtype=config.dtype))
        weights.update(
            get_tri_attn_node_weights(mapping, pairformer_state_dict,
                                      f"{layer_prefix}.tri_att_start",
                                      f"{layer_tbm_prefix}.tri_attn_start",
                                      dtype=config.dtype))
        weights.update(
            get_tri_attn_node_weights(mapping, pairformer_state_dict,
                                      f"{layer_prefix}.tri_att_end",
                                      f"{layer_tbm_prefix}.tri_attn_end",
                                      dtype=config.dtype))
        weights.update(
            get_tri_mul_node_weights(mapping, pairformer_state_dict,
                                     f"{layer_prefix}.tri_mul_out",
                                     f"{layer_tbm_prefix}.tri_mul_out",
                                     config.max_tri_mul_tp_size,
                                     dtype=config.dtype))
        weights.update(
            get_tri_mul_node_weights(mapping, pairformer_state_dict,
                                     f"{layer_prefix}.tri_mul_in",
                                     f"{layer_tbm_prefix}.tri_mul_in",
                                     config.max_tri_mul_tp_size,
                                     dtype=config.dtype))
        if not config.no_update_s:
            weights.update(
                get_transition_weights(mapping, pairformer_state_dict,
                                       f"{layer_prefix}.transition_s",
                                       f"{layer_tbm_prefix}.transition_s",
                                       config.max_transition_tp_size,
                                       config.token_s * 4, dtype=config.dtype))
        weights.update(
            get_transition_weights(mapping, pairformer_state_dict,
                                   f"{layer_prefix}.transition_z",
                                   f"{layer_tbm_prefix}.transition_z",
                                   config.max_transition_tp_size,
                                   config.token_z * 4, dtype=config.dtype))
    return weights


def convert_hf_token_transformer(config: DiffusionTransformerConfig,
                                 mapping: Mapping,
                                 local_checkpoint: str = None,
                                 model_name: str = "protenix"):
    """Convert Protenix diffusion_transformer weights to TRT engine format.

    Protenix checkpoint prefix: diffusion_diffusion_transformer
    Strategy: map Protenix names → boltz1 intermediate naming → reuse boltz1 extractors.

    Protenix adaln structure:
        layernorm_a.layernorm_s.weight  → adaln.s_norm.weight
        layernorm_a.linear_s.weight/bias → adaln.s_scale.weight/bias
        layernorm_a.linear_nobias_s.weight → adaln.s_bias.weight

    This mirrors OF3's mapping:
        layer_norm_a.layer_norm_s → adaln.s_norm
        layer_norm_a.linear_g    → adaln.s_scale
        layer_norm_a.linear_s    → adaln.s_bias
    """
    state_dict = _load_protenix_state_dict(local_checkpoint)

    prefix = "diffusion_module.diffusion_transformer.blocks"
    replace_prefix = "token_transformer.layers"

    tktn_state_dict = OrderedDict()
    for name, param in state_dict.items():
        if not name.startswith(prefix):
            continue
        name = name.replace(prefix, replace_prefix)

        if "attention_pair_bias" in name:
            # ── adaln (layernorm_a) ───────────────────────────────
            if "layernorm_a" in name:
                name = name.replace("attention_pair_bias.layernorm_a", "adaln")
                name = name.replace("layernorm_s", "s_norm")
                # linear_s (scale) must be mapped before linear_nobias_s (bias)
                name = name.replace("linear_nobias_s", "s_bias")
                name = name.replace("linear_s", "s_scale")
                tktn_state_dict[name] = param
                continue
            # ── output projection (linear_a_last) ─────────────────
            if "linear_a_last" in name:
                name = name.replace("attention_pair_bias.linear_a_last",
                                    "output_projection.0")
                tktn_state_dict[name] = param
                continue
            # ── attention projections ──────────────────────────────
            if "attention.linear_" in name:
                name = name.replace("attention_pair_bias.attention", "pair_bias_attn")
                name = name.replace("linear_q", "proj_q")
                name = name.replace("linear_k", "proj_k")
                name = name.replace("linear_v", "proj_v")
                name = name.replace("linear_o", "proj_o")
                name = name.replace("linear_g", "proj_g")
                tktn_state_dict[name] = param
                continue
            # ── pair bias (layernorm_z + linear_nobias_z) ─────────
            if "layernorm_z" in name or "linear_nobias_z" in name:
                name = name.replace("attention_pair_bias.layernorm_z",
                                    "pair_bias_attn.proj_z.0")
                name = name.replace("attention_pair_bias.linear_nobias_z",
                                    "pair_bias_attn.proj_z.1")
                tktn_state_dict[name] = param
                continue
            logger.warning(f"Protenix token_transformer missed: {name}")

        if "conditioned_transition_block" in name:
            # ── adaln ─────────────────────────────────────────────
            if "adaln." in name:
                name = name.replace("conditioned_transition_block.adaln",
                                    "transition.adaln")
                name = name.replace("layernorm_s", "s_norm")
                name = name.replace("linear_nobias_s", "s_bias")
                name = name.replace("linear_s", "s_scale")
                tktn_state_dict[name] = param
                continue
            # ── SwiGLU / transition ───────────────────────────────
            if "linear_nobias_a1" in name:
                name = name.replace("conditioned_transition_block.linear_nobias_a1",
                                    "transition.swish_gate.0")
                tktn_state_dict[name] = param
                continue
            if "linear_nobias_a2" in name:
                name = name.replace("conditioned_transition_block.linear_nobias_a2",
                                    "transition.a_to_b")
                tktn_state_dict[name] = param
                continue
            if "linear_nobias_b" in name:
                name = name.replace("conditioned_transition_block.linear_nobias_b",
                                    "transition.b_to_a")
                tktn_state_dict[name] = param
                continue
            if "linear_s" in name:
                name = name.replace("conditioned_transition_block.linear_s",
                                    "transition.output_projection.0")
                tktn_state_dict[name] = param
                continue
            logger.warning(f"Protenix token_transformer missed: {name}")

    # ── Extract using boltz1 functions ────────────────────────────────
    mapping = mapping if mapping is not None else Mapping()
    src_prefix = "token_transformer.layers"
    dst_prefix = "layers"
    weights = {}

    from tensorrt_bionemo.models.boltz1.convert import get_conditioned_transition_block_weights

    for i in range(config.num_blocks):
        layer_prefix = f"{src_prefix}.{i}"
        layer_tbm_prefix = f"{dst_prefix}.{i}"
        weights.update(
            get_adaln_weights(mapping, tktn_state_dict,
                              f"{layer_prefix}.adaln",
                              f"{layer_tbm_prefix}.adaln",
                              config.dim, config.dim_single_cond,
                              dtype=config.dtype))
        weights.update(
            get_pairwise_attn_weights(
                mapping, tktn_state_dict,
                f"{layer_prefix}.pair_bias_attn",
                f"{layer_tbm_prefix}.pair_bias_attn",
                max_attention_pairwise_tp_size=True,
                num_heads=config.num_heads,
                attention_initial_norm=config.attention_initial_norm,
                compute_pair_bias=config.version == "v1",
                dtype=config.dtype))
        weights.update(
            get_conditioned_transition_block_weights(
                mapping, tktn_state_dict,
                f"{layer_prefix}.transition",
                f"{layer_tbm_prefix}.transition",
                config.dim, config.dim_single_cond,
                expansion_factor=config.expansion_factor,
                dtype=config.dtype))
        weights.update(
            get_output_projection_weights(
                mapping, tktn_state_dict,
                f"{layer_prefix}.output_projection",
                f"{layer_tbm_prefix}.output_projection",
                dtype=config.dtype))
        weights.update(
            get_post_norm_weights(mapping, tktn_state_dict,
                                  f"{layer_prefix}.post_lnorm",
                                  f"{layer_tbm_prefix}.post_lnorm",
                                  dtype=config.dtype))
    return weights
