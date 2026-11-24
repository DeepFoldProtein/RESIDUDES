# SPDX-License-Identifier: Apache-2.0
"""OpenFold3 PT state_dict -> Haiku param tree rules.

PT key layout (verified against /checkpoints/of3-p2-155k.pt):

  pairformer_stack.blocks.{i}.pair_stack.tri_mul_{out,in}.{...}
  pairformer_stack.blocks.{i}.pair_stack.tri_att_{start,end}.{...}
  pairformer_stack.blocks.{i}.pair_stack.pair_transition.{...}
  pairformer_stack.blocks.{i}.attn_pair_bias.{...}
  pairformer_stack.blocks.{i}.single_transition.{...}

OF3 conventions (differ from AF3 reference):
  - Most linears have NO bias (`use_bias=False`).
  - Exception: attn_pair_bias.mha.linear_q.bias.
  - SwiGLU is split into two PT linears (linear_a + linear_b); we concat.
  - Tri-mul gating uses two split linears (linear_a_p + linear_b_p, linear_a_g
    + linear_b_g); we concat each pair.
"""

from __future__ import annotations

from typing import Sequence

from residudes_engine_jax.weights._rules import (
    CompositeRule,
    Rule,
    SimpleRule,
    concat_linear_w,
    identity,
    linear_b_to_haiku_multihead,
    linear_w_to_haiku,
    linear_w_to_haiku_multihead,
)


# --------- per-block rule generators ----------

def _tri_mul_block_rules(
    i: int, pt_name: str, hk_name: str, *,
    pt_root: str = "pairformer_stack",
    hk_root: str = "pairformer",
) -> list[Rule]:
    """Rules for one TriangleMultiplication submodule.

    pt_name: 'tri_mul_out' or 'tri_mul_in'
    hk_name: 'triangle_multiplication_outgoing' or '_incoming'
    """
    pt = f"{pt_root}.blocks.{i}.pair_stack.{pt_name}"
    hk = f"{hk_root}/pair_former_block_{i}/{hk_name}"
    return [
        # Layer norms: PT weight/bias -> HK scale/offset.
        SimpleRule(f"{pt}.layer_norm_in.weight", f"{hk}/left_norm_input/scale"),
        SimpleRule(f"{pt}.layer_norm_in.bias", f"{hk}/left_norm_input/offset"),
        SimpleRule(f"{pt}.layer_norm_out.weight", f"{hk}/center_norm/scale"),
        SimpleRule(f"{pt}.layer_norm_out.bias", f"{hk}/center_norm/offset"),
        # Composite: a/b projections concatenated.
        CompositeRule(
            [f"{pt}.linear_a_p.weight", f"{pt}.linear_b_p.weight"],
            f"{hk}/projection/w", concat_linear_w,
        ),
        CompositeRule(
            [f"{pt}.linear_a_g.weight", f"{pt}.linear_b_g.weight"],
            f"{hk}/gate/w", concat_linear_w,
        ),
        # Single linears.
        SimpleRule(f"{pt}.linear_z.weight", f"{hk}/output_projection/w", linear_w_to_haiku),
        SimpleRule(f"{pt}.linear_g.weight", f"{hk}/gating_linear/w", linear_w_to_haiku),
    ]


def _tri_att_block_rules(
    i: int, pt_name: str, hk_name: str, *,
    num_head: int = 4, head_dim: int = 32,
    pt_root: str = "pairformer_stack",
    hk_root: str = "pairformer",
) -> list[Rule]:
    """Rules for one TriangleAttention submodule.

    pt_name: 'tri_att_start' or 'tri_att_end'
    hk_name: 'pair_attention1' or 'pair_attention2'
    """
    pt = f"{pt_root}.blocks.{i}.pair_stack.{pt_name}"
    hk = f"{hk_root}/pair_former_block_{i}/{hk_name}"
    mh_w = linear_w_to_haiku_multihead(num_head, head_dim)
    return [
        SimpleRule(f"{pt}.layer_norm.weight", f"{hk}/act_norm/scale"),
        SimpleRule(f"{pt}.layer_norm.bias", f"{hk}/act_norm/offset"),
        # PT linear_z [out=num_head, in=c]; HK pair_bias_projection [in=c, out=num_head]
        SimpleRule(f"{pt}.linear_z.weight", f"{hk}/pair_bias_projection/w", linear_w_to_haiku),
        # MHA q/k/v: [out=H*D, in=c] -> [in=c, H, D]
        SimpleRule(f"{pt}.mha.linear_q.weight", f"{hk}/q_projection/w", mh_w),
        SimpleRule(f"{pt}.mha.linear_k.weight", f"{hk}/k_projection/w", mh_w),
        SimpleRule(f"{pt}.mha.linear_v.weight", f"{hk}/v_projection/w", mh_w),
        # MHA gate/out: stay 2-D
        SimpleRule(f"{pt}.mha.linear_g.weight", f"{hk}/gating_query/w", linear_w_to_haiku),
        SimpleRule(f"{pt}.mha.linear_o.weight", f"{hk}/output_projection/w", linear_w_to_haiku),
    ]


def _pair_transition_block_rules(
    i: int, *,
    pt_root: str = "pairformer_stack",
    hk_root: str = "pairformer",
) -> list[Rule]:
    pt = f"{pt_root}.blocks.{i}.pair_stack.pair_transition"
    hk = f"{hk_root}/pair_former_block_{i}/pair_transition"
    return [
        SimpleRule(f"{pt}.layer_norm.weight", f"{hk}/input_layer_norm/scale"),
        SimpleRule(f"{pt}.layer_norm.bias", f"{hk}/input_layer_norm/offset"),
        # SwiGLU: two PT linears -> one HK linear with stacked output
        # along axis=1 so tokamax.gated_linear_unit can fuse it.
        CompositeRule(
            [f"{pt}.swiglu.linear_a.weight", f"{pt}.swiglu.linear_b.weight"],
            f"{hk}/transition1/w", concat_linear_w,
        ),
        SimpleRule(f"{pt}.linear_out.weight", f"{hk}/transition2/w", linear_w_to_haiku),
    ]


def _attn_pair_bias_block_rules(
    i: int, *,
    num_head: int = 16, head_dim: int = 24,
    pt_root: str = "pairformer_stack",
    hk_root: str = "pairformer",
) -> list[Rule]:
    """Single-track attention with pair bias (only one with q-bias)."""
    pt = f"{pt_root}.blocks.{i}.attn_pair_bias"
    hk_block = f"{hk_root}/pair_former_block_{i}"
    mh_w = linear_w_to_haiku_multihead(num_head, head_dim)
    mh_b = linear_b_to_haiku_multihead(num_head, head_dim)
    return [
        # layer_norm_a -> single_attention_layer_norm
        SimpleRule(f"{pt}.layer_norm_a.weight", f"{hk_block}/single_attention_layer_norm/scale"),
        SimpleRule(f"{pt}.layer_norm_a.bias", f"{hk_block}/single_attention_layer_norm/offset"),
        # layer_norm_z + linear_z -> single_pair_logits_*
        SimpleRule(f"{pt}.layer_norm_z.weight", f"{hk_block}/single_pair_logits_norm/scale"),
        SimpleRule(f"{pt}.layer_norm_z.bias", f"{hk_block}/single_pair_logits_norm/offset"),
        SimpleRule(f"{pt}.linear_z.weight", f"{hk_block}/single_pair_logits_projection/w", linear_w_to_haiku),
        # mha q/k/v
        SimpleRule(f"{pt}.mha.linear_q.weight", f"{hk_block}/single_attention_q_projection/w", mh_w),
        SimpleRule(f"{pt}.mha.linear_q.bias", f"{hk_block}/single_attention_q_projection/b", mh_b),
        SimpleRule(f"{pt}.mha.linear_k.weight", f"{hk_block}/single_attention_k_projection/w", mh_w),
        SimpleRule(f"{pt}.mha.linear_v.weight", f"{hk_block}/single_attention_v_projection/w", mh_w),
        # mha gate/output
        SimpleRule(f"{pt}.mha.linear_g.weight", f"{hk_block}/single_attention_gating_query/w", linear_w_to_haiku),
        SimpleRule(f"{pt}.mha.linear_o.weight", f"{hk_block}/single_attention_transition2/w", linear_w_to_haiku),
    ]


def _single_transition_block_rules(
    i: int, *,
    pt_root: str = "pairformer_stack",
    hk_root: str = "pairformer",
) -> list[Rule]:
    pt = f"{pt_root}.blocks.{i}.single_transition"
    hk_prefix = f"{hk_root}/pair_former_block_{i}/single_transition_ffw"
    return [
        SimpleRule(f"{pt}.layer_norm.weight", f"{hk_prefix}_layer_norm/scale"),
        SimpleRule(f"{pt}.layer_norm.bias", f"{hk_prefix}_layer_norm/offset"),
        CompositeRule(
            [f"{pt}.swiglu.linear_a.weight", f"{pt}.swiglu.linear_b.weight"],
            f"{hk_prefix}_transition1/w", concat_linear_w,
        ),
        SimpleRule(f"{pt}.linear_out.weight", f"{hk_prefix}_transition2/w", linear_w_to_haiku),
    ]


def pairformer_rules(num_blocks: int) -> list[Rule]:
    """All rules to convert OF3 pairformer state_dict -> Haiku params."""
    out: list[Rule] = []
    for i in range(num_blocks):
        out += _tri_mul_block_rules(i, "tri_mul_out", "triangle_multiplication_outgoing")
        out += _tri_mul_block_rules(i, "tri_mul_in", "triangle_multiplication_incoming")
        out += _tri_att_block_rules(i, "tri_att_start", "pair_attention1")
        out += _tri_att_block_rules(i, "tri_att_end", "pair_attention2")
        out += _pair_transition_block_rules(i)
        out += _attn_pair_bias_block_rules(i)
        out += _single_transition_block_rules(i)
    return out


def _attention_pair_bias_token_block_rules(
    i: int, *,
    num_head: int = 16, head_dim: int = 48,
    pt_root: str,
    hk_root: str = "token_transformer",
) -> list[Rule]:
    """Per-block diffusion attention_pair_bias rules (AdaLN variant)."""
    pt = f"{pt_root}.blocks.{i}.attention_pair_bias"
    hk = f"{hk_root}/block_{i}"
    mh_w = linear_w_to_haiku_multihead(num_head, head_dim)
    mh_b = linear_b_to_haiku_multihead(num_head, head_dim)
    return [
        # AdaLN of x: PT layer_norm_a -> HK attn_single_cond_*
        SimpleRule(f"{pt}.layer_norm_a.layer_norm_s.weight", f"{hk}/attn_single_cond_layer_norm/scale"),
        SimpleRule(f"{pt}.layer_norm_a.linear_g.weight", f"{hk}/attn_single_cond_scale/w", linear_w_to_haiku),
        SimpleRule(f"{pt}.layer_norm_a.linear_g.bias", f"{hk}/attn_single_cond_scale/b"),
        SimpleRule(f"{pt}.layer_norm_a.linear_s.weight", f"{hk}/attn_single_cond_bias/w", linear_w_to_haiku),
        # Pair norm + bias projection
        SimpleRule(f"{pt}.layer_norm_z.weight", f"{hk}/attn_pair_norm/scale"),
        SimpleRule(f"{pt}.linear_z.weight", f"{hk}/attn_pair_bias_projection/w", linear_w_to_haiku),
        # AdaLN-zero output gate
        SimpleRule(f"{pt}.linear_ada_out.weight", f"{hk}/attn_adaptive_zero_cond/w", linear_w_to_haiku),
        SimpleRule(f"{pt}.linear_ada_out.bias", f"{hk}/attn_adaptive_zero_cond/b"),
        # MHA
        SimpleRule(f"{pt}.mha.linear_q.weight", f"{hk}/attn_q_projection/w", mh_w),
        SimpleRule(f"{pt}.mha.linear_q.bias", f"{hk}/attn_q_projection/b", mh_b),
        SimpleRule(f"{pt}.mha.linear_k.weight", f"{hk}/attn_k_projection/w", mh_w),
        SimpleRule(f"{pt}.mha.linear_v.weight", f"{hk}/attn_v_projection/w", mh_w),
        SimpleRule(f"{pt}.mha.linear_g.weight", f"{hk}/attn_gating_query/w", linear_w_to_haiku),
        SimpleRule(f"{pt}.mha.linear_o.weight", f"{hk}/attn_transition2/w", linear_w_to_haiku),
    ]


def _conditioned_transition_block_rules(
    i: int, *,
    pt_root: str,
    hk_root: str = "token_transformer",
) -> list[Rule]:
    """Per-block diffusion conditioned_transition (SwiGLU + AdaLN + AdaLN-zero)."""
    pt = f"{pt_root}.blocks.{i}.conditioned_transition"
    hk = f"{hk_root}/block_{i}"
    return [
        # AdaLN of x
        SimpleRule(f"{pt}.layer_norm.layer_norm_s.weight", f"{hk}/trans_ffw_single_cond_layer_norm/scale"),
        SimpleRule(f"{pt}.layer_norm.linear_g.weight", f"{hk}/trans_ffw_single_cond_scale/w", linear_w_to_haiku),
        SimpleRule(f"{pt}.layer_norm.linear_g.bias", f"{hk}/trans_ffw_single_cond_scale/b"),
        SimpleRule(f"{pt}.layer_norm.linear_s.weight", f"{hk}/trans_ffw_single_cond_bias/w", linear_w_to_haiku),
        # SwiGLU FFW (split a/b -> stacked along axis=1 for tokamax GLU)
        CompositeRule(
            [f"{pt}.swiglu.linear_a.weight", f"{pt}.swiglu.linear_b.weight"],
            f"{hk}/trans_ffw_transition1/w", concat_linear_w,
        ),
        SimpleRule(f"{pt}.linear_out.weight", f"{hk}/trans_ffw_transition2/w", linear_w_to_haiku),
        # AdaLN-zero output gate
        SimpleRule(f"{pt}.linear_g.weight", f"{hk}/trans_ffw_adaptive_zero_cond/w", linear_w_to_haiku),
        SimpleRule(f"{pt}.linear_g.bias", f"{hk}/trans_ffw_adaptive_zero_cond/b"),
    ]


def token_transformer_rules(
    num_blocks: int = 24,
    *,
    use_sample_diffusion_copy: bool = True,
) -> list[Rule]:
    """OF3 token (diffusion) transformer rules.

    The OF3 ckpt stores two copies of the diffusion stack:
      - ``diffusion_module.diffusion_transformer.*``       (training/loss path)
      - ``sample_diffusion.diffusion_module.diffusion_transformer.*`` (inference)
    The TRT compat layer hot-swaps the inference path, so by default we
    convert that copy. Set use_sample_diffusion_copy=False to convert the
    other.
    """
    pt_root = (
        "sample_diffusion.diffusion_module.diffusion_transformer"
        if use_sample_diffusion_copy
        else "diffusion_module.diffusion_transformer"
    )
    out: list[Rule] = []
    for i in range(num_blocks):
        out += _attention_pair_bias_token_block_rules(i, pt_root=pt_root)
        out += _conditioned_transition_block_rules(i, pt_root=pt_root)
    return out
