# SPDX-License-Identifier: Apache-2.0
"""Protenix v1 / v2 PT state_dict -> Haiku param tree rules.

Verified against /checkpoints/protenix-v2.pt (4174 keys).

PT key layout (Protenix v2):
  module.pairformer_stack.blocks.{i}.tri_mul_{out,in}.{...}
  module.pairformer_stack.blocks.{i}.tri_att_{start,end}.{...}
  module.pairformer_stack.blocks.{i}.pair_transition.{...}
  module.pairformer_stack.blocks.{i}.attention_pair_bias.{...}    # NOTE: was attn_pair_bias in OF3
  module.pairformer_stack.blocks.{i}.single_transition.{...}
  module.diffusion_module.diffusion_transformer.blocks.{i}.{...}
  module.confidence_head.pairformer_stack.blocks.{i}.{...}        # separate, optional

Naming differs from OF3:
  attention_pair_bias.attention.* (vs mha.*)
  layernorm_a.{weight,bias} (vs layer_norm_a)
  layernorm_z.{weight,bias} (vs layer_norm_z)
  linear_nobias_z.weight (vs linear_z)
  linear.weight in tri_att (vs linear_z)
  pair_transition.layernorm1 (vs layer_norm)
  pair_transition.linear_no_bias{,_a,_b}  (vs linear_out / swiglu.linear_{a,b})

  Diffusion AdaLN:
    layernorm_a.layernorm_s, layernorm_a.linear_nobias_s, layernorm_a.linear_s
    linear_a_last (vs linear_ada_out)
    conditioned_transition_block.adaln.* + linear_nobias_a1/a2/b + linear_s

v1 vs v2:
  v2: c_z=256, num_head_pair=8
  v1: c_z=128, num_head_pair=4
"""

from __future__ import annotations

from residudes_engine_jax.weights._rules import (
    CompositeRule,
    Rule,
    SimpleRule,
    concat_linear_w,
    linear_b_to_haiku_multihead,
    linear_w_to_haiku,
    linear_w_to_haiku_multihead,
)


# --------- pairformer block helpers ----------

def _tri_mul_block_rules(
    i: int, pt_name: str, hk_name: str, *,
    pt_root: str = "module.pairformer_stack",
    hk_root: str = "pairformer",
) -> list[Rule]:
    pt = f"{pt_root}.blocks.{i}.{pt_name}"  # tri_mul_{out,in} live at block level (not under pair_stack like OF3)
    hk = f"{hk_root}/pair_former_block_{i}/{hk_name}"
    return [
        SimpleRule(f"{pt}.layer_norm_in.weight", f"{hk}/left_norm_input/scale"),
        SimpleRule(f"{pt}.layer_norm_in.bias", f"{hk}/left_norm_input/offset"),
        SimpleRule(f"{pt}.layer_norm_out.weight", f"{hk}/center_norm/scale"),
        SimpleRule(f"{pt}.layer_norm_out.bias", f"{hk}/center_norm/offset"),
        CompositeRule(
            [f"{pt}.linear_a_p.weight", f"{pt}.linear_b_p.weight"],
            f"{hk}/projection/w", concat_linear_w,
        ),
        CompositeRule(
            [f"{pt}.linear_a_g.weight", f"{pt}.linear_b_g.weight"],
            f"{hk}/gate/w", concat_linear_w,
        ),
        SimpleRule(f"{pt}.linear_z.weight", f"{hk}/output_projection/w", linear_w_to_haiku),
        SimpleRule(f"{pt}.linear_g.weight", f"{hk}/gating_linear/w", linear_w_to_haiku),
    ]


def _tri_att_block_rules(
    i: int, pt_name: str, hk_name: str, *,
    num_head: int, head_dim: int,
    pt_root: str = "module.pairformer_stack",
    hk_root: str = "pairformer",
) -> list[Rule]:
    pt = f"{pt_root}.blocks.{i}.{pt_name}"
    hk = f"{hk_root}/pair_former_block_{i}/{hk_name}"
    mh_w = linear_w_to_haiku_multihead(num_head, head_dim)
    return [
        SimpleRule(f"{pt}.layer_norm.weight", f"{hk}/act_norm/scale"),
        SimpleRule(f"{pt}.layer_norm.bias", f"{hk}/act_norm/offset"),
        # Protenix uses .linear (not .linear_z)
        SimpleRule(f"{pt}.linear.weight", f"{hk}/pair_bias_projection/w", linear_w_to_haiku),
        SimpleRule(f"{pt}.mha.linear_q.weight", f"{hk}/q_projection/w", mh_w),
        SimpleRule(f"{pt}.mha.linear_k.weight", f"{hk}/k_projection/w", mh_w),
        SimpleRule(f"{pt}.mha.linear_v.weight", f"{hk}/v_projection/w", mh_w),
        SimpleRule(f"{pt}.mha.linear_g.weight", f"{hk}/gating_query/w", linear_w_to_haiku),
        SimpleRule(f"{pt}.mha.linear_o.weight", f"{hk}/output_projection/w", linear_w_to_haiku),
    ]


def _pair_transition_block_rules(
    i: int, *,
    pt_root: str = "module.pairformer_stack",
    hk_root: str = "pairformer",
) -> list[Rule]:
    pt = f"{pt_root}.blocks.{i}.pair_transition"
    hk = f"{hk_root}/pair_former_block_{i}/pair_transition"
    return [
        SimpleRule(f"{pt}.layernorm1.weight", f"{hk}/input_layer_norm/scale"),
        SimpleRule(f"{pt}.layernorm1.bias", f"{hk}/input_layer_norm/offset"),
        # Protenix: linear_no_bias_a (a) + linear_no_bias_b (b) -> stacked
        CompositeRule(
            [f"{pt}.linear_no_bias_a.weight", f"{pt}.linear_no_bias_b.weight"],
            f"{hk}/transition1/w", concat_linear_w,
        ),
        SimpleRule(f"{pt}.linear_no_bias.weight", f"{hk}/transition2/w", linear_w_to_haiku),
    ]


def _attention_pair_bias_block_rules(
    i: int, *,
    num_head: int = 16, head_dim: int = 24,
    pt_root: str = "module.pairformer_stack",
    hk_root: str = "pairformer",
) -> list[Rule]:
    pt = f"{pt_root}.blocks.{i}.attention_pair_bias"
    hk_block = f"{hk_root}/pair_former_block_{i}"
    mh_w = linear_w_to_haiku_multihead(num_head, head_dim)
    mh_b = linear_b_to_haiku_multihead(num_head, head_dim)
    return [
        SimpleRule(f"{pt}.layernorm_a.weight", f"{hk_block}/single_attention_layer_norm/scale"),
        SimpleRule(f"{pt}.layernorm_a.bias", f"{hk_block}/single_attention_layer_norm/offset"),
        SimpleRule(f"{pt}.layernorm_z.weight", f"{hk_block}/single_pair_logits_norm/scale"),
        SimpleRule(f"{pt}.layernorm_z.bias", f"{hk_block}/single_pair_logits_norm/offset"),
        SimpleRule(f"{pt}.linear_nobias_z.weight", f"{hk_block}/single_pair_logits_projection/w", linear_w_to_haiku),
        # mha (named .attention. in Protenix)
        SimpleRule(f"{pt}.attention.linear_q.weight", f"{hk_block}/single_attention_q_projection/w", mh_w),
        SimpleRule(f"{pt}.attention.linear_q.bias", f"{hk_block}/single_attention_q_projection/b", mh_b),
        SimpleRule(f"{pt}.attention.linear_k.weight", f"{hk_block}/single_attention_k_projection/w", mh_w),
        SimpleRule(f"{pt}.attention.linear_v.weight", f"{hk_block}/single_attention_v_projection/w", mh_w),
        SimpleRule(f"{pt}.attention.linear_g.weight", f"{hk_block}/single_attention_gating_query/w", linear_w_to_haiku),
        SimpleRule(f"{pt}.attention.linear_o.weight", f"{hk_block}/single_attention_transition2/w", linear_w_to_haiku),
    ]


def _single_transition_block_rules(
    i: int, *,
    pt_root: str = "module.pairformer_stack",
    hk_root: str = "pairformer",
) -> list[Rule]:
    pt = f"{pt_root}.blocks.{i}.single_transition"
    hk_prefix = f"{hk_root}/pair_former_block_{i}/single_transition_ffw"
    return [
        SimpleRule(f"{pt}.layernorm1.weight", f"{hk_prefix}_layer_norm/scale"),
        SimpleRule(f"{pt}.layernorm1.bias", f"{hk_prefix}_layer_norm/offset"),
        CompositeRule(
            [f"{pt}.linear_no_bias_a.weight", f"{pt}.linear_no_bias_b.weight"],
            f"{hk_prefix}_transition1/w", concat_linear_w,
        ),
        SimpleRule(f"{pt}.linear_no_bias.weight", f"{hk_prefix}_transition2/w", linear_w_to_haiku),
    ]


def pairformer_rules(
    num_blocks: int = 48, *,
    variant: str = "v2",  # 'v1' or 'v2'
    confidence_head: bool = False,
) -> list[Rule]:
    """Protenix pairformer (main or confidence) rules.

    Args:
      num_blocks: 48 for main pairformer, 4 for confidence head (v2).
      variant: 'v1' (c_z=128, num_head_pair=4) or 'v2' (c_z=256, num_head_pair=8).
      confidence_head: True to convert the smaller confidence pairformer.
    """
    pt_root = (
        "module.confidence_head.pairformer_stack"
        if confidence_head
        else "module.pairformer_stack"
    )
    if variant == "v2":
        num_head_pair, head_dim_pair = 8, 32   # 256/8=32
    elif variant == "v1":
        num_head_pair, head_dim_pair = 4, 32   # 128/4=32
    else:
        raise ValueError(f"unknown variant {variant!r}")
    out: list[Rule] = []
    for i in range(num_blocks):
        out += _tri_mul_block_rules(i, "tri_mul_out", "triangle_multiplication_outgoing", pt_root=pt_root)
        out += _tri_mul_block_rules(i, "tri_mul_in", "triangle_multiplication_incoming", pt_root=pt_root)
        out += _tri_att_block_rules(i, "tri_att_start", "pair_attention1",
                                    num_head=num_head_pair, head_dim=head_dim_pair, pt_root=pt_root)
        out += _tri_att_block_rules(i, "tri_att_end", "pair_attention2",
                                    num_head=num_head_pair, head_dim=head_dim_pair, pt_root=pt_root)
        out += _pair_transition_block_rules(i, pt_root=pt_root)
        out += _attention_pair_bias_block_rules(i, pt_root=pt_root)
        out += _single_transition_block_rules(i, pt_root=pt_root)
    return out


# --------- token (diffusion) transformer ----------

def _token_attention_pair_bias_block_rules(
    i: int, *,
    num_head: int = 16, head_dim: int = 48,
    pt_root: str,
    hk_root: str = "token_transformer",
) -> list[Rule]:
    """Protenix diffusion attention_pair_bias.

    AdaLN naming map (Protenix -> OF3-equivalent semantics):
      layernorm_a.layernorm_s         <-> single_cond LayerNorm (s norm)
      layernorm_a.linear_s            <-> single_cond_scale  (sigmoid-gated, has bias)
      layernorm_a.linear_nobias_s     <-> single_cond_bias   (additive shift, no bias)
      linear_a_last                   <-> adaptive_zero_cond (output gate)
    """
    pt = f"{pt_root}.blocks.{i}.attention_pair_bias"
    hk = f"{hk_root}/block_{i}"
    mh_w = linear_w_to_haiku_multihead(num_head, head_dim)
    mh_b = linear_b_to_haiku_multihead(num_head, head_dim)
    return [
        # AdaLN of x
        SimpleRule(f"{pt}.layernorm_a.layernorm_s.weight", f"{hk}/attn_single_cond_layer_norm/scale"),
        SimpleRule(f"{pt}.layernorm_a.linear_s.weight", f"{hk}/attn_single_cond_scale/w", linear_w_to_haiku),
        SimpleRule(f"{pt}.layernorm_a.linear_s.bias", f"{hk}/attn_single_cond_scale/b"),
        SimpleRule(f"{pt}.layernorm_a.linear_nobias_s.weight", f"{hk}/attn_single_cond_bias/w", linear_w_to_haiku),
        # Pair norm + bias projection
        SimpleRule(f"{pt}.layernorm_z.weight", f"{hk}/attn_pair_norm/scale"),
        SimpleRule(f"{pt}.linear_nobias_z.weight", f"{hk}/attn_pair_bias_projection/w", linear_w_to_haiku),
        # AdaLN-zero output gate (Protenix names linear_a_last)
        SimpleRule(f"{pt}.linear_a_last.weight", f"{hk}/attn_adaptive_zero_cond/w", linear_w_to_haiku),
        SimpleRule(f"{pt}.linear_a_last.bias", f"{hk}/attn_adaptive_zero_cond/b"),
        # MHA (named attention. in Protenix)
        SimpleRule(f"{pt}.attention.linear_q.weight", f"{hk}/attn_q_projection/w", mh_w),
        SimpleRule(f"{pt}.attention.linear_q.bias", f"{hk}/attn_q_projection/b", mh_b),
        SimpleRule(f"{pt}.attention.linear_k.weight", f"{hk}/attn_k_projection/w", mh_w),
        SimpleRule(f"{pt}.attention.linear_v.weight", f"{hk}/attn_v_projection/w", mh_w),
        SimpleRule(f"{pt}.attention.linear_g.weight", f"{hk}/attn_gating_query/w", linear_w_to_haiku),
        SimpleRule(f"{pt}.attention.linear_o.weight", f"{hk}/attn_transition2/w", linear_w_to_haiku),
    ]


def _conditioned_transition_block_rules(
    i: int, *,
    pt_root: str,
    hk_root: str = "token_transformer",
) -> list[Rule]:
    """Protenix conditioned_transition_block rules (SwiGLU + AdaLN + AdaLN-zero)."""
    pt = f"{pt_root}.blocks.{i}.conditioned_transition_block"
    hk = f"{hk_root}/block_{i}"
    return [
        # AdaLN of x  (under .adaln.)
        SimpleRule(f"{pt}.adaln.layernorm_s.weight", f"{hk}/trans_ffw_single_cond_layer_norm/scale"),
        SimpleRule(f"{pt}.adaln.linear_s.weight", f"{hk}/trans_ffw_single_cond_scale/w", linear_w_to_haiku),
        SimpleRule(f"{pt}.adaln.linear_s.bias", f"{hk}/trans_ffw_single_cond_scale/b"),
        SimpleRule(f"{pt}.adaln.linear_nobias_s.weight", f"{hk}/trans_ffw_single_cond_bias/w", linear_w_to_haiku),
        # SwiGLU FFW: linear_nobias_a1/a2 -> stacked, linear_nobias_b -> output
        CompositeRule(
            [f"{pt}.linear_nobias_a1.weight", f"{pt}.linear_nobias_a2.weight"],
            f"{hk}/trans_ffw_transition1/w", concat_linear_w,
        ),
        SimpleRule(f"{pt}.linear_nobias_b.weight", f"{hk}/trans_ffw_transition2/w", linear_w_to_haiku),
        # AdaLN-zero output gate (Protenix names linear_s at conditioned_transition_block level)
        SimpleRule(f"{pt}.linear_s.weight", f"{hk}/trans_ffw_adaptive_zero_cond/w", linear_w_to_haiku),
        SimpleRule(f"{pt}.linear_s.bias", f"{hk}/trans_ffw_adaptive_zero_cond/b"),
    ]


def token_transformer_rules(
    num_blocks: int = 24,
    *,
    variant: str = "v2",
) -> list[Rule]:
    """Protenix v1/v2 token (diffusion) transformer rules."""
    pt_root = "module.diffusion_module.diffusion_transformer"
    out: list[Rule] = []
    for i in range(num_blocks):
        out += _token_attention_pair_bias_block_rules(i, pt_root=pt_root)
        out += _conditioned_transition_block_rules(i, pt_root=pt_root)
    return out


# ─── Diffusion conditioning rules ─────────────────────────────────────────

def _transition_residual_rules(
    pt_prefix: str, hk_prefix: str,
) -> list[Rule]:
    """Protenix Transition (= TransitionResidual): layernorm1 + 3 linears."""
    return [
        SimpleRule(f"{pt_prefix}.layernorm1.weight", f"{hk_prefix}/layernorm1/scale"),
        SimpleRule(f"{pt_prefix}.layernorm1.bias", f"{hk_prefix}/layernorm1/offset"),
        SimpleRule(f"{pt_prefix}.linear_no_bias_a.weight",
                   f"{hk_prefix}/linear_no_bias_a/w", linear_w_to_haiku),
        SimpleRule(f"{pt_prefix}.linear_no_bias_b.weight",
                   f"{hk_prefix}/linear_no_bias_b/w", linear_w_to_haiku),
        SimpleRule(f"{pt_prefix}.linear_no_bias.weight",
                   f"{hk_prefix}/linear_no_bias/w", linear_w_to_haiku),
    ]


def _prefix_rules(rules: list[Rule], prefix: str) -> list[Rule]:
    """Prepend ``prefix/`` to every rule's hk_target."""
    out: list[Rule] = []
    for r in rules:
        if isinstance(r, SimpleRule):
            out.append(SimpleRule(r.pt_key, f"{prefix}/{r.hk_target}", r.transform))
        else:
            out.append(CompositeRule(r.pt_keys, f"{prefix}/{r.hk_target}", r.transform))
    return out


def diffusion_module_rules(
    *,
    transformer_num_blocks: int = 24,
    atom_n_blocks: int = 3,
    variant: str = "v2",
) -> list[Rule]:
    """Master rule list for the FULL JAX DiffusionModule.

    Combines all sub-module rules under ``diffusion_module/`` scope:
      - diffusion_conditioning
      - atom_attention_encoder (incl. atom_transformer)
      - atom_attention_decoder (incl. atom_transformer)
      - token_transformer (= diffusion_transformer)
      - top-level layernorm_s / linear_no_bias_s / layernorm_a
    """
    pt = "module.diffusion_module"
    hk = "diffusion_module"

    rules: list[Rule] = []
    rules += _prefix_rules(diffusion_conditioning_rules(), hk)
    rules += _prefix_rules(atom_attention_encoder_rules(n_blocks=atom_n_blocks), hk)
    rules += _prefix_rules(atom_attention_decoder_rules(n_blocks=atom_n_blocks), hk)
    # Token transformer rules already use 'token_transformer/' top-level
    rules += _prefix_rules(token_transformer_rules(num_blocks=transformer_num_blocks,
                                                    variant=variant), hk)
    # Top-level DiffusionModule params
    rules += [
        SimpleRule(f"{pt}.layernorm_s.weight", f"{hk}/layernorm_s/scale"),
        SimpleRule(f"{pt}.linear_no_bias_s.weight",
                   f"{hk}/linear_no_bias_s/w", linear_w_to_haiku),
        SimpleRule(f"{pt}.layernorm_a.weight", f"{hk}/layernorm_a/scale"),
    ]
    return rules


def atom_attention_encoder_rules(*, n_blocks: int = 3) -> list[Rule]:
    """Rules for AtomAttentionEncoder (Algorithm 5)."""
    pt = "module.diffusion_module.atom_attention_encoder"
    hk = "atom_attention_encoder"
    out: list[Rule] = [
        SimpleRule(f"{pt}.linear_no_bias_ref_pos.weight",
                   f"{hk}/linear_no_bias_ref_pos/w", linear_w_to_haiku),
        SimpleRule(f"{pt}.linear_no_bias_ref_charge.weight",
                   f"{hk}/linear_no_bias_ref_charge/w", linear_w_to_haiku),
        SimpleRule(f"{pt}.linear_no_bias_f.weight",
                   f"{hk}/linear_no_bias_f/w", linear_w_to_haiku),
        SimpleRule(f"{pt}.linear_no_bias_d.weight",
                   f"{hk}/linear_no_bias_d/w", linear_w_to_haiku),
        SimpleRule(f"{pt}.linear_no_bias_invd.weight",
                   f"{hk}/linear_no_bias_invd/w", linear_w_to_haiku),
        SimpleRule(f"{pt}.linear_no_bias_v.weight",
                   f"{hk}/linear_no_bias_v/w", linear_w_to_haiku),
        SimpleRule(f"{pt}.layernorm_s.weight", f"{hk}/layernorm_s/scale"),
        SimpleRule(f"{pt}.linear_no_bias_s.weight",
                   f"{hk}/linear_no_bias_s/w", linear_w_to_haiku),
        SimpleRule(f"{pt}.layernorm_z.weight", f"{hk}/layernorm_z/scale"),
        SimpleRule(f"{pt}.linear_no_bias_z.weight",
                   f"{hk}/linear_no_bias_z/w", linear_w_to_haiku),
        SimpleRule(f"{pt}.linear_no_bias_r.weight",
                   f"{hk}/linear_no_bias_r/w", linear_w_to_haiku),
        SimpleRule(f"{pt}.linear_no_bias_cl.weight",
                   f"{hk}/linear_no_bias_cl/w", linear_w_to_haiku),
        SimpleRule(f"{pt}.linear_no_bias_cm.weight",
                   f"{hk}/linear_no_bias_cm/w", linear_w_to_haiku),
        # small_mlp.{1,3,5} (Sequential indices skip ReLU at 0/2/4)
        SimpleRule(f"{pt}.small_mlp.1.weight",
                   f"{hk}/small_mlp_1/w", linear_w_to_haiku),
        SimpleRule(f"{pt}.small_mlp.3.weight",
                   f"{hk}/small_mlp_3/w", linear_w_to_haiku),
        SimpleRule(f"{pt}.small_mlp.5.weight",
                   f"{hk}/small_mlp_5/w", linear_w_to_haiku),
        SimpleRule(f"{pt}.linear_no_bias_q.weight",
                   f"{hk}/linear_no_bias_q/w", linear_w_to_haiku),
    ]
    # Atom transformer rules: patch hk_root to be inside encoder
    at_rules = atom_transformer_rules(encoder=True, n_blocks=n_blocks)
    # Wrap each rule's hk path under {hk}/atom_transformer
    for r in at_rules:
        if isinstance(r, SimpleRule):
            new_target = r.hk_target.replace("atom_transformer/", f"{hk}/atom_transformer/", 1)
            out.append(SimpleRule(r.pt_key, new_target, r.transform))
        else:  # Composite
            new_target = r.hk_target.replace("atom_transformer/", f"{hk}/atom_transformer/", 1)
            out.append(CompositeRule(r.pt_keys, new_target, r.transform))
    return out


def atom_attention_decoder_rules(*, n_blocks: int = 3) -> list[Rule]:
    """Rules for AtomAttentionDecoder (Algorithm 6)."""
    pt = "module.diffusion_module.atom_attention_decoder"
    hk = "atom_attention_decoder"
    out: list[Rule] = [
        SimpleRule(f"{pt}.linear_no_bias_a.weight",
                   f"{hk}/linear_no_bias_a/w", linear_w_to_haiku),
        SimpleRule(f"{pt}.layernorm_q.weight", f"{hk}/layernorm_q/scale"),
        SimpleRule(f"{pt}.linear_no_bias_out.weight",
                   f"{hk}/linear_no_bias_out/w", linear_w_to_haiku),
    ]
    at_rules = atom_transformer_rules(encoder=False, n_blocks=n_blocks)
    for r in at_rules:
        if isinstance(r, SimpleRule):
            new_target = r.hk_target.replace("atom_transformer/", f"{hk}/atom_transformer/", 1)
            out.append(SimpleRule(r.pt_key, new_target, r.transform))
        else:
            new_target = r.hk_target.replace("atom_transformer/", f"{hk}/atom_transformer/", 1)
            out.append(CompositeRule(r.pt_keys, new_target, r.transform))
    return out


def _atom_attention_pair_bias_rules(
    pt: str, hk: str, *, num_head: int = 4, head_dim: int = 32,
) -> list[Rule]:
    """One AtomTransformer attention_pair_bias block (cross-attention mode)."""
    pt_apb = f"{pt}.attention_pair_bias"
    hk_apb = hk  # block-level scope
    mh_w = linear_w_to_haiku_multihead(num_head, head_dim)
    mh_b = linear_b_to_haiku_multihead(num_head, head_dim)
    return [
        # AdaLN for q (layernorm_a)
        SimpleRule(f"{pt_apb}.layernorm_a.layernorm_s.weight",
                   f"{hk_apb}/attention_pair_bias_layernorm_a_single_cond_layer_norm/scale"),
        SimpleRule(f"{pt_apb}.layernorm_a.linear_s.weight",
                   f"{hk_apb}/attention_pair_bias_layernorm_a_single_cond_scale/w", linear_w_to_haiku),
        SimpleRule(f"{pt_apb}.layernorm_a.linear_s.bias",
                   f"{hk_apb}/attention_pair_bias_layernorm_a_single_cond_scale/b"),
        SimpleRule(f"{pt_apb}.layernorm_a.linear_nobias_s.weight",
                   f"{hk_apb}/attention_pair_bias_layernorm_a_single_cond_bias/w", linear_w_to_haiku),
        # AdaLN for kv (layernorm_kv)
        SimpleRule(f"{pt_apb}.layernorm_kv.layernorm_s.weight",
                   f"{hk_apb}/attention_pair_bias_layernorm_kv_single_cond_layer_norm/scale"),
        SimpleRule(f"{pt_apb}.layernorm_kv.linear_s.weight",
                   f"{hk_apb}/attention_pair_bias_layernorm_kv_single_cond_scale/w", linear_w_to_haiku),
        SimpleRule(f"{pt_apb}.layernorm_kv.linear_s.bias",
                   f"{hk_apb}/attention_pair_bias_layernorm_kv_single_cond_scale/b"),
        SimpleRule(f"{pt_apb}.layernorm_kv.linear_nobias_s.weight",
                   f"{hk_apb}/attention_pair_bias_layernorm_kv_single_cond_bias/w", linear_w_to_haiku),
        # Pair bias projection
        SimpleRule(f"{pt_apb}.layernorm_z.weight",
                   f"{hk_apb}/attention_pair_bias_layernorm_z/scale"),
        SimpleRule(f"{pt_apb}.linear_nobias_z.weight",
                   f"{hk_apb}/attention_pair_bias_linear_nobias_z/w", linear_w_to_haiku),
        # Attention q/k/v/g/o (flat, head-split done in JAX reshape)
        SimpleRule(f"{pt_apb}.attention.linear_q.weight",
                   f"{hk_apb}/attention_pair_bias_attention_linear_q/w", linear_w_to_haiku),
        SimpleRule(f"{pt_apb}.attention.linear_q.bias",
                   f"{hk_apb}/attention_pair_bias_attention_linear_q/b"),
        SimpleRule(f"{pt_apb}.attention.linear_k.weight",
                   f"{hk_apb}/attention_pair_bias_attention_linear_k/w", linear_w_to_haiku),
        SimpleRule(f"{pt_apb}.attention.linear_v.weight",
                   f"{hk_apb}/attention_pair_bias_attention_linear_v/w", linear_w_to_haiku),
        SimpleRule(f"{pt_apb}.attention.linear_g.weight",
                   f"{hk_apb}/attention_pair_bias_attention_linear_g/w", linear_w_to_haiku),
        SimpleRule(f"{pt_apb}.attention.linear_o.weight",
                   f"{hk_apb}/attention_pair_bias_attention_linear_o/w", linear_w_to_haiku),
        # AdaLN-zero gate (linear_a_last)
        SimpleRule(f"{pt_apb}.linear_a_last.weight",
                   f"{hk_apb}/attention_pair_bias_linear_a_last/w", linear_w_to_haiku),
        SimpleRule(f"{pt_apb}.linear_a_last.bias",
                   f"{hk_apb}/attention_pair_bias_linear_a_last/b"),
    ]


def _atom_conditioned_transition_rules(pt: str, hk: str) -> list[Rule]:
    """ConditionedTransitionBlock rules (Protenix variant: 3 linears + AdaLN-zero)."""
    pt_ct = f"{pt}.conditioned_transition_block"
    hk_ct = hk
    return [
        SimpleRule(f"{pt_ct}.adaln.layernorm_s.weight",
                   f"{hk_ct}/conditioned_transition_block_adaln_single_cond_layer_norm/scale"),
        SimpleRule(f"{pt_ct}.adaln.linear_s.weight",
                   f"{hk_ct}/conditioned_transition_block_adaln_single_cond_scale/w", linear_w_to_haiku),
        SimpleRule(f"{pt_ct}.adaln.linear_s.bias",
                   f"{hk_ct}/conditioned_transition_block_adaln_single_cond_scale/b"),
        SimpleRule(f"{pt_ct}.adaln.linear_nobias_s.weight",
                   f"{hk_ct}/conditioned_transition_block_adaln_single_cond_bias/w", linear_w_to_haiku),
        SimpleRule(f"{pt_ct}.linear_nobias_a1.weight",
                   f"{hk_ct}/conditioned_transition_block_linear_nobias_a1/w", linear_w_to_haiku),
        SimpleRule(f"{pt_ct}.linear_nobias_a2.weight",
                   f"{hk_ct}/conditioned_transition_block_linear_nobias_a2/w", linear_w_to_haiku),
        SimpleRule(f"{pt_ct}.linear_nobias_b.weight",
                   f"{hk_ct}/conditioned_transition_block_linear_nobias_b/w", linear_w_to_haiku),
        SimpleRule(f"{pt_ct}.linear_s.weight",
                   f"{hk_ct}/conditioned_transition_block_linear_s/w", linear_w_to_haiku),
        SimpleRule(f"{pt_ct}.linear_s.bias",
                   f"{hk_ct}/conditioned_transition_block_linear_s/b"),
    ]


def atom_transformer_rules(
    *,
    encoder: bool = True,   # if False -> decoder
    n_blocks: int = 3,
    n_heads: int = 4,
    head_dim: int = 32,
) -> list[Rule]:
    """Rules for AtomTransformer (= DiffusionTransformer cross-attention) inside
    AtomAttentionEncoder/Decoder.
    """
    parent = "atom_attention_encoder" if encoder else "atom_attention_decoder"
    pt_root = (
        f"module.diffusion_module.{parent}.atom_transformer.diffusion_transformer"
    )
    hk_root = "atom_transformer"
    out: list[Rule] = []
    for i in range(n_blocks):
        pt_block = f"{pt_root}.blocks.{i}"
        hk_block = f"{hk_root}/diffusion_transformer_blocks_{i}"
        out += _atom_attention_pair_bias_rules(
            pt_block, hk_block, num_head=n_heads, head_dim=head_dim,
        )
        out += _atom_conditioned_transition_rules(pt_block, hk_block)
    return out


def diffusion_conditioning_rules() -> list[Rule]:
    """Rules for Protenix DiffusionConditioning (Algorithm 21).

    Pairs with residudes_engine_jax.haiku.diffusion_conditioning.DiffusionConditioning.
    """
    pt = "module.diffusion_module.diffusion_conditioning"
    hk = "diffusion_conditioning"
    out: list[Rule] = [
        # FourierEmbedding (frozen params)
        SimpleRule(f"{pt}.fourier_embedding.w", f"{hk}/fourier_embedding/w"),
        SimpleRule(f"{pt}.fourier_embedding.b", f"{hk}/fourier_embedding/b"),
        # LayerNorms (scale-only, create_offset=False)
        SimpleRule(f"{pt}.layernorm_n.weight", f"{hk}/layernorm_n/scale"),
        SimpleRule(f"{pt}.layernorm_s.weight", f"{hk}/layernorm_s/scale"),
        SimpleRule(f"{pt}.layernorm_z.weight", f"{hk}/layernorm_z/scale"),
        # Single linear projections (no bias)
        SimpleRule(f"{pt}.linear_no_bias_n.weight",
                   f"{hk}/linear_no_bias_n/w", linear_w_to_haiku),
        SimpleRule(f"{pt}.linear_no_bias_s.weight",
                   f"{hk}/linear_no_bias_s/w", linear_w_to_haiku),
        SimpleRule(f"{pt}.linear_no_bias_z.weight",
                   f"{hk}/linear_no_bias_z/w", linear_w_to_haiku),
        # RelPE
        SimpleRule(f"{pt}.relpe.linear_no_bias.weight",
                   f"{hk}/relpe/linear_no_bias/w", linear_w_to_haiku),
    ]
    # Transition residuals (4 of them)
    for sub in ("transition_z1", "transition_z2", "transition_s1", "transition_s2"):
        out += _transition_residual_rules(f"{pt}.{sub}", f"{hk}/{sub}")
    return out


# --------- MSA module ----------
#
# PT layout (Protenix v2):
#   module.msa_blocks.{i}.outer_product_mean_msa.{layer_norm,linear_1,linear_2,linear_out}
#   module.msa_blocks.{i}.msa_stack.msa_pair_weighted_averaging.{layernorm_m,layernorm_z,linear_no_bias_{mv,mg,z,out}}
#   module.msa_blocks.{i}.msa_stack.transition_m.{layernorm1,linear_no_bias_{a,b},linear_no_bias}
#   module.msa_blocks.{i}.pair_stack.{tri_mul_out,tri_mul_in,tri_att_start,tri_att_end,pair_transition}
#   (no msa_stack on last block)
#
# JAX (haiku) layout under msa_module:
#   msa_module/msa_block_{i}/outer_product_mean_msa/...
#   msa_module/msa_block_{i}/msa_stack/msa_pair_weighted_averaging/...
#   msa_module/msa_block_{i}/msa_stack/transition_m/...
#   msa_module/msa_block_{i}/pair_stack/triangle_multiplication_outgoing/...
#   etc — same names the converter rules already produce, just under
#   msa_block_{i}/pair_stack/ instead of pair_former_block_{i}/.

def _opm_output_w_to_3d(c_hidden: int):
    """Reshape PT linear_out [c_z, c_hidden^2] → [c_hidden, c_hidden, c_z]
    matching AF3's 3-D output_w."""
    import numpy as _np
    def fn(arr: _np.ndarray) -> _np.ndarray:
        c_z = arr.shape[0]
        # PT stores `weight` as [out=c_z, in=c_hidden^2]. We want
        # [c_hidden, c_hidden, c_z].
        return arr.reshape(c_z, c_hidden, c_hidden).transpose(1, 2, 0)
    return fn


def _opm_block_rules(i: int, *, pt_root: str, hk_root: str,
                     c_hidden: int = 32) -> list[Rule]:
    pt = f"{pt_root}.{i}.outer_product_mean_msa"
    hk = f"{hk_root}/msa_block_{i}/outer_product_mean_msa"
    return [
        SimpleRule(f"{pt}.layer_norm.weight", f"{hk}/layer_norm/scale"),
        SimpleRule(f"{pt}.layer_norm.bias", f"{hk}/layer_norm/offset"),
        SimpleRule(f"{pt}.linear_1.weight", f"{hk}/linear_1/w", linear_w_to_haiku),
        SimpleRule(f"{pt}.linear_2.weight", f"{hk}/linear_2/w", linear_w_to_haiku),
        SimpleRule(f"{pt}.linear_out.weight", f"{hk}/output_w",
                   _opm_output_w_to_3d(c_hidden)),
        SimpleRule(f"{pt}.linear_out.bias", f"{hk}/output_b"),
    ]


def _msa_pwa_block_rules(i: int, *, pt_root: str, hk_root: str) -> list[Rule]:
    pt = f"{pt_root}.{i}.msa_stack.msa_pair_weighted_averaging"
    hk = f"{hk_root}/msa_block_{i}/msa_stack/msa_pair_weighted_averaging"
    return [
        SimpleRule(f"{pt}.layernorm_m.weight", f"{hk}/layernorm_m/scale"),
        SimpleRule(f"{pt}.layernorm_m.bias", f"{hk}/layernorm_m/offset"),
        SimpleRule(f"{pt}.layernorm_z.weight", f"{hk}/layernorm_z/scale"),
        SimpleRule(f"{pt}.layernorm_z.bias", f"{hk}/layernorm_z/offset"),
        SimpleRule(f"{pt}.linear_no_bias_mv.weight", f"{hk}/linear_no_bias_mv/w", linear_w_to_haiku),
        SimpleRule(f"{pt}.linear_no_bias_z.weight", f"{hk}/linear_no_bias_z/w", linear_w_to_haiku),
        SimpleRule(f"{pt}.linear_no_bias_mg.weight", f"{hk}/linear_no_bias_mg/w", linear_w_to_haiku),
        SimpleRule(f"{pt}.linear_no_bias_out.weight", f"{hk}/linear_no_bias_out/w", linear_w_to_haiku),
    ]


def _msa_transition_block_rules(i: int, *, pt_root: str, hk_root: str) -> list[Rule]:
    pt = f"{pt_root}.{i}.msa_stack.transition_m"
    hk = f"{hk_root}/msa_block_{i}/msa_stack/transition_m"
    return [
        SimpleRule(f"{pt}.layernorm1.weight", f"{hk}/input_layer_norm/scale"),
        SimpleRule(f"{pt}.layernorm1.bias", f"{hk}/input_layer_norm/offset"),
        CompositeRule(
            [f"{pt}.linear_no_bias_a.weight", f"{pt}.linear_no_bias_b.weight"],
            f"{hk}/transition1/w", concat_linear_w,
        ),
        SimpleRule(f"{pt}.linear_no_bias.weight", f"{hk}/transition2/w", linear_w_to_haiku),
    ]


def _msa_pair_stack_rules(
    i: int, *, num_head: int, head_dim: int,
    pt_root: str, hk_root: str,
) -> list[Rule]:
    """Pair-stack inside an MSA block. Same op layout as the main pairformer
    pair stack (tri_mul_out/in, tri_att_start/end, pair_transition) but the
    PT path has an extra ``pair_stack.`` segment, and the JAX scope is
    ``msa_block_{i}/pair_stack/...``."""
    pt_b = f"{pt_root}.{i}.pair_stack"
    hk_b = f"{hk_root}/msa_block_{i}/pair_stack"
    mh_w = linear_w_to_haiku_multihead(num_head, head_dim)

    def tri_mul(pt_name, hk_name):
        pt = f"{pt_b}.{pt_name}"
        hk = f"{hk_b}/{hk_name}"
        return [
            SimpleRule(f"{pt}.layer_norm_in.weight", f"{hk}/left_norm_input/scale"),
            SimpleRule(f"{pt}.layer_norm_in.bias", f"{hk}/left_norm_input/offset"),
            SimpleRule(f"{pt}.layer_norm_out.weight", f"{hk}/center_norm/scale"),
            SimpleRule(f"{pt}.layer_norm_out.bias", f"{hk}/center_norm/offset"),
            CompositeRule([f"{pt}.linear_a_p.weight", f"{pt}.linear_b_p.weight"],
                          f"{hk}/projection/w", concat_linear_w),
            CompositeRule([f"{pt}.linear_a_g.weight", f"{pt}.linear_b_g.weight"],
                          f"{hk}/gate/w", concat_linear_w),
            SimpleRule(f"{pt}.linear_z.weight", f"{hk}/output_projection/w", linear_w_to_haiku),
            SimpleRule(f"{pt}.linear_g.weight", f"{hk}/gating_linear/w", linear_w_to_haiku),
        ]

    def tri_att(pt_name, hk_name):
        pt = f"{pt_b}.{pt_name}"
        hk = f"{hk_b}/{hk_name}"
        return [
            SimpleRule(f"{pt}.layer_norm.weight", f"{hk}/act_norm/scale"),
            SimpleRule(f"{pt}.layer_norm.bias", f"{hk}/act_norm/offset"),
            SimpleRule(f"{pt}.linear.weight", f"{hk}/pair_bias_projection/w", linear_w_to_haiku),
            SimpleRule(f"{pt}.mha.linear_q.weight", f"{hk}/q_projection/w", mh_w),
            SimpleRule(f"{pt}.mha.linear_k.weight", f"{hk}/k_projection/w", mh_w),
            SimpleRule(f"{pt}.mha.linear_v.weight", f"{hk}/v_projection/w", mh_w),
            SimpleRule(f"{pt}.mha.linear_g.weight", f"{hk}/gating_query/w", linear_w_to_haiku),
            SimpleRule(f"{pt}.mha.linear_o.weight", f"{hk}/output_projection/w", linear_w_to_haiku),
        ]

    def pair_trans():
        pt = f"{pt_b}.pair_transition"
        hk = f"{hk_b}/pair_transition"
        return [
            SimpleRule(f"{pt}.layernorm1.weight", f"{hk}/input_layer_norm/scale"),
            SimpleRule(f"{pt}.layernorm1.bias", f"{hk}/input_layer_norm/offset"),
            CompositeRule(
                [f"{pt}.linear_no_bias_a.weight", f"{pt}.linear_no_bias_b.weight"],
                f"{hk}/transition1/w", concat_linear_w,
            ),
            SimpleRule(f"{pt}.linear_no_bias.weight", f"{hk}/transition2/w", linear_w_to_haiku),
        ]

    out: list[Rule] = []
    out += tri_mul("tri_mul_out", "triangle_multiplication_outgoing")
    out += tri_mul("tri_mul_in", "triangle_multiplication_incoming")
    out += tri_att("tri_att_start", "pair_attention1")
    out += tri_att("tri_att_end", "pair_attention2")
    out += pair_trans()
    return out


def msa_module_rules(
    num_blocks: int = 4, *, variant: str = "v2",
    pt_root: str = "module.msa_module.blocks",
    hk_root: str = "msa_module",
) -> list[Rule]:
    """Convert Protenix MSA module weights to Haiku params.

    Args:
      num_blocks: 4 for Protenix v2 main MSA module.
      variant: 'v2' (c_z=256, num_head_pair=8) only currently supported.
    """
    if variant != "v2":
        raise NotImplementedError(f"msa_module variant {variant!r} not supported")
    num_head, head_dim = 8, 32  # 256/8
    out: list[Rule] = []
    for i in range(num_blocks):
        out += _opm_block_rules(i, pt_root=pt_root, hk_root=hk_root)
        if i + 1 < num_blocks:  # last block has no msa_stack
            out += _msa_pwa_block_rules(i, pt_root=pt_root, hk_root=hk_root)
            out += _msa_transition_block_rules(i, pt_root=pt_root, hk_root=hk_root)
        out += _msa_pair_stack_rules(i, num_head=num_head, head_dim=head_dim,
                                     pt_root=pt_root, hk_root=hk_root)
    return out
