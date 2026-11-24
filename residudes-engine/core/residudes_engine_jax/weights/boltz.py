# SPDX-License-Identifier: Apache-2.0
"""Boltz-2 PT state_dict -> Haiku param tree rules.

Verified against /.boltz/boltz2_conf.ckpt (5102 keys, state_dict at top).

PT key layout:
  pairformer_module.layers.{i}.{...}                       # main structure pairformer (64 layers)
  confidence_module.pairformer_stack.layers.{i}.{...}      # confidence pairformer (8 layers)
  structure_module.score_model.token_transformer.layers.{i}.{...}  # diffusion (24 layers)

Boltz uses ``layers.{i}`` (vs OF3/Protenix's ``blocks.{i}``).

Naming differs from OF3 / Protenix:
  attention                             pair-biased single-track attention
    proj_q/k/v/g/o.weight (q has bias)
    proj_z.0.{weight, bias}             LayerNorm
    proj_z.1.weight                     Linear (no bias)
  pre_norm_s                            LayerNorm of single act (was layer_norm_a)
  transition_{s,z}
    norm.{weight, bias}                 LayerNorm
    fc1, fc2                            SwiGLU split (vs swiglu.linear_a/b)
    fc3                                 output projection (vs linear_out)
  tri_att_{start,end}.linear            pair bias projection (vs linear_z)
  tri_mul_{in,out}                      ALREADY-FUSED projections
    norm_in/norm_out                    LayerNorms (single 'norm_' prefix)
    p_in.weight (2C, C)                 fused [a_p; b_p] in one linear
    g_in.weight (2C, C)                 fused [a_g; b_g] in one linear
    p_out                               output projection (vs linear_z)
    g_out                               output gate (vs linear_g)

Token transformer in Boltz:
  Each layer's ``pair_bias_attn`` does NOT project pair bias internally —
  the pair bias projection happens once in the score_model, and a per-layer
  bias tensor is passed via the ``bias`` kwarg. This differs from our shared
  HK TokenTransformer (which projects per-block). Token transformer rules
  need a Boltz-specific HK variant, which is left as a follow-up.
"""

from __future__ import annotations

import numpy as np

from residudes_engine_jax.weights._rules import (
    Rule,
    SimpleRule,
    linear_b_to_haiku_multihead,
    linear_w_to_haiku,
    linear_w_to_haiku_multihead,
)


# --------- pairformer per-layer rule generators ----------

def _tri_mul_layer_rules(
    i: int, pt_name: str, hk_name: str, *,
    pt_root: str,
    hk_root: str = "pairformer",
) -> list[Rule]:
    """Boltz tri_mul: projections already fused (p_in, g_in are 2C x C)."""
    pt = f"{pt_root}.layers.{i}.{pt_name}"
    hk = f"{hk_root}/pair_former_block_{i}/{hk_name}"
    return [
        SimpleRule(f"{pt}.norm_in.weight", f"{hk}/left_norm_input/scale"),
        SimpleRule(f"{pt}.norm_in.bias", f"{hk}/left_norm_input/offset"),
        SimpleRule(f"{pt}.norm_out.weight", f"{hk}/center_norm/scale"),
        SimpleRule(f"{pt}.norm_out.bias", f"{hk}/center_norm/offset"),
        # Fused projections (PT [2C, C] -> HK [C, 2C] via simple transpose)
        SimpleRule(f"{pt}.p_in.weight", f"{hk}/projection/w", linear_w_to_haiku),
        SimpleRule(f"{pt}.g_in.weight", f"{hk}/gate/w", linear_w_to_haiku),
        # Output projection + final gate
        SimpleRule(f"{pt}.p_out.weight", f"{hk}/output_projection/w", linear_w_to_haiku),
        SimpleRule(f"{pt}.g_out.weight", f"{hk}/gating_linear/w", linear_w_to_haiku),
    ]


def _tri_att_layer_rules(
    i: int, pt_name: str, hk_name: str, *,
    num_head: int, head_dim: int,
    pt_root: str,
    hk_root: str = "pairformer",
) -> list[Rule]:
    pt = f"{pt_root}.layers.{i}.{pt_name}"
    hk = f"{hk_root}/pair_former_block_{i}/{hk_name}"
    mh_w = linear_w_to_haiku_multihead(num_head, head_dim)
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


def _pair_transition_layer_rules(
    i: int, *,
    pt_root: str,
    hk_root: str = "pairformer",
) -> list[Rule]:
    """Boltz transition_z (pair track): SwiGLU."""
    pt = f"{pt_root}.layers.{i}.transition_z"
    hk = f"{hk_root}/pair_former_block_{i}/pair_transition"
    # fc1 (out=2I, in=C) and fc2 (out=2I, in=C) need to be concatenated.
    # Wait - looking at shapes: fc1 (1536, 384) for transition_s, (512, 128) for
    # transition_z. So fc1 is [out=intermediate, in=C]. Both fc1 and fc2 produce
    # `intermediate` outputs (not 2*intermediate). We need to concat them.
    from residudes_engine_jax.weights._rules import CompositeRule, concat_linear_w
    return [
        SimpleRule(f"{pt}.norm.weight", f"{hk}/input_layer_norm/scale"),
        SimpleRule(f"{pt}.norm.bias", f"{hk}/input_layer_norm/offset"),
        CompositeRule(
            [f"{pt}.fc1.weight", f"{pt}.fc2.weight"],
            f"{hk}/transition1/w", concat_linear_w,
        ),
        SimpleRule(f"{pt}.fc3.weight", f"{hk}/transition2/w", linear_w_to_haiku),
    ]


def _attention_pair_bias_layer_rules(
    i: int, *,
    num_head: int = 16, head_dim: int = 24,
    pt_root: str,
    hk_root: str = "pairformer",
) -> list[Rule]:
    """Boltz pair-biased attention (single track)."""
    pt = f"{pt_root}.layers.{i}.attention"
    hk_block = f"{hk_root}/pair_former_block_{i}"
    pre_norm = f"{pt_root}.layers.{i}.pre_norm_s"
    mh_w = linear_w_to_haiku_multihead(num_head, head_dim)
    mh_b = linear_b_to_haiku_multihead(num_head, head_dim)
    return [
        # pre_norm_s: LayerNorm on s before attention
        SimpleRule(f"{pre_norm}.weight", f"{hk_block}/single_attention_layer_norm/scale"),
        SimpleRule(f"{pre_norm}.bias", f"{hk_block}/single_attention_layer_norm/offset"),
        # proj_z is Sequential[LayerNorm, Linear]; .0 = LN, .1 = Linear
        SimpleRule(f"{pt}.proj_z.0.weight", f"{hk_block}/single_pair_logits_norm/scale"),
        SimpleRule(f"{pt}.proj_z.0.bias", f"{hk_block}/single_pair_logits_norm/offset"),
        SimpleRule(f"{pt}.proj_z.1.weight", f"{hk_block}/single_pair_logits_projection/w", linear_w_to_haiku),
        # mha: proj_q/k/v/g/o (q has bias)
        SimpleRule(f"{pt}.proj_q.weight", f"{hk_block}/single_attention_q_projection/w", mh_w),
        SimpleRule(f"{pt}.proj_q.bias", f"{hk_block}/single_attention_q_projection/b", mh_b),
        SimpleRule(f"{pt}.proj_k.weight", f"{hk_block}/single_attention_k_projection/w", mh_w),
        SimpleRule(f"{pt}.proj_v.weight", f"{hk_block}/single_attention_v_projection/w", mh_w),
        SimpleRule(f"{pt}.proj_g.weight", f"{hk_block}/single_attention_gating_query/w", linear_w_to_haiku),
        SimpleRule(f"{pt}.proj_o.weight", f"{hk_block}/single_attention_transition2/w", linear_w_to_haiku),
    ]


def _single_transition_layer_rules(
    i: int, *,
    pt_root: str,
    hk_root: str = "pairformer",
) -> list[Rule]:
    pt = f"{pt_root}.layers.{i}.transition_s"
    hk_prefix = f"{hk_root}/pair_former_block_{i}/single_transition_ffw"
    from residudes_engine_jax.weights._rules import CompositeRule, concat_linear_w
    return [
        SimpleRule(f"{pt}.norm.weight", f"{hk_prefix}_layer_norm/scale"),
        SimpleRule(f"{pt}.norm.bias", f"{hk_prefix}_layer_norm/offset"),
        CompositeRule(
            [f"{pt}.fc1.weight", f"{pt}.fc2.weight"],
            f"{hk_prefix}_transition1/w", concat_linear_w,
        ),
        SimpleRule(f"{pt}.fc3.weight", f"{hk_prefix}_transition2/w", linear_w_to_haiku),
    ]


def pairformer_rules(
    num_layers: int = 64,
    *,
    confidence: bool = False,
) -> list[Rule]:
    """Boltz-2 pairformer rules.

    Args:
      num_layers: 64 for main structure pairformer, 8 for confidence head.
      confidence: True to convert ``confidence_module.pairformer_stack.layers``.
    """
    pt_root = (
        "confidence_module.pairformer_stack" if confidence else "pairformer_module"
    )
    out: list[Rule] = []
    for i in range(num_layers):
        out += _tri_mul_layer_rules(i, "tri_mul_out", "triangle_multiplication_outgoing", pt_root=pt_root)
        out += _tri_mul_layer_rules(i, "tri_mul_in", "triangle_multiplication_incoming", pt_root=pt_root)
        out += _tri_att_layer_rules(i, "tri_att_start", "pair_attention1",
                                    num_head=4, head_dim=32, pt_root=pt_root)
        out += _tri_att_layer_rules(i, "tri_att_end", "pair_attention2",
                                    num_head=4, head_dim=32, pt_root=pt_root)
        out += _pair_transition_layer_rules(i, pt_root=pt_root)
        out += _attention_pair_bias_layer_rules(i, pt_root=pt_root)
        out += _single_transition_layer_rules(i, pt_root=pt_root)
    return out


def _boltz_tt_layer_rules(
    i: int, *,
    num_head: int = 16, head_dim: int = 48,
    pt_root: str = "structure_module.score_model.token_transformer",
    hk_root: str = "token_transformer",
) -> list[Rule]:
    """Per-layer rules for Boltz token transformer.

    Pairs with residudes_engine_jax.haiku.boltz_token_transformer.BoltzTokenTransformer.
    """
    pt = f"{pt_root}.layers.{i}"
    hk = f"{hk_root}/layer_{i}"
    mh_w = linear_w_to_haiku_multihead(num_head, head_dim)
    mh_b = linear_b_to_haiku_multihead(num_head, head_dim)
    return [
        # AdaLN at layer level
        SimpleRule(f"{pt}.adaln.s_norm.weight", f"{hk}/adaln_single_cond_layer_norm/scale"),
        SimpleRule(f"{pt}.adaln.s_scale.weight", f"{hk}/adaln_single_cond_scale/w", linear_w_to_haiku),
        SimpleRule(f"{pt}.adaln.s_scale.bias", f"{hk}/adaln_single_cond_scale/b"),
        SimpleRule(f"{pt}.adaln.s_bias.weight", f"{hk}/adaln_single_cond_bias/w", linear_w_to_haiku),
        # pair_bias_attn (no LN, q has bias, others no bias)
        SimpleRule(f"{pt}.pair_bias_attn.proj_q.weight", f"{hk}/pair_bias_attn_q_projection/w", mh_w),
        SimpleRule(f"{pt}.pair_bias_attn.proj_q.bias", f"{hk}/pair_bias_attn_q_projection/b", mh_b),
        SimpleRule(f"{pt}.pair_bias_attn.proj_k.weight", f"{hk}/pair_bias_attn_k_projection/w", mh_w),
        SimpleRule(f"{pt}.pair_bias_attn.proj_v.weight", f"{hk}/pair_bias_attn_v_projection/w", mh_w),
        SimpleRule(f"{pt}.pair_bias_attn.proj_g.weight", f"{hk}/pair_bias_attn_gating_query/w", linear_w_to_haiku),
        SimpleRule(f"{pt}.pair_bias_attn.proj_o.weight", f"{hk}/pair_bias_attn_transition2/w", linear_w_to_haiku),
        # AdaLN-zero gate (output_projection.0 = same Linear stored under output_projection_linear)
        SimpleRule(f"{pt}.output_projection.0.weight", f"{hk}/output_projection/w", linear_w_to_haiku),
        SimpleRule(f"{pt}.output_projection.0.bias", f"{hk}/output_projection/b"),
        # Conditioned transition (3-linear gated FFW)
        SimpleRule(f"{pt}.transition.adaln.s_norm.weight", f"{hk}/transition_adaln_single_cond_layer_norm/scale"),
        SimpleRule(f"{pt}.transition.adaln.s_scale.weight", f"{hk}/transition_adaln_single_cond_scale/w", linear_w_to_haiku),
        SimpleRule(f"{pt}.transition.adaln.s_scale.bias", f"{hk}/transition_adaln_single_cond_scale/b"),
        SimpleRule(f"{pt}.transition.adaln.s_bias.weight", f"{hk}/transition_adaln_single_cond_bias/w", linear_w_to_haiku),
        SimpleRule(f"{pt}.transition.swish_gate.0.weight", f"{hk}/transition_swish_gate/w", linear_w_to_haiku),
        SimpleRule(f"{pt}.transition.a_to_b.weight", f"{hk}/transition_a_to_b/w", linear_w_to_haiku),
        SimpleRule(f"{pt}.transition.b_to_a.weight", f"{hk}/transition_b_to_a/w", linear_w_to_haiku),
        SimpleRule(f"{pt}.transition.output_projection.0.weight", f"{hk}/transition_output_projection/w", linear_w_to_haiku),
        SimpleRule(f"{pt}.transition.output_projection.0.bias", f"{hk}/transition_output_projection/b"),
    ]


def template_pairformer_rules(num_layers: int = 2) -> list[Rule]:
    """Boltz template_module's PairformerNoSeqModule (no single-track).

    PT path: ``template_module.pairformer.layers.{i}.{tri_*, transition_z}``.
    Layer count and channel widths differ from main pairformer (token_z=64,
    head_dim=32). The HK side uses a separate hk.transform scope, so the
    "pairformer/pair_former_block_{i}/..." names don't clash with the main
    pairformer's params dict — different ``apply_rules`` calls produce
    independent param trees.
    """
    pt_root = "template_module.pairformer"
    out: list[Rule] = []
    for i in range(num_layers):
        out += _tri_mul_layer_rules(i, "tri_mul_out", "triangle_multiplication_outgoing", pt_root=pt_root)
        out += _tri_mul_layer_rules(i, "tri_mul_in", "triangle_multiplication_incoming", pt_root=pt_root)
        out += _tri_att_layer_rules(i, "tri_att_start", "pair_attention1",
                                    num_head=4, head_dim=32, pt_root=pt_root)
        out += _tri_att_layer_rules(i, "tri_att_end", "pair_attention2",
                                    num_head=4, head_dim=32, pt_root=pt_root)
        out += _pair_transition_layer_rules(i, pt_root=pt_root)
    return out


def msa_pairformer_layer_rules(layer_idx: int) -> list[Rule]:
    """One MSA layer's pairformer_layer (no-seq, embedded inside MSAModule).

    PT path: ``msa_module.layers.{layer_idx}.pairformer_layer.{tri_*, transition_z}``.
    Each MSA layer carries one PairformerNoSeqLayer; they are NOT a single
    monolithic stack like template_module.pairformer.

    HK side: each rule batch builds a single-block stack so we get a clean
    "pairformer/pair_former_block_0/..." layout per MSA layer.
    """
    pt_root = f"msa_module.layers.{layer_idx}.pairformer_layer"
    # Inline reuse — the per-block helpers expect ``layers.{i}`` under pt_root,
    # but here pt_root IS already the layer. Simpler: pass i=0 with a custom
    # pt_root that omits ``.layers``.
    def _adapt(rules: list[Rule]) -> list[Rule]:
        # _tri_*_layer_rules build paths like ``{pt_root}.layers.0.tri_*``.
        # MSA layout has the components directly under pt_root (no ``.layers.0``
        # indirection). Strip the FORMATTER-added ``.layers.0`` (right after
        # pt_root) — not the first ``.layers.0`` overall (which is part of
        # pt_root for layer_idx=0).
        from residudes_engine_jax.weights._rules import CompositeRule as _C
        bad = f"{pt_root}.layers.0."
        good = f"{pt_root}."
        adapted: list[Rule] = []
        for r in rules:
            if isinstance(r, SimpleRule):
                adapted.append(SimpleRule(
                    r.pt_key.replace(bad, good, 1),
                    r.hk_target, r.transform,
                ))
            elif isinstance(r, _C):
                adapted.append(_C(
                    [k.replace(bad, good, 1) for k in r.pt_keys],
                    r.hk_target, r.transform,
                ))
        return adapted

    out: list[Rule] = []
    out += _adapt(_tri_mul_layer_rules(0, "tri_mul_out", "triangle_multiplication_outgoing", pt_root=pt_root))
    out += _adapt(_tri_mul_layer_rules(0, "tri_mul_in", "triangle_multiplication_incoming", pt_root=pt_root))
    out += _adapt(_tri_att_layer_rules(0, "tri_att_start", "pair_attention1",
                                       num_head=4, head_dim=32, pt_root=pt_root))
    out += _adapt(_tri_att_layer_rules(0, "tri_att_end", "pair_attention2",
                                       num_head=4, head_dim=32, pt_root=pt_root))
    out += _adapt(_pair_transition_layer_rules(0, pt_root=pt_root))
    return out


def token_transformer_rules(num_layers: int = 24) -> list[Rule]:
    """Boltz-2 token (diffusion) transformer rules.

    Pairs with residudes_engine_jax.haiku.boltz_token_transformer.BoltzTokenTransformer.
    The PT state_dict has duplicate ``output_projection_linear.*`` keys (a
    Sequential alias of ``output_projection.0.*``); we consume the canonical
    ``output_projection.0.*`` form and ignore the alias.
    """
    out: list[Rule] = []
    for i in range(num_layers):
        out += _boltz_tt_layer_rules(i)
    return out
