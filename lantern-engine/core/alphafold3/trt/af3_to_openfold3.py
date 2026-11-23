#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""AlphaFold 3 parameters → an OpenFold 3-named state dict.

The TensorRT engine recipes (tensorrt_bionemo) read OpenFold 3 checkpoint names
for the pairformer stack and the diffusion token transformer. AlphaFold 3 has the
same two blocks with the same shapes, so expressing its published parameters in
those names lets the OpenFold 3 converters, the engine builder and the engine's
own torch reference models consume them unchanged. Only two things need care:

* the triangle-multiplication projections interleave the two halves by channel
  (index 2c + {0, 1}) where OpenFold 3 keeps two separate Linears, and the
  incoming node's operands are swapped between the two code bases;
* the diffusion transformer projects pair logits once per super block of four
  layers (weights shaped (128, 4, 16)); every layer gets its own slice.

    python af3_to_openfold3.py --model-dir /path/with/af3.bin --out af3_as_openfold3.pt
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from af3torch.params import Params, Scope  # noqa: E402

NUM_PAIRFORMER, NUM_TOKEN_BLOCKS, SUPER_BLOCK = 48, 24, 4
C_S, C_Z, C_A = 384, 128, 768
# Which half of a SwiGLU projection the engine applies SiLU to, by OpenFold 3 name
# (checked numerically against the engine's torch reference in tests/debug_trt_reference.py).
PAIRFORMER_SILU_ON = "linear_a"
TOKEN_SILU_ON = "linear_a"


def _f(s: Scope, name: str, kind: str = "weights") -> torch.Tensor:
    return s.get(name, kind, dtype=torch.float32).detach().cpu().contiguous()


def _swiglu(w1: torch.Tensor, silu_on: str) -> tuple[torch.Tensor, torch.Tensor]:
    """AlphaFold 3's transition1 (in, 2·inter): swish on the first half. Returns (linear_a, linear_b)
    weights in torch Linear layout (out, in), placing the swish-gated half where the engine applies SiLU."""
    inter = w1.shape[1] // 2
    gated, plain = w1[:, :inter].T.contiguous(), w1[:, inter:].T.contiguous()
    return (gated, plain) if silu_on == "linear_a" else (plain, gated)


def _tri_attention(sd: dict, pre: str, s: Scope) -> None:
    sd[f"{pre}.layer_norm.weight"] = _f(s, "act_norm", "scale")
    sd[f"{pre}.layer_norm.bias"] = _f(s, "act_norm", "offset")
    sd[f"{pre}.linear_z.weight"] = _f(s, "pair_bias_projection").T.contiguous()            # (H, C_z)
    q, k = _f(s, "q_projection"), _f(s, "k_projection")                                      # (H, D, C_z): already (out, in)
    sd[f"{pre}.mha.linear_q.weight"] = q.reshape(-1, q.shape[-1]).contiguous()
    sd[f"{pre}.mha.linear_k.weight"] = k.reshape(-1, k.shape[-1]).contiguous()
    v = _f(s, "v_projection")                                                                 # (C_z, H, D)
    sd[f"{pre}.mha.linear_v.weight"] = v.reshape(v.shape[0], -1).T.contiguous()
    sd[f"{pre}.mha.linear_o.weight"] = _f(s, "output_projection").T.contiguous()            # (C_z, H·D)
    sd[f"{pre}.mha.linear_g.weight"] = _f(s, "gating_query").contiguous()                    # (H·D, C_z): already (out, in)


def _tri_multiplication(sd: dict, pre: str, s: Scope, *, swap: bool) -> None:
    proj, gate = _f(s, "projection"), _f(s, "gate")                                          # (C, 2C), channels interleaved
    a_p, b_p = proj[:, 0::2].T.contiguous(), proj[:, 1::2].T.contiguous()
    a_g, b_g = gate[:, 0::2].T.contiguous(), gate[:, 1::2].T.contiguous()
    if swap:                                                                                  # incoming: engine contracts a over (k,i)
        a_p, b_p, a_g, b_g = b_p, a_p, b_g, a_g
    sd[f"{pre}.linear_a_p.weight"], sd[f"{pre}.linear_b_p.weight"] = a_p, b_p
    sd[f"{pre}.linear_a_g.weight"], sd[f"{pre}.linear_b_g.weight"] = a_g, b_g
    sd[f"{pre}.layer_norm_in.weight"] = _f(s, "left_norm_input", "scale")
    sd[f"{pre}.layer_norm_in.bias"] = _f(s, "left_norm_input", "offset")
    sd[f"{pre}.layer_norm_out.weight"] = _f(s, "center_norm", "scale")
    sd[f"{pre}.layer_norm_out.bias"] = _f(s, "center_norm", "offset")
    sd[f"{pre}.linear_z.weight"] = _f(s, "output_projection").T.contiguous()
    sd[f"{pre}.linear_g.weight"] = _f(s, "gating_linear").T.contiguous()


def _transition(sd: dict, pre: str, s: Scope) -> None:
    sd[f"{pre}.layer_norm.weight"] = _f(s, "input_layer_norm", "scale")
    sd[f"{pre}.layer_norm.bias"] = _f(s, "input_layer_norm", "offset")
    a, b = _swiglu(_f(s, "transition1"), PAIRFORMER_SILU_ON)
    sd[f"{pre}.swiglu.linear_a.weight"], sd[f"{pre}.swiglu.linear_b.weight"] = a, b
    sd[f"{pre}.linear_out.weight"] = _f(s, "transition2").T.contiguous()


def pairformer_state_dict(params: Params, num_blocks: int = NUM_PAIRFORMER) -> dict[str, torch.Tensor]:
    sd: dict[str, torch.Tensor] = {}
    ev = params.root().child("diffuser").child("evoformer")
    for i in range(num_blocks):
        s = ev.stack("__layer_stack_no_per_layer_1", i).child("trunk_pairformer")
        pre = f"pairformer_stack.blocks.{i}"
        # single attention with pair bias
        ap = f"{pre}.attn_pair_bias"
        sd[f"{ap}.layer_norm_a.weight"] = _f(s, "single_attention_layer_norm", "scale")
        sd[f"{ap}.layer_norm_a.bias"] = _f(s, "single_attention_layer_norm", "offset")
        sd[f"{ap}.layer_norm_z.weight"] = _f(s, "single_pair_logits_norm", "scale")
        sd[f"{ap}.layer_norm_z.bias"] = _f(s, "single_pair_logits_norm", "offset")
        sd[f"{ap}.linear_z.weight"] = _f(s, "single_pair_logits_projection").T.contiguous()  # (H, C_z)
        q = _f(s, "single_attention_q_projection")                                            # (C_s, H, D)
        sd[f"{ap}.mha.linear_q.weight"] = q.reshape(C_S, -1).T.contiguous()
        sd[f"{ap}.mha.linear_q.bias"] = _f(s, "single_attention_q_projection", "bias").reshape(-1).contiguous()
        sd[f"{ap}.mha.linear_k.weight"] = _f(s, "single_attention_k_projection").reshape(C_S, -1).T.contiguous()
        sd[f"{ap}.mha.linear_v.weight"] = _f(s, "single_attention_v_projection").reshape(C_S, -1).T.contiguous()
        sd[f"{ap}.mha.linear_g.weight"] = _f(s, "single_attention_gating_query").T.contiguous()
        sd[f"{ap}.mha.linear_o.weight"] = _f(s, "single_attention_transition2").T.contiguous()
        _transition(sd, f"{pre}.single_transition", s.child("single_transition"))
        # pair stack
        _tri_attention(sd, f"{pre}.pair_stack.tri_att_start", s.child("pair_attention1"))
        _tri_attention(sd, f"{pre}.pair_stack.tri_att_end", s.child("pair_attention2"))
        _tri_multiplication(sd, f"{pre}.pair_stack.tri_mul_out", s.child("triangle_multiplication_outgoing"), swap=False)
        _tri_multiplication(sd, f"{pre}.pair_stack.tri_mul_in", s.child("triangle_multiplication_incoming"), swap=True)
        _transition(sd, f"{pre}.pair_stack.pair_transition", s.child("pair_transition"))
    return sd


def token_transformer_state_dict(params: Params, num_blocks: int = NUM_TOKEN_BLOCKS) -> dict[str, torch.Tensor]:
    sd: dict[str, torch.Tensor] = {}
    tr = params.root().child("diffuser").child("~").child("diffusion_head").child("transformer")
    ln_z = _f(tr, "pair_input_layer_norm", "scale")
    base = "sample_diffusion.diffusion_module.diffusion_transformer.blocks"
    for layer in range(num_blocks):
        sb, b = divmod(layer, SUPER_BLOCK)
        outer = tr.stack("__layer_stack_with_per_layer", sb)
        inner = outer.stack("__layer_stack_with_per_layer", b)
        plp = _f(outer, "pair_logits_projection")                                             # (C_z, 4, H)
        pre = f"{base}.{layer}"
        g = lambda n, k="weights": _f(inner, f"transformer{n}", k)  # noqa: E731
        ap = f"{pre}.attention_pair_bias"
        sd[f"{ap}.layer_norm_a.layer_norm_s.weight"] = g("single_cond_layer_norm", "scale")
        sd[f"{ap}.layer_norm_a.linear_g.weight"] = g("single_cond_scale").T.contiguous()
        sd[f"{ap}.layer_norm_a.linear_g.bias"] = g("single_cond_scale", "bias")
        sd[f"{ap}.layer_norm_a.linear_s.weight"] = g("single_cond_bias").T.contiguous()
        q = g("q_projection")                                                                 # (C_a, H, D)
        sd[f"{ap}.mha.linear_q.weight"] = q.reshape(C_A, -1).T.contiguous()
        sd[f"{ap}.mha.linear_q.bias"] = g("q_projection", "bias").reshape(-1).contiguous()
        sd[f"{ap}.mha.linear_k.weight"] = g("k_projection").reshape(C_A, -1).T.contiguous()
        sd[f"{ap}.mha.linear_v.weight"] = g("v_projection").reshape(C_A, -1).T.contiguous()
        sd[f"{ap}.mha.linear_g.weight"] = g("gating_query").T.contiguous()
        sd[f"{ap}.mha.linear_o.weight"] = g("transition2").T.contiguous()
        sd[f"{ap}.layer_norm_z.weight"] = ln_z.clone()                                        # shared across layers, no bias
        sd[f"{ap}.linear_z.weight"] = plp[:, b, :].T.contiguous()                             # (H, C_z)
        sd[f"{ap}.linear_ada_out.weight"] = g("adaptive_zero_cond").T.contiguous()
        sd[f"{ap}.linear_ada_out.bias"] = g("adaptive_zero_cond", "bias")
        ct = f"{pre}.conditioned_transition"
        sd[f"{ct}.layer_norm.layer_norm_s.weight"] = g("ffw_single_cond_layer_norm", "scale")
        sd[f"{ct}.layer_norm.linear_g.weight"] = g("ffw_single_cond_scale").T.contiguous()
        sd[f"{ct}.layer_norm.linear_g.bias"] = g("ffw_single_cond_scale", "bias")
        sd[f"{ct}.layer_norm.linear_s.weight"] = g("ffw_single_cond_bias").T.contiguous()
        a, b_ = _swiglu(g("ffw_transition1"), TOKEN_SILU_ON)
        sd[f"{ct}.swiglu.linear_a.weight"], sd[f"{ct}.swiglu.linear_b.weight"] = a, b_
        sd[f"{ct}.linear_out.weight"] = g("ffw_transition2").T.contiguous()
        sd[f"{ct}.linear_g.weight"] = g("ffw_adaptive_zero_cond").T.contiguous()
        sd[f"{ct}.linear_g.bias"] = g("ffw_adaptive_zero_cond", "bias")
    return sd


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", required=True, help="directory holding af3.bin")
    ap.add_argument("--out", required=True, help="output .pt (a plain state dict in OpenFold 3 names)")
    args = ap.parse_args()
    params = Params(pathlib.Path(args.model_dir), device="cpu")
    sd = pairformer_state_dict(params)
    sd.update(token_transformer_state_dict(params))
    torch.save(sd, args.out)
    print(f"{len(sd)} tensors → {args.out}")


if __name__ == "__main__":
    main()
