# SPDX-License-Identifier: CC-BY-NC-SA-4.0
"""The AlphaFold 3 trunk (alphafold3.model.network.evoformer) in PyTorch: pair and
single embeddings from target features, the previous recycle, relative positions,
bonds, templates, the MSA stack and 48 pairformer blocks."""
from __future__ import annotations

import os

import torch

from . import nn
from .atom_cross_attention import encoder
from .featurization import create_msa_feat, create_relative_encoding, create_target_feat, shuffle_and_truncate_msa
from .modules import evoformer_iteration, pairformer_iteration
from .nn import Scope
from .template import template_embedding

PAIR_CHANNEL, SEQ_CHANNEL, MSA_CHANNEL, NUM_MSA = 128, 384, 64, 1024
NUM_PAIRFORMER, NUM_MSA_STACK = 48, 4


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name, "")
    try:
        return float(v) if v else default
    except ValueError:
        return default


def create_target_feat_embedding(batch, s_root: Scope, dtype: torch.dtype) -> torch.Tensor:
    """Target features plus the per-atom conditioning encoder's token activations: (N, 447)."""
    target_feat = create_target_feat(batch).to(dtype)
    enc = encoder(None, None, None, batch, s_root, "evoformer_conditioning", per_token_channels=SEQ_CHANNEL)
    return torch.cat([target_feat, enc.token_act.to(dtype)], dim=-1)


def _embed_bonds(batch, pair: torch.Tensor, s: Scope) -> torch.Tensor:
    n = batch.token_features.token_index.shape[0]
    pl = batch.polymer_ligand_bond_info.tokens_to_polymer_ligand_bonds
    ll = batch.ligand_ligand_bond_info.tokens_to_ligand_ligand_bonds
    idx_pl = pl.gather_idxs * pl.gather_mask.long().prod(dim=1)[:, None]
    idx_ll = ll.gather_idxs * ll.gather_mask.long().prod(dim=1)[:, None]
    idx = torch.cat([idx_pl, idx_ll], dim=0)
    contact = torch.zeros((n, n), dtype=torch.float32, device=pair.device)
    if idx.numel():
        contact[idx[:, 0], idx[:, 1]] = 1.0
    contact[0, 0] = 0.0
    return pair + nn.linear(contact[:, :, None].to(pair.dtype), s, "bond_embedding")


def _embed_templates(batch, pair: torch.Tensor, pair_mask: torch.Tensor, s: Scope) -> torch.Tensor:
    tpl = batch.templates
    if tpl.aatype.shape[0] == 0:
        return pair
    dtype = pair.dtype
    asym_id = batch.token_features.asym_id
    tok_covered = tpl.atom_mask.bool().any(dim=-1).long()                                  # (T, N)
    cc_only = tok_covered * tpl.is_cross_chain.long()[:, None]
    cc = cc_only.to(torch.float32)                                                           # CUDA has no int64 matmul
    pair_covered = ((cc.T @ cc) > 0).to(dtype)
    block_diag = (asym_id[:, None] == asym_id[None, :]).to(dtype)
    multichain_mask = torch.maximum(block_diag, pair_covered)
    act = template_embedding(pair, tpl, pair_mask, multichain_mask, s.child("template_embedding"))
    w = _env_float("AF3_TPL_WEIGHT", 1.0)
    if w != 1.0:
        act = act * w
    return pair + act


def evoformer(batch, prev: dict, target_feat: torch.Tensor, s: Scope, *, gen: torch.Generator | None,
              dtype: torch.dtype = torch.bfloat16, engines=None) -> dict:
    tf = batch.token_features
    mask = tf.mask.float()
    left = nn.linear(target_feat, s, "left_single")[:, None]
    right = nn.linear(target_feat, s, "right_single")[None]
    pair = left + right
    pair_mask = (mask[:, None] * mask[None, :]).to(dtype)

    pair = pair + nn.linear(nn.layer_norm(prev["pair"].to(dtype), s, "prev_embedding_layer_norm"), s, "prev_embedding")
    rel = create_relative_encoding(tf, 32, 2, int(_env_float("AF3_CYCLIC_PERIOD_CHAIN", 0)),
                                   None if not os.environ.get("AF3_CYCLIC_CHAIN_BIN_AMPLIFY") else
                                   [float(x) for x in os.environ["AF3_CYCLIC_CHAIN_BIN_AMPLIFY"].split(",")])
    pair = pair + nn.linear(rel.to(dtype), s.child("~_relative_encoding"), "position_activations")
    pair = _embed_bonds(batch, pair, s)
    pair = _embed_templates(batch, pair, pair_mask, s)

    # MSA stack
    rows, msa_mask, deletion = shuffle_and_truncate_msa(batch.msa, NUM_MSA, gen)
    msa_feat = create_msa_feat(rows, deletion.float()).to(dtype)
    msa_act = nn.linear(msa_feat, s, "msa_activations") + nn.linear(target_feat, s, "extra_msa_target_feat")[None]
    msa_mask = msa_mask.to(dtype)
    for i in range(NUM_MSA_STACK):
        msa_act, pair = evoformer_iteration(msa_act, pair, msa_mask, pair_mask, s.stack("__layer_stack_no_per_layer", i).child("msa_stack"))
    del msa_act

    single = nn.linear(target_feat, s, "single_activations")
    single = single + nn.linear(nn.layer_norm(prev["single"].to(dtype), s, "prev_single_embedding_layer_norm"), s, "prev_single_embedding")
    seq_mask = mask.to(dtype)
    if engines is not None and engines.pairformer is not None:
        single, pair = engines.trunk(single, pair, seq_mask, pair_mask)
    else:
        for i in range(NUM_PAIRFORMER):
            pair, single = pairformer_iteration(pair, pair_mask, s.stack("__layer_stack_no_per_layer_1", i).child("trunk_pairformer"),
                                                single_act=single, seq_mask=seq_mask)
    return {"single": single, "pair": pair, "target_feat": target_feat}
