# SPDX-License-Identifier: CC-BY-NC-SA-4.0
"""Confidence and distogram heads (alphafold3.model.network.confidence_head / distogram_head)."""
from __future__ import annotations

import torch

from . import nn
from .modules import pairformer_iteration
from .nn import Scope
from .template import dgram_from_positions

NUM_CONF_LAYERS = 4


def _bin_centers(max_bin: float, num_bins: int, device) -> torch.Tensor:
    breaks = torch.linspace(0.0, max_bin, num_bins - 1, device=device)
    step = breaks[1] - breaks[0]
    centers = breaks + step / 2
    return torch.cat([centers, centers[-1:] + step])


def _tm_adjusted(num_tokens: torch.Tensor, centers: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
    clipped = torch.clamp(num_tokens.float(), min=19.0)
    d0 = 1.24 * (clipped - 15) ** (1.0 / 3) - 1.8
    tm_per_bin = 1.0 / (1 + centers[None, None, :].square() / d0[:, :, None].square())
    return (probs * tm_per_bin).sum(-1)


def confidence_head(dense_atom_positions: torch.Tensor, embeddings: dict, seq_mask: torch.Tensor,
                    token_atoms_to_pseudo_beta: nn.GatherInfo, asym_id: torch.Tensor, s: Scope,
                    *, dtype: torch.dtype = torch.bfloat16) -> dict:
    """One diffusion sample → confidence outputs (pLDDT, PAE, PDE, resolved, TM-adjusted PAE)."""
    device = seq_mask.device
    seq_mask_cast = seq_mask.to(dtype)
    pair_mask = (seq_mask_cast[:, None] * seq_mask_cast[None, :]).to(dtype)
    pair_act = embeddings["pair"].to(dtype)
    single_act = embeddings["single"].to(dtype)
    target_feat = embeddings["target_feat"].to(dtype)
    n = seq_mask.shape[0]

    ef = s.child("~_embed_features")
    emb = nn.linear(target_feat, ef, "left_target_feat_project")[None, :, :] + nn.linear(target_feat, ef, "right_target_feat_project")[:, None, :]
    positions = nn.convert(token_atoms_to_pseudo_beta, dense_atom_positions.float(), layout_axes=(-3, -2))
    dgram = dgram_from_positions(positions) * pair_mask.float()[..., None]
    emb = emb + nn.linear(dgram.to(dtype), ef, "distogram_feat_project")
    pair_act = pair_act + emb

    for i in range(NUM_CONF_LAYERS):
        pair_act, single_act = pairformer_iteration(pair_act, pair_mask, s.stack("__layer_stack_no_per_layer", i).child("confidence_pairformer"),
                                                    single_act=single_act, seq_mask=seq_mask.to(dtype))
    pair_act = pair_act.float()
    pair_mask_f = pair_mask.float()

    left = nn.linear(nn.layer_norm(pair_act, s, "logits_ln"), s, "left_half_distance_logits")
    distance_logits = left + left.transpose(0, 1)
    centers = _bin_centers(31.0, 64, device)
    probs = torch.softmax(distance_logits, dim=-1)
    pde = (probs * centers).sum(-1) * pair_mask_f
    average_pde = pde.sum() / pair_mask_f.sum()

    pae_logits = nn.linear(nn.layer_norm(pair_act, s, "pae_logits_ln"), s, "pae_logits")
    pae_probs = torch.softmax(pae_logits, dim=-1)
    seq_bool = seq_mask.bool()
    pair_bool = (seq_bool[:, None] & seq_bool[None, :])
    pae = (pae_probs * centers).sum(-1) * pair_bool.float()

    x = asym_id[None, :] == asym_id[:, None]
    num_chain_tokens = (x.long() * pair_bool.long()).sum(-1)
    num_interface = num_chain_tokens[None, :] + num_chain_tokens[:, None]
    num_interface = num_interface - x.long() * (num_interface // 2)
    num_interface = num_interface * pair_bool.long()
    num_global = torch.full(pair_bool.shape, int(seq_bool.sum()), device=device, dtype=torch.long)
    tm_global = _tm_adjusted(num_global, centers, pae_probs)
    tm_interface = _tm_adjusted(num_interface, centers, pae_probs)

    single_act = single_act.float()
    plddt_logits = nn.linear(nn.layer_norm(single_act, s, "plddt_logits_ln"), s, "plddt_logits")       # (N, D, 50)
    bw = 1.0 / plddt_logits.shape[-1]
    plddt_centers = torch.arange(0.5 * bw, 1.0, bw, device=device)
    predicted_lddt = (torch.softmax(plddt_logits, dim=-1) * plddt_centers).sum(-1) * 100.0
    resolved_logits = nn.linear(nn.layer_norm(single_act, s, "experimentally_resolved_ln"), s, "experimentally_resolved_logits")
    resolved = torch.softmax(resolved_logits, dim=-1)[..., 1]
    return {
        "predicted_lddt": predicted_lddt, "predicted_experimentally_resolved": resolved,
        "full_pde": pde, "full_pde_logits": distance_logits, "full_plddt_logits": plddt_logits, "average_pde": average_pde,
        "full_pae_logits": pae_logits, "full_pae": pae,
        "tmscore_adjusted_pae_global": tm_global, "tmscore_adjusted_pae_interface": tm_interface,
    }


def distogram_head(batch, embeddings: dict, s: Scope, *, return_distogram: bool) -> dict:
    pair_act = embeddings["pair"].float()
    seq_mask = batch.token_features.mask.bool()
    pair_mask = (seq_mask[:, None] & seq_mask[None, :]).float()
    half = nn.linear(pair_act, s, "half_logits")
    logits = half + half.transpose(0, 1)
    probs = torch.softmax(logits, dim=-1)
    breaks = torch.linspace(2.3125, 21.6875, 63, device=pair_act.device)
    bin_tops = torch.cat([breaks, (breaks[-1] + (breaks[-1] - breaks[-2]))[None]])
    is_contact = (bin_tops <= 8.0 + 1e-3).float()
    contact = torch.einsum("ijk,k->ij", probs, is_contact) * pair_mask
    out = {"bin_edges": breaks, "contact_probs": contact}
    if return_distogram:
        out["distogram"] = logits
    return out
