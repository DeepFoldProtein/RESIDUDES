# SPDX-License-Identifier: CC-BY-NC-SA-4.0
"""Template embedding (alphafold3.model.network.template_modules) in PyTorch."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from alphafold3.model import protein_data_processing as pdp

from . import nn
from .featurization import one_hot, POLYMER_TYPES
from .modules import pairformer_iteration
from .nn import Scope

DGRAM_MIN, DGRAM_MAX, DGRAM_BINS = 3.25, 50.75, 39


def dgram_from_positions(positions: torch.Tensor) -> torch.Tensor:
    """(N, 3) → (N, N, 39) one-hot distance bins (squared-distance thresholds, open last bin)."""
    lower = torch.linspace(DGRAM_MIN, DGRAM_MAX, DGRAM_BINS, device=positions.device, dtype=torch.float32).square()
    upper = torch.cat([lower[1:], torch.tensor([1e8], device=positions.device)])
    dist2 = (positions[:, None, :] - positions[None, :, :]).square().sum(-1, keepdim=True)
    return (dist2 > lower).float() * (dist2 < upper).float()


def pseudo_beta(aatype: torch.Tensor, positions: torch.Tensor, masks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """scoring.pseudo_beta_fn without ligand flags: CB (CA for glycine) per residue."""
    table = torch.as_tensor(pdp.RESTYPE_PSEUDOBETA_INDEX, device=aatype.device).long()
    idx = table[aatype.long()]
    ar = torch.arange(aatype.shape[0], device=aatype.device)
    return positions[ar, idx], masks[ar, idx].float()


def backbone_unit_vectors(aatype: torch.Tensor, positions: torch.Tensor, mask: torch.Tensor):
    """make_backbone_rigid followed by `rigid[:, None].inverse().apply_to_point(translations)`:
    for every residue pair, the unit vector to the other residue's backbone origin in this
    residue's frame. Returns ((N, N) x, y, z) and the per-residue frame mask (N,)."""
    table = torch.as_tensor(pdp.RESTYPE_RIGIDGROUP_DENSE_ATOM_IDX, device=aatype.device).long()
    bb = table[aatype.long()][:, 0]                                  # (N, 3) → c, b, a
    c_i, b_i, a_i = bb[:, 0], bb[:, 1], bb[:, 2]
    ar = torch.arange(aatype.shape[0], device=aatype.device)
    m = mask.float()
    rigid_mask = m[ar, a_i] * m[ar, b_i] * m[ar, c_i]
    pa, pb, pc = positions[ar, a_i], positions[ar, b_i], positions[ar, c_i]

    def normalized(v, eps=1e-6):
        return v / torch.sqrt(torch.clamp((v * v).sum(-1, keepdim=True), min=eps * eps))

    e0 = normalized(pc - pb)
    e1 = pa - pb
    e1 = normalized(e1 - (e1 * e0).sum(-1, keepdim=True) * e0)
    e2 = torch.cross(e0, e1, dim=-1)
    rot = torch.stack([e0, e1, e2], dim=-1)                          # columns e0, e1, e2: R[i] (3, 3)
    t = pb                                                           # translation (N, 3)
    diff = t[None, :, :] - t[:, None, :]                             # (N_i, N_j, 3): t_j - t_i
    local = torch.einsum("iab,ija->ijb", rot, diff)                  # R_iᵀ (t_j - t_i)
    unit = normalized(local)
    return unit[..., 0], unit[..., 1], unit[..., 2], rigid_mask


def single_template_embedding(query: torch.Tensor, aatype: torch.Tensor, atom_positions: torch.Tensor, atom_mask: torch.Tensor,
                              padding_mask_2d: torch.Tensor, multichain_mask_2d: torch.Tensor, s: Scope, *, num_channels: int = 64) -> torch.Tensor:
    dtype = query.dtype
    positions = atom_positions.float() * atom_mask.float()[..., None]
    pb_pos, pb_mask = pseudo_beta(aatype, positions, atom_mask)
    pb_mask_2d = pb_mask[:, None] * pb_mask[None, :] * multichain_mask_2d.float()
    dgram = (dgram_from_positions(pb_pos) * pb_mask_2d[..., None]).to(dtype)
    to_concat = [(dgram, 1), (pb_mask_2d.to(dtype), 0)]
    aa = one_hot(aatype, POLYMER_TYPES, dtype=dtype)
    to_concat.append((aa[None, :, :], 1))
    to_concat.append((aa[:, None, :], 1))
    ux, uy, uz, bb_mask = backbone_unit_vectors(aatype, positions, atom_mask)
    bb_mask_2d = (bb_mask[:, None] * bb_mask[None, :] * multichain_mask_2d.float()).to(dtype)
    for u in (ux, uy, uz):
        to_concat.append(((u.to(dtype) * bb_mask_2d), 0))
    to_concat.append((bb_mask_2d, 0))
    to_concat.append((nn.layer_norm(query, s, "query_embedding_norm"), 1))

    act = None
    for i, (x, n_in) in enumerate(to_concat):
        term = nn.linear(x, s, f"template_pair_embedding_{i}", num_input_dims=n_in)
        if n_in == 0:
            term = term.expand(query.shape[0], query.shape[1], num_channels) if term.ndim == 3 else term
        act = term if act is None else act + term
    for i in range(2):
        act = pairformer_iteration(act, padding_mask_2d, s.stack("__layer_stack_no_per_layer", i).child("template_embedding_iteration"),
                                   pair_transition_factor=2)
    return nn.layer_norm(act, s, "output_layer_norm")


def template_embedding(query: torch.Tensor, templates, padding_mask_2d: torch.Tensor, multichain_mask_2d: torch.Tensor,
                       s: Scope) -> torch.Tensor:
    """TemplateEmbedding: average of per-template embeddings, relu, projected back to the pair width."""
    n_templates = templates.aatype.shape[0]
    n = query.shape[0]
    summed = torch.zeros((n, n, 64), dtype=query.dtype, device=query.device)
    sub = s.child("single_template_embedding")
    for t in range(n_templates):
        summed = summed + single_template_embedding(query, templates.aatype[t], templates.atom_positions[t], templates.atom_mask[t],
                                                    padding_mask_2d, multichain_mask_2d, sub)
    emb = F.relu(summed / (1e-7 + n_templates))
    return nn.linear(emb, s, "output_linear")
