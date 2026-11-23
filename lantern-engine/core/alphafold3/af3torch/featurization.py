# SPDX-License-Identifier: CC-BY-NC-SA-4.0
"""Model-side feature construction (alphafold3.model.network.featurization) in PyTorch,
plus the batch container that moves the featuriser's numpy output to the device."""
from __future__ import annotations

import dataclasses
import math
import types

import numpy as np
import torch
import torch.nn.functional as F

from alphafold3.model import feat_batch
from alphafold3.model.atom_layout import atom_layout as af3_layout

from . import nn

POLYMER_TYPES = 31          # residue_names.POLYMER_TYPES_NUM_WITH_UNKNOWN_AND_GAP


# ── batch on the device ───────────────────────────────────────────────────

def _tensor(arr, device) -> torch.Tensor:
    a = np.asarray(arr)
    if a.dtype == np.bool_:
        return torch.from_numpy(a.copy()).to(device)
    if a.dtype.kind in "iu":
        return torch.from_numpy(a.astype(np.int64)).to(device)
    if a.dtype.kind == "f":
        return torch.from_numpy(a.astype(np.float32)).to(device)
    raise TypeError(f"unsupported feature dtype {a.dtype}")


def _convert(obj, device):
    if isinstance(obj, af3_layout.GatherInfo):
        return nn.GatherInfo(_tensor(obj.gather_idxs, device), _tensor(obj.gather_mask, device), np.asarray(obj.input_shape))
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        ns = types.SimpleNamespace()
        for f in dataclasses.fields(obj):
            v = getattr(obj, f.name)
            try:
                setattr(ns, f.name, _convert(v, device))
            except TypeError:
                setattr(ns, f.name, v)          # structures, layouts: kept as-is (used by post-processing only)
        return ns
    if isinstance(obj, (np.ndarray, np.generic)):
        return _tensor(obj, device)
    if hasattr(obj, "shape") and hasattr(obj, "dtype"):
        return _tensor(np.asarray(obj), device)
    raise TypeError(type(obj))


def batch_to_device(batch_dict: dict, device) -> types.SimpleNamespace:
    """feat_batch.Batch (numpy) → nested namespaces of tensors with the same attribute paths."""
    b = feat_batch.Batch.from_data_dict(batch_dict)
    ns = types.SimpleNamespace()
    for f in dataclasses.fields(b):
        v = getattr(b, f.name)
        try:
            setattr(ns, f.name, _convert(v, device))
        except TypeError:
            setattr(ns, f.name, v)
    ns.num_res = int(b.num_res)
    return ns


# ── features ──────────────────────────────────────────────────────────────

def one_hot(x: torch.Tensor, n: int, dtype=torch.float32) -> torch.Tensor:
    x = x.long().clamp(min=0, max=n - 1) if x.numel() else x.long()
    return F.one_hot(x, n).to(dtype)


def create_target_feat(batch) -> torch.Tensor:
    tf = batch.token_features
    return torch.cat([one_hot(tf.aatype, POLYMER_TYPES), batch.msa.profile.float(),
                      batch.msa.deletion_mean.float()[..., None]], dim=-1)


def create_msa_feat(rows: torch.Tensor, deletion_matrix: torch.Tensor) -> torch.Tensor:
    msa_1hot = one_hot(rows, POLYMER_TYPES + 1)
    has_deletion = deletion_matrix.clamp(0.0, 1.0)[..., None]
    deletion_value = (torch.arctan(deletion_matrix / 3.0) * (2.0 / math.pi))[..., None]
    return torch.cat([msa_1hot, has_deletion, deletion_value], dim=-1)


def create_relative_encoding(tf, max_relative_idx: int = 32, max_relative_chain: int = 2,
                             cyclic_period_chain: int = 0, cyclic_chain_bin_amplify=None) -> torch.Tensor:
    """Relative position / token / entity / chain features: (N, N, 139) float32."""
    asym_id, residue_index, token_index = tf.asym_id.long(), tf.residue_index.long(), tf.token_index.long()
    entity_id, sym_id = tf.entity_id.long(), tf.sym_id.long()
    feats = []
    offset = residue_index[:, None] - residue_index[None, :]
    clipped = (offset + max_relative_idx).clamp(0, 2 * max_relative_idx)
    same_asym = asym_id[:, None] == asym_id[None, :]
    final = torch.where(same_asym, clipped, torch.full_like(clipped, 2 * max_relative_idx + 1))
    feats.append(one_hot(final, 2 * max_relative_idx + 2))

    token_offset = token_index[:, None] - token_index[None, :]
    clipped_tok = (token_offset + max_relative_idx).clamp(0, 2 * max_relative_idx)
    residue_same = same_asym & (residue_index[:, None] == residue_index[None, :])
    final_tok = torch.where(residue_same, clipped_tok, torch.full_like(clipped_tok, 2 * max_relative_idx + 1))
    feats.append(one_hot(final_tok, 2 * max_relative_idx + 2))

    entity_same = entity_id[:, None] == entity_id[None, :]
    feats.append(entity_same.float()[..., None])

    rel_sym = (sym_id[:, None] - sym_id[None, :]).float()
    if cyclic_period_chain > 0:
        rel_sym = rel_sym - cyclic_period_chain * torch.round(rel_sym / cyclic_period_chain)
    clipped_chain = (rel_sym + max_relative_chain).clamp(0, 2 * max_relative_chain).long()
    final_chain = torch.where(entity_same, clipped_chain, torch.full_like(clipped_chain, 2 * max_relative_chain + 1))
    rel_chain = one_hot(final_chain, 2 * max_relative_chain + 2)
    if cyclic_chain_bin_amplify is not None:
        rel_chain = rel_chain * torch.tensor(cyclic_chain_bin_amplify, dtype=rel_chain.dtype, device=rel_chain.device)
    feats.append(rel_chain)
    return torch.cat(feats, dim=-1)


def shuffle_and_truncate_msa(msa, num_msa: int, gen: torch.Generator | None):
    """shuffle_msa + truncate_msa_batch: a Gumbel-argsort permutation over rows with at
    least one unmasked position, then the first num_msa rows. Returns (rows, mask, deletion_matrix)."""
    mask = msa.mask.float()
    logits = (mask.sum(dim=-1).clamp(0.0, 1.0) - 1.0) * 1e6
    if gen is None:
        perm = torch.argsort(logits, descending=True, stable=True)
    else:
        u = torch.rand(logits.shape, generator=gen, device=gen.device if hasattr(gen, "device") else None).to(logits.device)
        gumbel = -torch.log(torch.clamp_min(-torch.log(u.clamp_min(1e-20)), 1e-20))   # -log(-log u); the inner clamp must follow the negation
        perm = torch.argsort(logits + gumbel, descending=True)
    idx = torch.arange(num_msa, device=perm.device).clamp(max=perm.shape[0] - 1)
    sel = perm[idx]
    return msa.rows[sel], msa.mask[sel], msa.deletion_matrix[sel]
