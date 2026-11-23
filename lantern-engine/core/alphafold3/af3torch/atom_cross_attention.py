# SPDX-License-Identifier: CC-BY-NC-SA-4.0
"""Per-atom cross attention encoder / decoder (alphafold3.model.network.atom_cross_attention).

`s` is the scope the Haiku caller lived in (`diffuser` for the trunk's
`evoformer_conditioning`, `diffuser/~/diffusion_head` for `diffusion`); the
Linear names are `<name>_...` strings in that scope and the transformers are
sub-modules `<name>_atom_transformer_encoder` / `_decoder`. Haiku created the
pair-conditioning Linears twice (once in `_per_atom_conditioning`, whose pair
output is discarded), so the encoder's live copies carry a `_1` suffix."""
from __future__ import annotations

import dataclasses

import torch
import torch.nn.functional as F

from . import nn
from .diffusion_transformer import cross_att_transformer
from .featurization import one_hot
from .nn import Scope


@dataclasses.dataclass
class EncoderOutput:
    token_act: torch.Tensor
    skip_connection: torch.Tensor
    queries_mask: torch.Tensor
    queries_single_cond: torch.Tensor
    keys_mask: torch.Tensor
    keys_single_cond: torch.Tensor
    pair_cond: torch.Tensor


def per_atom_conditioning(batch, s: Scope, name: str) -> torch.Tensor:
    """Single conditioning for every atom of every token from the reference conformer: (T, D, 128)."""
    ref = batch.ref_structure
    mask = ref.mask.float()
    act = nn.linear(ref.positions.float(), s, f"{name}_embed_ref_pos")
    act = act + nn.linear(mask[:, :, None], s, f"{name}_embed_ref_mask")
    act = act + nn.linear(one_hot(ref.element, 128), s, f"{name}_embed_ref_element")
    act = act + nn.linear(torch.arcsinh(ref.charge.float())[:, :, None], s, f"{name}_embed_ref_charge")
    chars = one_hot(ref.atom_name_chars, 64)
    t, d = act.shape[:2]
    act = act + nn.linear(chars.reshape(t, d, -1), s, f"{name}_embed_ref_atom_name")
    return act * mask[:, :, None]


def encoder(token_atoms_act: torch.Tensor | None, trunk_single_cond: torch.Tensor | None,
            trunk_pair_cond: torch.Tensor | None, batch, s: Scope, name: str, *, per_token_channels: int) -> EncoderOutput:
    aca = batch.atom_cross_att
    token_atoms_single_cond = per_atom_conditioning(batch, s, name)
    token_atoms_mask = batch.predicted_structure_info.atom_mask.float()
    queries_single_cond = nn.convert(aca.token_atoms_to_queries, token_atoms_single_cond, layout_axes=(-3, -2))
    queries_mask = nn.convert(aca.token_atoms_to_queries, token_atoms_mask, layout_axes=(-2, -1))

    if trunk_single_cond is not None:
        tsc = nn.linear(nn.layer_norm(trunk_single_cond.float(), s, f"{name}_lnorm_trunk_single_cond", offset=False),
                        s, f"{name}_embed_trunk_single_cond")
        queries_single_cond = queries_single_cond + nn.convert(aca.tokens_to_queries, tsc, layout_axes=(-2,))

    if token_atoms_act is None:
        queries_act = queries_single_cond
    else:
        queries_act = nn.convert(aca.token_atoms_to_queries, token_atoms_act.float(), layout_axes=(-3, -2))
        queries_act = nn.linear(queries_act, s, f"{name}_atom_positions_to_features")
        queries_act = queries_act * queries_mask[..., None]
        queries_act = queries_act + queries_single_cond

    keys_single_cond = nn.convert(aca.queries_to_keys, queries_single_cond, layout_axes=(-3, -2))
    keys_mask = nn.convert(aca.queries_to_keys, queries_mask, layout_axes=(-2, -1))

    row_act = nn.linear(F.relu(queries_single_cond), s, f"{name}_single_to_pair_cond_row_1")
    pair_cond_keys_input = nn.convert(aca.queries_to_keys, queries_single_cond, layout_axes=(-3, -2))
    col_act = nn.linear(F.relu(pair_cond_keys_input), s, f"{name}_single_to_pair_cond_col_1")
    pair_act = row_act[:, :, None, :] + col_act[:, None, :, :]

    if trunk_pair_cond is not None:
        tpc = nn.linear(nn.layer_norm(trunk_pair_cond.float(), s, f"{name}_lnorm_trunk_pair_cond", offset=False),
                        s, f"{name}_embed_trunk_pair_cond")
        num_tokens = tpc.shape[0]
        tq, tk = aca.tokens_to_queries, aca.tokens_to_keys
        gi = nn.GatherInfo(num_tokens * tq.gather_idxs[:, :, None] + tk.gather_idxs[:, None, :],
                           tq.gather_mask[:, :, None] & tk.gather_mask[:, None, :], (num_tokens, num_tokens))
        pair_act = pair_act + nn.convert(gi, tpc, layout_axes=(-3, -2))

    ref = batch.ref_structure
    queries_ref_pos = nn.convert(aca.token_atoms_to_queries, ref.positions.float(), layout_axes=(-3, -2))
    queries_ref_space_uid = nn.convert(aca.token_atoms_to_queries, ref.ref_space_uid, layout_axes=(-2, -1))
    keys_ref_pos = nn.convert(aca.queries_to_keys, queries_ref_pos, layout_axes=(-3, -2))
    keys_ref_space_uid = nn.convert(aca.queries_to_keys, ref.ref_space_uid, layout_axes=(-2, -1))
    offsets_valid = queries_ref_space_uid[:, :, None] == keys_ref_space_uid[:, None, :]
    valid = offsets_valid[..., None].float()
    offsets = queries_ref_pos[:, :, None, :] - keys_ref_pos[:, None, :, :]
    pair_act = pair_act + nn.linear(offsets, s, f"{name}_embed_pair_offsets_1") * valid
    sq = offsets.square().sum(-1)
    pair_act = pair_act + nn.linear(1.0 / (1 + sq[..., None]), s, f"{name}_embed_pair_distances_1") * valid
    pair_act = pair_act + nn.linear(valid, s, f"{name}_embed_pair_offsets_valid")

    pair_act2 = nn.linear(F.relu(pair_act), s, f"{name}_pair_mlp_1")
    pair_act2 = nn.linear(F.relu(pair_act2), s, f"{name}_pair_mlp_2")
    pair_act = pair_act + nn.linear(F.relu(pair_act2), s, f"{name}_pair_mlp_3")

    queries_act = cross_att_transformer(
        queries_act, queries_mask, aca.queries_to_keys, keys_mask, queries_single_cond, keys_single_cond, pair_act,
        s.child(f"{name}_atom_transformer_encoder"), prefix=f"{name}_atom_transformer_encoder")
    queries_act = queries_act * queries_mask[..., None]
    skip_connection = queries_act

    queries_act = nn.linear(queries_act, s, f"{name}_project_atom_features_for_aggr")
    token_atoms_act = nn.convert(aca.queries_to_token_atoms, queries_act, layout_axes=(-3, -2))
    token_act = nn.mask_mean(token_atoms_mask[..., None], F.relu(token_atoms_act), dim=-2)
    return EncoderOutput(token_act, skip_connection, queries_mask, queries_single_cond, keys_mask, keys_single_cond, pair_act)


def decoder(token_act: torch.Tensor, enc: EncoderOutput, batch, s: Scope, name: str) -> torch.Tensor:
    aca = batch.atom_cross_att
    token_act = nn.linear(token_act, s, f"{name}_project_token_features_for_broadcast")
    num_token, max_atoms = aca.queries_to_token_atoms.shape
    token_atom_act = token_act[..., :, None, :].expand(*token_act.shape[:-1], max_atoms, token_act.shape[-1])
    queries_act = nn.convert(aca.token_atoms_to_queries, token_atom_act, layout_axes=(-3, -2))
    queries_act = queries_act + enc.skip_connection
    queries_act = queries_act * enc.queries_mask[..., None]
    queries_act = cross_att_transformer(
        queries_act, enc.queries_mask, aca.queries_to_keys, enc.keys_mask, enc.queries_single_cond, enc.keys_single_cond,
        enc.pair_cond, s.child(f"{name}_atom_transformer_decoder"), prefix=f"{name}_atom_transformer_decoder")
    queries_act = queries_act * enc.queries_mask[..., None]
    queries_act = nn.layer_norm(queries_act, s, f"{name}_atom_features_layer_norm", offset=False)
    update = nn.linear(queries_act, s, f"{name}_atom_features_to_position_update")
    return nn.convert(aca.queries_to_token_atoms, update, layout_axes=(-3, -2))
