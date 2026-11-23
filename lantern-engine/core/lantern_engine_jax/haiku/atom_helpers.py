# SPDX-License-Identifier: Apache-2.0
"""Atom-level utilities for the diffusion module port.

These helpers handle the atom <-> token mapping and the windowing layout
needed for AtomTransformer's sequence-local cross-attention.

Reference:
  models/protenix/protenix/model/utils.py::broadcast_token_to_atom
  models/protenix/protenix/model/utils.py::aggregate_atom_to_token
  models/protenix/protenix/model/modules/primitives.py::rearrange_qk_to_dense_trunk
  models/protenix/protenix/model/modules/primitives.py::broadcast_token_to_local_atom_pair
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def broadcast_token_to_atom(
    x_token: jax.Array,            # [..., N_token, d]
    atom_to_token_idx: jax.Array,  # [..., N_atom]
) -> jax.Array:
    """Gather token embeddings for each atom.

    Returns: ``x_atom[..., a, :] = x_token[..., atom_to_token_idx[..., a], :]``
    """
    # Simple case: 1-D index, broadcast over leading dims
    if atom_to_token_idx.ndim == 1:
        return x_token[..., atom_to_token_idx, :]
    # Batched gather along the token axis
    return jnp.take_along_axis(
        x_token, atom_to_token_idx[..., None], axis=-2,
    )


def aggregate_atom_to_token(
    x_atom: jax.Array,             # [..., N_atom, d]
    atom_to_token_idx: jax.Array,  # [..., N_atom]
    *,
    n_token: int,
    reduce: str = "mean",
) -> jax.Array:
    """Scatter-aggregate atom features into per-token features.

    Equivalent to torch_scatter's ``scatter`` with reduce='mean' or 'sum'.
    """
    feat_dim = x_atom.shape[-1]
    # Reshape so we can use segment_sum: collapse leading dims into a single
    # batch dim, then re-expand at the end.
    leading_shape = x_atom.shape[:-2]
    n_atom = x_atom.shape[-2]
    if leading_shape:
        # Use jnp.zeros + add via .at[...].add for general case
        # For each leading-dim "batch" we need an independent scatter.
        # Easiest: batch the indices to a global flat index space.
        batch_size = 1
        for d in leading_shape:
            batch_size *= d
        # Flatten leading dims
        x_flat = x_atom.reshape(batch_size, n_atom, feat_dim)
        if atom_to_token_idx.ndim == 1:
            # Same indices for each batch
            idx_flat = jnp.broadcast_to(atom_to_token_idx, (batch_size, n_atom))
        else:
            idx_flat = atom_to_token_idx.reshape(batch_size, n_atom)
        # For each batch, scatter into [n_token, feat_dim]
        out = jnp.zeros((batch_size, n_token, feat_dim), dtype=x_atom.dtype)
        out = out.at[jnp.arange(batch_size)[:, None], idx_flat, :].add(x_flat)
        if reduce == "mean":
            count = jnp.zeros((batch_size, n_token), dtype=x_atom.dtype)
            count = count.at[jnp.arange(batch_size)[:, None], idx_flat].add(1.0)
            out = out / jnp.maximum(count[..., None], 1.0)
        return out.reshape(*leading_shape, n_token, feat_dim)
    else:
        # No leading dims
        out = jnp.zeros((n_token, feat_dim), dtype=x_atom.dtype)
        out = out.at[atom_to_token_idx, :].add(x_atom)
        if reduce == "mean":
            count = jnp.zeros((n_token,), dtype=x_atom.dtype)
            count = count.at[atom_to_token_idx].add(1.0)
            out = out / jnp.maximum(count[..., None], 1.0)
        return out


def _pad_at_dim(x: jax.Array, dim: int, pad_length: tuple[int, int]) -> jax.Array:
    """Pad ``x`` along ``dim`` by ``(left, right)`` with zeros."""
    if dim < 0:
        dim = x.ndim + dim
    pad_widths = [(0, 0)] * x.ndim
    pad_widths[dim] = pad_length
    return jnp.pad(x, pad_widths)


def rearrange_qk_to_dense_trunk(
    q: jax.Array,                # [..., N, d]  (q at axis -2)
    k: jax.Array,                # [..., N, d]  (k at axis -2)
    *,
    n_queries: int = 32,
    n_keys: int = 128,
    compute_mask: bool = True,
):
    """Build dense windowed (q, k) layout for sequence-local attention.

    Produces:
        q_trunked: [..., n_blocks, n_queries, d]
        k_trunked: [..., n_blocks, n_keys, d]
        info: {'mask_trunked': [n_blocks, n_queries, n_keys] (bool)}

    where:
        n_blocks = ceil(N / n_queries)
        Each window centers the q chunk around the matching k chunk; the k
        chunk is wider (n_keys vs n_queries) so neighboring windows overlap.
    """
    assert n_queries % 2 == 0 and n_keys % 2 == 0
    assert n_keys >= n_queries
    n = q.shape[-2]
    n_blocks = (n + n_queries - 1) // n_queries
    q_pad_len = n_blocks * n_queries - n

    # Pad q along the last-but-one axis to a multiple of n_queries
    q_padded = _pad_at_dim(q, dim=-2, pad_length=(0, q_pad_len))  # [..., n_blocks*n_queries, d]
    q_trunked = q_padded.reshape(*q.shape[:-2], n_blocks, n_queries, q.shape[-1])

    # Pad k symmetrically: left pad (n_keys - n_queries)/2, right pad to fit
    pad_left = (n_keys - n_queries) // 2
    # Right pad so that for the last block, k spans up to its full window.
    # Total k length needed: (n_blocks - 1) * n_queries + n_keys
    total_k_len = (n_blocks - 1) * n_queries + n_keys
    pad_right = total_k_len - pad_left - n
    k_padded = _pad_at_dim(k, dim=-2, pad_length=(pad_left, pad_right))

    # Slide windows of size n_keys, stepping by n_queries
    # Construct via gather: for each block i, k[i*n_queries : i*n_queries + n_keys]
    block_starts = jnp.arange(n_blocks) * n_queries  # [n_blocks]
    offsets = jnp.arange(n_keys)                      # [n_keys]
    gather_idx = block_starts[:, None] + offsets[None, :]  # [n_blocks, n_keys]
    k_trunked = jnp.take(k_padded, gather_idx, axis=-2)
    # take returns shape [..., n_blocks, n_keys, d]

    info = {}
    if compute_mask:
        # Mark valid (within original [0, n) range and not pad on q side)
        q_valid = (jnp.arange(n_blocks * n_queries) < n).reshape(n_blocks, n_queries)
        k_valid = ((gather_idx >= pad_left) & (gather_idx < pad_left + n))
        mask = q_valid[..., :, None] & k_valid[..., None, :]
        info["mask_trunked"] = mask
        info["q_pad_length"] = q_pad_len
        info["pad_left"] = pad_left
    return q_trunked, k_trunked, info


def broadcast_token_to_local_atom_pair(
    z_token: jax.Array,            # [..., N_token, N_token, c_z]
    atom_to_token_idx: jax.Array,  # [..., N_atom]
    *,
    n_queries: int,
    n_keys: int,
    compute_mask: bool = False,
):
    """Lift a token-pair tensor to atom-pair WINDOWED layout.

    For each atom-pair window block, look up the (token_q, token_k) pair
    indices and gather z_token at those positions.

    Returns:
        z_atom_windowed: [..., n_blocks, n_queries, n_keys, c_z]
        info: dict (mask if requested)
    """
    # Build atom-side windowed indices for q and k
    n_atom = atom_to_token_idx.shape[-1]
    n_blocks = (n_atom + n_queries - 1) // n_queries
    pad_left = (n_keys - n_queries) // 2
    # q-side: pad atom_to_token_idx to multiple of n_queries
    q_pad_len = n_blocks * n_queries - n_atom
    q_idx_padded = _pad_at_dim(atom_to_token_idx, dim=-1, pad_length=(0, q_pad_len))
    q_idx = q_idx_padded.reshape(*atom_to_token_idx.shape[:-1], n_blocks, n_queries)
    # k-side: pad with pad_left on left, enough on right
    total_k_len = (n_blocks - 1) * n_queries + n_keys
    pad_right = total_k_len - pad_left - n_atom
    k_idx_padded = _pad_at_dim(atom_to_token_idx, dim=-1, pad_length=(pad_left, pad_right))
    block_starts = jnp.arange(n_blocks) * n_queries
    offsets = jnp.arange(n_keys)
    gather_idx_atom = block_starts[:, None] + offsets[None, :]
    k_idx = jnp.take(k_idx_padded, gather_idx_atom, axis=-1)  # [..., n_blocks, n_keys]

    # Now z_token[..., q_idx, k_idx, :] for each window
    # z_token shape: [..., N_token, N_token, c_z]
    # q_idx: [..., n_blocks, n_queries]; k_idx: [..., n_blocks, n_keys]
    # Need: [..., n_blocks, n_queries, n_keys, c_z]
    # Approach: broadcast q_idx to [..., n_blocks, n_queries, 1] and k_idx
    # to [..., n_blocks, 1, n_keys], then gather both axes.
    qi = q_idx[..., :, :, None]      # [..., n_blocks, n_queries, 1]
    ki = k_idx[..., :, None, :]      # [..., n_blocks, 1, n_keys]
    # Broadcast to [..., n_blocks, n_queries, n_keys]
    qi_b, ki_b = jnp.broadcast_arrays(qi, ki)
    # Pull z_token[..., qi_b, ki_b, :]
    z_out = z_token[..., qi_b, ki_b, :]
    info = {}
    if compute_mask:
        q_valid = (jnp.arange(n_blocks * n_queries) < n_atom).reshape(n_blocks, n_queries)
        k_valid = (gather_idx_atom >= pad_left) & (gather_idx_atom < pad_left + n_atom)
        info["mask_trunked"] = q_valid[..., :, None] & k_valid[..., None, :]
    return z_out, info
