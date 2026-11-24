# Copyright 2024 DeepMind Technologies Limited
# SPDX-License-Identifier: Apache-2.0
"""CP-aware variants of the heavy pair-track modules.

Each class here mirrors a counterpart in :mod:`alphafold3.model.network.modules`
(same Haiku parameter names, so the same checkpoint loads) but replaces the
single heavy einsum with a sharded version when :func:`parallelism.is_enabled`
returns ``True``. When CP is disabled the modules fall back to the upstream
implementation verbatim.

Implementation philosophy (mirrors boltz-cp, see
``references/boltz-cp/docs/...`` and ``parallelism.py`` docstring):

* pair tensor sharded 2-D over ``(cp_i, cp_j)``; channels replicated;
* central contraction uses ``all_gather`` on one of the two pair axes +
  local einsum + ``psum_scatter`` on the other — cheaper than Cannon's
  per-row alignment shift for the small CP sizes we care about and fits
  cleanly into one ``shard_map`` body;
* the **incoming** triangle equation is reduced to the **outgoing** path
  via an axis swap (``swapaxes(1, 2)`` on the GLU output and again on the
  einsum output).
"""

from __future__ import annotations

from functools import partial
from typing import Literal

import haiku as hk
import jax
import jax.numpy as jnp
from jax.experimental.shard_map import shard_map
from jax.sharding import NamedSharding, PartitionSpec as P

from alphafold3.common import base_config
from alphafold3.model import model_config
from alphafold3.model.components import haiku_modules as hm
from alphafold3.model.network import parallelism

try:
    import tokamax
except ImportError:  # pragma: no cover — only used when CP is disabled
    tokamax = None  # type: ignore[assignment]


# ── Central CP einsum primitive ───────────────────────────────────────────


def cp_pair_einsum_outgoing(a, b):
    """Compute ``einsum('cik,cjk->cij', a, b)`` over a 2-D CP mesh.

    Args:
      a, b: ``[c, N, N]`` tensors sharded as ``(None, cp_i, cp_j)``.

    Returns:
      ``[c, N, N]`` sharded as ``(None, cp_i, cp_j)`` — same layout as input.

    Algorithm (see module docstring):
      1. all-gather ``b`` along ``cp_i`` so the j-axis (axis 1) is full;
      2. local einsum produces partial summed over the local k tile;
      3. ``psum_scatter`` over ``cp_j`` sums the k tiles and reshards
         the j axis (axis 2) back onto ``cp_j``.

    Temporary memory: ``b_full`` is √cp × larger than the local ``b``
    (2× on a 2×2 grid), which we accept in exchange for two single-axis
    collectives instead of Cannon's √cp ring iterations.
    """
    mesh = parallelism.get_mesh()
    assert mesh is not None, "cp_pair_einsum_outgoing called without CP mesh"

    spec_pair = P(None, parallelism.axis_i(), parallelism.axis_j())

    @partial(shard_map, mesh=mesh, in_specs=(spec_pair, spec_pair),
             out_specs=spec_pair, check_rep=False)
    def _body(a_local, b_local):
        # b_local: [c, N/P_i, N/P_j].  axis 1 == j on cp_i.
        b_full_i = jax.lax.all_gather(
            b_local, parallelism.axis_i(), axis=1, tiled=True
        )                                              # [c, N, N/P_j]
        partial_out = jnp.einsum("cik,cjk->cij", a_local, b_full_i)
        out = jax.lax.psum_scatter(
            partial_out, parallelism.axis_j(),
            scatter_dimension=2, tiled=True,
        )
        return out

    return _body(a, b)


def cp_pair_einsum(a, b, equation: str):
    """Dispatch to the outgoing kernel, reducing incoming via axis swap.

    ``incoming(act) == outgoing(swapaxes(act, 0, 1))`` because the two
    triangle equations differ only in which act-N axis is named ``k``.
    Working in the post-GLU shape ``[c, N, N]`` the swap is on axes (1, 2).
    """
    if equation == "cik,cjk->cij":
        return cp_pair_einsum_outgoing(a, b)
    if equation == "ckj,cki->cij":
        a_sw = jnp.swapaxes(a, 1, 2)
        b_sw = jnp.swapaxes(b, 1, 2)
        out_sw = cp_pair_einsum_outgoing(a_sw, b_sw)
        return jnp.swapaxes(out_sw, 1, 2)
    raise ValueError(f"unsupported triangle equation: {equation!r}")


# ── Haiku module ──────────────────────────────────────────────────────────


class TriangleMultiplicationCP(hk.Module):
    """Drop-in CP-aware replacement for ``modules.TriangleMultiplication``.

    Parameter layout (``left_norm_input.{scale,offset}``, ``projection.*``,
    ``gate.*``, ``center_norm.*``, ``output_projection.*``, ``gating_linear.*``)
    is identical to the upstream module so a pre-trained checkpoint loads
    without remapping.

    Behaviour:

    * When :func:`parallelism.is_enabled` is False, the implementation is
      bit-equivalent to ``modules.TriangleMultiplication``: same einsum,
      same kernel selections, no comm.
    * Otherwise the central ``jnp.einsum(equation, a, b)`` is replaced
      with :func:`cp_pair_einsum`, which performs all-gather + local
      einsum + psum_scatter inside a ``shard_map`` block.
    """

    class Config(base_config.BaseConfig):
        equation: Literal["ikc,jkc->ijc", "kjc,kic->ijc"]
        use_glu_kernel: bool = True

    def __init__(
        self,
        config: Config,
        global_config: model_config.GlobalConfig,
        *,
        name,
    ):
        super().__init__(name=name)
        self.config = config
        self.global_config = global_config

    def __call__(self, act, mask):
        mask = mask[None, ...]
        num_channels = act.shape[-1]
        equation = {
            "ikc,jkc->ijc": "cik,cjk->cij",
            "kjc,kic->ijc": "ckj,cki->cij",
        }[self.config.equation]

        act = hm.LayerNorm(name="left_norm_input")(act)
        input_act = act

        # tokamax.gated_linear_unit produces broken results when running
        # under our CP shard_map composition (verified by bisection on
        # T1313/A1: with this branch active, plddt drops from 65 → 51 and
        # ranking_score crashes to -99 from clash penalty). The non-GLU
        # path below is numerically identical to the GLU kernel for the
        # serial case and composes correctly under shard_map, so we route
        # CP through it unconditionally.
        _use_glu = self.config.use_glu_kernel and not parallelism.is_enabled()

        if _use_glu and tokamax is not None:
            weights_projection, _ = hm.haiku_linear_get_params(
                act, num_output=num_channels * 2, name="projection"
            )
            weights_gate, _ = hm.haiku_linear_get_params(
                act,
                num_output=num_channels * 2,
                initializer=self.global_config.final_init,
                name="gate",
            )
            weights_glu = jnp.stack([weights_gate, weights_projection], axis=1)
            projection = tokamax.gated_linear_unit(
                act, weights_glu, activation=jax.nn.sigmoid
            )
            projection = jnp.transpose(projection, (2, 0, 1))
            projection *= mask
        else:
            projection = hm.Linear(num_channels * 2, name="projection")(act)
            projection = jnp.transpose(projection, (2, 0, 1))
            projection *= mask
            gate = hm.Linear(
                num_channels * 2,
                name="gate",
                bias_init=1.0,
                initializer=self.global_config.final_init,
            )(act)
            gate = jnp.transpose(gate, (2, 0, 1))
            projection *= jax.nn.sigmoid(gate)

        projection = projection.reshape(num_channels, 2, *projection.shape[1:])
        a, b = jnp.split(projection, 2, axis=1)
        a, b = jnp.squeeze(a, axis=1), jnp.squeeze(b, axis=1)

        # ── Central contraction: CP or serial ──────────────────────────────
        if parallelism.is_enabled():
            act = cp_pair_einsum(a, b, equation)
        else:
            act = jnp.einsum(equation, a, b)

        act = hm.LayerNorm(name="center_norm", axis=0, param_axis=0)(act)
        act = jnp.transpose(act, (1, 2, 0))
        act = hm.Linear(
            num_channels,
            initializer=self.global_config.final_init,
            name="output_projection",
        )(act)

        gate_out = hm.Linear(
            num_channels,
            name="gating_linear",
            bias_init=1.0,
            initializer=self.global_config.final_init,
        )(input_act)
        act *= jax.nn.sigmoid(gate_out)
        # XLA optimization barrier: without it, the compiler appears to
        # produce a wrong-result fusion on the CP path (plddt 65→49 on
        # T1313/A1 with all CP modules active). The same code with jax.debug.
        # print barriers inserted produced correct results, which narrowed
        # the bug to compiler reordering — optimization_barrier is the
        # minimal way to force the right ordering without the print overhead.
        if parallelism.is_enabled():
            act = jax.lax.optimization_barrier(act)
        return act


# ── Whole-module CP TriangleMultiplication ────────────────────────────────


class TriangleMultiplicationCPWhole(hk.Module):
    """All-in-one-shard_map variant of :class:`TriangleMultiplicationCP`.

    Same Haiku parameter names as the upstream serial module, so any
    checkpoint loads. The difference from :class:`TriangleMultiplicationCP`
    is that **every op runs inside a single shard_map body**, not just the
    central einsum:

    * GSPMD never has to bridge sharded → unsharded across module boundaries
      (no implicit reshards or "broadcast back to device 0" round-trips).
    * Linear / LayerNorm operate on per-device local tiles
      ``[N/P_i, N/P_j, c]``. Their params are channel-axis-only and stay
      replicated, so init under serial (full shape) yields the same shapes
      as apply under CP (local shape) — same param dict loads both.
    * Only collectives are the row all-gather + column psum_scatter at the
      central einsum (identical to :func:`cp_pair_einsum_outgoing`).

    Intent: a measurement target to see whether the per-module GSPMD
    overhead is what's making the existing CP path 10× slower than serial
    on PCIe-only / cross-NUMA setups. If this variant is materially faster,
    promote it to the production module.
    """

    class Config(base_config.BaseConfig):
        equation: Literal["ikc,jkc->ijc", "kjc,kic->ijc"]
        use_glu_kernel: bool = False  # GLU kernel inside shard_map is untested

    def __init__(
        self,
        config: Config,
        global_config: model_config.GlobalConfig,
        *,
        name,
    ):
        super().__init__(name=name)
        self.config = config
        self.global_config = global_config

    def __call__(self, act, mask):
        num_channels = act.shape[-1]
        # Translate the AF3 equation to the (c, i, j) layout used internally.
        eqn = {
            "ikc,jkc->ijc": "cik,cjk->cij",
            "kjc,kic->ijc": "ckj,cki->cij",
        }[self.config.equation]

        # Module instances created up-front so param registration happens in
        # a deterministic order regardless of whether shard_map is entered.
        ln_in = hm.LayerNorm(name="left_norm_input")
        proj_linear = hm.Linear(num_channels * 2, name="projection")
        gate_linear = hm.Linear(
            num_channels * 2,
            name="gate",
            bias_init=1.0,
            initializer=self.global_config.final_init,
        )
        ln_center = hm.LayerNorm(name="center_norm", axis=0, param_axis=0)
        out_proj = hm.Linear(
            num_channels,
            initializer=self.global_config.final_init,
            name="output_projection",
        )
        gating_lin = hm.Linear(
            num_channels,
            name="gating_linear",
            bias_init=1.0,
            initializer=self.global_config.final_init,
        )

        def _body(act_l, mask_l):
            """Local-tile compute: same math as upstream, but on a tile."""
            mask_l = mask_l[None, ...]  # broadcast over channels
            x = ln_in(act_l)
            input_act = x

            projection = proj_linear(x)
            projection = jnp.transpose(projection, (2, 0, 1))
            projection *= mask_l
            gate_t = gate_linear(x)
            gate_t = jnp.transpose(gate_t, (2, 0, 1))
            projection *= jax.nn.sigmoid(gate_t)

            projection = projection.reshape(num_channels, 2, *projection.shape[1:])
            a, b = jnp.split(projection, 2, axis=1)
            a, b = jnp.squeeze(a, axis=1), jnp.squeeze(b, axis=1)

            if parallelism.is_enabled():
                # Inline CP einsum — stay inside the shard_map body so there
                # are no GSPMD-managed reshards between this and the rest of
                # the module body.
                ai = parallelism.axis_i()
                aj = parallelism.axis_j()
                if eqn == "cik,cjk->cij":
                    # a, b: [c, N/P_i, N/P_j].  a axis 1 = i (cp_i), axis 2 = k (cp_j).
                    # Gather b along cp_i (axis 1) → b has full j-axis.
                    b_full = jax.lax.all_gather(b, ai, axis=1, tiled=True)
                    partial = jnp.einsum("cik,cjk->cij", a, b_full)
                    # psum_scatter on cp_j folds the contracted k axis AND
                    # reshards output axis 2 (= j) onto cp_j.
                    central = jax.lax.psum_scatter(
                        partial, aj, scatter_dimension=2, tiled=True
                    )
                else:  # eqn == "ckj,cki->cij"
                    # a, b: [c, N/P_i, N/P_j].  a axis 1 = k (cp_i), axis 2 = j (cp_j).
                    # b axis 1 = k (cp_i),                   axis 2 = i (cp_j).
                    # The swap-then-outgoing trick relied on GSPMD inserting
                    # an all_to_all reshard, which doesn't happen inside a
                    # bare shard_map — so do the right collectives directly.
                    # Gather b along cp_j (axis 2) → b has full i-axis.
                    b_full = jax.lax.all_gather(b, aj, axis=2, tiled=True)
                    partial = jnp.einsum("ckj,cki->cij", a, b_full)
                    # psum_scatter on cp_i folds the contracted k axis AND
                    # reshards output axis 1 (= i) onto cp_i.
                    central = jax.lax.psum_scatter(
                        partial, ai, scatter_dimension=1, tiled=True
                    )
            else:
                central = jnp.einsum(eqn, a, b)

            central = ln_center(central)
            central = jnp.transpose(central, (1, 2, 0))
            central = out_proj(central)

            gate_out = gating_lin(input_act)
            return central * jax.nn.sigmoid(gate_out)

        if not parallelism.is_enabled():
            return _body(act, mask)

        mesh = parallelism.get_mesh()
        pair_spec = P(parallelism.axis_i(), parallelism.axis_j(), None)
        mask_spec = P(parallelism.axis_i(), parallelism.axis_j())
        wrapped = shard_map(
            _body, mesh=mesh,
            in_specs=(pair_spec, mask_spec),
            out_specs=pair_spec,
            check_rep=False,
        )
        return wrapped(act, mask)


# ── CP attention primitive ────────────────────────────────────────────────


def _cp_attention(q, k, v, mask, bias, transpose: bool, gc):
    """Distributed self-attention with all-gather + local DPA + slice.

    Args:
      q, k, v: ``[N, N, H, d]`` sharded as pair (``cp_i, cp_j, None, None`` for
        the row-attn / outgoing case, ``cp_j, cp_i, None, None`` after the
        upstream axis swap for the column-attn / incoming case).
      mask: ``[N, 1, 1, N]`` sharded matching q's batch (axis 0) and gather
        axis (axis 3). For ``transpose=False`` the mask is
        ``(cp_i, None, None, cp_j)``; for ``transpose=True`` it is
        ``(cp_j, None, None, cp_i)``.  This relies on the pair mask being
        symmetric (``mask[i, j] == mask[j, i]``) — true for the standard
        AF3 pair masks.
      bias: ``[1, H, N, N]`` sharded ``(None, None, cp_i, cp_j)``. The bias
        is shared across all attentions and is broadcast over the batch dim
        of DPA, so its sharding is fixed regardless of ``transpose``.
      transpose: ``True`` for column / incoming attention (the upstream
        flag).
      gc: ``model_config.GlobalConfig`` — only its
        ``flash_attention_implementation`` field is consulted.

    Returns:
      ``[N, N, H, d]`` sharded as the input q (same layout).

    Algorithm (mirrors triangle-mul CP):

    1. ``all_gather`` q, k, v's seq axis (axis 1) along the gather mesh dim
       so each device owns a full ``S_k = N``; this keeps q's batch sharded
       on the orthogonal mesh dim (cheaper than gathering the entire pair).
    2. ``all_gather`` the bias on **both** mesh dims (replicate it) — bias
       memory is tiny compared to q/k/v.
    3. Local DPA (``tokamax`` when available, ``jax.nn`` otherwise).
    4. ``dynamic_slice`` the output's seq axis back to the local tile to
       restore the original pair sharding.
    """
    mesh = parallelism.get_mesh()
    if transpose:
        keep, gather = parallelism.axis_j(), parallelism.axis_i()
    else:
        keep, gather = parallelism.axis_i(), parallelism.axis_j()

    qkv_spec = P(keep, gather, None, None)
    mask_spec = P(keep, None, None, gather)
    bias_spec = P(None, None, parallelism.axis_i(), parallelism.axis_j())
    out_spec = P(keep, gather, None, None)

    @partial(shard_map, mesh=mesh,
             in_specs=(qkv_spec, qkv_spec, qkv_spec, mask_spec, bias_spec),
             out_specs=out_spec, check_rep=False)
    def _body(q_l, k_l, v_l, m_l, b_l):
        # Gather seq axis (axis 1) to full S_k=N.
        q_g = jax.lax.all_gather(q_l, gather, axis=1, tiled=True)
        k_g = jax.lax.all_gather(k_l, gather, axis=1, tiled=True)
        v_g = jax.lax.all_gather(v_l, gather, axis=1, tiled=True)
        # Mask: gather axis 3 (S_k).
        m_g = jax.lax.all_gather(m_l, gather, axis=3, tiled=True)
        # Bias: gather both mesh dims to fully replicate.
        b_g = jax.lax.all_gather(b_l, parallelism.axis_i(), axis=2, tiled=True)
        b_g = jax.lax.all_gather(b_g, parallelism.axis_j(), axis=3, tiled=True)

        if tokamax is not None:
            wa = tokamax.dot_product_attention(
                q_g, k_g, v_g, mask=m_g, bias=b_g,
                implementation=gc.flash_attention_implementation,
            )
        else:
            wa = jax.nn.dot_product_attention(q_g, k_g, v_g, mask=m_g, bias=b_g)

        # Re-shard axis 1 back to the local tile of `gather`.
        p_idx = jax.lax.axis_index(gather)
        tile = wa.shape[1] // parallelism._axis_size_static(gather)
        return jax.lax.dynamic_slice_in_dim(wa, p_idx * tile, tile, axis=1)

    return _body(q, k, v, mask, bias)


# ── GridSelfAttention CP ──────────────────────────────────────────────────


class GridSelfAttentionCP(hk.Module):
    """Drop-in CP-aware replacement for ``modules.GridSelfAttention``.

    Same Haiku parameter names as upstream (``act_norm``, ``pair_bias_projection``,
    ``q_projection``, ``k_projection``, ``v_projection``, ``gating_query``,
    ``output_projection``), so a pre-trained checkpoint loads without remapping.

    With CP disabled the central DPA call is exactly the upstream
    ``modules.GridSelfAttention._attention`` path; with CP enabled the DPA is
    replaced by :func:`_cp_attention`. The upstream's
    ``mapping.inference_subbatch`` chunking is intentionally not applied in
    the CP path — the all-gather already produces a single per-device batch
    of size ``N / P_keep``, which is the per-device working set we want the
    attention kernel to operate on directly.
    """

    class Config(base_config.BaseConfig):
        num_head: int = 4

    def __init__(
        self,
        config: Config,
        global_config: model_config.GlobalConfig,
        transpose: bool,
        *,
        name: str,
    ):
        super().__init__(name=name)
        self.config = config
        self.global_config = global_config
        self.transpose = transpose

    @hk.transparent
    def _attention(self, act, mask, bias):
        """Mirror of ``modules.GridSelfAttention._attention``."""
        num_channels = act.shape[-1]
        assert num_channels % self.config.num_head == 0
        qkv_dim = max(num_channels // self.config.num_head, 16)
        qkv_shape = (self.config.num_head, qkv_dim)

        q = hm.Linear(qkv_shape, use_bias=False, name="q_projection",
                      transpose_weights=True)(act)
        k = hm.Linear(qkv_shape, use_bias=False, name="k_projection",
                      transpose_weights=True)(act)
        v = hm.Linear(qkv_shape, use_bias=False, name="v_projection")(act)

        bias = jnp.expand_dims(bias, 0)

        if parallelism.is_enabled():
            weighted_avg = _cp_attention(
                q, k, v, mask, bias, self.transpose, self.global_config,
            )
        elif tokamax is not None:
            weighted_avg = tokamax.dot_product_attention(
                q, k, v, mask=mask, bias=bias,
                implementation=self.global_config.flash_attention_implementation,
            )
        else:
            weighted_avg = jax.nn.dot_product_attention(
                q, k, v, mask=mask, bias=bias,
            )
        weighted_avg = jnp.reshape(weighted_avg, weighted_avg.shape[:-2] + (-1,))

        gate_values = hm.Linear(
            self.config.num_head * qkv_dim,
            bias_init=1.0,
            initializer="zeros",
            transpose_weights=True,
            name="gating_query",
        )(act)
        weighted_avg *= jax.nn.sigmoid(gate_values)
        return hm.Linear(
            num_channels,
            initializer=self.global_config.final_init,
            name="output_projection",
        )(weighted_avg)

    def __call__(self, act, pair_mask):
        assert act.ndim == 3
        assert pair_mask.ndim == 2

        pair_mask = jnp.swapaxes(pair_mask, -1, -2)
        act = hm.LayerNorm(name="act_norm")(act)

        nonbatched_bias = hm.Linear(
            self.config.num_head, use_bias=False, name="pair_bias_projection"
        )(act)
        nonbatched_bias = jnp.transpose(nonbatched_bias, [2, 0, 1])

        if self.transpose:
            act = jnp.swapaxes(act, -2, -3)

        pair_mask = pair_mask[:, None, None, :].astype(jnp.bool_)

        attn = self._attention(act, pair_mask, nonbatched_bias)

        if self.transpose:
            attn = jnp.swapaxes(attn, -2, -3)
        # XLA optimization barrier — defensive. Same rationale as
        # TriangleMultiplicationCP: without it the compiler can produce a
        # wrong-result fusion across the sharded-collective → serial-op
        # boundary (T1313/A1 plddt 65→49 with all CP modules active).
        if parallelism.is_enabled():
            attn = jax.lax.optimization_barrier(attn)
        return attn


# ── OuterProductMean CP ───────────────────────────────────────────────────


def _opm_cp(left_act, right_act, mask, output_w, output_b):
    """CP-aware OuterProductMean inner compute.

    Inputs:
      left_act, right_act: ``[S, N, c_outer]`` sharded ``(cp_j, cp_i, None)``
        (MSA sharding — depth on cp_j, residue on cp_i).
      mask:                ``[S, N, 1]``       sharded ``(cp_j, cp_i, None)``.
      output_w:            ``[c_outer, c_outer, c_out]`` replicated.
      output_b:            ``[c_out]``                  replicated.

    Returns:
      ``[N, N, c_out]`` sharded ``(cp_i, cp_j, None)`` (pair sharding), already
      divided by the masked normalisation, ready for downstream pair-track ops.

    Algorithm: for each ``(p_i, p_j)`` device, materialise the slices
    ``left[:, p_i_tile, :]`` and ``right[:, p_j_tile, :]`` over the **full** S
    axis. That's two collectives per device:

      * gather right's N along cp_i and slice to the local cp_j tile (this
        moves right's N axis from cp_i sharding onto cp_j sharding);
      * gather S along cp_j on both left and right so the depth-reduction
        becomes a purely local einsum.

    No psum is required afterwards — each device owns the full S range that
    its output tile needs.
    """
    mesh = parallelism.get_mesh()
    ai = parallelism.axis_i()
    aj = parallelism.axis_j()

    msa_spec = P(aj, ai, None)
    pair_spec = P(ai, aj, None)
    repl_spec = P()

    @partial(shard_map, mesh=mesh,
             in_specs=(msa_spec, msa_spec, msa_spec, repl_spec, repl_spec),
             out_specs=pair_spec, check_rep=False)
    def _body(l_l, r_l, m_l, ow, ob):
        # Gather S along cp_j first → every device in a cp_i column owns the
        # same full-S, N/P_i slice (left's natural layout for the einsum).
        l_s  = jax.lax.all_gather(l_l, aj, axis=0, tiled=True)   # [S, N/P_i, c_outer]
        r_s  = jax.lax.all_gather(r_l, aj, axis=0, tiled=True)   # [S, N/P_i, c_outer]
        ml_s = jax.lax.all_gather(m_l, aj, axis=0, tiled=True)   # [S, N/P_i, 1]

        # Re-shard right's N: gather along cp_i to materialise full N, then
        # slice on cp_j. The S-first ordering matters — the previous slice's
        # data is well-defined per cp_j only once S is replicated.
        def _n_to_cp_j(t):
            t_full = jax.lax.all_gather(t, ai, axis=1, tiled=True)
            p_j_idx = jax.lax.axis_index(aj)
            tile = t_full.shape[1] // parallelism._axis_size_static(aj)
            return jax.lax.dynamic_slice_in_dim(t_full, p_j_idx * tile, tile, axis=1)

        r_j  = _n_to_cp_j(r_s)    # [S, N/P_j, c_outer]
        mr_j = _n_to_cp_j(ml_s)   # [S, N/P_j, 1]

        # Local outer product (full S, no psum needed).
        l_t = jnp.transpose(l_s, [0, 2, 1])                          # [S, c_outer, N/P_i]
        partial = jnp.einsum("acb,ade->dceb", l_t, r_j)              # [N/P_j, c_outer, c_outer, N/P_i]
        out = jnp.einsum("dceb,cef->dbf", partial, ow) + ob          # [N/P_j, N/P_i, c_out]
        out = jnp.transpose(out, [1, 0, 2])                          # [N/P_i, N/P_j, c_out]

        # Norm: einsum('abc,adc->bdc', mask, mask) → [N, N, 1].
        m_t = jnp.transpose(ml_s, [0, 2, 1])                         # [S, 1, N/P_i]
        norm_p = jnp.einsum("acb,ade->dceb", m_t, mr_j)              # [N/P_j, 1, 1, N/P_i]
        norm = jnp.transpose(norm_p[:, 0, 0, :], [1, 0])[..., None]  # [N/P_i, N/P_j, 1]

        return out / (1e-3 + norm)

    return _body(left_act, right_act, mask, output_w, output_b)


class OuterProductMeanCP(hk.Module):
    """Drop-in CP-aware replacement for ``modules.OuterProductMean``.

    Same Haiku parameter names as upstream — ``layer_norm_input``,
    ``left_projection``, ``right_projection``, ``output_w``, ``output_b`` —
    so a pre-trained checkpoint loads without remapping.

    The non-CP path inlines the same einsum upstream uses, but without
    ``mapping.inference_subbatch`` chunking; that's numerically equivalent
    when ``N <= chunk_size`` and matters only for large-N memory tuning.
    """

    class Config(base_config.BaseConfig):
        chunk_size: int = 128
        num_outer_channel: int = 32

    def __init__(
        self,
        config: Config,
        global_config: model_config.GlobalConfig,
        num_output_channel: int,
        *,
        name: str,
    ):
        super().__init__(name=name)
        self.config = config
        self.global_config = global_config
        self.num_output_channel = num_output_channel

    def __call__(self, act, mask):
        mask = mask[..., None]
        act = hm.LayerNorm(name="layer_norm_input")(act)

        left_act = mask * hm.Linear(
            self.config.num_outer_channel,
            initializer="linear",
            name="left_projection",
        )(act)
        right_act = mask * hm.Linear(
            self.config.num_outer_channel,
            initializer="linear",
            name="right_projection",
        )(act)

        if self.global_config.final_init == "zeros":
            w_init = hk.initializers.Constant(0.0)
        else:
            w_init = hk.initializers.VarianceScaling(scale=2.0, mode="fan_in")

        output_w = hk.get_parameter(
            "output_w",
            shape=(
                self.config.num_outer_channel,
                self.config.num_outer_channel,
                self.num_output_channel,
            ),
            dtype=act.dtype,
            init=w_init,
        )
        output_b = hk.get_parameter(
            "output_b",
            shape=(self.num_output_channel,),
            dtype=act.dtype,
            init=hk.initializers.Constant(0.0),
        )

        if parallelism.is_enabled():
            out = _opm_cp(left_act, right_act, mask, output_w, output_b)
            # XLA optimization barrier — see TriangleMultiplicationCP for
            # rationale (wrong-result fusion across sharded-collective →
            # serial-op boundary on T1313/A1 dropped plddt 65→49).
            return jax.lax.optimization_barrier(out)

        # Non-CP path: explicit einsum (numerically identical to the upstream
        # inference_subbatch path for N <= chunk_size).
        l_t = jnp.transpose(left_act, [0, 2, 1])
        partial = jnp.einsum("acb,ade->dceb", l_t, right_act)
        out = jnp.einsum("dceb,cef->dbf", partial, output_w) + output_b
        out = jnp.transpose(out, [1, 0, 2])
        norm = jnp.einsum("abc,adc->bdc", mask, mask)
        return out / (1e-3 + norm)
