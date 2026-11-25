# SPDX-License-Identifier: Apache-2.0
"""Boltz-2 JAX hot-module backend.

Mirrors ``core/boltz/compat.py`` but routes pairformers through Haiku/JAX.

Module attribute paths:
  structure pairformer:    model.pairformer_module
  confidence pairformer:   model.confidence_module.pairformer_stack
  token transformer:       model.structure_module.score_model.token_transformer

Boltz pairformer signature: forward(s, z, mask, pair_mask, ...).
Tensors are in batch-first format [1, N, c_s], [1, N, N, c_z], etc.

The token_transformer hot-swap is NOT yet implemented because Boltz's
score_model passes pair bias in via a ``bias`` kwarg per-block (not
projected internally). A Boltz-specific HK variant is needed.
See ``residudes_engine_jax/weights/boltz.py`` docstring.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

import haiku as hk
import jax
import jax.numpy as jnp
import torch
import torch.nn as nn

from residudes_engine_jax.bridge import torch_to_jax, jax_to_torch, get_jax_device
from residudes_engine_jax.haiku.boltz_token_transformer import (
    BoltzTokenTransformer,
    BoltzTokenTransformerVariant,
)
from residudes_engine_jax.haiku.pairformer import PairformerStack, PairformerVariant
from residudes_engine_jax.protenix import _stack_per_block_params
from residudes_engine_jax.weights._rules import apply_rules
from residudes_engine_jax.weights.boltz import (
    pairformer_rules,
    template_pairformer_rules,
    msa_pairformer_layer_rules,
    token_transformer_rules,
)


@dataclass
class BoltzJAXConfig:
    """Boltz-2 JAX config."""
    structure_pairformer_num_layers: int = 64
    confidence_pairformer_num_layers: int = 8
    token_transformer_num_layers: int = 24
    token_transformer_num_head: int = 16
    template_pairformer_num_layers: int = 2  # template_module.pairformer (no-seq)
    msa_num_layers: int = 4  # msa_module.layers count (each has pairformer_layer)
    multiplicity: int = 5
    dtype: jnp.dtype = jnp.float32
    jit: bool = True
    # When False, only the structure pairformer is swapped.
    swap_confidence: bool = True
    swap_token_transformer: bool = True
    # Template + MSA module no-seq pairformers — the trimer-scale memory
    # bottleneck (tri_att_start in template_module.pairformer fired the OOM
    # at N≈2880 because PT activations alone hit 170 GB before alloc).
    swap_template_pairformer: bool = True
    swap_msa_pairformer: bool = True

    @classmethod
    def from_checkpoint(cls, path: str) -> "BoltzJAXConfig":
        import re
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        sd = ckpt.get("state_dict", ckpt)
        s_n = sorted(set(int(re.search(r"\.layers\.(\d+)\.", k).group(1)) for k in sd
                          if k.startswith("pairformer_module.layers.")))
        c_n = sorted(set(int(re.search(r"\.layers\.(\d+)\.", k).group(1)) for k in sd
                          if k.startswith("confidence_module.pairformer_stack.layers.")))
        t_n = sorted(set(int(re.search(r"\.layers\.(\d+)\.", k).group(1)) for k in sd
                          if k.startswith("structure_module.score_model.token_transformer.layers.")))
        tmpl_n = sorted(set(int(re.search(r"\.layers\.(\d+)\.", k).group(1)) for k in sd
                          if k.startswith("template_module.pairformer.layers.")))
        msa_n = sorted(set(int(re.search(r"msa_module\.layers\.(\d+)\.", k).group(1)) for k in sd
                          if re.match(r"msa_module\.layers\.\d+\.pairformer_layer\.", k)))
        return cls(
            structure_pairformer_num_layers=len(s_n),
            confidence_pairformer_num_layers=len(c_n),
            token_transformer_num_layers=len(t_n),
            template_pairformer_num_layers=len(tmpl_n),
            msa_num_layers=len(msa_n),
        )


# ----------- Compat wrapper (one class, used for both pairformers) -----------

class _CompatPairformerJAXBoltz(nn.Module):
    """Boltz pairformer signature: forward(s, z, mask, pair_mask, ...)."""

    def __init__(self, params: dict, config: BoltzJAXConfig, num_layers: int):
        super().__init__()
        self._config = config
        variant = PairformerVariant.boltz2(num_blocks=num_layers)
        # Same knobs the Protenix wrapper honours (RESIDUDES_ENGINE_JAX_PAIRFORMER_SCAN /
        # _REMAT). With scan the 64 blocks share one compiled body via
        # hk.experimental.layer_stack instead of being compiled one by one.
        remat, use_scan = _pairformer_knobs()
        if num_layers < 2:
            use_scan = False
        def fn(z, pair_mask, s, single_mask):
            return PairformerStack(variant=variant, with_single=True, remat=remat,
                                   use_scan=use_scan)(z, pair_mask, s, single_mask)
        # Don't shadow nn.Module._apply (used by .cpu() / .to(...) traversal).
        self._jax_apply = hk.transform(fn).apply
        if config.jit:
            self._jax_apply = jax.jit(self._jax_apply)
        if use_scan:
            params = _stack_per_block_params(
                params, num_blocks=num_layers, stack_root="pairformer",
                old_prefix="pair_former_block_", new_name="pair_former_block")
        self._params = jax.device_put(
            jax.tree_util.tree_map(
                lambda x: jnp.asarray(x, dtype=_param_dtype(x, config.dtype)),
                params,
            ),
            get_jax_device(),
        )

    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        mask: torch.Tensor,
        pair_mask: torch.Tensor,
        **kwargs,
    ):
        s_b = s.dim() == 3
        z_b = z.dim() == 4
        s_in = s.squeeze(0) if s_b else s
        z_in = z.squeeze(0) if z_b else z
        sm_in = mask.squeeze(0) if mask.dim() == 2 else mask
        pm_in = pair_mask.squeeze(0) if pair_mask.dim() == 3 else pair_mask

        orig = s.dtype
        s_j = torch_to_jax(_act(s_in, self._config.dtype))
        z_j = torch_to_jax(_act(z_in, self._config.dtype))
        sm_j = torch_to_jax(sm_in.to(torch.float32))
        pm_j = torch_to_jax(pm_in.to(torch.float32))

        s_out, z_out = self._jax_apply(self._params, None, z_j, pm_j, s_j, sm_j)
        s_t = jax_to_torch(s_out, dtype=orig)
        z_t = jax_to_torch(z_out, dtype=orig)
        if s_b: s_t = s_t.unsqueeze(0)
        if z_b: z_t = z_t.unsqueeze(0)
        return s_t, z_t


def _param_dtype(x, target):
    a = jnp.asarray(x)
    if jnp.issubdtype(a.dtype, jnp.floating):
        return target
    return a.dtype


_INPUT_FP32 = os.environ.get("RESIDUDES_ENGINE_JAX_INPUT_DTYPE", "compute") == "fp32"


_NOSEQ_APPLY: dict = {}


def _shared_noseq_apply(variant: PairformerVariant, jit: bool):
    key = (variant, jit)
    fn_apply = _NOSEQ_APPLY.get(key)
    if fn_apply is None:
        remat, _ = _pairformer_knobs()
        def fn(z, pair_mask):
            return PairformerStack(variant=variant, with_single=False, remat=remat)(z, pair_mask)
        fn_apply = hk.transform(fn).apply
        if jit:
            fn_apply = jax.jit(fn_apply)
        _NOSEQ_APPLY[key] = fn_apply
    return fn_apply


def _pairformer_knobs() -> tuple[bool, bool]:
    """(remat, use_scan) from RESIDUDES_ENGINE_JAX_PAIRFORMER_REMAT / _SCAN, as in protenix.py."""
    on = ("1", "true", "yes", "on")
    remat = os.environ.get("RESIDUDES_ENGINE_JAX_PAIRFORMER_REMAT", "1").lower() in on
    use_scan = os.environ.get("RESIDUDES_ENGINE_JAX_PAIRFORMER_SCAN", "0").lower() in on
    return remat, use_scan


def _act(t: torch.Tensor, dtype) -> torch.Tensor:
    """Activations cross into JAX in the swapped modules' compute dtype.

    Upcasting them to fp32 (the behaviour until 2026-11) made every matmul
    fp32 (the bf16 weights were promoted) and, because cuDNN flash attention
    only takes bf16/fp16, pushed the triangle attention onto XLA's
    materialised path: a 4·H·N³-byte fp32 logits tensor (36 GB at 1315
    tokens) and a 2–3× slower pairformer. Protenix's wrapper never upcast,
    which is why its JAX path was the only one faster than PyTorch.
    RESIDUDES_ENGINE_JAX_INPUT_DTYPE=fp32 restores the old behaviour. Masks stay fp32."""
    if _INPUT_FP32 or dtype == jnp.float32:
        return t.to(torch.float32)
    return t.to(torch.bfloat16)


class _CompatPairformerNoSeqJAXBoltz(nn.Module):
    """Boltz no-seq pairformer (z-only): forward(z, pair_mask, **kw).

    Used for template_module.pairformer (2 layers, token_z=64) and each
    msa_module.layers.{i}.pairformer_layer (1 layer, token_z=128). The HK
    side is just PairformerStack(with_single=False) with the right variant.
    """

    def __init__(self, params: dict, variant: PairformerVariant, dtype: jnp.dtype, jit: bool = True):
        super().__init__()
        self._variant = variant
        self._dtype = dtype
        # One jitted apply per variant, shared by every instance: the four MSA
        # pairformer layers (and the template stack) have identical shapes, so
        # sharing the function object lets jax's cache compile the body once
        # instead of once per layer.
        self._jax_apply = _shared_noseq_apply(variant, jit)
        self._params = jax.device_put(
            jax.tree_util.tree_map(
                lambda x: jnp.asarray(x, dtype=_param_dtype(x, dtype)),
                params,
            ),
            get_jax_device(),
        )

    def forward(
        self,
        z: torch.Tensor,
        pair_mask: torch.Tensor,
        chunk_size_tri_attn=None,  # accepted positional, ignored (JAX path)
        use_kernels: bool = False,
        use_cuequiv_mul: bool = False,
        use_cuequiv_attn: bool = False,
        **kwargs,
    ):
        # z: [B*T, N, N, c] (template) or [B, N, N, c] (msa). Squeeze leading
        # batch axis if present, run, restore.
        z_dim = z.dim()
        pm_dim = pair_mask.dim()
        # The HK PairformerBlock expects z [N, N, c] and pair_mask [N, N].
        # Boltz passes z [B, N, N, c] and pair_mask [B, N, N]. Loop over batch.
        orig_dtype = z.dtype
        if z_dim == 4 and pm_dim == 3:
            outs = []
            for b in range(z.shape[0]):
                z_j = torch_to_jax(_act(z[b], self._dtype))
                pm_j = torch_to_jax(pair_mask[b].to(torch.float32))
                z_out = self._jax_apply(self._params, None, z_j, pm_j)
                outs.append(jax_to_torch(z_out, dtype=orig_dtype))
            return torch.stack(outs, dim=0)
        else:
            z_j = torch_to_jax(_act(z, self._dtype))
            pm_j = torch_to_jax(pair_mask.to(torch.float32))
            z_out = self._jax_apply(self._params, None, z_j, pm_j)
            return jax_to_torch(z_out, dtype=orig_dtype)


class _CompatTokenTransformerJAXBoltz(nn.Module):
    """Boltz token transformer (score_model.token_transformer)."""

    def __init__(self, params: dict, config: BoltzJAXConfig):
        super().__init__()
        self._config = config
        variant = BoltzTokenTransformerVariant(
            num_layers=config.token_transformer_num_layers,
            num_head=config.token_transformer_num_head,
        )
        def fn(a, s, b, mask):
            return BoltzTokenTransformer(variant=variant)(a, s, b, mask)
        # Avoid the nn.Module._apply shadow trap.
        self._jax_apply = hk.transform(fn).apply
        if config.jit:
            self._jax_apply = jax.jit(self._jax_apply)
        self._params = jax.device_put(
            jax.tree_util.tree_map(
                lambda x: jnp.asarray(x, dtype=_param_dtype(x, config.dtype)),
                params,
            ),
            get_jax_device(),
        )

    def forward(
        self,
        a: torch.Tensor,
        s: torch.Tensor,
        z: torch.Tensor = None,
        mask: torch.Tensor = None,
        bias: torch.Tensor = None,
        **kwargs,
    ):
        # Boltz passes bias under name 'bias' (or 'z' from compat). Accept both.
        if bias is None and z is not None:
            bias = z
        # Boltz layer expects: a [B, N, c_a], s [B, N, c_s], bias [B, N, N, L*H], mask [B, N]
        # Squeeze batch.
        a_in = a.squeeze(0) if a.dim() == 3 else a
        s_in = s.squeeze(0) if s.dim() == 3 else s
        b_in = bias.squeeze(0) if bias.dim() == 4 else bias
        m_in = mask.squeeze(0) if mask is not None and mask.dim() == 2 else mask
        if m_in is None:
            m_in = torch.ones(a_in.shape[0], dtype=a.dtype, device=a.device)

        orig = a.dtype
        a_j = torch_to_jax(_act(a_in, self._config.dtype))
        s_j = torch_to_jax(_act(s_in, self._config.dtype))
        b_j = torch_to_jax(_act(b_in, self._config.dtype))
        m_j = torch_to_jax(m_in.to(torch.float32))
        out = self._jax_apply(self._params, None, a_j, s_j, b_j, m_j)
        out_t = jax_to_torch(out, dtype=orig)
        if a.dim() == 3:
            out_t = out_t.unsqueeze(0)
        return out_t


class BoltzJAX:
    """Boltz-2 PT model -> JAX hot-swap.

    Only the structure + confidence pairformers are swapped here. The
    diffusion token transformer is left as a TODO until the Boltz-specific
    HK variant lands (different pair-bias interface).
    """

    def __init__(
        self,
        model: nn.Module,
        config: BoltzJAXConfig,
        *,
        ckpt_path: str | None = None,
        ckpt_state: dict | None = None,
    ):
        self._model = model
        self._config = config
        if ckpt_state is None:
            assert ckpt_path is not None
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            ckpt_state = ckpt.get("state_dict", ckpt)
        self._ckpt = ckpt_state
        self._original: dict[str, Any] = {}

    def optimize(self) -> None:
        # Structure pairformer
        sp = _CompatPairformerJAXBoltz(
            apply_rules(
                pairformer_rules(num_layers=self._config.structure_pairformer_num_layers,
                                 confidence=False),
                self._ckpt,
            ),
            self._config,
            self._config.structure_pairformer_num_layers,
        )
        self._original["pairformer_module"] = self._model.pairformer_module
        self._model.pairformer_module = sp

        # Confidence pairformer
        if self._config.swap_confidence:
            cp = _CompatPairformerJAXBoltz(
                apply_rules(
                    pairformer_rules(num_layers=self._config.confidence_pairformer_num_layers,
                                     confidence=True),
                    self._ckpt,
                ),
                self._config,
                self._config.confidence_pairformer_num_layers,
            )
            self._original["confidence_pairformer"] = (
                self._model.confidence_module.pairformer_stack
            )
            self._model.confidence_module.pairformer_stack = cp

        # Token transformer
        if self._config.swap_token_transformer:
            tt = _CompatTokenTransformerJAXBoltz(
                apply_rules(
                    token_transformer_rules(
                        num_layers=self._config.token_transformer_num_layers,
                    ),
                    self._ckpt,
                ),
                self._config,
            )
            score_model = self._model.structure_module.score_model
            self._original["token_transformer"] = score_model.token_transformer
            score_model.token_transformer = tt

        # Template module's no-seq pairformer (the trimer-OOM target).
        if self._config.swap_template_pairformer and hasattr(self._model, "template_module"):
            tmpl_variant = PairformerVariant.boltz2_template(
                num_blocks=self._config.template_pairformer_num_layers
            )
            tmpl = _CompatPairformerNoSeqJAXBoltz(
                apply_rules(
                    template_pairformer_rules(
                        num_layers=self._config.template_pairformer_num_layers,
                    ),
                    self._ckpt,
                ),
                tmpl_variant, self._config.dtype, self._config.jit,
            )
            self._original["template_pairformer"] = self._model.template_module.pairformer
            self._model.template_module.pairformer = tmpl

        # MSA module's per-layer pairformer_layer — swap each independently.
        if self._config.swap_msa_pairformer and hasattr(self._model, "msa_module"):
            msa_variant = PairformerVariant.boltz2_msa(num_blocks=1)
            for layer_idx in range(self._config.msa_num_layers):
                msa_pl = _CompatPairformerNoSeqJAXBoltz(
                    apply_rules(
                        msa_pairformer_layer_rules(layer_idx),
                        self._ckpt,
                    ),
                    msa_variant, self._config.dtype, self._config.jit,
                )
                key = f"msa_pairformer_layer_{layer_idx}"
                self._original[key] = self._model.msa_module.layers[layer_idx].pairformer_layer
                self._model.msa_module.layers[layer_idx].pairformer_layer = msa_pl

    def restore(self) -> None:
        if "pairformer_module" in self._original:
            self._model.pairformer_module = self._original.pop("pairformer_module")
        if "confidence_pairformer" in self._original:
            self._model.confidence_module.pairformer_stack = (
                self._original.pop("confidence_pairformer")
            )
        if "token_transformer" in self._original:
            self._model.structure_module.score_model.token_transformer = (
                self._original.pop("token_transformer")
            )
        if "template_pairformer" in self._original:
            self._model.template_module.pairformer = self._original.pop("template_pairformer")
        for key in [k for k in self._original if k.startswith("msa_pairformer_layer_")]:
            layer_idx = int(key.rsplit("_", 1)[1])
            self._model.msa_module.layers[layer_idx].pairformer_layer = self._original.pop(key)
