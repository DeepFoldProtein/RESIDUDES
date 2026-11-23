# SPDX-License-Identifier: Apache-2.0
"""OpenFold3 JAX hot-module backend.

Mirrors ``Bio-TRT/openfold3/compat.py`` (the TRT version) but routes pairformer
and token transformer through JAX (Haiku) modules with DLPack bridge instead
of TRT engines.

Module attribute paths:
  pairformer:        model.pairformer_stack
  token_transformer: model.sample_diffusion.diffusion_module.diffusion_transformer

Usage::

    from openfold3.entry_points.runners import InferenceExperimentRunner
    from lantern_engine_jax.openfold3 import OpenFold3JAX, OpenFold3JAXConfig

    runner = InferenceExperimentRunner(...)
    runner.setup()  # PyTorch model loaded
    cfg = OpenFold3JAXConfig.from_checkpoint('/path/to/of3-p2-155k.pt')
    wrapper = OpenFold3JAX(runner.model, cfg)
    wrapper.optimize()  # in-place swap
    # runner.model now uses JAX for the two hot modules.

Caveats:
  - JAX must run on the same GPU as PyTorch (DLPack zero-copy assumes shared
    pointer). Set CUDA_VISIBLE_DEVICES consistently or call set_jax_device().
  - With CPU jaxlib, this works but is much slower than the TRT path. CUDA
    jaxlib is required for any real perf parity; see Bio-TRT README.
  - Engines are converted from PT weights at __init__ time; no .npz cache yet.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from typing import Any

import haiku as hk
import jax
import jax.numpy as jnp
import torch
import torch.nn as nn

from lantern_engine_jax.bridge import torch_to_jax, jax_to_torch, get_jax_device
from lantern_engine_jax.haiku.pairformer import PairformerStack, PairformerVariant
from lantern_engine_jax.protenix import _stack_per_block_params
from lantern_engine_jax.haiku.token_transformer import (
    TokenTransformer,
    TokenTransformerVariant,
)
from lantern_engine_jax.weights._rules import apply_rules
from lantern_engine_jax.weights.openfold3 import pairformer_rules, token_transformer_rules


@dataclass
class OpenFold3JAXConfig:
    """Configuration for the JAX hot-swap wrapper."""

    # Number of pairformer blocks to instantiate (must match checkpoint).
    pairformer_num_blocks: int = 48
    # Number of diffusion-transformer blocks.
    token_transformer_num_blocks: int = 24
    # Multiplicity (max diffusion samples per JIT call). The wrapper loops in
    # PyTorch over batches of this many samples to amortize trace cost.
    multiplicity: int = 5
    # FP32 vs BF16 for the JAX core. Default FP32 for parity; switch to BF16
    # for speed once verified.
    dtype: jnp.dtype = jnp.float32
    # If False, disable jax.jit (debug only).
    jit: bool = True

    @classmethod
    def from_checkpoint(cls, path: str) -> "OpenFold3JAXConfig":
        """Inspect a PT checkpoint to auto-detect block counts."""
        import re
        sd = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(sd, dict) and "model" in sd:
            sd = sd["model"]
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        pf = sorted(
            set(int(re.search(r"\.blocks\.(\d+)\.", k).group(1)) for k in sd
                if k.startswith("pairformer_stack.blocks."))
        )
        tt = sorted(
            set(int(re.search(r"\.blocks\.(\d+)\.", k).group(1)) for k in sd
                if k.startswith("sample_diffusion.diffusion_module."
                                "diffusion_transformer.blocks."))
        )
        return cls(
            pairformer_num_blocks=len(pf),
            token_transformer_num_blocks=len(tt),
        )


# ----------- Compat (PyTorch nn.Module) wrappers -----------

class _CompatPairformerJAXOF3(nn.Module):
    """OF3 pairformer signature: forward(s, z, single_mask, pair_mask, ...)."""

    def __init__(self, params: dict, config: OpenFold3JAXConfig):
        super().__init__()
        self._config = config
        variant = PairformerVariant.openfold3(num_blocks=config.pairformer_num_blocks)
        # BIO_TRT_JAX_PAIRFORMER_SCAN / _REMAT, as in the Protenix wrapper: with
        # scan the 48 blocks share one compiled body (hk.experimental.layer_stack).
        on = ("1", "true", "yes", "on")
        remat = os.environ.get("BIO_TRT_JAX_PAIRFORMER_REMAT", "1").lower() in on
        use_scan = os.environ.get("BIO_TRT_JAX_PAIRFORMER_SCAN", "0").lower() in on
        def fn(z, pair_mask, s, single_mask):
            return PairformerStack(variant=variant, with_single=True, remat=remat,
                                   use_scan=use_scan)(z, pair_mask, s, single_mask)
        # Don't name this `_apply` — that shadows nn.Module._apply, which torch
        # uses to recurse into children for .cpu() / .cuda() / .to(...).
        self._jax_apply = hk.transform(fn).apply
        if config.jit:
            self._jax_apply = jax.jit(self._jax_apply)
        if use_scan:
            params = _stack_per_block_params(
                params, num_blocks=config.pairformer_num_blocks, stack_root="pairformer",
                old_prefix="pair_former_block_", new_name="pair_former_block")
        # Cast params to engine dtype, push to JAX device.
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
        single_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        **kwargs,
    ):
        # OF3 PT shapes: s [1, N, c_s], z [1, N, N, c_z], masks [1, N], [1, N, N].
        s_b = s.dim() == 3
        z_b = z.dim() == 4
        if s_b: s_in = s.squeeze(0)
        else:   s_in = s
        if z_b: z_in = z.squeeze(0)
        else:   z_in = z
        if single_mask.dim() == 2: sm_in = single_mask.squeeze(0)
        else:                      sm_in = single_mask
        if pair_mask.dim() == 3:   pm_in = pair_mask.squeeze(0)
        else:                      pm_in = pair_mask

        orig_dtype = s.dtype
        s_j = torch_to_jax(_act(s_in, self._config.dtype))
        z_j = torch_to_jax(_act(z_in, self._config.dtype))
        sm_j = torch_to_jax(sm_in.to(torch.float32))
        pm_j = torch_to_jax(pm_in.to(torch.float32))

        s_out, z_out = self._jax_apply(self._params, None, z_j, pm_j, s_j, sm_j)
        s_t = jax_to_torch(s_out, dtype=orig_dtype)
        z_t = jax_to_torch(z_out, dtype=orig_dtype)
        if s_b: s_t = s_t.unsqueeze(0)
        if z_b: z_t = z_t.unsqueeze(0)
        return s_t, z_t


class _CompatTokenTransformerJAXOF3(nn.Module):
    """OF3 token transformer signature: forward(a, s, z, mask, ...)."""

    def __init__(self, params: dict, config: OpenFold3JAXConfig):
        super().__init__()
        self._config = config
        variant = TokenTransformerVariant.openfold3(num_blocks=config.token_transformer_num_blocks)
        # BIO_TRT_JAX_TOKEN_TRANSFORMER_SCAN, as in the Protenix wrapper (24 blocks → one body).
        use_scan = os.environ.get("BIO_TRT_JAX_TOKEN_TRANSFORMER_SCAN", "0").lower() in ("1", "true", "yes", "on")
        def fn(a, s, z, mask):
            return TokenTransformer(variant=variant, use_scan=use_scan)(a, s, z, mask)
        # Avoid the nn.Module._apply shadow trap (see _CompatPairformerJAXOF3).
        self._jax_apply = hk.transform(fn).apply
        if config.jit:
            self._jax_apply = jax.jit(self._jax_apply)
        if use_scan:
            params = _stack_per_block_params(
                params, num_blocks=config.token_transformer_num_blocks,
                stack_root="token_transformer", old_prefix="block_", new_name="block")
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
        **kwargs,
    ):
        # OF3 PT calls token_transformer (DiffusionTransformer) with broadcast
        # leading dims. Per the OF3 docstring:
        #   a    : [*, N, C_a]   — typically [B, n_samples, N, c_a] in inference
        #   s    : [*, N, C_s]   — broadcast over samples → [B, N, c_s]
        #   z    : [*, N, N, C_z] — broadcast over samples → [B, N, N, c_z]
        #   mask : [*, N]        — broadcast over samples → [B, N]
        # The single track (s, z, mask) is shared across diffusion samples,
        # only `a` carries the n_samples axis. We strip leading size-1 dims
        # to canonical shapes and loop on the n_samples axis of `a`.
        def _strip_leading_ones(t: torch.Tensor, target_ndim: int) -> torch.Tensor:
            while t.ndim > target_ndim and t.shape[0] == 1:
                t = t.squeeze(0)
            return t

        # a → [n_samples, N, c_a]
        a_in = _strip_leading_ones(a, 3)
        if a_in.ndim == 2:
            a_in = a_in.unsqueeze(0)         # [1, N, c_a] (single sample)
        n_samples = a_in.shape[0]

        # Single track is shared across samples — no sample axis to slice.
        s_in = _strip_leading_ones(s, 2)     # → [N, c_s]
        z_in = _strip_leading_ones(z, 3)     # → [N, N, c_z]
        m_in = _strip_leading_ones(mask, 1)  # → [N]

        orig_dtype = a.dtype
        z_j = torch_to_jax(_act(z_in, self._config.dtype))
        s_j = torch_to_jax(_act(s_in, self._config.dtype))
        m_j = torch_to_jax(m_in.to(torch.float32))

        outs = []
        for k in range(n_samples):
            a_j = torch_to_jax(_act(a_in[k], self._config.dtype))
            out = self._jax_apply(self._params, None, a_j, s_j, z_j, m_j)
            outs.append(jax_to_torch(out, dtype=orig_dtype))
        out_t = torch.stack(outs, dim=0)               # [n_samples, N, c_a]
        # Restore the original a leading-dim layout.
        # OF3 expects [*, n_samples, N, c_a] back; reproduce it from a's shape.
        for _ in range(a.ndim - out_t.ndim):
            out_t = out_t.unsqueeze(0)
        return out_t


def _param_dtype(x, target):
    a = jnp.asarray(x)
    if jnp.issubdtype(a.dtype, jnp.floating):
        return target
    return a.dtype


_INPUT_FP32 = os.environ.get("BIO_TRT_JAX_INPUT_DTYPE", "compute") == "fp32"


def _act(t: torch.Tensor, dtype) -> torch.Tensor:
    """Activations cross into JAX in the swapped modules' compute dtype (see
    lantern_engine_jax.boltz._act for why the old fp32 upcast cost 2–3× and 4·H·N³ bytes).
    BIO_TRT_JAX_INPUT_DTYPE=fp32 restores the old behaviour. Masks stay fp32."""
    if _INPUT_FP32 or dtype == jnp.float32:
        return t.to(torch.float32)
    return t.to(torch.bfloat16)


# ----------- Top-level wrapper that orchestrates the swap -----------

class OpenFold3JAX:
    """Wraps an OF3 PyTorch model and swaps its hot modules for JAX equivalents.

    Usage::

        wrapper = OpenFold3JAX(model, OpenFold3JAXConfig.from_checkpoint(ckpt_path))
        wrapper.optimize()
        # model.pairformer_stack and ....diffusion_transformer are now JAX-backed
    """

    def __init__(
        self,
        model: nn.Module,
        config: OpenFold3JAXConfig,
        *,
        ckpt_path: str | None = None,
        ckpt_state: dict | None = None,
    ):
        self._model = model
        self._config = config
        if ckpt_state is None:
            assert ckpt_path is not None, "Provide ckpt_path or ckpt_state"
            sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            if isinstance(sd, dict) and "model" in sd:
                sd = sd["model"]
            if isinstance(sd, dict) and "state_dict" in sd:
                sd = sd["state_dict"]
            ckpt_state = sd
        self._ckpt = ckpt_state
        self._original: dict[str, nn.Module] = {}

    def optimize(self) -> None:
        """In-place swap. Saves originals so restore() can revert."""
        # Pairformer
        pf_rules = pairformer_rules(num_blocks=self._config.pairformer_num_blocks)
        pf_params = apply_rules(pf_rules, self._ckpt)
        pf = _CompatPairformerJAXOF3(pf_params, self._config)
        self._original["pairformer_stack"] = self._model.pairformer_stack
        self._model.pairformer_stack = pf

        # Token transformer
        tt_rules = token_transformer_rules(
            num_blocks=self._config.token_transformer_num_blocks,
            use_sample_diffusion_copy=True,
        )
        tt_params = apply_rules(tt_rules, self._ckpt)
        tt = _CompatTokenTransformerJAXOF3(tt_params, self._config)
        diff_mod = self._model.sample_diffusion.diffusion_module
        self._original["diffusion_transformer"] = diff_mod.diffusion_transformer
        diff_mod.diffusion_transformer = tt

    def restore(self) -> None:
        """Revert the swap, putting the original PyTorch modules back."""
        if "pairformer_stack" in self._original:
            self._model.pairformer_stack = self._original.pop("pairformer_stack")
        if "diffusion_transformer" in self._original:
            self._model.sample_diffusion.diffusion_module.diffusion_transformer = (
                self._original.pop("diffusion_transformer")
            )
