# SPDX-License-Identifier: Apache-2.0
"""Protenix (v1 + v2) JAX hot-module backend.

Mirrors ``core/protenix/compat.py`` but routes through Haiku/JAX.

Module attribute paths (Protenix):
  pairformer:        model.pairformer_stack
  token_transformer: model.diffusion_module.diffusion_transformer
                      (NOT sample_diffusion.diffusion_module.*)

Forward signatures differ from OF3:
  - PairformerStack.forward(s, z, pair_mask=None, ...) — no single_mask;
    derives single mask from pair_mask diagonal (or all-ones if None).
  - DiffusionTransformer.forward(a, s, z, ...) — no explicit mask arg.
  - Tensors are WITHOUT a leading batch dim (s [N, c_s], z [N, N, c_z]).
  - Diffusion z may arrive in [...,C,N,N] layout when
    enable_efficient_fusion=True; we detect and permute.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import haiku as hk
import jax
import jax.numpy as jnp
import torch
import torch.nn as nn

from residudes_engine_jax.bridge import torch_to_jax, jax_to_torch, get_jax_device
from residudes_engine_jax.haiku.msa_module import MSAModuleStack, MsaModuleVariant
from residudes_engine_jax.haiku.pairformer import PairformerStack, PairformerVariant
from residudes_engine_jax.haiku.token_transformer import (
    TokenTransformer,
    TokenTransformerVariant,
)
from residudes_engine_jax.weights._rules import apply_rules
from residudes_engine_jax.weights.protenix import (
    msa_module_rules,
    pairformer_rules,
    token_transformer_rules,
)


@dataclass
class ProtenixJAXConfig:
    """Protenix JAX hot-swap config."""
    variant: Literal["v1", "v2"] = "v2"
    pairformer_num_blocks: int = 48
    token_transformer_num_blocks: int = 24
    msa_module_num_blocks: int = 4
    multiplicity: int = 5
    dtype: jnp.dtype = jnp.float32
    jit: bool = True

    @classmethod
    def from_checkpoint(cls, path: str, *, variant: str = "v2") -> "ProtenixJAXConfig":
        import re
        sd = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(sd, dict) and "model" in sd:
            sd = sd["model"]
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        pf = sorted(set(int(re.search(r"\.blocks\.(\d+)\.", k).group(1)) for k in sd
                        if k.startswith("module.pairformer_stack.blocks.")))
        tt = sorted(set(int(re.search(r"\.blocks\.(\d+)\.", k).group(1)) for k in sd
                        if k.startswith("module.diffusion_module.diffusion_transformer.blocks.")))
        return cls(
            variant=variant,
            pairformer_num_blocks=len(pf),
            token_transformer_num_blocks=len(tt),
        )


# ----------- Compat wrappers -----------


def _stack_per_block_params(
    params: dict, *, num_blocks: int, stack_root: str,
    old_prefix: str, new_name: str,
) -> dict:
    """Migrate per-index Haiku params (``pair_former_block_{i}/...``) into
    ``hk.experimental.layer_stack``-compatible stacked params
    (``pair_former_block/...`` with leading axis ``num_blocks``).

    Operates in the Haiku flat dict format: top-level keys are scope paths,
    values are inner ``{"w": ..., "b": ...}`` dicts. We rewrite only the
    block scopes that live under ``stack_root``.
    """
    out: dict = {}
    # Group block scopes by their suffix (everything after the index part).
    grouped: dict[str, list[tuple[int, dict]]] = {}
    for full_scope, inner in params.items():
        if not full_scope.startswith(stack_root + "/"):
            out[full_scope] = inner
            continue
        rest = full_scope[len(stack_root) + 1:]
        if not rest.startswith(old_prefix):
            out[full_scope] = inner
            continue
        # rest = pair_former_block_{i}[/sub/scope]
        head_and_tail = rest[len(old_prefix):].split("/", 1)
        try:
            idx = int(head_and_tail[0])
        except ValueError:
            out[full_scope] = inner
            continue
        sub = head_and_tail[1] if len(head_and_tail) == 2 else ""
        new_scope = (f"{stack_root}/__layer_stack_no_per_layer/{new_name}"
                     + (f"/{sub}" if sub else ""))
        grouped.setdefault(new_scope, []).append((idx, inner))

    import numpy as _np
    for new_scope, items in grouped.items():
        items.sort(key=lambda kv: kv[0])
        if [i for i, _ in items] != list(range(num_blocks)):
            raise ValueError(
                f"missing block params for {new_scope}: indices "
                f"{[i for i, _ in items]} != range({num_blocks})"
            )
        keys = items[0][1].keys()
        stacked = {
            k: _np.stack([inner[k] for _, inner in items], axis=0)
            for k in keys
        }
        out[new_scope] = stacked
    return out


def _pad_missing_msa_stack(
    params: dict, *, num_blocks: int, stack_root: str,
) -> dict:
    """Add zero-init msa_stack params for any MSA block that doesn't have
    them (last block in PT Protenix has no msa_stack). Required for
    ``hk.experimental.layer_stack`` to see homogeneous block params.

    With zero weights, the unused MSAStack call on the last block produces
    a no-op (msa_pair_weighted_averaging output = 0, transition_m output = 0,
    so m += 0 → m unchanged). Output ``z`` is unaffected.
    """
    import numpy as _np
    out = dict(params)
    # Find any msa_block_X scope that has msa_stack params.
    template: dict[str, dict] = {}
    for scope, inner in params.items():
        if not scope.startswith(f"{stack_root}/msa_block_"):
            continue
        # Extract sub-path after msa_block_X/
        rest = scope[len(stack_root) + len("/msa_block_"):]
        idx_str, _, sub = rest.partition("/")
        if not sub.startswith("msa_stack/"):
            continue
        if sub not in template:
            template[sub] = {k: _np.zeros_like(v) for k, v in inner.items()}

    if not template:
        return out

    # For each block, add missing msa_stack params from template (zero-init).
    for i in range(num_blocks):
        for sub, zero_inner in template.items():
            scope = f"{stack_root}/msa_block_{i}/{sub}"
            if scope not in out:
                out[scope] = dict(zero_inner)
    return out


class _CompatPairformerJAXProtenix(nn.Module):
    """Protenix pairformer signature: forward(s, z, pair_mask=None, ...)."""

    def __init__(self, params: dict, config: ProtenixJAXConfig):
        super().__init__()
        self._config = config
        if config.variant == "v2":
            variant = PairformerVariant.protenix_v2(num_blocks=config.pairformer_num_blocks)
        else:
            variant = PairformerVariant.protenix_v1(num_blocks=config.pairformer_num_blocks)
        import os as _os
        _remat = _os.environ.get(
            "RESIDUDES_ENGINE_JAX_PAIRFORMER_REMAT", "1").lower() in (
                "1", "true", "yes", "on")
        _use_scan = _os.environ.get(
            "RESIDUDES_ENGINE_JAX_PAIRFORMER_SCAN", "0").lower() in (
                "1", "true", "yes", "on")
        def fn(z, pair_mask, s, single_mask):
            return PairformerStack(
                variant=variant, with_single=True, remat=_remat,
                use_scan=_use_scan,
            )(z, pair_mask, s, single_mask)
        self._apply = hk.transform(fn).apply
        if config.jit:
            self._apply = jax.jit(self._apply)
        if _use_scan:
            params = _stack_per_block_params(
                params, num_blocks=config.pairformer_num_blocks,
                stack_root="pairformer",
                old_prefix="pair_former_block_",
                new_name="pair_former_block",
            )
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
        pair_mask: torch.Tensor = None,
        **kwargs,
    ):
        # Protenix may pass tensors with or without batch. Normalize.
        s_b = s.dim() == 3      # [1, N, c_s] vs [N, c_s]
        z_b = z.dim() == 4
        s_in = s.squeeze(0) if s_b else s
        z_in = z.squeeze(0) if z_b else z
        n_res = z_in.shape[0]

        if pair_mask is None:
            pm_in = torch.ones(n_res, n_res, dtype=s.dtype, device=s.device)
        else:
            pm_in = pair_mask.squeeze(0) if pair_mask.dim() > 2 else pair_mask
        # Single mask = pair_mask diagonal (when provided), else ones.
        sm_in = pm_in.diagonal(dim1=-2, dim2=-1)

        orig = s.dtype
        # Pass tensors in their native dtype (BF16/FP32). JAX side will cast
        # if needed. Avoid the costly .to(torch.float32) upcast — it adds
        # ~2s/call for N=588 and doubles peak memory.
        s_j = torch_to_jax(s_in)
        z_j = torch_to_jax(z_in)
        sm_j = torch_to_jax(sm_in)
        pm_j = torch_to_jax(pm_in)

        s_out, z_out = self._apply(self._params, None, z_j, pm_j, s_j, sm_j)
        s_t = jax_to_torch(s_out, dtype=orig)
        z_t = jax_to_torch(z_out, dtype=orig)
        if s_b: s_t = s_t.unsqueeze(0)
        if z_b: z_t = z_t.unsqueeze(0)
        return s_t, z_t


class _CompatTokenTransformerJAXProtenix(nn.Module):
    """Protenix token transformer: forward(a, s, z, ...).

    Protenix shapes:
      a: [B?, N_sample, N_token, c_a]  (e.g. [1, 5, N, 768])
      s: [B?, N_sample, N_token, c_s]  OR [N_sample, N, c_s]
      z: [B?, N_token, N_token, c_z]   OR [B?, c_z, N_token, N_token] if
                                            enable_efficient_fusion=True

    All N_sample samples are processed in ONE jax.vmap'd JIT call to
    amortize compilation across the 200-step diffusion loop.
    """

    def __init__(self, params: dict, config: ProtenixJAXConfig):
        super().__init__()
        self._config = config
        if config.variant == "v2":
            variant = TokenTransformerVariant.protenix_v2(num_blocks=config.token_transformer_num_blocks)
        else:
            variant = TokenTransformerVariant.protenix_v1(num_blocks=config.token_transformer_num_blocks)
        import os as _os
        _use_scan = _os.environ.get(
            "RESIDUDES_ENGINE_JAX_TOKEN_TRANSFORMER_SCAN", "0").lower() in (
                "1", "true", "yes", "on")
        def fn(a, s, z, mask):
            return TokenTransformer(variant=variant, use_scan=_use_scan)(a, s, z, mask)
        apply_unbatched = hk.transform(fn).apply
        # vmap over the n_samples leading axis of (a, s); z and mask shared.
        # Signature: apply(params, rng, a, s, z, mask)
        apply_vmapped = jax.vmap(
            apply_unbatched, in_axes=(None, None, 0, 0, None, None)
        )
        if config.jit:
            self._apply = jax.jit(apply_vmapped)
        else:
            self._apply = apply_vmapped
        if _use_scan:
            params = _stack_per_block_params(
                params, num_blocks=config.token_transformer_num_blocks,
                stack_root="token_transformer",
                old_prefix="block_",
                new_name="block",
            )
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
        **kwargs,
    ):
        # In real Protenix inference, ``a`` arrives as [B, N_sample, N, c_a],
        # ``s`` as [B, N_sample, N, c_s] (or [N_sample, N, c_s]), and ``z``
        # as [B, N, N, c_z] OR [B, c_z, N, N] (efficient_fusion).
        # Squeeze the leading batch=1 dim if present.
        if a.dim() == 4: a = a.squeeze(0)
        if s.dim() == 4: s = s.squeeze(0)
        if z.dim() == 4: z = z.squeeze(0)

        n_token = a.shape[1]

        # Detect [...,C,N,N] layout (enable_efficient_fusion=True). The last 3
        # dims of z should be [N, N, C]; if last == second-to-last but != third,
        # z is in [C, N, N] layout — permute to [N, N, C].
        if z.dim() >= 3 and z.shape[-1] == z.shape[-2] and z.shape[-3] != z.shape[-1]:
            ndim = z.dim()
            z = z.permute(*range(ndim - 3), ndim - 2, ndim - 1, ndim - 3).contiguous()
        # All-ones mask (Protenix doesn't pass one)
        mask_t = torch.ones(n_token, dtype=a.dtype, device=a.device)

        orig = a.dtype
        # Native dtype passthrough (avoid FP32 upcast).
        a_j = torch_to_jax(a)
        s_j = torch_to_jax(s)
        z_j = torch_to_jax(z)
        m_j = torch_to_jax(mask_t)
        out = self._apply(self._params, None, a_j, s_j, z_j, m_j)  # vmap'd
        return jax_to_torch(out, dtype=orig)        # [n_samples, N, c_a]


class _CompatMsaModuleJAXProtenix(nn.Module):
    """Protenix MSAModule hot-swap.

    The PT preprocessing (MSA sampling, one-hot encoding, the input
    embedding linears) stays in PyTorch — it's ~50ms on 9fm7 and uses
    Protenix's own helpers. Only the 4-block loop runs in JAX, where
    cuex tri_mul / tri_attn make it 5-10x faster than PT torch kernels.
    """

    def __init__(self, original_msa: nn.Module, params: dict,
                 config: ProtenixJAXConfig):
        super().__init__()
        self._config = config
        # Keep reference to original module — we call its preprocess modules
        # (sample_msa_feature_dict_random_without_replacement / one_hot_fp32 /
        # linear_no_bias_m / linear_no_bias_s) on the PT side.
        self._orig = original_msa

        variant = MsaModuleVariant.protenix_v2(
            num_blocks=config.msa_module_num_blocks,
        )

        import os as _os
        _msa_remat = _os.environ.get(
            "RESIDUDES_ENGINE_JAX_MSA_REMAT", "0").lower() in ("1","true","yes","on")
        _msa_scan = _os.environ.get(
            "RESIDUDES_ENGINE_JAX_MSA_SCAN", "1").lower() in ("1","true","yes","on")

        def fn(m, z, pair_mask, m_mask):
            return MSAModuleStack(
                variant=variant, use_scan=_msa_scan, remat=_msa_remat,
            )(m, z, pair_mask, m_mask)

        self._apply = hk.transform(fn).apply
        if config.jit:
            self._apply = jax.jit(self._apply)

        if _msa_scan:
            # layer_stack expects homogeneous block params: per-block dicts
            # with the SAME set of leaves. The last MSA block has no
            # msa_stack in PT, so the converter omits those entries for
            # block N-1. We pad them with zero-init copies (the layer_stack
            # body still runs MSAStack on the last block, but its m output
            # is discarded — zero weights → identity-like update → no
            # numerical effect on the surfaced z).
            params = _pad_missing_msa_stack(
                params, num_blocks=config.msa_module_num_blocks,
                stack_root="msa_module",
            )
            params = _stack_per_block_params(
                params, num_blocks=config.msa_module_num_blocks,
                stack_root="msa_module",
                old_prefix="msa_block_",
                new_name="msa_block",
            )
        self._params = jax.device_put(
            jax.tree_util.tree_map(
                lambda x: jnp.asarray(x, dtype=_param_dtype(x, config.dtype)),
                params,
            ),
            get_jax_device(),
        )

    def forward(
        self,
        input_feature_dict: dict,
        z: torch.Tensor,
        s_inputs: torch.Tensor,
        pair_mask: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        # 1-5: PT preprocess (mirrors models/protenix/protenix/model/modules/pairformer.py:809)
        from protenix.model.modules.pairformer import (
            sample_msa_feature_dict_random_without_replacement,
        )
        if self._orig.n_blocks < 1 or "msa" not in input_feature_dict:
            return z
        if input_feature_dict["msa"].dim() < 2:
            return z

        # AF3 caps MSA depth at 1024 by default. Protenix allows up to
        # 16384, but the JAX path scales poorly with S (no fused MSA-aware
        # kernels). We cap the test cutoff used for sampling — env var
        # ``RESIDUDES_ENGINE_JAX_MSA_DEPTH_CAP`` overrides (default 1024).
        import os as _os
        try:
            cutoff_cap = int(_os.environ.get("RESIDUDES_ENGINE_JAX_MSA_DEPTH_CAP", "1024"))
        except ValueError:
            cutoff_cap = 1024
        eff_cutoff = (self._orig.msa_configs["train_cutoff"]
                      if self._orig.training
                      else self._orig.msa_configs["test_cutoff"])
        eff_cutoff = min(eff_cutoff, cutoff_cap)
        msa_feat = sample_msa_feature_dict_random_without_replacement(
            feat_dict=input_feature_dict,
            dim_dict={k: -2 for k in self._orig.input_feature},
            cutoff=eff_cutoff,
            lower_bound=(self._orig.msa_configs["train_lowerb"]
                         if self._orig.training
                         else self._orig.msa_configs["test_lowerb"]),
            strategy=self._orig.msa_configs["strategy"],
        )
        if (not self._orig.training) and z.shape[-2] > 2000:
            msa_feat["msa"] = self._orig.one_hot_fp32(
                msa_feat["msa"],
                num_classes=self._orig.input_feature["msa"],
            )
        else:
            msa_feat["msa"] = torch.nn.functional.one_hot(
                msa_feat["msa"],
                num_classes=self._orig.input_feature["msa"],
            )
        target_shape = msa_feat["msa"].shape[:-1]
        msa_sample = torch.cat(
            [msa_feat[name].reshape(*target_shape, d)
             for name, d in self._orig.input_feature.items()],
            dim=-1,
        )
        if not self._orig.training:
            del msa_feat
        msa_sample = self._orig.linear_no_bias_m(msa_sample)
        msa_sample = msa_sample + self._orig.linear_no_bias_s(s_inputs)

        # Squeeze leading batch dim if present.
        if msa_sample.dim() == 4: msa_sample = msa_sample.squeeze(0)
        if z.dim() == 4: z = z.squeeze(0)
        if pair_mask is None:
            n_res = z.shape[-2]
            pair_mask = torch.ones(
                n_res, n_res, dtype=z.dtype, device=z.device)
        elif pair_mask.dim() > 2:
            pair_mask = pair_mask.squeeze(0)

        # Memory gate: fall back to original PT block loop when MSA depth
        # exceeds a threshold. The JAX path's intermediate tensors scale
        # as S * N * c_intermediate (8 * c_m at MSAPairWeightedAveraging /
        # transition) and OOM on a 24 GiB 4090 above S≈1000 with N=644.
        # The PT path uses chunk_size=2048 inside its kernels and stays
        # within memory. Tunable via RESIDUDES_ENGINE_JAX_MSA_MAX_S.
        import os as _os
        try:
            max_s = int(_os.environ.get("RESIDUDES_ENGINE_JAX_MSA_MAX_S", "2048"))
        except ValueError:
            max_s = 2048
        if msa_sample.shape[0] > max_s:
            # Fall back to PT block loop. Reuse the original module's blocks
            # via _prep_blocks + checkpoint_blocks (same code path as the
            # un-swapped MSAModule.forward).
            from protenix.model.utils import checkpoint_blocks
            blocks = self._orig._prep_blocks(pair_mask=pair_mask)
            _, z = checkpoint_blocks(blocks, args=(msa_sample, z),
                                     blocks_per_ckpt=None)
            return z

        # 6: JAX 4-block stack
        m_mask = torch.ones(msa_sample.shape[:-1],
                            dtype=msa_sample.dtype, device=msa_sample.device)
        orig = z.dtype
        m_j = torch_to_jax(msa_sample)
        z_j = torch_to_jax(z)
        pm_j = torch_to_jax(pair_mask)
        mm_j = torch_to_jax(m_mask)
        z_out = self._apply(self._params, None, m_j, z_j, pm_j, mm_j)
        z_t = jax_to_torch(z_out, dtype=orig)
        return z_t


def _param_dtype(x, target):
    a = jnp.asarray(x)
    if jnp.issubdtype(a.dtype, jnp.floating):
        return target
    return a.dtype


class ProtenixJAX:
    """Protenix v1/v2 PT model -> JAX hot-swap."""

    def __init__(
        self,
        model: nn.Module,
        config: ProtenixJAXConfig,
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
        pf = _CompatPairformerJAXProtenix(
            apply_rules(
                pairformer_rules(num_blocks=self._config.pairformer_num_blocks,
                                 variant=self._config.variant),
                self._ckpt,
            ),
            self._config,
        )
        self._original["pairformer_stack"] = self._model.pairformer_stack
        self._model.pairformer_stack = pf

        tt = _CompatTokenTransformerJAXProtenix(
            apply_rules(
                token_transformer_rules(
                    num_blocks=self._config.token_transformer_num_blocks,
                    variant=self._config.variant,
                ),
                self._ckpt,
            ),
            self._config,
        )
        diff_mod = self._model.diffusion_module
        self._original["diffusion_transformer"] = diff_mod.diffusion_transformer
        diff_mod.diffusion_transformer = tt

        # MSA module hot-swap — opt-in (default OFF) via RESIDUDES_ENGINE_JAX_MSA_MODULE=1.
        # The 4-block JAX path is ~10x faster than PT *steady state* (3ms vs
        # 30ms/call), but the first 1-2 calls pay 4-5s of CUDA kernel autotune
        # per new shape. With only 10 cycles per inference, the autotune cost
        # exceeds the steady-state savings. Net loss for one-off jobs; potential
        # win for long-running services where autotune amortizes across many
        # proteins of similar shape. Large MSA (S>1024) falls back to PT to
        # avoid OOM (S * N * c_intermediate doesn't fit in 24 GiB). Set
        # RESIDUDES_ENGINE_JAX_MSA_MAX_S to tune the fallback threshold.
        import os as _os
        if _os.environ.get("RESIDUDES_ENGINE_JAX_MSA_MODULE", "0").lower() in (
                "1", "true", "yes", "on"):
            try:
                msa = _CompatMsaModuleJAXProtenix(
                    self._model.msa_module,
                    apply_rules(
                        msa_module_rules(
                            num_blocks=self._config.msa_module_num_blocks,
                            variant=self._config.variant,
                        ),
                        self._ckpt,
                    ),
                    self._config,
                )
                self._original["msa_module"] = self._model.msa_module
                self._model.msa_module = msa
            except Exception as exc:
                # Fall back to PT-side MSA module on any error so we don't
                # block the rest of the JAX hot-swap.
                import logging as _logging
                _logging.getLogger(__name__).warning(
                    "MSA module JAX hot-swap disabled: %s", exc)

    def restore(self) -> None:
        if "msa_module" in self._original:
            self._model.msa_module = self._original.pop("msa_module")
        if "pairformer_stack" in self._original:
            self._model.pairformer_stack = self._original.pop("pairformer_stack")
        if "diffusion_transformer" in self._original:
            self._model.diffusion_module.diffusion_transformer = (
                self._original.pop("diffusion_transformer")
            )
