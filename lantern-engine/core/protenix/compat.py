# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Protenix TRT compatibility layer.

Protenix has a DIFFERENT interface from OpenFold3:
  - PairformerStack.forward(s, z, pair_mask=None, ...) — no single_mask
  - DiffusionTransformer.forward(a, s, z, ...) — no mask arg
  - model.diffusion_module.diffusion_transformer (not sample_diffusion.diffusion_module)

The TRT engine I/O is the same (openfold3 config from tensorrt_bionemo), so the
engine files are shared. Only the compat wrappers differ.

Model attribute paths (Protenix):
  pairformer:          model.pairformer_stack
  token_transformer:   model.diffusion_module.diffusion_transformer
"""

from typing import Callable, Any

import torch
import torch.nn as nn

from tensorrt_bionemo._trt.module_wrappers import (PairformerTRT,
                                                   TokenTransformerTRT)
from tensorrt_bionemo.models.helper import (AcceleratedConfig,
                                            AcceleratedModules,
                                            OptimizedModuleSetterMixin)
from tensorrt_bionemo.runtime.misc import ensure_contiguous


def _get_total_memory(device_id: int) -> float:
    props = torch.cuda.get_device_properties(device_id)
    return props.total_memory / (1024 ** 3)


class CompatPairformerTRT(PairformerTRT):
    """TRT pairformer wrapper matching Protenix's PairformerStack.forward signature.

    Protenix calls: pairformer_stack(s, z, pair_mask=None, triangle_multiplicative=..., ...)
    TRT engine expects: s, z, mask (single), pair_mask
    """

    @ensure_contiguous
    def forward_udf(self, s: torch.Tensor, z: torch.Tensor,
                    pair_mask: torch.Tensor = None,
                    **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        # Protenix passes pair_mask=None and tensors WITHOUT batch dim:
        #   s: [N_token, c_s],  z: [N_token, N_token, c_z]
        # Track whether input had a batch dim so we can match the output shape.
        s_had_batch = (s.dim() == 3)  # [1, N, C] vs [N, C]
        z_had_batch = (z.dim() == 4)  # [1, N, N, C] vs [N, N, C]

        if s_had_batch:
            s = s.squeeze(0)
        if z_had_batch:
            z = z.squeeze(0)
        n_res = z.shape[0]

        # Generate masks (Protenix doesn't pass them)
        if pair_mask is None:
            single_mask = torch.ones(n_res, device=s.device, dtype=s.dtype)
            pair_mask = torch.ones(n_res, n_res, device=s.device, dtype=s.dtype)
        else:
            if pair_mask.dim() > 2:
                pair_mask = pair_mask.squeeze(0)
            single_mask = pair_mask.diagonal(dim1=-2, dim2=-1)

        # Empty cache for large sequences
        if n_res >= 1408:
            total_memory = _get_total_memory(z.device.index)
            if total_memory <= 48:
                torch.cuda.empty_cache()
            elif total_memory <= 80 and n_res >= 1920:
                torch.cuda.empty_cache()

        original_dtype = s.dtype

        # Engine built with support_batch=True expects batch dim: [1, N, C]
        support_batch = getattr(self.config, 'support_batch', False)
        if support_batch:
            s = s.unsqueeze(0)
            z = z.unsqueeze(0)
            single_mask = single_mask.unsqueeze(0)
            pair_mask = pair_mask.unsqueeze(0)

        inputs = {
            "s": s.to(self.config.torch_dtype),
            "z": z.to(self.config.torch_dtype),
            "mask": single_mask.to(self.config.torch_dtype),
            "pair_mask": pair_mask.to(self.config.torch_dtype),
        }

        allocator = self._context_memory_allocator
        outputs = allocator.forward(self, inputs)
        s = outputs["output_s"].to(original_dtype)
        z = outputs["output_z"].to(original_dtype)

        # Remove batch dim added for engine
        if support_batch:
            s = s.squeeze(0)
            z = z.squeeze(0)

        # Restore batch dim only if input had one
        if s_had_batch:
            s = s.unsqueeze(0)
        if z_had_batch:
            z = z.unsqueeze(0)
        return s, z


class CompatTokenTransformerTRT(TokenTransformerTRT):
    """TRT token transformer wrapper matching Protenix's DiffusionTransformer.forward signature.

    Protenix calls: diffusion_transformer(a=..., s=..., z=..., inplace_safe=..., ...)
    TRT engine expects: a, s, z, mask
    """

    @ensure_contiguous
    def forward_udf(self,
                    a: torch.Tensor,
                    s: torch.Tensor,
                    z: torch.Tensor = None,
                    **kwargs) -> torch.Tensor:
        # Protenix DiffusionModule calls diffusion_transformer with shapes
        # (NO batch dim, unlike OF3):
        #   a: [N_sample, N_token, c_token]   e.g. [10, 76, 768]
        #   s: [N_sample, N_token, c_s]       e.g. [10, 76, 384]
        #   z: [N_token, N_token, c_z]        e.g. [76, 76, 128]
        #
        # TRT engine expects:
        #   a: [mult, N_token, c_token]  (mult <= multiplicity, e.g. 5)
        #   s: [mult, N_token, c_s]
        #   z: [1, N_token, N_token, c_z]
        #   mask: [mult, N_token]

        n_samples = a.shape[0]
        n_token = a.shape[1]
        max_mult = self.config.multiplicity
        original_dtype = a.dtype

        # Protenix with enable_efficient_fusion=True permutes z to [...,C,N,N].
        # TRT engine expects [...,N,N,C] where C = dim_pairwise (128).
        # Detect: z's last 3 dims should be [N,N,C]. If last dim == second-to-last
        # but differs from third-to-last, z is in [C,N,N] layout.
        if z.dim() >= 3 and z.shape[-1] == z.shape[-2] and z.shape[-3] != z.shape[-1]:
            # z is in [...,C,N,N] layout → permute last 3 dims to [...,N,N,C]
            ndim = z.dim()
            z = z.permute(*range(ndim - 3), ndim - 2, ndim - 1, ndim - 3).contiguous()
        # z: add batch dim for engine [N,N,C] → [1,N,N,C]
        z_in = z.unsqueeze(0) if z.dim() == 3 else z

        # Generate mask (all-ones)
        mask_1 = torch.ones(1, n_token, device=a.device, dtype=a.dtype)

        outputs = []
        niters = (n_samples + max_mult - 1) // max_mult
        for i in range(niters):
            start = i * max_mult
            end = min(start + max_mult, n_samples)
            n = end - start

            a_i = a[start:end]                           # [n, N_token, c_token]
            s_i = s[start:end]                           # [n, N_token, c_s]
            mask_i = mask_1.expand(n, -1).contiguous()   # [n, N_token]

            inputs = {
                "a": a_i.to(self.config.torch_dtype),
                "s": s_i.to(self.config.torch_dtype),
                "z": z_in.to(self.config.torch_dtype),
                "mask": mask_i.to(self.config.torch_dtype),
            }
            allocator = self._context_memory_allocator
            outputs.append(allocator.forward(self, inputs)["output_a"])

        return torch.cat(outputs, dim=0).to(original_dtype)


class ProtenixAcceleratedModules(AcceleratedModules):

    def get_supported_modules(self) -> dict[str, tuple[nn.Module, Callable]]:

        def structure_pairformer_setter(mod: nn.Module,
                                        optimized: nn.Module) -> nn.Module:
            org = mod._model.pairformer_stack
            setattr(mod._model, "pairformer_stack", optimized)
            return org

        def token_transformer_setter(mod: nn.Module,
                                     optimized: nn.Module) -> nn.Module:
            # Protenix path: model.diffusion_module.diffusion_transformer
            # (NOT model.sample_diffusion.diffusion_module.diffusion_transformer)
            org = mod._model.diffusion_module.diffusion_transformer
            setattr(mod._model.diffusion_module,
                    "diffusion_transformer", optimized)
            return org

        return {
            "structure_pairformer": (CompatPairformerTRT, structure_pairformer_setter),
            "token_transformer": (CompatTokenTransformerTRT, token_transformer_setter),
        }


class ProtenixTRT(nn.Module, OptimizedModuleSetterMixin):
    """Thin wrapper around a Protenix model for TRT acceleration.

    Usage::

        runner = InferenceRunner(configs)
        wrapped = ProtenixTRT(runner.model)
        wrapped.optimize(accelerated_configs)
        # runner.model now has TRT-accelerated pairformer/transformer
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self._model = model

    def get_optimized_modules(
        self, accelerated_configs: dict[str, AcceleratedConfig]
    ) -> ProtenixAcceleratedModules:
        return ProtenixAcceleratedModules(accelerated_configs)

    def forward(self, **kwargs) -> Any:
        return self._model(**kwargs)
