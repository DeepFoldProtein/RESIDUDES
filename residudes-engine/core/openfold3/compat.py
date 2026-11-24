# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
OpenFold3 TRT compatibility layer.

OF3 interface (different from Protenix):
  - PairFormerStack.forward(s, z, single_mask, pair_mask, ...)
  - model.sample_diffusion.diffusion_module.diffusion_transformer
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
    """OF3 pairformer: forward(s, z, single_mask, pair_mask, ...)"""

    @ensure_contiguous
    def forward_udf(self, s: torch.Tensor, z: torch.Tensor,
                    single_mask: torch.Tensor, pair_mask: torch.Tensor,
                    **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.config.support_batch:
            s = s.squeeze(0)
            z = z.squeeze(0)
            single_mask = single_mask.squeeze(0)
            pair_mask = pair_mask.squeeze(0)
            n_res = z.shape[0]
        else:
            n_res = z.shape[1]

        if n_res >= 1408:
            total_memory = _get_total_memory(z.device.index)
            if total_memory <= 48:
                torch.cuda.empty_cache()
            elif total_memory <= 80 and n_res >= 1920:
                torch.cuda.empty_cache()

        original_dtype = s.dtype
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

        if not self.config.support_batch:
            s = s.unsqueeze(0)
            z = z.unsqueeze(0)
        return s, z


class CompatTokenTransformerTRT(TokenTransformerTRT):
    @ensure_contiguous
    def forward_udf(self, a: torch.Tensor, s: torch.Tensor,
                    z: torch.Tensor = None, mask: torch.Tensor = None,
                    **kwargs) -> torch.Tensor:
        diffusion_samples = a.shape[1]
        max_diffusion_samples = self.config.multiplicity
        s = s.squeeze(1)
        z = z.squeeze(1)
        mask = mask.squeeze(1)
        a = a.squeeze(0)

        outputs = []
        niters = (diffusion_samples + max_diffusion_samples - 1) // max_diffusion_samples
        for i in range(niters):
            n_repeat = min(diffusion_samples - i * max_diffusion_samples,
                           max_diffusion_samples)
            a_i = a[i * max_diffusion_samples:(i + 1) * max_diffusion_samples, ...]
            mask_i = mask.repeat_interleave(n_repeat, 0)
            s_i = s.repeat_interleave(n_repeat, 0)
            original_dtype = s.dtype
            inputs = {
                "a": a_i.to(self.config.torch_dtype),
                "s": s_i.to(self.config.torch_dtype),
                "z": z.to(self.config.torch_dtype),
                "mask": mask_i.to(self.config.torch_dtype),
            }
            allocator = self._context_memory_allocator
            outputs.append(allocator.forward(self, inputs)["output_a"])

        return torch.cat(outputs, dim=0).to(original_dtype).unsqueeze(0)


class OpenFold3AcceleratedModules(AcceleratedModules):
    def get_supported_modules(self) -> dict[str, tuple[nn.Module, Callable]]:
        def structure_pairformer_setter(mod, optimized):
            org = mod._model.pairformer_stack
            setattr(mod._model, "pairformer_stack", optimized)
            return org

        def token_transformer_setter(mod, optimized):
            # OF3: model.sample_diffusion.diffusion_module.diffusion_transformer
            org = mod._model.sample_diffusion.diffusion_module.diffusion_transformer
            setattr(mod._model.sample_diffusion.diffusion_module,
                    "diffusion_transformer", optimized)
            return org

        return {
            "structure_pairformer": (CompatPairformerTRT, structure_pairformer_setter),
            "token_transformer": (CompatTokenTransformerTRT, token_transformer_setter),
        }


class OpenFold3TRT(nn.Module, OptimizedModuleSetterMixin):
    def __init__(self, model: nn.Module):
        super().__init__()
        self._model = model

    def get_optimized_modules(self, accelerated_configs):
        return OpenFold3AcceleratedModules(accelerated_configs)

    def forward(self, **kwargs) -> Any:
        return self._model(**kwargs)
