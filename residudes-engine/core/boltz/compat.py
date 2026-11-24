# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Boltz2 TRT compatibility layer.

Model attribute paths:
  structure pairformer:    model.pairformer_module
  confidence pairformer:   model.confidence_module.pairformer_stack
  token transformer:       model.structure_module.score_model.token_transformer
                           (or model.structure_module.token_transformer for older boltz)
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
    @ensure_contiguous
    def forward_udf(self, s: torch.Tensor, z: torch.Tensor,
                    mask: torch.Tensor, pair_mask: torch.Tensor,
                    **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.config.support_batch:
            s = s.squeeze(0)
            z = z.squeeze(0)
            mask = mask.squeeze(0)
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
            "mask": mask.to(self.config.torch_dtype),
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
    def forward_udf(self,
                    a: torch.Tensor,
                    s: torch.Tensor,
                    z: torch.Tensor = None,
                    mask: torch.Tensor = None,
                    bias: torch.Tensor = None,
                    **kwargs) -> torch.Tensor:
        # Boltz-2 (DiffusionModule.forward in boltz/model/modules/diffusionv2.py) calls
        #   token_transformer(a, mask=mask.float(), s=s, bias=token_trans_bias.float(),
        #                     multiplicity=m)
        # with the diffusion batch already flattened into the leading axis:
        #   a    [b·m, N, dim]              s    [b·m, N, dim_single_cond]
        #   mask [b·m, N]                   bias [b,   N, N, heads·blocks]  (b = 1 at inference;
        #                                        the per-block pair biases are precomputed once
        #                                        per structure by DiffusionConditioning)
        # The Boltz-2 engine is built with version="v2", bias_proj=False, so it takes exactly
        # that precomputed bias as its `z` input ([1, N, N, heads·blocks]) and a/s/mask with
        # the multiplicity as the leading (dynamic, ≤ config.multiplicity) axis — no
        # re-projection inside the engine. We only need to chunk b·m into engine-sized
        # pieces. `to_keys` / `model_cache` are atom-level / cache arguments Boltz does not
        # pass here; they are accepted and ignored.
        if z is None:
            z = bias
        if z is None:
            raise ValueError("Boltz token transformer TRT: need `bias` (or `z`)")
        if a.dim() == 4:                      # defensive: [b, m, N, dim] → [b·m, N, dim]
            a = a.reshape(-1, *a.shape[2:])
            s = s.reshape(-1, *s.shape[2:])
            mask = mask.reshape(-1, mask.shape[-1]) if mask is not None else None
        bm, n_tok = a.shape[0], a.shape[1]
        if z.dim() == 3:
            z = z.unsqueeze(0)                # [N, N, HB] → [1, N, N, HB]
        if z.shape[0] != 1:
            raise ValueError(f"Boltz token transformer TRT: expected batch 1 bias, got {tuple(z.shape)}")
        if mask is None:
            mask = torch.ones(bm, n_tok, device=a.device, dtype=a.dtype)

        max_mult = int(self.config.multiplicity or 1)
        original_dtype = a.dtype
        eng_dtype = self.config.torch_dtype
        z_in = z.to(eng_dtype)
        outputs = []
        for start in range(0, bm, max_mult):
            end = min(start + max_mult, bm)
            inputs = {
                "a": a[start:end].to(eng_dtype),
                "s": s[start:end].to(eng_dtype),
                "z": z_in,
                "mask": mask[start:end].to(eng_dtype),
            }
            outputs.append(self._context_memory_allocator.forward(self, inputs)["output_a"])
        out = outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=0)
        return out.to(original_dtype)


class Boltz2AcceleratedModules(AcceleratedModules):

    def get_supported_modules(self) -> dict[str, tuple[nn.Module, Callable]]:

        def structure_pairformer_setter(mod: nn.Module,
                                        optimized: nn.Module) -> nn.Module:
            org = mod._model.pairformer_module
            setattr(mod._model, "pairformer_module", optimized)
            return org

        def confidence_pairformer_setter(mod: nn.Module,
                                         optimized: nn.Module) -> nn.Module:
            org = mod._model.confidence_module.pairformer_stack
            setattr(mod._model.confidence_module, "pairformer_stack", optimized)
            return org

        def token_transformer_setter(mod: nn.Module,
                                     optimized: nn.Module) -> nn.Module:
            # boltz 2.x: structure_module.score_model.token_transformer
            score_model = mod._model.structure_module.score_model
            org = score_model.token_transformer
            setattr(score_model, "token_transformer", optimized)
            return org

        return {
            "structure_pairformer": (CompatPairformerTRT, structure_pairformer_setter),
            "confidence_pairformer": (CompatPairformerTRT, confidence_pairformer_setter),
            "token_transformer": (CompatTokenTransformerTRT, token_transformer_setter),
        }


class BoltzTRT(nn.Module, OptimizedModuleSetterMixin):
    """Thin wrapper around a Boltz2 model for TRT acceleration.

    Usage::

        wrapped = BoltzTRT(boltz_model)
        wrapped.set_optimized_modules(accelerated_configs)
        # wrapped._model now has TRT-accelerated pairformer/transformer
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self._model = model

    def get_optimized_modules(
        self, accelerated_configs: dict[str, AcceleratedConfig]
    ) -> Boltz2AcceleratedModules:
        return Boltz2AcceleratedModules(accelerated_configs)

    def forward(self, **kwargs) -> Any:
        return self._model(**kwargs)
