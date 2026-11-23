# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from abc import ABCMeta, abstractmethod

import torch

from .._common import default_net
from .._utils import set_obj_attrs, str_dtype_to_torch, str_dtype_to_trt
from ..functional import Tensor, cast, matmul
from ..module import Module
from ..parameter import Parameter


class LinearBase(Module, metaclass=ABCMeta):

    def __init__(
        self,
        local_in_features,
        local_out_features,
        bias=True,
        dtype=None,
        share_weight=None,
        strict_dtype=False,
        pad_lda=0,
        pad_ldc=0,
        prefer_managed_weight=True,
    ):
        super().__init__()
        self.in_features = local_in_features
        self.out_features = local_out_features
        self.dtype = dtype
        self.pad_lda = pad_lda
        self.pad_ldc = pad_ldc
        self.prefer_managed_weight = prefer_managed_weight

        self.share_weight = share_weight
        if not share_weight:
            self.weight = Parameter(
                shape=(self.out_features, self.in_features),
                dtype=dtype,
                prefer_managed=self.prefer_managed_weight,
            )
            set_obj_attrs(
                self.weight,
                {
                    "weight_loader": self.weight_loader,
                },
            )
        else:
            self.weight = share_weight

        self.strict_dtype = str_dtype_to_trt(
            self.dtype) if strict_dtype else None

        if bias:
            self.bias = Parameter(shape=(self.out_features, ), dtype=dtype)
            assert pad_ldc == 0, "not support pad_ldc with bias"
        else:
            self.register_parameter("bias", None)

    def weight_loader(self, param: Parameter,
                      loaded_weight: torch.Tensor) -> None:
        param.value = loaded_weight

    def get_weight(self) -> Tensor:
        return self.weight.get_constant_tensor(network=default_net())

    @abstractmethod
    def collect_and_bias(self, x: Tensor) -> Tensor:
        pass

    def multiply_collect(self, x, weight, **kwargs):
        x = matmul(x, weight, transb=True)
        return self.collect_and_bias(x, **kwargs)

    def forward(self, x, **kwargs) -> Tensor:
        return self.multiply_collect(x, self.get_weight(), **kwargs)


class Linear(LinearBase):

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        dtype=None,
        share_weight=None,
        strict_dtype=False,
        pad_lda=0,
        pad_ldc=0,
        prefer_managed_weight=True,
        is_qkv=False,
    ):
        super().__init__(
            local_in_features=in_features,
            local_out_features=out_features,
            bias=bias,
            dtype=dtype,
            share_weight=share_weight,
            strict_dtype=strict_dtype,
            pad_lda=pad_lda,
            pad_ldc=pad_ldc,
            prefer_managed_weight=prefer_managed_weight,
        )
        self.is_qkv = is_qkv
        if bias:
            set_obj_attrs(
                self.bias,
                {
                    "weight_loader": self.weight_loader,
                },
            )

    def collect_and_bias(self, x, **kwargs):

        if self.bias is not None:
            bias = cast(self.bias.value, x.dtype)
            x = x + bias
        return x

    def postprocess(self, tllm_key, weights, **kwargs):
        using_head_as_leading_dim = kwargs.get("using_head_as_leading_dim",
                                               False)
        config = kwargs.get("config", None)
        if self.is_qkv:
            if isinstance(weights, list):
                head_size = config.hidden_size // config.num_attention_heads if config.head_size is None else config.head_size
                if getattr(config, "remove_duplicated_kv_heads", False):
                    if config.remove_duplicated_kv_heads:
                        k, v = weights[1:]
                        k = k.reshape([
                            k.shape[0] // head_size // 2, 2, head_size,
                            self.in_features
                        ])
                        v = v.reshape([
                            v.shape[0] // head_size // 2, 2, head_size,
                            self.in_features
                        ])
                        assert (k[:, 0] == k[:, 1]).all()
                        assert (v[:, 0] == v[:, 1]).all()
                        k = k[:, 0].reshape([-1, self.in_features])
                        v = v[:, 0].reshape([-1, self.in_features])
                        weights[1] = k
                        weights[2] = v
                # Duplicate kv heads in case of invalid TP size
                config.num_key_value_heads

                weights = torch.cat(weights)
            if using_head_as_leading_dim:
                # Reorder [n_head, 3, head_dim, ...] into [3, n_head, head_dim, ...]
                assert config.num_attention_heads == config.num_key_value_heads, "using_head_as_leading_dim require head_size to be multiple of 3."
                num_heads = config.num_attention_heads
                head_dim = self.out_features // (3 * num_heads)
                w = weights.reshape(num_heads, 3, head_dim, -1)
                w = w.transpose(0, 1)
                if w.shape[-1] > 1:
                    weights = w.reshape(-1, self.in_features)  # Weight
                else:
                    weights = w.reshape(-1)  # Bias

        if isinstance(weights, list):
            weights = torch.cat(weights)

        weights = weights.to(str_dtype_to_torch(self.dtype))
        return {tllm_key: weights}
