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

from ..functional import layer_norm
from ..module import Module
from ..parameter import Parameter


class LayerNorm(Module):

    def __init__(self,
                 normalized_shape,
                 eps=1e-05,
                 elementwise_affine=True,
                 bias=True,
                 dtype=None):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape, )
        self.normalized_shape = tuple(normalized_shape)
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = Parameter(shape=self.normalized_shape, dtype=dtype)
            if bias:
                self.bias = Parameter(shape=self.normalized_shape, dtype=dtype)
            else:
                self.register_parameter('bias', None)
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.eps = eps
        self.dtype = dtype

    def forward(self, x, normalized_shape=None):
        weight = 1. if self.weight is None else self.weight.value
        bias = 0. if self.bias is None else self.bias.value
        if normalized_shape is None:
            normalized_shape = self.normalized_shape
        return layer_norm(x, normalized_shape, weight, bias, self.eps)
