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
from enum import IntEnum
from typing import Callable, Optional, Union

import numpy as np
import tensorrt as trt
import tensorrt.plugin as trtp
from tensorrt_llm_lite._common import default_trtnet
from tensorrt_llm_lite._utils import fp32_array
from tensorrt_llm_lite.functional import (Tensor, _create_tensor, cast, concat,
                                          constant, floordiv, shape)


class AttentionBiasType(IntEnum):
    triangle = 0
    pairwise = 1


def _plugin_v3_output(net, layer, idx=0) -> Tensor:
    """Wrap a plugin v3 layer output with an identity layer.

    Workaround for TRT plugin v3 output buffer reuse issue (GitHub #3313):
    TRT may reuse the plugin's output buffer before downstream layers read it.
    Adding a native identity layer forces TRT to copy the output, preventing
    buffer corruption.
    """
    raw = layer.get_output(idx)
    identity = net.add_identity(raw)
    return _create_tensor(identity.get_output(0), identity)


def _get_plugin_namespace():
    """Return the plugin namespace to use: 'trifast' if available, else 'cuequivariance'."""
    try:
        _ = trtp.op.trifast.identity_2
        return "trifast"
    except AttributeError:
        return "cuequivariance"


def identity_sz(s: Tensor,
                z: Tensor,
                use_identity_plugin: bool = True) -> Tensor:
    '''
    Add an identity operation.

    This plugin is used to break Myelin layers. See NVBUGs: https://nvbugspro.nvidia.com/bug/5300894

    Parameters:
        s : Tensor
            The first input tensor.
        z : Tensor
            The second input tensor.

    Returns:
        The tensors produced by this identity operation.
    '''
    if not use_identity_plugin:
        return s, z
    # Use C++ IdentitySZ plugin (from libnvinfer_plugin_tensorrt_bionemo.so)
    # to break Myelin fusion. Python plugin v3 has output buffer issues,
    # so the C++ plugin is required for correctness.
    from .plugin import TRT_BNM_PLUGIN_NAMESPACE
    plg_creator = trt.get_plugin_registry().get_plugin_creator(
        'IdentitySZ', '1', TRT_BNM_PLUGIN_NAMESPACE)
    assert plg_creator is not None, (
        "IdentitySZ plugin not found. Ensure libnvinfer_plugin_tensorrt_bionemo.so "
        "is in thirdparty/tensorrt_bionemo/libs/ and LD_LIBRARY_PATH includes /opt/conda/lib"
    )
    pfc = trt.PluginFieldCollection()
    id_plug = plg_creator.create_plugin("identity_sz", pfc)
    plug_inputs = [s.trt_tensor, z.trt_tensor]
    layer = default_trtnet().add_plugin_v2(plug_inputs, id_plug)
    from tensorrt_llm_lite.functional import _add_plugin_info
    _add_plugin_info(layer, plg_creator, "identity_sz", pfc)

    return _create_tensor(layer.get_output(0), layer), \
        _create_tensor(layer.get_output(1), layer)


def chunk_loop(tensors: list[Tensor],
               chunk_size: int = 0,
               loop_body: Callable = None,
               reshape_output: bool = True,
               name: str = "chunk_loop_output") -> Tensor:
    if chunk_size == 0 or chunk_size is None:
        if loop_body is not None:
            return loop_body(tensors)
        return tensors
    x = tensors[0]
    bs = shape(x, 0)
    bs = cast(bs, trt.int64)
    chunk_size_tensor = constant(np.array(chunk_size, dtype=np.int64))
    niters = cast(floordiv(bs, chunk_size_tensor), trt.int64)

    # Reshape tensors to have the first dimension be the iteration dimension
    reshaped_tensors = []
    for t in tensors:
        s = []
        for i in range(1, t.ndim()):
            s.append(shape(t, i))
        t = t.view(concat([niters, chunk_size_tensor, *s]))
        reshaped_tensors.append(t)
    chunk_loop = default_trtnet().add_loop()
    chunk_loop.add_trip_limit(niters.trt_tensor, trt.TripLimit.COUNT)

    data = []
    for t in reshaped_tensors:
        # For each tensor, we need to add a loop iterator
        iterator = chunk_loop.add_iterator(t.trt_tensor, 0, False)
        data.append(_create_tensor(iterator.get_output(0), iterator))

    if loop_body is not None:
        loop_body_output = loop_body(data)
    else:
        assert len(data) == 1
        loop_body_output = data[0]

    trt_output_layer = chunk_loop.add_loop_output(loop_body_output.trt_tensor,
                                                  trt.LoopOutput.CONCATENATE,
                                                  0)
    trt_output_layer.name = name
    trt_output_layer.set_input(1, niters.trt_tensor)
    output = _create_tensor(trt_output_layer.get_output(0), trt_output_layer)
    if reshape_output:
        s = []
        for i in range(2, output.ndim()):
            s.append(shape(output, i))
        output = output.view(concat([bs, *s]))
    return output


def triangle_attention(q: Tensor,
                       k: Tensor,
                       v: Tensor,
                       bias: Tensor,
                       mask: Optional[Tensor],
                       scale: float,
                       num_heads: int = 0,
                       backend: str = "TRIFAST") -> Tensor:
    """Triangle attention using Python TRT plugins.

    Backends:
        TRIFAST: trifast Triton kernel (numerically stable, recommended)
        CUEQUIV: cuEquivariance CUDA kernel (has BF16 precision issues on long sequences)

    Parameters:
        q : Tensor of shape (B, N, H, Q, D)
        k : Tensor of shape (B, N, H, K, D)
        v : Tensor of shape (B, N, H, K, D)
        bias : Tensor of shape (B, 1, H, Q, K)
        mask : Optional[Tensor] of shape (B, N, 1, 1, K), bool
        scale : float, typically 1.0 / sqrt(head_dim)
        num_heads : int, number of attention heads (required for TRIFAST)
        backend : str, "TRIFAST" or "CUEQUIV"

    Returns:
        Output tensor of shape (B, N, H, Q, D)
    """
    net = default_trtnet()
    assert mask is not None, "triangle_attention requires mask"

    if backend == "TRIFAST":
        assert num_heads > 0, "num_heads required for TRIFAST backend"
        inputs, shape_inputs, plugin = trtp.op.trifast.triangle_attention(
            q.trt_tensor, k.trt_tensor, v.trt_tensor, bias.trt_tensor,
            mask.trt_tensor, scale=scale, num_heads=num_heads)(net)
    else:  # CUEQUIV
        inputs, shape_inputs, plugin = trtp.op.cuequivariance.triangle_attention(
            q.trt_tensor, k.trt_tensor, v.trt_tensor, bias.trt_tensor,
            mask.trt_tensor, scale=scale)(net)

    layer = net.add_plugin_v3(inputs, shape_inputs, plugin)
    return _plugin_v3_output(net, layer)


def dynamic_const_tensor(
        shape: Tensor,
        value: float = 0.0,
        dtype: Union[str, trt.DataType] = 'float32') -> Tensor:
    """ Create a dynamic tensor of zeros. """
    low = constant(fp32_array(float(0.0)))
    high = constant(fp32_array([float(0.0)] * shape.shape[0]))

    layer = default_trtnet().add_fill([0], trt.FillOperation.LINSPACE,
                                      trt.float32)

    layer.set_input(0, shape.trt_tensor)
    layer.set_input(1, low.trt_tensor)
    layer.set_input(2, high.trt_tensor)

    if value != 0.0:
        c = constant(fp32_array(value))
        return cast(_create_tensor(layer.get_output(0), layer) + c, dtype)
    return cast(_create_tensor(layer.get_output(0), layer), dtype)
