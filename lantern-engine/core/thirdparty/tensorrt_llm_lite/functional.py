# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import math
import weakref
from collections import OrderedDict
from enum import IntEnum
from functools import partial
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

# isort: off
import tensorrt as trt
# isort: on

from ._common import default_net, default_trtnet, precision
from ._utils import (bf16_array, bool_array, dim_resolve_negative,
                     dim_to_trt_axes, dims_array, fp16_array, fp32_array,
                     int32_array, int64_array, np_dtype_to_trt,
                     str_dtype_to_trt, trt_dtype_to_np, trt_dtype_to_str)
from .network import PluginInfo, set_np_weight, set_plugin_info


class DimRange(object):
    '''
        One DimRange object stores the ranges of all the dimensions of one tensor in one optimization profile.
        For example, tensor has 2 dimensions. Then the data members are:
            self.min = [dim 0 min, dim 1 min]
            self.opt = [dim 0 opt, dim 1 opt]
            self.max = [dim 0 max, dim 1 max]
        For static dimension, it has min==opt==max, thus the shape param in the ctor can be an integer
    '''

    def __init__(self, shape: List[Union[int, List[int], Tuple[int, int,
                                                               int]]],
                 names: List[str]):
        '''
        Parameters:
            shape: a list with length N, each element is an integer or a 3-elements tuple/list of int,
                where N is the number of dimensions for a tensor.
                When one element is an integer, it means that dimension is static.
                Otherwise, when one element is a tuple/list, it means the dimension is dynamic.
                The 3 elements in one tuple/list is ordered by (min, opt, max), and this function asserts
                0 <= min <= opt <= max.

                Example, for a 3 rank tensor, with 1st dimension being static and has value 16, and second dimension being dynamic with
                min/opt/max values being 1/8/32, and 3rd dimension being static and has value 8.
                The shape parameter could be:
                    [16, (1, 8, 32), 8]
                It has same semantics of
                    [(16, 16, 16), (1, 8, 32), (8, 8, 8)]
        '''
        self.min = []
        self.opt = []
        self.max = []
        self.dimension_names = names
        assert len(names) == len(
            shape
        ), "Expecting shape list and name list must have same length, got {shape=}, {name=}"
        for dim in shape:
            if isinstance(dim, (list, tuple)):
                assert len(dim) == 3 and 0 <= dim[0] <= dim[1] <= dim[2], \
                "Each dimension must specify a 3-elements tuple or list in the order of (min,opt,max), got {dim=}"
                self.min.append(dim[0])
                self.opt.append(dim[1])
                self.max.append(dim[2])
            elif isinstance(dim, int):
                self.min.append(dim)
                self.opt.append(dim)
                self.max.append(dim)
            else:
                raise AttributeError(
                    f'Dimension should be [min, opt, max] (dynamic shape) or int (specific value). Got {type(dim)}'
                )

    def __eq__(self, __value: object) -> bool:
        return isinstance(__value, DimRange) and \
            self.dimension_names == __value.dimension_names and \
            self.min == __value.min and self.opt == __value.opt and self.max == __value.max

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return f"{self.dimension_names=} {self.min=}, {self.opt=}, {self.max=})"

    def __hash__(self) -> int:
        return hash(str(self))


class Tensor(object):
    '''
    The class to represent dense tensors.

    A dense tensor is named, has a shape and contains typed elements. Each
    dimension of a tensor can either be static or dynamic. Static dimensions
    are known at engine compilation by TensorRT. Dynamic dimensions can take
    values determined at runtime. The tensor can be located on the host (CPU)
    or the device (GPU).
    '''

    def __init__(self,
                 name=None,
                 dtype=None,
                 shape=None,
                 dim_range=None,
                 is_network_input=True,
                 location=trt.TensorLocation.DEVICE,
                 network=None,
                 trt_tensor=None):
        '''
        Parameters:
            name : str
                The name of the tensor.

            dtype : tensorrt.DataType
                The type of the elements of the tensor. See the TensorRT
                documentation for list of supported data types.

            shape : tensorrt.Dims
                The dimensions of the tensor. In TensorRT-LLM, tensors can have
                static or dynamic dimensions (it is possible to mix static and
                dynamic dimensions).  A static dimension is known when the
                TensorRT engine is built. A dynamic dimension can be set when
                the engine is executed. Use -1 for dynamic dimensions.

            dim_range : OrderedDict
                An ordered dictionary (the positions of the elements matter)
                that associates a name and a range of values to the dimensions.
                For a static dimension, the range must be limited to a single
                value. For a dynamic dimension, the range is defined by three
                values [min, opt, max] where min and max are, respectively, the
                smallest and largest possible values of that dimension.  The
                opt value is used by TensorRT to optimize the engine for the
                most common case.

                Assume there is N optimization profiles, each item dim_range dict is ordered by:
                 (dynamic dimension name : [profile 0 (min, opt, max), profile 1 (min, opt, max), ... profile N(min, opt, max)])
                or it's following when the dimension is static (can think as min==opt==max):
                 (static dimension name : [profile 0 value, profile 1 value, ... profile N value])
                For static dimension the profile 0-N value must be same, (TODO: can it be simplified to be only 1 value?)
                And number of keys is equal to number of optimization profiles.

            is_network_input : bool
                A boolean indicating if that tensor is an input of the network.
                Inputs must be provided by the user to run the engine.

            location : tensorrt.TensorLocation
                A flag to indicate where the tensor will be located. It can be
                on the host (CPU) or the device (GPU).

            network: Network
                A parent Network instance, that helps to fine the users of this tensor.

            trt_tensor: trt.ITensor
                Construct with the ITensor instance directly, and no shape profiles are required.
        '''

        # Layout of self.profiles
        # Opt profile 0: dim 0 (min, opt, max), dim 1 (min, opt, max) ... dim M
        # Opt profile 1: dim 0 (min, opt, max), dim 1 (min, opt, max) ... dim M
        # ...
        # Opt profile N: dim 0 ...                                        dim M

        # So from the dim_range arg to self.profiles conversion, there is a layout transpose
        # dim_range arg is: {M dimension x N profiles}, while self.profiles layout is {N profiles x M dimensions}
        if isinstance(dtype, str):
            dtype = str_dtype_to_trt(dtype)

        self.profiles = []

        self.is_tensor_wrapper = False  # specially for the graph rewriter

        # work as a wrapper for a trt.ITensor, this is used specially in the graph rewriter
        if trt_tensor is not None:
            self.is_tensor_wrapper = True
            assert network is not None
            self.trt_tensor = trt_tensor
            self._network = weakref.ref(network)
            assert not is_network_input, "is_network_input should be False when trt_tensor is not None"
            return

        # be cautious here, the weakref is critical to avoid circular referencing before Network and Tensor
        # using strong reference will likely cause significant peak memory increase, since Network objects
        # holds the weights data.
        self._network = weakref.ref(default_net())
        self.is_network_input = is_network_input
        if is_network_input:
            if dim_range is not None:
                assert isinstance(dim_range, OrderedDict)
                assert len(
                    dim_range
                ) >= 1, f"Each input tensor shall have at least one dimension, tensor '{name}' found {dim_range=}"

                found_profiles = [
                    len(ranges) for _, ranges in dim_range.items()
                ]
                assert all(
                    [x == found_profiles[0] for x in found_profiles]
                ), f"Expecting all the dimensions in the dim_range has same number of profiles, tensor '{name}' got {dim_range=}"

                num_opt_profile = len(list(dim_range.items())[0][1])
                assert num_opt_profile >= 1
                for i in range(num_opt_profile):
                    range_shape = []
                    dimension_names = []
                    for dim, ranges in dim_range.items():
                        assert isinstance(ranges, (list, tuple))
                        range_shape.append(ranges[i])
                        dimension_names.append(dim)
                    self.profiles.append(DimRange(range_shape,
                                                  dimension_names))

            default_net()._add_input(self, name, dtype, shape, dim_range)
            self.name = name
            self.dtype = dtype
            self.shape = shape
            self.location = location

    @property
    def network(self):
        return self._network()

    @property
    def name(self):
        '''
        The name of the tensor.
        '''
        return self.trt_tensor.name

    @name.setter
    def name(self, name):
        '''
        Set the name of the tensor.
        '''
        if name is not None:
            self.trt_tensor.name = name

    @property
    def dtype(self):
        '''
        The type of the elements in the tensor.
        '''
        return self.trt_tensor.dtype

    @dtype.setter
    def dtype(self, dtype):
        '''
        Set the type of the elements in the tensor.
        '''
        if dtype is not None:
            self.trt_tensor.dtype = dtype

    @property
    def shape(self):
        '''
        The shape of the tensor.
        '''
        return self.size()

    @shape.setter
    def shape(self, shape):
        '''
        Set the shape of the tensor. See __init__.
        '''
        if shape is not None:
            self.trt_tensor.shape = shape

    @property
    def location(self):
        '''
        The physical location of the tensor (on the host or the device).
        '''
        return self.trt_tensor.location

    @location.setter
    def location(self, location):
        '''
        Set the physical location of the tensor (on the host or the device). See __init__.
        '''
        if location is not None:
            self.trt_tensor.location = location

    def mark_output(self,
                    name: Optional[str] = None,
                    dtype: Optional[Union[str, trt.DataType]] = None):
        '''
        Mark a tensor as a network output.

        When a tensor is marked as an output, its content can be obtained after
        the execution of the TensorRT engine. The user is responsible for
        allocating buffers to store the output tensors when preparing the
        execution of the TensorRT engine.
        '''
        if name is None:
            name = self.name

        if isinstance(dtype, str):
            dtype = str_dtype_to_trt(dtype)

        assert dtype is None or isinstance(dtype, trt.DataType)
        default_net()._mark_output(self, name, dtype)

    def __add__(self, b):
        '''
        See functional.add.
        '''
        return add(self, b)

    def __radd__(self, b):
        '''
        See functional.add.
        '''
        return add(b, self)

    def __sub__(self, b):
        '''
        See functional.sub.
        '''
        return sub(self, b)

    def __rsub__(self, b):
        '''
        See functional.sub.
        '''
        return sub(b, self)

    def __mul__(self, b):
        '''
        See functional.mul.
        '''
        return mul(self, b)

    def __rmul__(self, b):
        '''
        See functional.mul.
        '''
        return mul(b, self)

    def __truediv__(self, b):
        '''
        See functional.div.
        '''
        return div(self, b)

    def __floordiv__(self, b):
        '''
        See functional.floordiv.
        '''
        return floordiv(self, b)

    def __mod__(self, b):
        '''
        See functional.floordiv.
        '''
        return modulo(self, b)

    def __lt__(self, b):
        '''
        See functional.lt.
        '''
        return lt(self, b)

    def __gt__(self, b):
        '''
        See functional.gt.
        '''
        return gt(self, b)

    def __eq__(self, b):
        '''
        See functional.eq.
        '''
        if self.is_tensor_wrapper:
            # for graph rewriter
            return hash(self) == hash(b)
        else:
            # for creating the network
            return eq(self, b)

    def __ge__(self, b):
        '''
        Maps to functional.gt or functional.eq.
        '''
        return op_or(self.__gt__(b), self.__eq__(b))

    def __le__(self, b):
        '''
        Maps to functional.lt or functional.eq.
        '''
        return op_or(self.__lt__(b), self.__eq__(b))

    def view(self, shape, zero_is_placeholder=True):
        '''
        See functional.view.
        '''
        return view(self, shape, zero_is_placeholder)

    def flatten(self, start_dim=0, end_dim=-1):
        '''
        See functional.flatten.
        '''
        return flatten(self, start_dim, end_dim)

    def permute(self, dims):
        '''
        See functional.permute.
        '''
        return permute(self, dims)

    def transpose(self, dim0, dim1):
        '''
        See functional.transpose.
        '''
        return transpose(self, dim0, dim1)

    def mean(self, dim, keepdim=False):
        '''
        See functional.mean.
        '''
        return mean(self, dim, keepdim)

    def max(self, dim, keepdim=False):
        '''
        See functional.max.
        '''
        return max(self, dim, keepdim)

    def abs(self):
        '''
        See functional.abs.
        '''
        return abs(self)

    def sqrt(self):
        '''
        See functional.sqrt.
        '''
        return sqrt(self)

    def squeeze(self, dim, zero_is_placeholder):
        '''
        See functional.squeeze.
        '''
        return squeeze(self, dim, zero_is_placeholder)

    def unsqueeze(self, dim):
        '''
        See functional.squeeze.
        '''
        return unsqueeze(self, dim)

    def log(self):
        '''
        See functional.log.
        '''
        return log(self)

    def cast(self, dtype):
        '''
        See functional.cast.
        '''
        return cast(self, dtype)

    def size(self, dim=None):
        '''
        Returns the shape of the tensor if the dim parameter is None.
        Otherwise, returns a size of the dimension indicated by dim. The
        behavior is undefined if dim is negative or exceeds the rank of the
        tensor.
        '''
        if dim is None:
            return self.trt_tensor.shape

        return self.trt_tensor.shape[dim]

    def rank(self):
        '''
        Returns the rank (i.e. the number of dimensions) of the tensor.
        '''
        return len(self.trt_tensor.shape)

    def ndim(self):
        '''
        Returns the rank (i.e. the number of dimensions) of the tensor.
        '''
        return self.rank()

    def split(self, split_size_or_sections, dim=0):
        '''
        See functional.split.
        '''
        return split(self, split_size_or_sections, dim)

    def select(self, dim, index):
        '''
        See functional.select.
        '''
        return select(self, dim, index)

    def unbind(self, dim=0):
        '''
        See functional.unbind.
        '''
        return unbind(self, dim)

    def repeat(self, sizes):
        '''
        See functional.repeat
        '''
        return repeat(self, sizes)

    def is_dynamic(self, dim=None):
        '''
        If the argument 'dim' is None, that function returns a boolean that
        indicates if the tensor contains a dynamic dimension (True) or not
        (False). In that case, the first dimension is excluded (as it usually
        corresponds to the batch size).  If the argument is an integer, that
        functions returns a boolean that indicates if the dimension 'dim' is
        dynamic (True) or not (False).
        '''
        if dim is not None:
            return self.trt_tensor.shape[dim] == -1

        for i, s in enumerate(self.trt_tensor.shape):
            if i != 0 and s == -1:
                return True

        return False

    # graph writer related functions

    def get_parent(self):
        ''' Get the layer that produces this tensor.  '''
        return self.network.get_tensor_parent(self)

    def get_users(self):
        ''' Get the layers that use this tensor as an input.  '''
        return self.network.get_tensor_users(self)

    def is_trt_wrapper(self):
        '''
        Check if there is a trt.ITensor member inside, which is required for
        graph rewriter. In order to differentiate usages, it may be necessary
        to have an inheritance hierarchy.
        '''
        if hasattr(self, 'trt_tensor'):
            return True
        else:
            return False

    def __hash__(self):
        if self.is_trt_wrapper():
            return id(self.trt_tensor)
        else:
            return id(None)

    def __repr__(self):
        return f"TensorRT-LLM Tensor: {self.name=} {self.dtype=} {self.shape=}"

    def __xor__(self, b):
        '''
        Maps to functional.gt or functional.eq.
        '''
        print(f"self.shape: {self.shape}, b.shape: {b.shape}")
        a, b = broadcast_helper(self, b)
        print(f"a.shape: {a.shape}, b.shape: {b.shape}")
        return op_xor(a, b)


def _create_tensor(trt_tensor: trt.ITensor, producer: trt.ILayer) -> Tensor:
    '''
    A helper function to create a TensorRT-LLM Tensor object that encapsulates
    the connection between the TensorRT tensor (trt.ITensor) and the layer
    (trt.ILayer) that produces it.

    That function is expected to be used as:

        # Insert a new layer in the network using the TensorRT API:
        layer = default_trtnet().add_<some_layer>(...)
        # Extract the first output of that layer and connect it to the layer.
        return _create_tensor(layer.get_output(0), layer)

    That function also sets the precision of the layer/producer to the default
    precision of the network.

    Parameters:
        trt_tensor : trt.ITensor
            The TensorRT tensor to connect to its producer (the layer).

        producer : trt.ILayer
            The producer.

    Returns:
        The TensorRT-LLM tensor (functional.Tensor) that encapsulates the
        TensorRT tensor and the layer that produces it. The former is
        accessible through the attribute 'trt_tensor' and the latter using the
        attribute 'producer'.
    '''
    assert trt_tensor is not None
    assert producer is not None

    # Set the layer name since this is the only
    # centralized location to pass the name from
    # module space to the TRT IR
    default_net()._set_layer_name(producer)

    assert trt_tensor.shape.__len__(
    ) >= 0, f"tensor {trt_tensor.name} has an invalid shape"
    tensor = Tensor(name=trt_tensor.name,
                    dtype=trt_tensor.dtype,
                    shape=trt_tensor.shape,
                    is_network_input=False)
    tensor.trt_tensor = trt_tensor
    tensor.producer = producer

    # tb.print_stack(limit=10) # FOR DEBUGGING: filter producer.name if needed
    if default_net().dtype is not None and not default_net().strongly_typed:
        if producer.type not in [
                trt.LayerType.SHAPE, trt.LayerType.CONSTANT,
                trt.LayerType.GATHER, trt.LayerType.CONCATENATION
        ]:
            producer.precision = default_net().dtype
    assert tensor is not None

    return tensor


def _add_plugin_info(layer, plugin_creator: trt.IPluginCreator,
                     plugin_name: str, pfc: trt.PluginFieldCollection) -> None:
    plugin_info = PluginInfo(plugin_creator, plugin_name, pfc)
    set_plugin_info(default_net().trt_network, layer.name, plugin_info)


class RotaryScalingType(IntEnum):
    none = 0
    linear = 1
    dynamic = 2
    longrope = 3
    llama3 = 4
    yarn = 5
    mrope = 6

    @staticmethod
    def from_string(s):
        try:
            return RotaryScalingType[s]
        except KeyError:
            raise ValueError(f'Unsupported rotary scaling type: {s}')


class PositionEmbeddingType(IntEnum):
    learned_absolute = 0
    rope_gptj = 1
    rope_gpt_neox = 2
    long_rope = 3
    alibi = 4
    alibi_with_scale = 5
    relative = 6
    chatglm = 7
    yarn = 8
    mrope = 9
    deferred = 10  # Apply customized positional embedding by using an external embedder. K will be cached before embedding.

    def is_rope(self) -> bool:
        return self in [
            self.rope_gptj, self.rope_gpt_neox, self.long_rope, self.mrope
        ]

    def is_mrope(self) -> bool:
        return self in [self.mrope]

    def is_alibi(self) -> bool:
        return self in [self.alibi, self.alibi_with_scale]

    def is_deferred(self) -> bool:
        return self in [self.deferred]

    @staticmethod
    def choices() -> List[str]:
        return [embedding.name for embedding in PositionEmbeddingType]

    def __str__(self):
        return self.name

    @staticmethod
    def from_string(s):
        try:
            return PositionEmbeddingType[s]
        except KeyError:
            raise ValueError(f'Unsupported position embedding type: {s}')


class AttentionMaskType(IntEnum):
    padding = 0
    causal = 1
    sliding_window_causal = 2
    bidirectional = 3
    bidirectionalglm = 4  # TODO: merge this mask into bidirectional
    blocksparse = 5
    custom_mask = 6


class LayerNormType(IntEnum):
    LayerNorm = 0
    RmsNorm = 1
    GroupNorm = 2


class LayerNormPositionType(IntEnum):
    pre_layernorm = 0
    post_layernorm = 1


class MLPType(IntEnum):
    MLP = 0
    GatedMLP = 1
    FusedGatedMLP = 2


class SliceInputType(IntEnum):
    data = 0
    start = 1
    size = 2
    stride = 3
    fill_value = 4
    axes = 5


def activation(input: Tensor, act_type: trt.ActivationType) -> Tensor:
    '''
    Add an activation function.

    Parameters:
        input : Tensor
            The input tensor on which the activation function is applied.

        act_type : trt.ActivationType
            The type of the activation (RELU, TANH, SIGMOID, ...).

    The following closures are defined in functional.*:

        relu    for op=trt.ActivationType.RELU
        tanh    for op=trt.ActivationType.TANH
        sigmoid for op=trt.ActivationType.SIGMOID

    Returns:
        The tensor produced by the activation layer.
    '''
    layer = default_trtnet().add_activation(input.trt_tensor, act_type)
    return _create_tensor(layer.get_output(0), layer)


def int_clip(input: Tensor, lower: int, upper: int) -> Tensor:
    assert lower <= upper, f"Lower bound must be less than or equal to upper bound i.e. {lower} <= {upper}"
    res = minimum(input, upper)
    res = maximum(res, lower)
    return res


def clip(input: Tensor, alpha: float, beta: float) -> Tensor:
    '''
    Add a CLIP operation that sets the range to [alpha, beta].

    Parameters:
        input : Tensor
            The input tensor on which the activation function is applied.

        alpha : float
            The lower bound of the CLIP function.

        beta : float
            The upper bound of the CLIP function.

    Returns:
        The tensor produced by the activation layer.
    '''
    layer = default_trtnet().add_activation(input.trt_tensor,
                                            trt.ActivationType.CLIP)
    layer.alpha = alpha
    layer.beta = beta
    return _create_tensor(layer.get_output(0), layer)


relu = partial(activation, act_type=trt.ActivationType.RELU)
tanh = partial(activation, act_type=trt.ActivationType.TANH)
sigmoid = partial(activation, act_type=trt.ActivationType.SIGMOID)


def silu(input: Tensor) -> Tensor:
    '''
    Add a SiLU (`x * sigmoid(x)`) operation.

    Parameters:
        input : Tensor
            The input tensor on which the activation function is applied.

    Returns:
        The tensor produced by the activation layer.
    '''
    return input * sigmoid(input)


def swiglu(input: Tensor) -> Tensor:
    '''
    Add a SwiGLU (`x * SiLU(gate)`) operation.

    That function takes a tensor, splits it into two halves along the last
    dimension, applies SiLU to the second half and multiply the results. The
    behavior is undefined if the last dimension is not even.

    Parameters:
        input : Tensor
            The input tensor on which the activation function is applied.

    Returns:
        The tensor produced by the activation layer.
    '''
    x, gate = chunk(input, 2, dim=-1)
    return silu(gate) * x


def squared_relu(x: Tensor) -> Tensor:
    '''
    Add a Squared ReLU operation.

    This function applies ReLU and squares the output.

    Parameters:
        input : Tensor
            The input tensor on which the activation function is applied.

    Returns:
        The tensor produced by the activation layer.
    '''
    return pow(relu(x), 2.0)


def cast(input: Tensor, dtype: Union[str, trt.DataType]):
    '''
    Add a cast operation.

    For an input tensor of type INT8, this function sets the dynamic range of
    the input to [-127, 127] for automatic dequantization. For a cast into
    INT8, that function sets the dynamic range of the output to [-127, 127] for
    automatic quantization.

    Parameters:
        input : Tensor
            The input tensor on which the cast is applied.

        dtype : str or trt.DataType
            The data type of the output tensor after the cast. When 'dtype' is
            provided as a string, it must be a name amongst the valid names.
            See _str_to_trt_dtype_dict in _utils.py for a list of supported
            types and type names.

    Returns:
        The tensor produced by the inserted layer.
    '''
    if isinstance(dtype, str):
        cvt_dtype = str_dtype_to_trt(dtype)
    elif isinstance(dtype, trt.DataType):
        cvt_dtype = dtype
    else:
        raise TypeError("%s is not supported" % type(dtype))

    if input.dtype == cvt_dtype:
        # If input type and cast dtype are the same, do nothing
        return input

    layer = default_trtnet().add_cast(input.trt_tensor, cvt_dtype)
    if not default_net().strongly_typed:
        layer.set_output_type(0, cvt_dtype)
    output = _create_tensor(layer.get_output(0), layer)
    if not default_net().strongly_typed:
        if input.dtype == str_dtype_to_trt('int8'):
            layer.get_input(0).set_dynamic_range(-127, 127)
        if cvt_dtype == str_dtype_to_trt('int8'):
            layer.get_output(0).set_dynamic_range(-127, 127)

    return output


def flip(input: Tensor, dims: Sequence[int]) -> Tensor:
    '''
    Reverses the order of an n-D tensor along given axis in dims.

    That flip operation maps to a TensorRT ISliceLayer. For the dimensions
    listed in dims it copies the elements from the last one to the first one
    (from (N-1) down to 0 with a step of -1). For the dimensions not in 'dims',
    it copies the elements from the first one to the last one (from 0 to N-1
    with a step of 1).

    Parameters:
        input : Tensor
            The input tensor on which the cast is applied.

        dims : list or tuple
            The axes to flip. Negative indices are supported.

    Returns:
        The tensor produced by the inserted layer.
    '''
    assert not input.is_dynamic()

    ndim = input.ndim()

    for index, value in enumerate(dims):
        assert -ndim <= value < ndim
        if -ndim <= value < 0:
            dims[index] += ndim

    assert len(dims) == len(set(dims))

    start_values = [
        input.size()[i] - 1 if i in dims else 0 for i in range(ndim)
    ]
    stride_values = [-1 if i in dims else 1 for i in range(ndim)]

    layer = default_trtnet().add_slice(input.trt_tensor,
                                       start=start_values,
                                       shape=input.size(),
                                       stride=stride_values)

    return _create_tensor(layer.get_output(0), layer)


def interpolate(input: Tensor,
                size: Union[int, List[int]] = None,
                scale_factor: Union[float, List[float]] = None,
                mode: str = 'nearest',
                align_corners: bool = False,
                recompute_scale_factor: bool = False,
                antialias: bool = False) -> Tensor:
    ##
    ## TODO: Document that function!
    ##

    assert not input.is_dynamic()

    input_ndim = input.ndim()

    assert 2 < input_ndim < 6, "Only 3D, 4D and 5D input Tensors supported"
    assert (size is not None) ^ (
        scale_factor
        is not None), "Only one of out_shape or scales should be defined"

    assert mode in ('nearest', 'linear', 'bilinear', 'bicubic', 'trilinear',
                    'nearest-exact')

    if mode == 'trilinear' and input_ndim != 5:
        raise ValueError("trilinear only supports 5D tensor")

    if mode == "bilinear" and input_ndim != 4:
        raise ValueError("bilinear only supports 4D tensor")

    if mode == "linear" and input_ndim != 3:
        raise ValueError("linear only supports 3D tensor")

    layer = default_trtnet().add_resize(input.trt_tensor)

    input_shape = input.size()

    updated_shape = []
    if scale_factor:
        scale_len = 1 if isinstance(scale_factor,
                                    (float, int)) else len(scale_factor)
        if scale_len == 1 and isinstance(scale_factor, (float, int)):
            updated_scale = [scale_factor for _ in range(input_ndim - 2)]

        else:
            updated_scale = scale_factor
        updated_shape = [
            int(math.floor(updated_scale[i - 2] *
                           input_shape[i])) if i > 1 else input_shape[i]
            for i in range(input_ndim)
        ]

    else:
        size_len = 1 if isinstance(size, int) else len(size)
        assert size_len == input_ndim - 2
        if size_len == 1 and isinstance(size, int):
            updated_size = [size for _ in range(input_ndim - 2)]
        else:
            updated_size = size

        updated_shape = [
            input_shape[i] if i < 2 else updated_size[i - 2]
            for i in range(input_ndim)
        ]
    layer.shape = updated_shape

    if mode in ['nearest', 'nearest-exact'] or mode is None:
        layer.resize_mode = trt.InterpolationMode.NEAREST
        layer.coordinate_transformation = trt.ResizeCoordinateTransformation.ASYMMETRIC
    elif mode in ['linear', 'bilinear', 'trilinear']:
        layer.resize_mode = trt.InterpolationMode.LINEAR
        if align_corners:
            layer.coordinate_transformation = trt.ResizeCoordinateTransformation.ALIGN_CORNERS
        else:
            layer.coordinate_transformation = trt.ResizeCoordinateTransformation.HALF_PIXEL
        # TODO, need to confirm the align_corners effect on bilinear mode.
        if mode == 'bilinear':
            layer.coordinate_transformation = trt.ResizeCoordinateTransformation.HALF_PIXEL

    elif mode in ['bicubic']:
        layer.resize_mode = trt.InterpolationMode.CUBIC

        layer.coordinate_transformation = trt.ResizeCoordinateTransformation.HALF_PIXEL

    else:
        layer.resize_mode = trt.InterpolationMode.NEAREST
        layer.coordinate_transformation = trt.ResizeCoordinateTransformation.ASYMMETRIC

    return _create_tensor(layer.get_output(0), layer)


def matmul(input: Tensor,
           mat2: Tensor,
           transa: bool = False,
           transb: bool = False,
           use_fp32_acc: bool = True) -> Tensor:
    '''
    Add a matrix multiplication.

    That operation maps to a tensorrt.IMatrixMultiplyLayer layer. As explained
    in the TensorRT documentation, it computes the inner product between the
    two inputs after applying an optional transposition on the inputs.

    Parameters:
        input : Tensor
            The first tensor (often called A).

        mat2 : Tensor
            The second tensor (often called B).

        transa : bool
            Is the first input transposed? Set to 'True' if you want the first
            input to be transposed, 'False' otherwise.

        transb : bool
            Is the second input transposed? Set to 'True' if you want the
            second input to be transposed, 'False' otherwise.

        use_fp32_acc: bool
            Set to 'True' if for accuracy reason, this fp16 matmul needs to use
            fp32 accumulation. This can be a per model and per matmul decision.
    Returns:
        The tensor produced by the inserted layer.
    '''
    # This option is only supported for fp16, but not bf16 or any other precisions.
    use_fp32_acc = use_fp32_acc and input.dtype == trt.DataType.HALF and mat2.dtype == trt.DataType.HALF

    if use_fp32_acc:
        input = cast(input, 'float32')
        mat2 = cast(mat2, 'float32')

    input, mat2 = broadcast_helper(input, mat2)
    op0 = trt.MatrixOperation.TRANSPOSE if transa \
        else trt.MatrixOperation.NONE
    op1 = trt.MatrixOperation.TRANSPOSE if transb \
        else trt.MatrixOperation.NONE
    layer = default_trtnet().add_matrix_multiply(input.trt_tensor, op0,
                                                 mat2.trt_tensor, op1)
    output = _create_tensor(layer.get_output(0), layer)
    if use_fp32_acc:
        output = cast(output, "float16")

    return output


def constant(ndarray: np.ndarray,
             as_dtype: trt.DataType | None = None,
             as_shape=None) -> Tensor:
    '''
    Add a constant layer.

    TensorRT graphs encapsulate constant values in the form of constant layers
    (tensorrt.IConstantLayer). That function creates such a layer from a Numpy
    array of values. After compilation of the network by TensorRT, those
    weights are stored in the serialized TensorRT engine.

    Parameters:
        ndarray : numpy.ndarray
            The array of values (weights) encapsulated by this constant layer.

    Returns:
        The tensor produced by the inserted layer.
    '''
    trt_dtype = np_dtype_to_trt(
        ndarray.dtype) if as_dtype is None else as_dtype
    trt_shape = trt.Dims(
        ndarray.shape) if as_shape is None else trt.Dims(as_shape)
    trt_count = 1
    for i in range(len(trt_shape)):
        trt_count *= trt_shape[i]
    weights = trt.Weights(trt_dtype, ndarray.ctypes.data, trt_count)
    # Prevent underlying numpy array from going out of scope
    default_net().register_ndarray(ndarray)
    layer = default_trtnet().add_constant(trt_shape, weights)
    if not default_net().strongly_typed:
        layer.set_output_type(0, trt_dtype)
    tensor = _create_tensor(layer.get_output(0), layer)
    # TODO: remove this WAR after https://nvbugs/4359151 fixed.
    set_np_weight(default_trtnet(), layer.name, ndarray)
    return tensor


# TODO: TensorRT uses sizes of the output dimensions.
# DL framework uses ends usually. Will change it to ends.
def slice(input: Tensor,
          starts: Union[Tensor, Sequence[int]],
          sizes: Union[Tensor, Sequence[int]],
          strides: Union[Tensor, Sequence[int]] = None,
          mode: trt.SampleMode = None,
          fill_value: Union[float, Tensor] = None) -> Tensor:
    '''
    Add an operation to extract a slice from a tensor.

    As described in the TensorRT documentation of the ISliceLayer, the slice
    layer has two variants: Static and dynamic.

    For static slicing, this function takes the starts and sizes values in the
    different dimensions to slice at layer creation time via a sequence of
    integers. For dynamic slicing, it accepts starts and sizes as
    tensorrt.ITensor`s.

    The slice layer selects for each dimension a start location from within the
    input tensor, and copies elements to the output tensor using a stride of 1
    across the input tensor. Start and size tensors must be 1-D int32 shape
    tensors if not specified as a sequence of integers.

    As an example, on input = [[0, 2, 4], [1, 3, 5]], the call to

        slice(input, start=[1, 0], size=[1, 2])

    will produce the tensor [[1, 3]] as output. The slice operator when
    executed by TensorRT will copy one row (because size[0] == 1) starting from
    the 2nd row (because start[0] == 1) and two columns (size[1] == 2) starting
    from the 1st column (because start[1] == 0).

    In pseudo-code the behavior of that operation can be described as follows
    for a 2D tensor (and easily be extended to more dimensions):

        output = Tensor(shape=sizes)
        for ii in range(sizes[0]):
            for jj in range(sizes[1]):
                output[ii][jj] = input[starts[0]+ii][starts[1]+jj]

    Note that it is common in deep-learning frameworks to use ranges
    [start:end] for similar operations. It can be emulated by setting the sizes
    argument such that in each dimension [start:start+size] == [start:end] i.e.
    size = end-start.

    TensorRT supports different slice modes but that function restricts that
    choice to `mode == tensorrt.SampleMode.STRICT_BOUNDS`.

    Parameters:
        input : Tensor
            The input tensor on which the slicing is performed.

        starts : Union[Tensor, Sequence[int]]
            The starting points, in the input tensor, and each dimension.

        sizes : Union[Tensor, Sequence[int]]
            The number of elements in each dimension of the sliced tensor (output).

        strides : Union[Tensor, Sequence[int]]
            The step be taken from start, in input tensor.

        mode : trt.SampleMode
            The mode that controls how the slice operation handles out of bounds coordinates.

    Returns:
        The tensor produced by the slice layer.
    '''
    input_ndim = input.ndim()

    trt_starts = starts
    if isinstance(starts, Tensor):
        trt_starts = [0 for _ in range(input_ndim)]  # unused dummy value

    trt_sizes = sizes
    if isinstance(sizes, Tensor):
        trt_sizes = [1 for _ in range(input_ndim)]  # unused dummy value

    trt_strides = strides
    if isinstance(strides, Tensor) or strides is None:
        trt_strides = [1 for _ in range(input_ndim)]

    if fill_value is not None and isinstance(fill_value, float):
        fill_value = constant(fp32_array(fill_value))

    layer = default_trtnet().add_slice(input.trt_tensor,
                                       start=trt_starts,
                                       shape=trt_sizes,
                                       stride=trt_strides)
    if mode is not None:
        layer.mode = mode

    if isinstance(starts, Tensor):
        layer.set_input(1, starts.trt_tensor)

    if isinstance(sizes, Tensor):
        layer.set_input(2, sizes.trt_tensor)

    if isinstance(strides, Tensor):
        layer.set_input(3, strides.trt_tensor)

    if mode is trt.SampleMode.FILL and isinstance(fill_value, Tensor):
        layer.set_input(4, fill_value.trt_tensor)

    return _create_tensor(layer.get_output(0), layer)


def pad(input: Tensor,
        pad: Union[Sequence[int], Tensor],
        mode: str = 'constant',
        value: Optional[float] = None) -> Tensor:
    '''
    Add a pad layer.

    The padding layer adds zero-padding at the start and end of the input tensor. And the
    padding size by which to pad some dimensions of input are described starting from the
    last dimension and moving forward.

    `[len(pad) / 2]` dimensions of input will be padded. For example, to pad only the last
    dimension of the input tensor, then pad has the form [padding_left, padding_right]; to
    pad the last 2 dimensions of the input tensor, then use [padding_left, padding_right,
    padding_top, padding_bottom]; to pad the last 3 dimensions, use [padding_left,
    padding_right, padding_top, padding_bottom, padding_front, padding_back].

    Parameters:
        input : Tensor
            The input tensor on which the padding_2d is performed.
        pad : sequence of int
            An m-elements tuple for padding, where its length m meets the requirement that
            m <= 2*input dimensions, and m is even.
        mode : str
            Only \'constant\' is supported.
        value : float
            Fill value for 'constant' padding. Default: 0.

    Returns:
        The tensor produced by the inserted layer.
    '''
    assert mode == "constant", "Only `'constant'` is supported now."

    if isinstance(pad, list) or isinstance(pad, tuple):
        assert (
            len(pad) % 2 == 0 and len(pad) <= 2 * input.ndim()
        ), "The length of `pad` should be even and less than 2*input.ndim"
        pad = constant(np.array(pad).astype(np.int32)).view([-1, 2])
    elif isinstance(pad, Tensor):
        pad = pad.flatten()
        assert (
            pad.size(0) % 2 == 0 and pad.size(0) <= 2 * input.ndim()
        ), "The length of `pad` should be even and less than 2*input.ndim"
        pad = pad.cast("int32").view([-1, 2])
    else:
        raise NotImplementedError(f"pad type {type(pad)} not supported")
    if value is None:
        value = 0

    pad = concat([constant(np.zeros((1, 2), dtype=np.int32)),
                  pad])  # pre-padding the indices
    padding_index = [0] * input.ndim()
    padding_index[-(pad.size(0) - 1):] = list(range(pad.size(0) - 1, 0,
                                                    -1))  # reverse the indices
    pad = index_select(pad,
                       dim=0,
                       index=constant(np.array(padding_index, dtype=np.int32)))
    pre_padding, post_padding = chunk(pad, chunks=2, dim=1)
    start = (pre_padding.flatten() * (-1)).cast('int32')
    extend_size = (pre_padding + post_padding).flatten()
    size = (extend_size + shape(input)).cast('int32')
    layer = default_trtnet().add_slice(input.trt_tensor,
                                       start=[0] * input.ndim(),
                                       shape=[0] * input.ndim(),
                                       stride=[1] * input.ndim())
    layer.mode = trt.SampleMode.FILL
    layer.set_input(SliceInputType.start, start.trt_tensor)
    layer.set_input(SliceInputType.size, size.trt_tensor)
    layer.set_input(SliceInputType.fill_value,
                    constant_to_tensor_(value, dtype=input.dtype).trt_tensor)
    output = _create_tensor(layer.get_output(0), layer)
    return output


def rand(shape: Tensor,
         low: float = 0,
         high: float = 1,
         dtype: Union[str, trt.DataType] = 'float32') -> Tensor:
    '''
    This operation adds a fill layer that generates a random (uniform) tensor with the specified shape and data type.

    Parameters:
        shape: Tensor
            The shape of the tensor needed to be generated.
        low: float
            The minimum value (inclusive) of the range used for random.
        high: float
            The maximum value (inclusive) of the range used for random.
        dtype: Union[str, trt.DataType]
            The desired data type for the output tensor.
    Returns:
        The generated random tensor produced by the fill layer.
    '''
    # NOTE: DISABLED FOR NOW UNTIL THE FILL LAYER (RANDOM_UNIFORM) in TRT IS FIXED
    assert False, "The rand() op is temporarily disabled."
    low = constant(fp32_array(low))
    high = constant(fp32_array(high))
    trt_dtype = dtype if isinstance(dtype,
                                    trt.DataType) else str_dtype_to_trt(dtype)

    layer = default_trtnet().add_fill([0], trt.FillOperation.RANDOM_UNIFORM,
                                      trt_dtype)

    layer.set_input(0, shape.trt_tensor)
    layer.set_input(1, low.trt_tensor)
    layer.set_input(2, high.trt_tensor)
    return _create_tensor(layer.get_output(0), layer)


def categorical_sample(probs: Tensor, rand_data: Tensor = None) -> Tensor:
    '''
    This is a sampling operation and an equivalent of torch.distributions.Categorical.sample()
    i.e. given a probability distribution tensor, it samples an index of that tensor.
    See: https://pytorch.org/docs/stable/distributions.html#torch.distributions.categorical.Categorical.sample
    NOTE: This assumes that the given probabilities are **not** normalized.

    Parameters:
        probs: Tensor
            A 1-D floating point tensor representing the probability distributions.
        rand_data: Tensor (optional)
            A random tensor of same shape as `probs` tensor.
            If not provided, this function will add a rand() op to generate it and use for sampling.
    Returns:
        A tensor containing a single index of the `probs` tensor representing the sample.
    '''
    probs = probs / sum(probs, dim=-1, keepdim=True)
    rand_shape = []
    assert probs.ndim() > 0
    for i in range(probs.ndim() - 1):
        rand_shape.append(shape(probs, i))
    rand_shape = concat(rand_shape)
    if rand_data is None:
        rand_data = rand(rand_shape, low=0, high=1, dtype=probs.dtype)
    assert rand_shape == shape(rand_data)
    rand_data = expand(unsqueeze(rand_data, -1), shape(probs))
    cum_probs = cumsum(probs, dim=-1)
    cmp = cast(cum_probs >= rand_data, probs.dtype)
    samples = argmax(cmp, dim=-1)
    return samples


class Conditional:
    '''
    Add an operation to conditionally execute two code paths/subgraphs.

    Usage:
        1. conditional = Conditional(condition)
        2. input_1_ = conditional.add_input(input_1)
           ...
           input_n_ = conditional.add_input(input_n)
        3. Construct the graph to get true_output_value and false_output_value using input_1_, ..., input_n_
        4. output = conditional.add_output(true_output_value, false_output_value)
    '''

    def __init__(self, condition: Tensor):
        self.layer = default_trtnet().add_if_conditional()
        if condition.ndim() > 0:
            condition = view(condition, [])
        self.layer.set_condition(condition.trt_tensor)

    def add_input(self, input: Tensor) -> Tensor:
        in_node = self.layer.add_input(input.trt_tensor)
        return _create_tensor(in_node.get_output(0), in_node)

    def add_output(self, true_value: Tensor, false_value: Tensor) -> Tensor:
        out_node = self.layer.add_output(true_value.trt_tensor,
                                         false_value.trt_tensor)
        return _create_tensor(out_node.get_output(0), out_node)


# TODO: support step.
def arange(start: Union[Tensor, int], end: Union[Tensor, int],
           dtype: str) -> Tensor:
    '''
    Add an operation to fill a 1D tensor.

    The tensor is filled with the values between start and end with a step of 1
    between the different elements. In pseudo-code, it corresponds to a tensor
    populated with the values:

        output = Tensor([dtype(ii) for ii in range(start, end, 1)])

    For example, a call to arange(3, 6, 'int32') will add an operation to the
    TensorRT graph that will produce [3, 4, 5] when executed. The call to
    arange(2, 5, 'float32') will add a layer to generate [2.0, 3.0, 4.0].

    This operation is implemented using a tensorrt.IFillLayer in
    trt.FillOperation.LINSPACE mode.

    Parameters:
        start : Union[Tensor, int]
            The starting point of the range.

        end : Union[Tensor, int]
            The end point of the range.

        dtype : str
            The type of the elements. See _str_to_trt_dtype_dict in _utils.py
            for a list of supported types and type names.

    Returns:
        The tensor produced by the fill layer. It is a 1D tensor containing
        `end-start` elements of type `dtype`.
    '''
    res_dtype = str_dtype_to_trt(dtype)
    if isinstance(start, int):
        assert isinstance(end, int)
        array_func = int32_array if res_dtype == trt.int32 else int64_array
        start = constant(array_func(start))
        end = constant(array_func(end))
    elif isinstance(start, Tensor):
        assert isinstance(end, Tensor)
        assert start.dtype == trt.int32 or start.dtype == trt.int64
        assert end.dtype == trt.int32 or end.dtype == trt.int64
        if start.dtype != end.dtype:
            if start.dtype == trt.int32:  # end == trt.int64
                if res_dtype == trt.int32:
                    end = cast(end, "int32")
                else:
                    start = cast(start, "int64")
            else:  # start == trt.int64 and end == trt.int32
                if res_dtype == trt.int32:
                    start = cast(start, "int32")
                else:
                    end = cast(end, "int64")
    else:
        raise TypeError("%s is not supported" % type(start))

    assert start.dtype == end.dtype, f"start type ({start.dtype}) != end type ({end.dtype})"
    step = constant_to_tensor_(1, dtype=start.dtype, to_array=True)

    num = end - start
    num = num.view([1]).cast(trt.int64)

    layer = default_trtnet().add_fill([0], trt.FillOperation.LINSPACE,
                                      start.dtype)
    layer.set_input(0, num.trt_tensor)  # rank = 1
    layer.set_input(1, start.trt_tensor)  # rank = 0
    layer.set_input(2, step.trt_tensor)  # rank = 1
    tensor = _create_tensor(layer.get_output(0), layer)
    if tensor.dtype != res_dtype:
        tensor = tensor.cast(dtype)
    return tensor


def expand(input: Tensor, expand_shape: Tensor) -> Tensor:
    '''
    Add an operation to expand a tensor.

    The operation expands the input tensor in the singleton dimensions to the
    size indicated by the corresponding dimension in the `expand_shape` tensor.
    In other words, given an input tensor with dimensions of size 1, those
    dimensions will be expanded to the size in `expand_shape`.

    For example, a tensor of shape [4, 3, 1, 3] will be expanded to a tensor of
    shape [4, 3, 2, 3] by the layer created using expand(input, [4, 3, 2, 3]).

    The expansion may either replicate the values or be mapped to a view with a
    stride of 0 in the expanded dimensions. For example, for a tensor [[3, 2]] of
    shape [1, 2],

        expand([[3, 2]], [2, 2])

    can be used to expand the input to [[3, 2], [3, 2]].

    This operation is implemented using a tensorrt.ISliceLayer. The current
    implementation does not verify that non singleton dimensions are not
    shrunk. In other words, for an input of shape [4, 1, 2],

        expand(input, [3, 2, 2])

    will produce a tensor of shape [3, 2, 2]. That behavior is subject to
    change in the future.

    Parameters:
        input : Tensor
            The input tensor.

        expand_shape : Tensor
            The new shape of the expanded tensor.

    Returns:
        The tensor produced by the expand layer.
    '''
    ndim = input.rank()
    layer = default_trtnet().add_slice(
        input.trt_tensor,
        start=[0 for _ in range(ndim)],
        shape=[1 for _ in range(ndim)],  # unused dummy value
        stride=[1 for _ in range(ndim)]  # unused dummy value
    )

    # The stride is either:
    #   0 for dimensions of size 1 (i.e. shape(input, i) - 1 == 1 - 1 == 0) or,
    #   1 for dimensions of size > 1 since minimum(value >= 1, 1) == 1.
    stride_tensor = concat(
        [minimum((shape(input, i) - 1), 1) for i in range(ndim)])

    layer.set_input(2, expand_shape.trt_tensor)
    layer.set_input(3, stride_tensor.trt_tensor)
    return _create_tensor(layer.get_output(0), layer)


def einsum(einsum_eq: str, inputs: Sequence[Tensor]) -> Tensor:
    '''
    Add an Einsum operation.

    That operation maps to tensorrt.IEinsumLayer. As explained in the TensorRT
    documentation, this layer implements a summation over the elements of the
    inputs along dimensions specified by the equation parameter, based on the
    Einstein summation convention. The layer can have one or more inputs of
    rank >= 0.  All the inputs must be of same data type. This layer supports
    all TensorRT data types except bool. There is one output tensor of the same
    type as the input tensors. The shape of output tensor is determined by the
    equation.

    The equation specifies ASCII lower-case letters for each dimension in the
    inputs in the same order as the dimensions, separated by comma for each
    input. The dimensions labeled with the same subscript must match or be
    able to be broadcasted. Repeated subscript labels in one input take the diagonal.
    Repeating a label across multiple inputs means that those axes will be
    multiplied. Omitting a label from the output means values along those axes
    will be summed. In implicit mode, the indices which appear once in the
    expression will be part of the output in increasing alphabetical order. In
    explicit mode, the output can be controlled by specifying output subscript
    labels by adding an arrow (‘->’) followed by subscripts for the output. For
    example, “ij,jk->ik” is equivalent to “ij,jk”. Ellipsis (‘…’) can be used
    in place of subscripts to broadcast the dimensions. See the TensorRT
    Developer Guide for more details on equation syntax.

    Many common operations can be expressed using the Einsum equation. For
    example:
        Matrix Transpose: ij->ji
        Sum: ij-> Matrix-Matrix
        Multiplication: ik,kj->ij
        Dot Product: i,i->
        Matrix-Vector Multiplication: ik,k->i
        Batch Matrix Multiplication: ijk,ikl->ijl
        Batch Diagonal: …ii->…i

    Note that TensorRT does not support ellipsis or diagonal operations so,
    neither, does TensorRT-LLM.

    Parameters:
        einsum_eq : str
            The Einsum equation.

        inputs: Sequence[Tensor]
            The sequence of inputs consumed by the Einsum operation.

    Returns:
        The tensor produced by the Einsum operation.
    '''
    layer = default_trtnet().add_einsum([i.trt_tensor for i in inputs],
                                        einsum_eq)
    return _create_tensor(layer.get_output(0), layer)


def permute(input: Tensor, dims: Sequence[int]) -> Tensor:
    '''
    Add an operation to permute the dimensions of a tensor.

    The dimensions of the input tensor are permuted according to the sequence
    of dimensions in 'dims'. That operation maps to tensorrt.IShuffleLayer where
    the second transposition is described by the indices in 'dims'.

    Given a tensor of rank N, the result of the permutation is a tensor of rank
    N in which the i-th input dimension maps to the dims[i]-th dimension.

    For example, permute(input, [1, 0]) will transpose a 2D tensor by permuting
    the rows and columns.

    Parameters:
        input : Tensor
            The input tensor to permute.

        dims : Sequence[int]
            The description of the permutation.

    Returns:
        The tensor produced by the permutation layer.
    '''
    dims = dim_resolve_negative(tuple(dims), input.ndim())
    layer = default_trtnet().add_shuffle(input.trt_tensor)
    layer.second_transpose = dims
    return _create_tensor(layer.get_output(0), layer)


def transpose(input: Tensor, dim0: int, dim1: int) -> Tensor:
    '''
    Add an operation to transpose two dimensions of a tensor.

    That operation produces a tensor in which the dimensions 'dim0' and 'dim1'
    are permuted. The other dimensions, if the rank of the tensor is greater
    than 2, remain untouched.

    That function is a helper built on the 'functional.permute' function.

    Parameters:
        input : Tensor
            The input tensor to transpose.

        dim0 : int
            The first dimension to transpose.

        dim1 : int
            The second dimension to transpose.

    Returns:
        The tensor produced by the permutation layer.
    '''
    permutation = list(range(input.ndim()))
    permutation[dim0] = dim1
    permutation[dim1] = dim0

    return permute(input, permutation)


def view(input: Tensor,
         shape: Union[Tensor, Sequence[int]],
         zero_is_placeholder: bool = True) -> Tensor:
    '''
    Add an operation to create a view of a tensor.

    That operation adds a tensorrt.IShuffleLayer to the network. If the 'shape'
    parameter is a Tensor, that view is dynamic. Otherwise, it is a static
    view.

    Note that TensorRT limits the number of inferred dimensions to 1. It means
    that the shape sequence or tensor cannot contain more than one -1. This
    function enforces that constraint and will assert if it is not respected.

    Parameters:
        input : Tensor
            The input tensor to transpose.

        shape : Union[Tensor, Sequence[int]]
            The shape of the new tensor.

        zero_is_placeholder : bool
            When that parameter is True, the 0s in 'shape' are replaced by the
            sizes of the corresponding dimensions from the 'input'. Otherwise,
            the dimensions corresponding to 0s are shrunk.

    Returns:
        The tensor produced by the view/shuffle layer.
    '''

    # TensorRT demands that at most one dimension is permitted to be specified as -1
    def assert_no_more_than_one_inferred_dim(list):
        inferred_dim_list = [i for i in list if i == -1]
        assert len(inferred_dim_list) <= 1

    layer = default_trtnet().add_shuffle(input.trt_tensor)
    layer.zero_is_placeholder = zero_is_placeholder
    if isinstance(shape, Tensor):
        assert_no_more_than_one_inferred_dim(shape.shape)
        layer.set_input(1, shape.trt_tensor)
    elif isinstance(shape, (list, tuple)):
        assert_no_more_than_one_inferred_dim(shape)
        layer.reshape_dims = tuple(shape)
    else:
        raise TypeError("%s is not supported" % type(shape))
    return _create_tensor(layer.get_output(0), layer)


def flatten(input: Tensor, start_dim: int = 0, end_dim: int = -1):
    '''
    Flattens input by reshaping it into a one-dimensional tensor.

    If start_dim or end_dim are passed, only dimensions starting with start_dim and
    ending with end_dim are flattened. The order of elements in input is unchanged.

    Parameters:
        input : Tensor
            The input tensor to flatten.

        start_dim : int
            The first dim to flatten.

        end_dim : int
            The last dim to flatten.

    Returns:
        The tensor produced by the flatten layer.

    '''
    shape = input.shape
    ndim = input.ndim()
    if start_dim < 0: start_dim += ndim
    if end_dim < 0: end_dim += ndim
    new_shape = list()
    for i in range(start_dim):
        new_shape.append(shape[i])
    if end_dim - start_dim >= 0:
        flat_dim = 1
        for i in range(start_dim, end_dim + 1):
            flat_dim *= shape[i]
        new_shape.append(flat_dim)
    for i in range(end_dim + 1, ndim):
        new_shape.append(shape[i])
    return view(input, new_shape)


def expand_dims(input: Tensor,
                dim: Union[int, Sequence[int]],
                shape_cast_dtype=None) -> Tensor:
    '''
    Add an operation to expand the tensor shape with singleton dimensions.

    That function adds a tensorrt.IShuffleLayer to the network. Given an 'input'
    of rank N and a sequence of M dimensions, the output tensor produced by
    this operation (when executed by TensorRT) will have a rank of N+M. Singleton
    dimensions will be inserted at the different positions in 'dim'.

    The pseudo-code for that operation is:

        new_shape, ii = [], 0
        for jj in range(input.rank() + len(dim)):
            new_shape.append(1 if jj in dims else input.shape[ii++])

    For example, for a tensor of shape [3, 4, 1, 5]

        expand_dims(input, [0, 2])

    will produce a tensor of shape [1, 3, 1, 4, 1, 5].

    Parameters:
        input : Tensor
            The input tensor to expand.

        dim : Union[int, Sequence[int]]
            The positions in the output tensor where to insert singleton
            dimensions.

    Returns:
        The tensor produced by the shuffle layer.
    '''
    if isinstance(dim, int):
        dim = (dim, )

    out_ndim = len(dim) + input.ndim()

    input_shape = shape(input, cast_to_dtype=shape_cast_dtype)
    out_shapes = []
    j = 0
    for i in range(out_ndim):
        if i in dim:
            out_shapes.append(1)
        else:
            out_shapes.append(gather(input_shape, 0, j))
            j = j + 1

    out_shape = concat(out_shapes)

    return view(input, out_shape, zero_is_placeholder=False)


# NOTE: Jointly added with Apple
def squeeze(input: Tensor,
            dim: Optional[Union[int, Sequence[int]]] = None,
            zero_is_placeholder: bool = False):
    '''
    Add an operation to remove singleton dimensions of a tensor.

    This functions creates an operation that removes singleton dimension
    (dimension of size 1) at positions 'dim' in the input tensor. It works with
    negative values for the 'dim'.

    For example, for a tensor 'input' of shape [1, 4, 1, 4]:

        squeeze(input,  0) will produce an output of shape [4, 1, 4],
        squeeze(input,  2) will produce an output of shape [1, 4, 4],
        squeeze(input, [0, 2]) will produce an output of shape [4, 4],
        squeeze(input, [-2]) will produce an output of shape [1, 4, 4],

    Parameters:
        input : Tensor
            The input tensor for which the singleton dimensions will be removed.

        dim : Union[int, Sequence[int]]
            The index of the singleton dimensions in the input tensor.

    Returns:
        The tensor produced by the layer.
    '''
    if dim is None:
        dim = list(range(input.ndim()))
    if isinstance(dim, int):
        dim = (dim, )
    dim = dim_resolve_negative(dim, input.ndim())

    new_shape = []
    for i, s in enumerate(input.shape):
        if s == 1 and i in dim:
            continue
        new_shape.append(shape(input, i))

    new_shape = concat(new_shape) if len(new_shape) > 0 else []
    input = input.view(new_shape, zero_is_placeholder=zero_is_placeholder)
    return input


def unsqueeze(input: Tensor, axis: int):
    '''
    Add an operation to insert a singleton dimension to a tensor.

    That functions creates an operation that insert a singleton dimension
    (dimension of size 1) at position 'axis' in the output tensor. It works with
    negative values for the 'axis'.

    For example, for a tensor 'input' of shape [4, 4]:

        unsqueeze(input,  0) will produce an output of shape [1, 4, 4],
        unsqueeze(input,  1) will produce an output of shape [4, 1, 4],
        unsqueeze(input, -1) will produce an output of shape [4, 4, 1],
        unsqueeze(input, -2) will produce an output of shape [4, 1, 4],

    Parameters:
        input : Tensor
            The input tensor to expand with a singleton dimension.

        axis : int
            The index of the singleton dimension in the output tensor.

    Returns:
        The tensor produced by the layer.
    '''
    if axis < 0:
        axis = axis + input.ndim() + 1

    return expand_dims(input, axis)


def stack(inputs: Sequence[Tensor], dim: int = 0) -> Tensor:
    '''
    Add an operation to contact input tensors along a new dimension.

    The function creates an operation that creates a new dim for all the
    input tensors and then concatenates them along that new dim.
.

    All the tensors in 'inputs' must have the same shape.

        for ii in range(inputs[0].rank()):
            assert all(inp.shape[ii] == inputs[0].shape[ii] for inp in inputs)

    The shape of the output tensor is defined as:

        output.rank() = inputs[0].rank() + 1

        output.shape[dim] = len(inputs)

        for ii in range(inputs[0].rank()):
            if ii < dim:
                output.shape[ii] = inputs[0].shape[ii]
            else:
                output.shape[ii+1] = inputs[0].shape[ii]

    For example, given a sequence of two 2D tensors [[0, 1], [2, 3]] and
    [[4, 5], [6, 7]] both of shape [2, 2],

        stack(inputs, 0)

    will produce [[[0, 1], [2, 3]], [[4, 5], [6, 7]]] of shape [2, 2, 2] and

        stack(inputs, 1)

    will produce [[[0, 1], [4, 5]], [[2, 3], [6, 7]]] of shape [2, 2, 2].

    Parameters:
        inputs : Sequence[Tensor]
            The sequence of tensors to stack.

        dim : int
            The dimension in which the stack is performed.

    Returns:
        A tensor that contains the input tensors stacked along a new dimension.
    '''
    return concat([unsqueeze(inp, axis=dim) for inp in inputs], dim=dim)


def expand_dims_like(left: Union[Tensor, int, float], right: Tensor) -> Tensor:
    '''
    Add an operation to expand the first tensor to the same rank as the second
    tensor.

    That function takes a first tensor. It also accepts an integer or a float,
    in which case it creates a constant tensor from it. In both cases, the rank
    of that first tensor is compared to the rank of the second tensor. If they
    are of the same rank, the first tensor is returned. Otherwise, the first
    tensor is expanded on the left to match the rank of the second tensor.

    Note that the shapes do not have to match, only the rank is considered in
    that function.

    For example, for a pair of tensors of shapes [3, 4] and [4, 3, 2], the
    first tensor will be expanded to a tensor of rank 3 and shape [1, 3, 4].

    Parameters:
        left : Union[Tensor, int, float]
            The first tensor to expand. When a scalar value is provided as a
            parameter, that function first creates a tensor before expanding it
            (if needed).

        right : Tensor
            The reference tensor to match.

    Returns:
        The tensor produced by the shuffle layer.
    '''
    if isinstance(left, int):
        left = constant(dims_array([left]))
    elif isinstance(left, float):
        if isinstance(right, Tensor) and right.dtype == trt.DataType.HALF:
            left = constant(fp16_array([left]))
        else:
            left = constant(fp32_array([left]))
    left_ndim = left.ndim()
    right_ndim = right.ndim()
    if right_ndim > left_ndim:
        new_ndim = list(range(right_ndim - left_ndim))
        return expand_dims(left, new_ndim)
    return left


# If dim is None, return a 1-D TensorRT-LLM tensor of the size
# If dim is not None, return a 0-D TensorRT-LLM tensor of the dimension size
def shape(input: Tensor,
          dim: Optional[int] = None,
          cast_to_dtype: Optional[Union[str, trt.DataType]] = None,
          clip_before_cast: Sequence[int] = None) -> Tensor:
    '''
    Add an operation to create a shape tensor.

    The shape tensor can either be the shape of the input tensor when the
    parameter dim is None or a scalar (tensor of rank 0) that corresponds to
    the size of dim-th dimension.

    Parameters:
        input : Tensor
            The input tensor from which we want to extract the shape or the
            size in one dimension.

        dim : Optional[int]
            The dimension from which to extract the size. If it is None, the
            entire shape of the input tensor is returned.

    Returns:
        A tensor that contains the shape of the input tensor (if 'dim' is None)
        or the size in the dimension 'dim' of the input tensor. If 'dim' is
        'None', that tensor has the same rank as the input tensor, otherwise
        its rank is 0.
    '''
    layer = default_trtnet().add_shape(input.trt_tensor)
    res = _create_tensor(layer.get_output(0), layer)
    if cast_to_dtype is not None:
        if clip_before_cast is not None and (cast_to_dtype == 'int32'
                                             or cast_to_dtype == trt.int32):
            assert len(
                clip_before_cast
            ) == 2, f"This parameter only expects a tuple of 2 integers (lower, upper) but got {clip_before_cast}"
            res = int_clip(res, clip_before_cast[0], clip_before_cast[1])
        res = cast(res, cast_to_dtype)

    if dim is None:
        return res

    return gather(res, dim=0, indices=dim).view([])


def gather(input: Tensor, dim: int, indices: Union[Tensor, int]) -> Tensor:
    '''
    Add an operation to gather elements from a tensor.

    That function implements the GatherElements operator from the ONNX
    specification as described in

        https://github.com/onnx/onnx/blob/main/docs/Operators.md#GatherElements

    The input and indices arguments must have the same rank >= 1. The operation
    will produce a tensor with the same shape as the indices tensor. The axis
    is the dimension to gather on.

    As shown in the ONNX description, for a 3D tensor, the output is:

        out[i][j][k] = input[indices[i][j][k]][j][k] if axis = 0,
        out[i][j][k] = input[i][indices[i][j][k]][k] if axis = 1,
        out[i][j][k] = input[i][j][indices[i][j][k]] if axis = 2.

    For example,

        gather([[4, 2], [5, 3]], 0, [[1, 0], [0, 1]])

    will produce [[5, 2], [4, 3]].

        gather([[1, 2, 3], [4, 5, 6], 1, [[1], [0]])

    will produce [[2], [4]]. See the ONNX documentation for more examples.

    That operation maps to the TensorRT IGatherLayer.

    Parameters:
        input : Tensor
            The input tensor to gather elements from.

        dim : int
            The dimension to gather on.

        indices : Union[Tensor, int]
            The positions in the 'dim' dimension to gather from.

    Returns:
        The tensor containing the gathered elements. It has the same shape as
        the indices tensor.
    '''
    if isinstance(indices, int):
        indices = constant(int32_array([indices]))

    # The input and indices tensors must have the same rank.
    assert input.rank() == indices.rank()

    layer = default_trtnet().add_gather_v2(input.trt_tensor,
                                           indices.trt_tensor,
                                           mode=trt.GatherMode.ELEMENT)

    if dim < 0:
        dim = input.ndim() + dim
    layer.axis = dim
    return _create_tensor(layer.get_output(0), layer)


def select(input: Tensor, dim: int, index: Union[Tensor, int]) -> Tensor:
    '''
    Add an operation to select a slice of elements from a tensor.

    Given an input tensor, that function creates an operation that selects the
    index-th slice of elements in the dimension 'dim' to create a new tensor.
    The output tensor has a shape in which the input dimension 'dim' is
    removed.

    The 'index' can either be an integer or a 1D tensor containing a single
    element.

    For example, on input=[[4, 2, 5], [2, 1, 2], [4, 7, 1]], which has a shape
    [3, 3],

        select(input, 0, 1)

    will create a tensor of shape [3] that contains the [2, 1, 2].

    Regarding the shape of the output tensor, the dimension 'dim' is removed.
    It means that for a tensor of shape [4, 2, 6, 3],

        select(input, 2, 4)

    will select the 5th slice (index == 4) from the 3rd dimension (dim == 2)
    and return a tensor of shape [4, 2, 3] (i.e. the 3rd dimension is removed).

    That operation maps to the TensorRT IGatherLayer.

    Parameters:
        input : Tensor
            The input tensor to select from.

        dim : int
            The dimension to select from.

        index : Union[Tensor, int]
            The index of the slice in the 'dim' dimension to select.

    Returns:
        The tensor containing the selected slice.
    '''
    if isinstance(index, int):
        index = constant(int32_array([index]))
    assert index.rank() == 1 and index.size(
        0) == 1, f"index should have rank 1, got {index.rank()}"

    new_shape = []
    for i in range(input.rank()):
        if i != dim:
            new_shape.append(shape(input, i))

    layer = default_trtnet().add_gather(input.trt_tensor, index.trt_tensor,
                                        dim)
    return _create_tensor(layer.get_output(0), layer).view(concat(new_shape))


def index_select(input: Tensor, dim: int, index: Tensor) -> Tensor:
    '''
    Add an operation to select slices of elements from a tensor.

    Given an input tensor, that function creates an operation that selects the
    slices of elements in the dimension 'dim' at the indices listed in 'index'
    to create a new tensor.  The output tensor has the same rank as the input
    tensor.

    The 'index' is a tensor of rank 1.

    For example, on input=[[4, 2, 5], [2, 1, 2], [4, 7, 1]], which has a shape
    [3, 3],

        index_select(input, 0, [0, 1])

    will create a tensor of shape [2, 3] that contains the [[4, 2, 5], [2, 1, 2]].

    Regarding the shape of the output tensor, the dimension 'dim' has the same
    size as the 'index' tensor. It means that for a input tensor of shape [4, 2, 6, 3],

        index_select(input, 2, [1, 4])

    will select the 2nd and 5th slices (index == 1 or 4) from the 3rd dimension
    (dim == 2) and return a tensor of shape [4, 2, 2, 3] (i.e. the 3rd
    dimension is shrunk to 2).

    Note that this operation can also be used to expand a tensor in the 'dim'
    dimension, for example, on input [[0, 1], [2, 3]],

        index_select(input, 1, [0, 0, 0])

    will produce a tensor of shape [2, 3] containing [[0, 0, 0], [2, 2, 2]].

    That operation maps to the TensorRT IGatherLayer.

    Parameters:
        input : Tensor
            The input tensor to select from.

        dim : int
            The dimension to select from.

        index : Tensor
            The indices of the slices in the 'dim' dimension to select.

    Returns:
        The tensor containing the selected slices.
    '''
    assert index.rank() == 1, f"index should have rank 1, got {index.rank()}"

    new_shape = []
    for i in range(input.rank()):
        if i != dim:
            new_shape.append(shape(input, i))
        else:
            new_shape.append(shape(index, 0))

    layer = default_trtnet().add_gather(input.trt_tensor, index.trt_tensor,
                                        dim)
    return _create_tensor(layer.get_output(0), layer).view(concat(new_shape))


# NOTE: Jointly added with Apple
def scatter(input: Tensor, dim: int, indices: Tensor,
            updates: Tensor) -> Tensor:
    '''
    This operation adds a layer that creates an output tensor by element-wise
    copying values from the input tensor and then updating values by the given
     `indices` and `updates` tensors.
     For a 2D input tensor, it first copies the input to output,
     then updates the output tensor like the following for each entry in `updates`:
        output[indices[i][j]][j] = updates[i][j] if dim=0
        output[i][indices[i][j]] = updates[i][j] if dim=1
     If the `input` tensor is [[1, 2, 3], [4, 5, 6]],
     the indices tensor is [[1, 2], [0, 1]],
     the updates tensor is [[-1, -2], [-3, -4]], and dim=1
     the output tensor will be [[1, -1, -2], [-3, -4, 6]].
     Parameters:
        input: Tensor
            The input data that needs to be updated.
        dim: int
            The axis on which the scatter is to be performed.
        indices: Tensor
            An integer tensor of the same rank as input that indicates the positions to be updated.
        updates: Tensor
            A data tensor of same shape as the `indices` tensor that contains the update values.
     Returns:
        A tensor created by the element-wise scatter layer.
    '''
    layer = default_trtnet().add_scatter(input.trt_tensor,
                                         indices.trt_tensor,
                                         updates.trt_tensor,
                                         mode=trt.ScatterMode.ELEMENT)
    layer.axis = dim
    return _create_tensor(layer.get_output(0), layer)


def gather_nd(input: Tensor, indices: Tensor, batch_dims: int = 1) -> Tensor:
    '''
    Adds a layer that performs a gather with some element-wise dimensions.
    See: https://onnx.ai/onnx/operators/onnx__GatherND.html
    The gather is performed on dim=batch_dims.

    Parameters:
        input: Tensor
            The tensor on which the gather operation is performed.
        indices: Tensor
            The tensor that indicates which entries to be gathered.
        batch_dims: int
            The number of first dimensions that should be skipped before gather starts.
    Returns:
        A tensor created by the gather layer with GatherMode.ND.
    '''
    gather_layer = default_trtnet().add_gather_v2(input.trt_tensor,
                                                  indices.trt_tensor,
                                                  mode=trt.GatherMode.ND)
    gather_layer.num_elementwise_dims = batch_dims
    return _create_tensor(gather_layer.get_output(0), gather_layer)


def nonzero(input: Tensor) -> Tensor:
    '''
    Adds a layer that finds the indices of non-zero values of the input tensor.

    Parameters:
        input: Tensor
            The input tensor for which we need to find the indices of non-zero values.
    Returns:
        A tensor of shape [D, C] where D is the number of dimensions of `input` and
        C is the number of non-zero values in it.
        Each column of this 2D tensor represents the index tuple for each non-zero value.
    '''
    non_zero_layer = default_trtnet().add_non_zero(input.trt_tensor)
    return _create_tensor(non_zero_layer.get_output(0), non_zero_layer)


def masked_select(input: Tensor, mask: Tensor) -> Tensor:
    '''
    Add an operation to select elements from a tensor according to a boolean
    mask tensor.

    Given an input tensor, that function creates an operation that selects
    elements at the indices indicated by the boolean mask tensor to create
    a new tensor. The output tensor is a 1-D tensor.

    The input tensor must have rank >= 1. The shapes of the input tensor and
    the mask tensor don’t need to match, but they must be able to be broadcasted.

    For example, on input=[[4, 2, 5], [2, 1, 2], [4, 7, 1]], which has a shape
    [3, 3],

        masked_select(input, [[True, False, True], [False, True, False], [True, False, True]])

    will create a tensor of shape [5] that contains the [4, 5, 1, 4, 1].

        masked_select(input, [[True], [False], [True]])

    will create a tensor of shape [6] that contains the [4, 2, 5, 4, 7, 1].

        masked_select(input, [[False, False, True]])

    will create a tensor of shape [3] that contains the [5, 2, 1].

        masked_select(input, [False])

    will create a tensor of shape [0] which is empty.

    That operation is implemented by NonZero, Shuffle and GatherV2 layers
    in TensorRT.

    Parameters:
        input : Tensor
            The input tensor to select from.

        mask : Tensor
            The boolean mask tensor that indicates elements to select.

    Returns:
        The 1-D tensor containing the selected elements.
    '''
    assert input.rank() >= 1, "input should have rank >= 1"
    input, mask = broadcast_helper(input, mask)
    expanded_mask = expand(mask, shape(input))

    non_zero_layer = default_trtnet().add_non_zero(expanded_mask.trt_tensor)

    shuffle_layer = default_trtnet().add_shuffle(non_zero_layer.get_output(0))
    shuffle_layer.second_transpose = (1, 0)

    gather_layer = default_trtnet().add_gather_v2(input.trt_tensor,
                                                  shuffle_layer.get_output(0),
                                                  mode=trt.GatherMode.ND)
    return _create_tensor(gather_layer.get_output(0), gather_layer)


def masked_scatter(input: Tensor, mask: Tensor, source: Tensor) -> Tensor:
    '''
    Add the masked_scatter base on PyTorch definition.

    See https://pytorch.org/docs/stable/generated/torch.Tensor.masked_scatter_.html#torch-tensor-masked-scatter for a
    description of that function.

    Parameters:
        input : Tensor
            The input tensor.

        mask : Tensor
            The boolean mask tensor that indicates elements to select.

        source: Tensor
            The tensor to copy from
    Returns:
        The tensor containing the source tensor selected by mask.

    '''
    assert input.rank() >= 1, "input should have rank >= 1"
    input, mask = broadcast_helper(input, mask)
    expanded_mask = expand(mask, shape(input))

    non_zero_layer = default_trtnet().add_non_zero(expanded_mask.trt_tensor)

    shuffle_layer = default_trtnet().add_shuffle(non_zero_layer.get_output(0))
    shuffle_layer.second_transpose = (1, 0)
    source = source.view([-1])

    scatter_layer = default_trtnet().add_scatter(input.trt_tensor,
                                                 shuffle_layer.get_output(0),
                                                 source.trt_tensor,
                                                 mode=trt.ScatterMode.ND)

    return _create_tensor(scatter_layer.get_output(0), scatter_layer)


def concat(inputs: Sequence[Union[Tensor, int]], dim: int = 0) -> Tensor:
    '''
    Add an operation to concatenate tensors.

    The function creates an operation that concatenates the tensors from the
    sequence 'inputs'. The concatenation is done along the dimension 'dim'.

    All the tensors in 'inputs' must have the same shape expect for the
    dimension 'dim'.

        for ii in range(inputs[0].rank()):
            assert (ii == dim) or all(inp.shape[ii] == inputs[0].shape[ii] for inp in inputs)

    The shape of the output tensor is defined as:

        for ii in range(inputs[0].rank()):
            # Same size as all the inputs in dimension ii != dim.
            output.shape[ii] = inputs[0].shape[ii]

            # Sum of the sizes in the different inputs in dimension 'dim'.
            if ii == dim:
                for jj in range(1, len(inputs)):
                    output.shape[ii] += inputs[jj].shape[ii]

    For example, given a sequence of two 2D tensors [[0, 1], [2, 3]] and
    [[4, 5], [6, 7]] both of shape [2, 2],

        concat(inputs, 0)

    will produce [[0, 1], [2, 3], [4, 5], [6, 7]] of shape [4, 2] and

        concat(inputs, 1)

    will produce [[0, 1, 4, 5], [2, 3, 6, 7]] of shape [2, 4].

    Parameters:
        inputs : Sequence[Union[Tensor, int]]
            The sequence of tensors to concatenate. For integers, that function
            creates constant tensors.

        dim : int
            The dimension in which the concatenation is performed.

    Returns:
        A tensor that contains the concatenation of the tensors.
    '''
    assert len(
        inputs
    ) > 0, f"Number of inputs ({len(inputs)}) to the concatenation layer must be > 0."
    tmp = []
    inputs = constants_to_tensors_(*inputs)
    for i in inputs:
        if i.rank() == 0:
            tmp.append(i.view([1]))
        else:
            tmp.append(i)

    layer = default_trtnet().add_concatenation([i.trt_tensor for i in tmp])
    layer.axis = dim_resolve_negative(dim, tmp[0].ndim())[0]
    return _create_tensor(layer.get_output(0), layer)


def softmax(input: Tensor, dim: Optional[int] = None) -> Tensor:
    '''
    Add an operation to compute softmax on a tensor.

    That operation computes the softmax on the input tensor in the dimension
    'dim' if specified. Otherwise, it is applied on the last dimension.

    It inserts a ISoftmaxLayer to the TensorRT graph.

    Parameters:
        input : Tensor
            The input tensor on which to apply softmax.

        dim : Optional[int]
            The dimension used to apply softmax.

    Returns:
        The output tensor of the softmax layer.
    '''
    if dim is None:
        dim = input.ndim() - 1
    if dim < 0:
        dim = input.ndim() + dim
    axes = dim_to_trt_axes(dim)

    layer = default_trtnet().add_softmax(input.trt_tensor)
    layer.axes = axes

    return _create_tensor(layer.get_output(0), layer)


def embedding(input: Tensor,
              weight: Tensor,
              sharding_dim=0,
              padding=None) -> Tensor:
    '''
    Add an operation to perform embedding lookup.

    Parameters:
        input : Tensor
            The input tensor the contains the indices to perform the lookup.

        weight : Tensor
            The table to gather from.

        sharding_dim : int
            sharding_dim = 0 means that we shard the embedding table in vocab dim;
            sharding_dim = 1 means that we shard the embedding table in embedding dim.

        padding: Tensor
            Additional padding added to the end of the embedding table before feeding into gather op.

    Returns:
        The tensor produced by the embedding lookup layer.
    '''

    if padding is not None:
        padded_weight = concat([weight, padding], dim=0)
    else:
        padded_weight = weight

    layer = default_trtnet().add_gather(padded_weight.trt_tensor,
                                        input.trt_tensor, 0)
    x = _create_tensor(layer.get_output(0), layer)
    return x


def constant_to_tensor_(input: Union[Tensor, int, float, bool],
                        dtype: Union[trt.DataType, str] = None,
                        to_array=True) -> Tensor:
    if dtype is None:
        # deduce the type from the given value
        # NOTE: bool is a subtype of int, so bool needs to be checked first
        if isinstance(input, bool):
            dtype = trt.bool
        elif isinstance(input, int):
            dtype = trt.int32
        else:
            dtype = trt.float32

    if not isinstance(input, Tensor):
        if isinstance(dtype, str):
            dtype = str_dtype_to_trt(dtype)
        array_fn_dict = {
            trt.int64: int64_array,
            trt.int32: int32_array,
            trt.float32: fp32_array,
            trt.float16: fp16_array,
            trt.bfloat16: bf16_array,
            trt.bool: bool_array,
        }
        assert dtype in array_fn_dict
        return constant(array_fn_dict[dtype]([input] if to_array else input))

    return input


def constants_to_tensors_(
        *inputs: Union[Tensor, int, float]) -> Tuple[Tensor, ...]:
    '''
    Helper function to create tensors from multiple inputs.

    For each inputs, that function first creates a constant tensor if the input
    is an integer or a float. Then, if any input is int64, it upcasts other
    integer inputs to int64.

    Parameters:
        inputs : Tuple[Union[Tensor, int, float], ...]
            The inputs to create tensors from.

    Returns:
        A tuple of tensors.
    '''
    has_int64: bool = False
    for i in inputs:
        if isinstance(i, int) and (i >= 2**31 or i < -2**31)\
                or isinstance(i, Tensor) and i.dtype == trt.int64:
            has_int64 = True
            break

    if not has_int64:
        return tuple(constant_to_tensor_(i) for i in inputs)

    result = []
    for i in inputs:
        if isinstance(i,
                      int) or isinstance(i, Tensor) and i.dtype == trt.int32:
            result.append(
                constant_to_tensor_(i, trt.int64 if has_int64 else trt.int32))
        else:
            result.append(constant_to_tensor_(i))
    return tuple(result)


def broadcast_helper(
        left: Union[Tensor, int, float],
        right: Union[Tensor, int, float]) -> Tuple[Tensor, Tensor]:
    '''
    Helper function to perform a broadcast.

    For each input, that function first creates a constant tensor if the input
    is an integer or a float. Then, if needed, it expands the smaller tensor to
    make sure its rank is the same as the larger one.

    Parameters:
        left : Union[Tensor, int, float]
            The first input. If that input is an integer or a float, the
            function creates a constant tensor.

        right : Union[Tensor, int, float]
            The second input. If that input is an integer or a float, the
            function creates a constant tensor.

    Returns:
        A pair of tensors of same rank.
    '''
    if not default_net().strongly_typed:
        left = constant_to_tensor_(left)
        right = constant_to_tensor_(right)
    else:
        left = constant_to_tensor_(
            left, right.dtype if isinstance(right, Tensor) else None)
        right = constant_to_tensor_(right, left.dtype)

    if left.rank() == right.rank():
        return (left, right)

    if left.rank() < right.rank():
        left = expand_dims_like(left, right)
        return (left, right)

    if left.rank() > right.rank():
        right = expand_dims_like(right, left)
        return (left, right)


def elementwise_binary(left: Union[Tensor, int,
                                   float], right: Union[Tensor, int, float],
                       op: trt.ElementWiseOperation) -> Tensor:
    '''
    Add an elementwise operation with two inputs.

    For each input, that function first creates a constant tensor if the input
    is an integer or a float. Then, if needed, it expands the smaller tensor to
    make sure its rank is the same as the larger one. Then, it performs the
    elementwise operation 'op'.

    The following closures are defined in functional.*:

        add      for op=trt.ElementWiseOperation.SUM
        sub      for op=trt.ElementWiseOperation.SUB
        mul      for op=trt.ElementWiseOperation.PROD
        div      for op=trt.ElementWiseOperation.DIV
        floordiv for op=trt.ElementWiseOperation.FLOOR_DIV
        gt       for op=trt.ElementWiseOperation.GREATER
        lt       for op=trt.ElementWiseOperation.LESS
        op_and   for op=trt.ElementWiseOperation.AND
        op_or    for op=trt.ElementWiseOperation.OR
        eq       for op=trt.ElementWiseOperation.EQUAL
        minimum  for op=trt.ElementWiseOperation.MIN
        maximum  for op=trt.ElementWiseOperation.MAX
        pow      for op=trt.ElementWiseOperation.POW

    It is implemented using the IElementWiseLayer from TensorRT.

    Parameters:
        left : Union[Tensor, int, float]
            The first input. If that input is an integer or a float, the
            function creates a constant tensor.

        right : Union[Tensor, int, float]
            The second input. If that input is an integer or a float, the
            function creates a constant tensor.

        op : trt.ElementWiseOperation
            The binary operation to perform.

    Returns:
        The tensor produced by this elementwise operation.
    '''
    left, right = broadcast_helper(left, right)
    if left.dtype == trt.int32 and right.dtype == trt.int64:
        left = cast(left, trt.int64)
    if left.dtype == trt.int64 and right.dtype == trt.int32:
        right = cast(right, trt.int64)
    layer = default_trtnet().add_elementwise(left.trt_tensor, right.trt_tensor,
                                             op)
    return _create_tensor(layer.get_output(0), layer)


add = partial(elementwise_binary, op=trt.ElementWiseOperation.SUM)
sub = partial(elementwise_binary, op=trt.ElementWiseOperation.SUB)
mul = partial(elementwise_binary, op=trt.ElementWiseOperation.PROD)
div = partial(elementwise_binary, op=trt.ElementWiseOperation.DIV)
floordiv = partial(elementwise_binary, op=trt.ElementWiseOperation.FLOOR_DIV)
gt = partial(elementwise_binary, op=trt.ElementWiseOperation.GREATER)
lt = partial(elementwise_binary, op=trt.ElementWiseOperation.LESS)
op_and = partial(elementwise_binary, op=trt.ElementWiseOperation.AND)
op_or = partial(elementwise_binary, op=trt.ElementWiseOperation.OR)
eq = partial(elementwise_binary, op=trt.ElementWiseOperation.EQUAL)
minimum = partial(elementwise_binary, op=trt.ElementWiseOperation.MIN)
maximum = partial(elementwise_binary, op=trt.ElementWiseOperation.MAX)
pow = partial(elementwise_binary, op=trt.ElementWiseOperation.POW)
op_xor = partial(elementwise_binary, op=trt.ElementWiseOperation.XOR)


def modulo(x: Tensor, y: Union[Tensor, int]) -> Tensor:
    '''
    This function adds an element-wise modulo (x % y) operation for a given tensor.
    Since there is no TensorRT layer that can directly perform this,
    this function implements it using some of the basic operations.

    Returns:
        A tensor that represents (x % y) modulo operation.
    '''
    return x - (x // y) * y


def where(condition: Union[Tensor, bool], left: Union[Tensor, int, float],
          right: Union[Tensor, int, float]) -> Tensor:
    '''
    Add a where (aka select or if-then-else) operation.

    Assuming the three input parameters have the same shape, that function creates
    the operation to compute a tensor of the same shape such that:

        for ii in range(mul(condition.shape)):
            output[ii] = left[ii] if condition[ii] else right[ii]

    For each input, that function first creates a constant tensor if the
    condition is boolean or the left/right input is an integer or a float.
    Then, if needed, it expands the smaller tensor to make sure its
    rank is the same as the larger one. Then, it performs the selection.

    It is implemented using the ISelectLayer from TensorRT.

    Parameters:
        condition : Union[Tensor, bool]
            The condition. If that input is a boolean, the function
            creates a constant tensor.

        left : Union[Tensor, int, float]
            The first input. If that input is an integer or a float, the
            function creates a constant tensor.

        right : Union[Tensor, int, float]
            The second input. If that input is an integer or a float, the
            function creates a constant tensor.

    Returns:
        The tensor produced by this where operation.
    '''
    # Convert to tensors.
    condition = constant_to_tensor_(condition)
    left, right = constants_to_tensors_(left, right)

    # Find the tensor with the largest rank of the three.
    largest = condition
    if largest.rank() < left.rank():
        largest = left
    if largest.rank() < right.rank():
        largest = right

    # Expand the tensors to match the largest one.
    if condition is not largest:
        condition = expand_dims_like(condition, largest)
    if left is not largest:
        left = expand_dims_like(left, largest)
    if right is not largest:
        right = expand_dims_like(right, largest)

    # Insert the operation.
    layer = default_trtnet().add_select(condition.trt_tensor, left.trt_tensor,
                                        right.trt_tensor)
    return _create_tensor(layer.get_output(0), layer)


def unary(input: Tensor, op: trt.UnaryOperation) -> Tensor:
    '''
    Add an elementwise operation on a single input.

    The following closures are defined in functional.*:

        round   for op=trt.UnaryOperation.ROUND
        sqrt    for op=trt.UnaryOperation.SQRT
        exp     for op=trt.UnaryOperation.EXP
        sin     for op=trt.UnaryOperation.SIN
        cos     for op=trt.UnaryOperation.COS
        abs     for op=trt.UnaryOperation.ABS
        log     for op=trt.UnaryOperation.LOG

    It is implemented using the IUnaryLayer from TensorRT.

    Parameters:
        input : Tensor
            The input tensor.

        op : trt.UnaryOperation
            The unary operation to perform.

    Returns:
        The tensor produced by this elementwise operation.
    '''
    layer = default_trtnet().add_unary(input.trt_tensor, op)
    return _create_tensor(layer.get_output(0), layer)


round = partial(unary, op=trt.UnaryOperation.ROUND)
sqrt = partial(unary, op=trt.UnaryOperation.SQRT)
exp = partial(unary, op=trt.UnaryOperation.EXP)
sin = partial(unary, op=trt.UnaryOperation.SIN)
cos = partial(unary, op=trt.UnaryOperation.COS)
abs = partial(unary, op=trt.UnaryOperation.ABS)
log = partial(unary, op=trt.UnaryOperation.LOG)
not_op = partial(unary, op=trt.UnaryOperation.NOT)


def log_softmax(input: Tensor, dim: int) -> Tensor:
    '''
    This function is equivalent of torch.nn.functional.log_softmax() i.e.
    it performs log(softmax(input)) in a safer and faster way.

    Parameters:
        input: Tensor
            The data tensor on which log_softmax to be computed.
        dim: int
            The dimension of the input tensor along which log_softmax will be computed.
    Returns:
        A tensor of same shape as input with log_softmax computed on the specified dim.
    '''
    x_max = max(input, dim=dim, keepdim=True)
    x = input - x_max
    return x - log(sum(exp(x), dim=dim, keepdim=True))


def reduce(input: Tensor,
           op: trt.ReduceOperation,
           dim: Union[int, Tuple[int]],
           keepdim: bool = False) -> Tensor:
    '''
    Add an reduction operation to do along a dimension.

    It is implemented using the IReduceLayer from TensorRT.

    Parameters:
        input : Tensor
            The input tensor.

        op : trt.ReduceOperation
            The reduction operation to perform.
            Options: SUM, PROD, MAX, MIN, AVG

        dim : int
            The dimension along which the reduction is performed.

        keepdim : bool
            Is the dimension kept in the reduced tensor? When True the
            dimension is kept, it is removed from the shape otherwise.

    Returns:
        The tensor produced by this reduction operation.
    '''
    dim = dim_resolve_negative(dim, input.ndim())
    axes = dim_to_trt_axes(dim)

    layer = default_trtnet().add_reduce(input.trt_tensor,
                                        op,
                                        axes,
                                        keep_dims=keepdim)
    return _create_tensor(layer.get_output(0), layer)


prod = partial(reduce, op=trt.ReduceOperation.PROD)
min = partial(reduce, op=trt.ReduceOperation.MIN)


def mean(input: Tensor,
         dim: Union[int, Tuple[int]],
         keepdim: bool = False) -> Tensor:
    '''
    Add an operation to compute the mean along a dimension.

    Computes the mean along the dimension 'dim' of the input tensor.

    It is implemented using the IReduceLayer from TensorRT.

    Parameters:
        input : Tensor
            The input tensor.

        dim : int
            The dimension along which the mean is computed.

        keepdim : bool
            Is the dimension kept in the reduced tensor? When True the
            dimension is kept, it is removed from the shape otherwise.

    Returns:
        The tensor produced by this reduction operation.
    '''
    return reduce(input, op=trt.ReduceOperation.AVG, dim=dim, keepdim=keepdim)


def max(input: Tensor, dim: int, keepdim: bool = False) -> Tensor:
    '''
    Add an operation to compute the max along a dimension.

    Computes the max along the dimension 'dim' of the input tensor.

    It is implemented using the IReduceLayer from TensorRT.

    Parameters:
        input : Tensor
            The input tensor.

        dim : int
            The dimension along which the mean is computed.

        keepdim : bool
            Is the dimension kept in the reduced tensor? When True the
            dimension is kept, it is removed from the shape otherwise.

    Returns:
        The tensor produced by this reduction operation.
    '''
    return reduce(input, op=trt.ReduceOperation.MAX, dim=dim, keepdim=keepdim)


def sum(input: Tensor, dim: int, keepdim: bool = False) -> Tensor:
    '''
    Add an operation to compute the sum along a dimension.

    Computes the sum along the dimension 'dim' of the input tensor.

    It is implemented using the IReduceLayer from TensorRT.

    Parameters:
        input : Tensor
            The input tensor.

        dim : int
            The dimension along which the mean is computed.

        keepdim : bool
            Is the dimension kept in the reduced tensor? When True the
            dimension is kept, it is removed from the shape otherwise.

    Returns:
        The tensor produced by this reduction operation.
    '''
    return reduce(input, op=trt.ReduceOperation.SUM, dim=dim, keepdim=keepdim)


def argmax(input: Tensor, dim: int, keepdim: bool = False) -> Tensor:
    '''
    Add an argmax operation.

    As explained in the ONNX documentation,

        https://github.com/onnx/onnx/blob/main/docs/Operators.md#argmax

    that function creates a layer computing the indices of the max elements of
    the input tensor's element along the provided dim. The resulting tensor
    has the same rank as the input if keepdims is True. If keepdims is False,
    then the resulting tensor has the reduced dimension pruned.

    Parameters:
        input : Tensor
            The input tensor.

        dim : int
            The dimension in which to compute the argmax indices.

        keepdim : bool
            Do we keep the dimension along which the reduction is performed?
            Yes, if set to True, no otherwise.

    Returns:
        The tensor produced by this argmax operation.
    '''
    dim = dim_resolve_negative(dim, input.ndim())
    axes = dim_to_trt_axes(dim)

    layer = default_trtnet().add_topk(input.trt_tensor, trt.TopKOperation.MAX,
                                      1, axes)
    output = layer.get_output(1)

    if keepdim:
        return _create_tensor(output, layer)

    output = _create_tensor(output, layer)
    a = list(range(input.ndim()))
    for d in dim:
        a.pop(d)
    indices = constant(int32_array(a))
    output_shape = shape(output)
    new_shape = gather(output_shape, 0, indices)
    return view(output, new_shape)


def gelu(x: Tensor) -> Tensor:
    '''
    Add a GELU operation.

    Parameters:
        input : Tensor
            The input tensor on which the activation function is applied.

    Returns:
        The tensor produced by the activation layer.
    '''
    return 0.5 * x * (
        tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * pow(x, 3.0))) + 1.0)


def geglu(x: Tensor) -> Tensor:
    '''
    Add a Gated-GELU operation.

    That function takes a tensor, splits it into two halves along the last
    dimension, applies GELU to the second half and multiply the results. The
    behavior is undefined if the last dimension is not even.

    Parameters:
        input : Tensor
            The input tensor on which the activation function is applied.

    Returns:
        The tensor produced by the activation layer.
    '''
    a, b = chunk(x, 2, dim=-1)
    return a * gelu(b)


def quick_gelu(x: Tensor) -> Tensor:
    return x * sigmoid(1.702 * x)


def gegelu(x: Tensor, limit: Optional[float] = None) -> Tensor:
    # a, b = x[..., ::2], x[..., 1::2]
    ndim = x.ndim()
    a_starts = [0 for i in range(ndim)]
    b_starts = [1 if i == (ndim - 1) else 0 for i in range(ndim)]
    shapes = concat([
        shape(x, i) / 2 if i == (ndim - 1) else shape(x, i)
        for i in range(ndim)
    ])
    strides = [2 if i == (ndim - 1) else 1 for i in range(ndim)]

    a = slice(x, a_starts, shapes, strides)
    b = slice(x, b_starts, shapes, strides)

    if limit is not None:
        a = clip(a, alpha=float(-1e20), beta=limit)
        b = clip(b, alpha=-limit, beta=limit)

    # C = B + 1
    const1 = arange(constant(int32_array(1)), constant(int32_array(2)),
                    trt_dtype_to_str(b.dtype))
    for _ in range(ndim - 1):
        const1 = expand_dims(const1, 0)

    b_shape = concat([shape(b, i) for i in range(ndim)])
    const1_arr = expand(const1, b_shape)

    return quick_gelu(a) * (b + const1_arr)


def group_norm(input: Tensor,
               num_groups: int,
               weight: Optional[Tensor] = None,
               bias: Optional[Tensor] = None,
               eps: float = 1e-05):

    ##
    ## TODO: Document that function!
    ##

    assert not input.is_dynamic(1)
    num_channels = input.size()[1]

    ndim = input.ndim()
    old_shape = shape(input)
    new_shape = concat([
        input.size(0),
        num_groups,
        num_channels // num_groups,
    ] + [input.size(i) for i in range(2, ndim)])
    x = input.view(new_shape)

    # instance norm
    w_shape = [1, num_groups] + [1 for i in range(ndim - 1)]
    instance_weight = constant(np.ones(w_shape,
                                       dtype=trt_dtype_to_np(x.dtype)))
    instance_bias = constant(np.zeros(w_shape, dtype=trt_dtype_to_np(x.dtype)))
    axes_mask = 0
    for i in range(2, x.ndim()):
        axes_mask |= 1 << i
    layer = default_trtnet().add_normalization(x.trt_tensor,
                                               instance_weight.trt_tensor,
                                               instance_bias.trt_tensor,
                                               axes_mask)
    layer.epsilon = eps
    y = _create_tensor(layer.get_output(0), layer)
    y = y.view(old_shape)

    new_shape = concat([num_channels] + [1 for _ in range(2, ndim)])
    if weight is not None:
        y = y * weight.view(new_shape)
    if bias is not None:
        y = y + bias.view(new_shape)

    return y


def softplus(input: Tensor, beta: float, threshold: float) -> Tensor:
    '''
    Add the softplus activation base on PyTorch definition.

    See https://pytorch.org/docs/stable/generated/torch.nn.functional.softplus.html#torch-nn-functional-softplus for a
    description of that function.

    Parameters:
        input : Tensor
            Input TensorRT-LLM Tensor.
        beta : float
            The parameter for softplus computation.
        threshold : float
            The threshold for reverting to the linear function when input * beta > threshold

    Returns:
        The output tensor created by that layer.
    '''
    sf_layer = default_trtnet().add_activation(input.trt_tensor,
                                               trt.ActivationType.SOFTPLUS)
    sf_layer.alpha = 1 / beta
    sf_layer.beta = beta

    prod_tensor = input * beta
    result = prod_tensor > threshold

    return where(result, input, _create_tensor(sf_layer.get_output(0),
                                               sf_layer))


def outer(input: Tensor, vec2: Tensor) -> Tensor:
    '''
    Add an operation to compute the outer product between two tensors.

    That operation creates an Einsum node.

    Parameters:
        input : Tensor
            The first input tensor.

        vec2 : Tensor
            The second input tensor.

    Returns:
        The output tensor produced by this layer.
    '''
    return einsum('i,j->ij', [input, vec2])


def avg_pool2d(input: Tensor,
               kernel_size: Tuple[int],
               stride: Optional[Tuple[int]] = None,
               padding: Optional[Tuple[int]] = (0, 0),
               ceil_mode: bool = False,
               count_include_pad: bool = True) -> Tensor:

    ##
    ## TODO: Document that function!
    ##

    assert not input.is_dynamic()
    ndim = input.ndim()
    if ndim == 3:
        input = expand_dims(input, 0)

    layer = default_trtnet().add_pooling_nd(input.trt_tensor,
                                            trt.PoolingType.AVERAGE,
                                            kernel_size)
    if stride is None:
        stride = kernel_size
    layer.stride_nd = stride

    output = _create_tensor(layer.get_output(0), layer)

    if ndim == 3:
        return output.view(
            concat([output.size(1),
                    output.size(2),
                    output.size(3)]))

    return output


def conv1d(input: Tensor,
           weight: Tensor,
           bias: Optional[Tensor] = None,
           stride: int = 1,
           padding: int = 0,
           dilation: int = 1,
           groups: int = 1) -> Tensor:

    noutput = weight.size()[0]
    kernel_size = weight.size()[-2]
    is_weight_constant = (weight.producer is not None
                          and weight.producer.type == trt.LayerType.CONSTANT)
    weight = weight.producer.weights if is_weight_constant else trt.Weights()

    if bias is not None:
        is_bias_constant = (bias.producer is not None
                            and bias.producer.type == trt.LayerType.CONSTANT)
        bias = bias.producer.weights if is_bias_constant else trt.Weights()

    input_shuffled = stack([input], dim=input.ndim())
    kernel_size = trt.Dims([kernel_size, 1])

    layer = default_trtnet().add_convolution_nd(input_shuffled.trt_tensor,
                                                noutput, kernel_size, weight,
                                                bias)
    layer.stride_nd = (stride, 2)
    layer.padding_nd = (padding, 0)
    layer.dilation_nd = (dilation, 2)
    layer.num_groups = groups

    if not is_weight_constant:
        layer.set_input(1, weight.trt_tensor)
    if bias is not None and not is_bias_constant:
        layer.set_input(2, bias.trt_tensor)

    output_2d = _create_tensor(layer.get_output(0), layer)
    output_1d = squeeze(output_2d, dim=-1)
    return output_1d


def conv2d(input: Tensor,
           weight: Tensor,
           bias: Optional[Tensor] = None,
           stride: Tuple[int, int] = (1, 1),
           padding: Tuple[int, int] = (0, 0),
           dilation: Tuple[int, int] = (1, 1),
           groups: int = 1,
           pre_padding: Optional[Tuple[int, int]] = None,
           post_padding: Optional[Tuple[int, int]] = None) -> Tensor:
    ##
    ## TODO: Document that function!
    ##

    ndim = input.ndim()
    if ndim == 3:
        input = expand_dims(input, 0)

    noutput = weight.size()[0]
    kernel_size = (weight.size()[-2], weight.size()[-1])

    is_weight_constant = (weight.producer is not None
                          and weight.producer.type == trt.LayerType.CONSTANT)
    weight = weight.producer.weights if is_weight_constant else trt.Weights()

    if bias is not None:
        is_bias_constant = (bias.producer is not None
                            and bias.producer.type == trt.LayerType.CONSTANT)
        bias = bias.producer.weights if is_bias_constant else trt.Weights()

    layer = default_trtnet().add_convolution_nd(input.trt_tensor, noutput,
                                                kernel_size, weight, bias)
    layer.stride_nd = stride
    layer.padding_nd = padding
    layer.dilation_nd = dilation
    layer.num_groups = groups
    layer.dilation_nd = dilation
    if pre_padding:
        layer.pre_padding = pre_padding
    if post_padding:
        layer.post_padding = post_padding

    if not is_weight_constant:
        layer.set_input(1, weight.trt_tensor)
    if bias is not None and not is_bias_constant:
        layer.set_input(2, bias.trt_tensor)

    output = _create_tensor(layer.get_output(0), layer)

    if ndim == 3:
        return output.view(
            concat([output.size(1),
                    output.size(2),
                    output.size(3)]))

    return output


def conv3d(input: Tensor,
           weight: Tensor,
           bias: Optional[Tensor] = None,
           stride: Union[int, Tuple[int, int]] = (1, 1, 1),
           padding: Union[int, Tuple[int, int]] = (0, 0, 0),
           dilation: Union[int, Tuple[int, int]] = (1, 1, 1),
           groups: int = 1) -> Tensor:
    ##
    ## TODO: Document this function!
    ##

    ndim = input.ndim()
    # TRT requires the input of Conv3D layer to be 5-dimentional tensor.
    if ndim == 4:
        input = expand_dims(input, 0)
    assert input.ndim() == 5

    if isinstance(stride, int):
        stride = tuple([stride] * 3)
    if isinstance(padding, int):
        padding = tuple([padding] * 3)
    if isinstance(dilation, int):
        dilation = tuple([dilation] * 3)

    noutput = weight.size()[0]
    kernel_size = (weight.size()[-3], weight.size()[-2], weight.size()[-1])

    is_weight_constant = (weight.producer is not None
                          and weight.producer.type == trt.LayerType.CONSTANT)
    weight = weight.producer.weights if is_weight_constant else trt.Weights()

    if bias is not None:
        is_bias_constant = (bias.producer is not None
                            and bias.producer.type == trt.LayerType.CONSTANT)
        bias = bias.producer.weights if is_bias_constant else trt.Weights()

    layer = default_trtnet().add_convolution_nd(input.trt_tensor, noutput,
                                                kernel_size, weight, bias)
    layer.stride_nd = stride
    layer.padding_nd = padding
    layer.dilation_nd = dilation
    layer.num_groups = groups
    layer.dilation_nd = dilation

    if not is_weight_constant:
        layer.set_input(1, weight.trt_tensor)
    if bias is not None and not is_bias_constant:
        layer.set_input(2, bias.trt_tensor)

    output = _create_tensor(layer.get_output(0), layer)
    return output


def conv_transpose2d(input: Tensor,
                     weight: Tensor,
                     bias: Optional[Tensor] = None,
                     stride: Tuple[int, int] = (1, 1),
                     padding: Tuple[int, int] = (0, 0),
                     output_padding: Tuple[int, int] = (0, 0),
                     dilation: Tuple[int, int] = (1, 1),
                     groups: int = 1) -> Tensor:
    ##
    ## TODO: Document that function!
    ##

    assert not input.is_dynamic()

    ndim = input.ndim()
    if ndim == 3:
        input = expand_dims(input, 0)

    noutput = weight.size()[1]
    kernel_size = (weight.size()[-2], weight.size()[-1])

    is_weight_constant = (weight.producer is not None
                          and weight.producer.type == trt.LayerType.CONSTANT)
    weight = weight.producer.weights if is_weight_constant else trt.Weights()

    if bias is not None:
        is_bias_constant = (bias.producer is not None
                            and bias.producer.type == trt.LayerType.CONSTANT)
        bias = bias.producer.weights if is_bias_constant else trt.Weights()

    layer = default_trtnet().add_deconvolution_nd(input.trt_tensor, noutput,
                                                  kernel_size, weight, bias)
    layer.stride_nd = stride
    layer.padding_nd = padding
    layer.num_groups = groups

    if not is_weight_constant:
        layer.set_input(1, weight.trt_tensor)
    if bias is not None and not is_bias_constant:
        layer.set_input(2, bias.trt_tensor)

    output = _create_tensor(layer.get_output(0), layer)

    if ndim == 3:
        return output.view(
            concat([output.size(1),
                    output.size(2),
                    output.size(3)]))

    return output


def split(tensor: Tensor,
          split_size_or_sections: Union[int, Sequence[int]],
          dim: int = 0) -> Sequence[Tensor]:
    '''
    Add an operation that splits a tensor into sub-tensors.

    This operation creates a list of tensors that are obtained from the input
    tensor by slicing it along the dimension 'dim'. If 'split_size_or_sections'
    is an integer, the tensor is split into 'input.shape[dim] /
    split_size_or_sections' slices. If 'split_size_or_sections' is a list of
    sizes, the tensor is split into 'len(split_size_or_sections)' slices and
    the size of the ith slice is given by 'split_size_or_sections[i]'.

    There are several constraints with the current implementation:

        - The input tensor must be static (no dynamic dimension),
        - If 'split_size_or_sections' is an integer, the number of elements in
          the 'dim' dimension of the input must be a multiple of
          'split_size_or_sections': 'input.shape[dim] % split_size_or_sections == 0'.
        - If 'split_size_or_sections' is a sequence, the sum of the elements in
          'split_size_or_sections' must be equal to the size in the dimension
          'dim': 'input.shape[dim] == sum(ii for ii in split_size_or_sections)'.

    That operation is implemented using a 'slice' operation for each output
    slice.

    Parameters:
        tensor : Tensor
            The input tensor to slice.

        split_size_or_sections : Union[int, Sequence[int]]
            If it is an integer, it encodes the size of each slice. Otherwise,
            if it is a sequence, it is the size of each slice.

        dim : int
            The dimension of the tensor to slice.

    Returns:
        The list of tensors produced by the different operations.
    '''
    assert not tensor.is_dynamic(dim)

    ndim = tensor.ndim()
    if dim < 0:
        dim += ndim
    dim_value = tensor.size()[dim]
    starts = [constant(dims_array([0])) for _ in range(ndim)]
    sizes = [shape(tensor, i) for i in range(ndim)]

    if isinstance(split_size_or_sections, int):
        # TODO: support non-divisible cases
        assert dim_value % split_size_or_sections == 0
        num_sections = dim_value // split_size_or_sections
        sizes[dim] = constant(dims_array([split_size_or_sections]))

        outputs = []
        for i in range(num_sections):
            starts[dim] = constant(dims_array([split_size_or_sections * i]))
            outputs.append(slice(tensor, concat(starts), concat(sizes)))
        return outputs
    else:
        total_size = 0
        for i in split_size_or_sections:
            total_size += i
        assert dim_value == total_size
        num_sections = len(split_size_or_sections)

        outputs = []
        for i in range(num_sections):
            if i > 0:
                starts[dim] = starts[dim] + sizes[dim]
            sizes[dim] = constant(dims_array([split_size_or_sections[i]]))
            outputs.append(slice(tensor, concat(starts), concat(sizes)))
        return outputs


def chunk(tensor: Tensor, chunks: int, dim: int = 0) -> Tensor:
    '''
    Add an operation that splits a tensor into sub-tensors.

    This operation creates a list of tensors that are obtained from the input
    tensor by chunking it along the dimension 'dim'. It produces 'chunks'
    sub-tensors.

    That operation is only defined for static tensors (no dynamic dimension)
    and the size of the tensor in the dimension 'dim' must be a multiple of
    'chunks': 'input.shape[dim] % chunks == 0'.

    It maps to 'split' with 'split_size = input.shape[dim] / chunks'.

    Parameters:
        tensor : Tensor
            The input tensor to slice.

        chunks : int
            The number of slices to split the input tensor into.

        dim : int
            The dimension of the tensor to slice.

    Returns:
        The list of tensors produced by the different operations.
    '''
    assert not tensor.is_dynamic(dim)

    ndim = tensor.ndim()
    if dim < 0:
        dim += ndim
    dim_value = tensor.size()[dim]
    assert dim_value % chunks == 0

    return split(tensor, dim_value // chunks, dim)


def unbind(input: Tensor, dim: int = 0):
    '''
    Removes a tensor dimension.

    Returns a tuple of all slices along a given dimension, already without it.
    '''
    ndim = input.ndim()
    outputs = split(input, 1, dim)
    output_shape = [input.shape[i] for i in range(ndim) if i != dim]
    return [output.view(output_shape) for output in outputs]


def assertion(condition: Tensor, message: str = '') -> None:
    default_trtnet().add_assertion(condition.trt_tensor, message)


def layer_norm(input: Tensor,
               normalized_shape: Union[int, Tuple[int]],
               weight: Optional[Tensor] = None,
               bias: Optional[Tensor] = None,
               eps: float = 1e-05,
               use_diff_of_squares: bool = True) -> Tensor:
    '''
    Add a layer-norm operation on a tensor.

    That operation applies the layer-normalization to its input tensor. In its
    simplest form, for large language models, the 'normalized_shape' should be
    set to the hidden dimension of the activation tensor. Otherwise, it is the
    shape of the normalized fraction of the tensor (starting from the
    right-most dimension).

    The 'weight' tensor corresponds to 'gamma' in the layer-norm formula and
    'bias' is 'beta'. The 'eps' value is added to the variance before computing
    the squared-root.

    This implementation (when using the plugin) supports an additional flag to
    enable/disable the use of a difference of squares ('Var = Mean(X^2) -
    Mean(X)^2').

    Parameters:
        input : Tensor
            The tensor to normalize.

        normalized_shape : Union[int, Tuple[int]]
            The shape of the sub-tensor that is normalized. Use 'hidden_dim' to
            normalize the inner-most dimension of an activation tensor in LLMs.

        weight : Optional[Tensor] = None
            The 'gamma' term in layer-norm. Its shape must be
            'normalized_shape'.

        bias : Optional[Tensor] = None
            The 'beta' term in layer-norm. Its shape must be
            'normalized_shape'.

        eps : float
            The epsilon term to be added to the variance in the squared-root.

        use_diff_of_squares : bool
            Does the plugin use the difference of squares to compute the
            variance?

    Returns:
        The output tensor of that operation.
    '''
    input, weight = broadcast_helper(input, weight)
    input, bias = broadcast_helper(input, bias)
    if isinstance(normalized_shape, int):  # FIXME: better way?
        axis = input.ndim() - 1
    else:
        axis = input.ndim() - len(normalized_shape)
    axes_mask = 0
    for i in range(axis, input.ndim()):
        axes_mask |= 1 << i
    layer = default_trtnet().add_normalization(input.trt_tensor,
                                               weight.trt_tensor,
                                               bias.trt_tensor, axes_mask)
    layer.epsilon = eps
    return _create_tensor(layer.get_output(0), layer)


def rms_norm(input: Tensor,
             normalized_shape: Union[int, Tuple[int]],
             num_groups: int = 1,
             weight: Optional[Tensor] = None,
             eps: float = 1e-06) -> Tensor:
    '''
    Add a RMS norm operation on a tensor.

    That operation applies the rms-normalization to its input tensor. In its
    simplest form, for large language models, the 'normalized_shape' should be
    set to the hidden dimension of the activation tensor. Otherwise, it is the
    shape of the normalized fraction of the tensor (starting from the
    right-most dimension).

    The 'weight' tensor corresponds to 'gamma' in the rms-norm formula.
    The 'eps' value is added to the variance before computing the squared-root.

    Parameters:
        input: Tensor
            The tensor to normalize.

        normalized_shape : Union[int, Tuple[int]]
            The shape of the sub-tensor that is normalized. Use 'hidden_dim' to
            normalize the inner-most dimension of an activation tensor in LLMs.

        num_groups: int = 1
            The group size.

        weight : Optional[Tensor] = None
            The 'gamma' term in layer-norm. Its shape must be
            'normalized_shape'.

        eps : float
            The epsilon term to be added to the variance in the squared-root.weig
    Returns:
        The output tensor of that operation.
    '''
    normalized_shape = [normalized_shape] if isinstance(
        normalized_shape, int) else normalized_shape

    dim = tuple([-i - 1 for i in range(len(normalized_shape))])

    if num_groups > 1:
        assert len(normalized_shape) == 1
        num_channels = input.size()[-1]
        ndim = input.ndim()
        old_shape = shape(input)
        new_shape = concat([input.size(i) for i in range(ndim - 1)] +
                           [num_groups, num_channels // num_groups])
        input = input.view(new_shape)

    with precision("float32"):
        input_dtype = input.dtype
        fp32_input = cast(input, "float32")
        varx = pow(fp32_input, 2.0)

        varx = varx.mean(dim=dim, keepdim=True)
        denom = varx + eps
        denom = denom.sqrt()
        fp32_y = fp32_input / denom
        y = cast(fp32_y, input_dtype)

    if num_groups > 1:
        y = y.view(old_shape)

    if weight is not None:
        y = y * weight

    return y


def rearrange(inputs: Union[Tensor, Sequence[Tensor]], expression: str,
              **kwargs) -> Tensor:
    '''
    Add a rearrange operation on a tensor.

    This operation is a reader-friendly smart element reordering for multidimensional tensors,
    including functionality of transpose (axes permutation), reshape (view), squeeze, unsqueeze,
    stack, concatenate and other operations. Please see: https://einops.rocks/api/rearrange/

    For example, if the shape of input tensor is [32, 30, 40, 3], and run:
        `rearrange(x, 'b (h h1) (w w1) c -> b h w 1 (c h1 w1) 1', h1=2, w1=2)`
    it would produce a tensor with shape as [32, 15, 20, 1, 12, 1].

    Parameters:
        input: Union[Tensor, Sequence[Tensor]]
            If it is a tensor, it will directly operate on it.
            Otherwise, if it is a sequence, it will concat it to a tensor and then
            operates on it.

        expression : str
            The expression about how to reorder the tensor in a reader-friendly way.

        kwargs:
            Keyword arguments to set some identifiers with specific values.

    Returns:
        The output tensor of this operation.
    '''
    import re

    def _init_expression(expr):
        expr_items = expr.split(" ")
        tmp_name_index = 0
        for idx, item in enumerate(expr_items):
            values = re.findall(r'\b\d+\b', item)
            if len(values) > 0:
                prefix = "(" if "(" in item else ""
                subfix = ")" if ")" in item else ""
                expr_items[
                    idx] = f"{prefix}NumericId{tmp_name_index}Val{values[0]}{subfix}"
                tmp_name_index += 1
        return " ".join(expr_items)

    def _get_all_identifier(expr):
        return re.findall(r'\b[a-zA-Z_]+\d*\b', expr)

    def _get_all_symbols(expr):
        return re.findall(r'\b\w+\b', expr)

    def _get_dim_expr(expr):
        return [
            _get_all_symbols(match.group())
            for match in re.finditer(r'\b\w+\b|\(.*?\)', expr)
        ]

    src_shape_expr, _, dst_shape_expr = expression.partition("->")
    unknown_identifiers = re.findall(r'[^a-zA-Z0-9_\(\)]',
                                     src_shape_expr + dst_shape_expr)
    assert len(
        unknown_identifiers) > 0, f"Unknown identifiers: {unknown_identifiers}"
    src_identifiers = _get_all_identifier(src_shape_expr)
    dst_identifiers = _get_all_identifier(dst_shape_expr)
    assert (len(src_identifiers) == len(set(src_identifiers))
            and len(dst_identifiers) == len(set(dst_identifiers))
            ), "Indexing expression contains duplicate dimension."
    assert (
        set(src_identifiers) == set(dst_identifiers)
    ), "Identifiers only on one side of expression (should be on both)."

    new_expression = _init_expression(expression)
    src_shape_expr, _, dst_shape_expr = new_expression.partition("->")

    # concat if inputs are sequence of tensors
    if isinstance(inputs, Sequence):
        inputs = concat([unsqueeze(t, 0) for t in inputs], dim=0)
    assert (
        inputs.ndim() == len(_get_dim_expr(src_shape_expr))
    ), f"inputs.ndim() is {inputs.ndim()} while indexing expression has {len(_get_dim_expr(src_shape_expr))}"

    src_symbols = _get_all_symbols(src_shape_expr)
    dst_symbols = _get_all_symbols(dst_shape_expr)

    # find all the symbols-values mapping and store them in symbol_map
    symbol_map = {
        symbol: {
            "updated": False,
            "value": None
        }
        for symbol in set(src_symbols + dst_symbols)
    }
    for symbol in symbol_map:
        if "NumericId" in symbol:
            symbol_map[symbol]["value"] = int(symbol.partition("Val")[-1])
            symbol_map[symbol]["updated"] = True
    for symbol, value in kwargs.items():
        symbol_map[symbol]["value"] = value
        symbol_map[symbol]["updated"] = True

    for idx, dim_expr in enumerate(_get_dim_expr(src_shape_expr)):
        if len(dim_expr) == 1:
            symbol = dim_expr[0]
            if not symbol_map[symbol]["updated"]:
                symbol_map[symbol]["value"] = shape(inputs, idx)
                symbol_map[symbol]["updated"] = True
        else:
            divisors = []
            unknown_symbol = None
            for symbol in dim_expr:
                if not symbol_map[symbol]["updated"]:
                    unknown_symbol = symbol
                else:
                    divisors.append(symbol_map[symbol]["value"])
            if unknown_symbol is not None:
                assert len(divisors) > 0
                divisor = prod(cast(concat(divisors), "int64"), dim=-1)
                symbol_map[unknown_symbol]["value"] = shape(inputs,
                                                            idx) / divisor
                symbol_map[unknown_symbol]["updated"] = True

    for symbol, item in symbol_map.items():
        assert (item["updated"]
                ), f"{symbol} cannot be inferred, please set it manually"

    dst_dims = []
    for dim_expr in _get_dim_expr(dst_shape_expr):
        if len(dim_expr) == 1:
            dst_dims.append(symbol_map[dim_expr[0]]["value"])
        else:
            accumulator = prod(cast(
                concat([symbol_map[symbol]["value"] for symbol in dim_expr]),
                "int64"),
                               dim=-1)
            dst_dims.append(accumulator)
    dst_dims = cast(concat(dst_dims, dim=-1), "int64")

    src_indices = {symbol: idx for idx, symbol in enumerate(src_identifiers)}
    permute_dims = [src_indices[symbol] for symbol in dst_identifiers]

    symbol_shape = cast(
        concat([symbol_map[symbol]["value"] for symbol in src_identifiers],
               dim=-1), "int64")
    tensor = inputs.view(symbol_shape)
    tensor = permute(tensor, permute_dims)
    tensor = tensor.view(dst_dims)
    return tensor


def repeat(input: Tensor, sizes: Sequence[int]) -> Tensor:
    '''
    Repeats the tensor along the specified dimensions.

    Parameters:
        input : Tensor
            The tensor to be repeated.
        sizes : Sequence[int]
            The number of times to repeat the tensor along each dimension.

    Returns:
        A tensor except for repeated input tensors along specified dim.

    '''
    assert input.ndim() <= len(sizes), \
        "Number of dimensions of repeat dims can not be smaller than number of dimensions of tensor"
    repeated_tensor = input
    for k in range(-1, -len(sizes) - 1, -1):
        repeated_tensor = concat([repeated_tensor] * sizes[k], dim=k)
    return repeated_tensor


def repeat_interleave(tensor: Tensor, repeats: int, dim: int) -> Tensor:
    '''
    Repeats elements of a tensor along an axis.

    Parameters:
        repeats : int
            The number of repetitions along axis specified.
        dim : int
            The dimension along which repetitions are performed.

    Returns:
        A tensor with the same shape as input except for repeated elements along specified dim.

    TODO: Allow repeats to be a list of integers and dim to be unspecified.
    '''
    expanded_tensor = expand_dims(tensor, dim + 1)
    tile_output_size = concat([
        repeats if i == (dim + 1) else shape(expanded_tensor, i)
        for i in range(expanded_tensor.ndim())
    ])
    tile = expand(expanded_tensor, tile_output_size)
    tile_reshape_size = [shape(tensor, i) for i in range(tensor.ndim())]
    tile_reshape_size[dim] = tile_reshape_size[dim] * repeats
    tensor = tile.view(concat(tile_reshape_size))
    return tensor


def meshgrid2d(x: Tensor, y: Tensor) -> Tuple[Tensor]:
    '''
    Creates grids (2D) of coordinates specified by the 1D inputs (only supports `indexing=\'xy\'`).

    Parameters:
        x : Tensor
            The first input (1D) tensor.
        y : Tensor
            The second input (1D) tensor.

    Returns:
        The tuple of two tensors produced.

    TODO: Add full support for torch.meshgrid.
          See https://pytorch.org/docs/stable/generated/torch.meshgrid.html#torch-meshgrid
    '''
    if x.ndim() == 1:
        x = expand_dims(x, 0)
    if y.ndim() == 1:
        y = expand_dims(y, 0)
    grid_x = repeat_interleave(x, shape(y, 1),
                               1).view([x.shape[-1], y.shape[-1]])
    grid_y = repeat(y, (x.shape[-1], 1))
    return (grid_x, grid_y)


def generate_logn_scaling(seq_length: int = 8192,
                          max_position_embeddings: int = 32768) -> np.ndarray:
    '''
    Compute the Log-N scaling vector for Qwen inference extrapolation

    Parameters:
        seq_length : int
            The max seq length in training (default to 8192 in Qwen-1)
        max_position_embeddings : int
            The max position embeddings. (default to 32768 in Qwen-1)

    Returns:
        A constant np.ndarray that contains logn scaling vector
    '''
    logn_list = [
        math.log(i, seq_length) if i > seq_length else 1
        for i in range(1, max_position_embeddings + 1)
    ]
    return np.asarray(logn_list, dtype=np.float32)


def generate_alibi_slopes(num_heads: int,
                          tp_size: int = 1,
                          tp_rank: int = 0,
                          alibi_scale: float = 1.0,
                          alibi_bias_max: int = 8) -> np.ndarray:
    '''
    Compute the ALiBi slopes as described in https://arxiv.org/abs/2211.05100.

    Parameters:
        num_heads : int
            The number of heads.
        dtype : trt.DataType
            The data type of the returned slopes
        tp_size : int
            The tensor parallelism size
        tp_rank : int
            The tensor parallelism rank

    Returns:
        A constant tensor that contains the ALiBi slopes.
    '''
    start_head_id = 0
    end_head_id = num_heads

    if tp_size > 1:
        rank_heads = num_heads // tp_size
        start_head_id = rank_heads * tp_rank
        end_head_id = start_head_id + rank_heads

    closest_power_of_2 = 2**np.floor(np.log2(num_heads))
    # FT's implementation
    # https://github.com/NVIDIA/FasterTransformer/blob/main/src/fastertransformer/kernels/gen_relative_pos_bias.cu#L248
    slopes_ft = []
    for h_id in range(start_head_id, end_head_id):
        if h_id < closest_power_of_2:
            slopes_ft.append(
                np.power(
                    2**(-(2**-(np.log2(closest_power_of_2) -
                               np.log2(alibi_bias_max)))), h_id + 1))
        else:
            slopes_ft.append(
                np.power(
                    2**(-(2**-(np.log2(closest_power_of_2 * 2) -
                               np.log2(alibi_bias_max)))),
                    (h_id - closest_power_of_2) * 2 + 1))
    slopes = np.asarray(slopes_ft, dtype=np.float32)

    slopes = alibi_scale * slopes
    slopes = slopes.reshape(1, (end_head_id - start_head_id), 1, 1)
    return slopes


def generate_alibi_biases(slopes: Tensor, key_length: Tensor) -> Tensor:
    '''
    Compute the ALiBi biases as described in https://arxiv.org/abs/2211.05100.

    The ALiBi biases are added to the result of the Q*K^T product in the
    multi-head attention block.

    Parameters:
        slopes : Tensor
            The slopes.

        key_length : Tensor
            The size of the K vector per head.

    Returns:
        A constant tensor that contains the ALiBi biases.
    '''
    # We don't need to care about the batch size or query length since we can just broadcast
    # across the batch and query dimensions

    trt_0 = constant(int32_array(0))
    arange_shape = concat([1, 1, 1, key_length])

    arange_tensor = arange(trt_0, key_length, "float32").view(arange_shape)
    return slopes * arange_tensor


def expand_mask(mask: Tensor, tgt_len: Optional[Tensor] = None) -> Tensor:
    '''
    Expand an attention mask.

    That function adds the sequence of operations to expand from a tensor of
    shape '[batch_size, src_seq_len]' to a tensor of shape
    '[batch_size, 1, tgt_seq_len, src_seq_len]'. It can be used to create the
    mask applied to the Q*K^T product before the softmax operation in the
    multi-head attention block.

    Parameters:
        mask : Tensor
            The input mask

        tgt_len : Optional[Tensor]
            The dimension of the 3rd dimension in the output tensor. If None,
            the 2nd dimension of the input is used.

    Returns:
        The tensor created by that sequence of operations.
    '''
    bsz = shape(mask, 0)
    src_len = shape(mask, 1)
    tgt_len = tgt_len if tgt_len is not None else src_len

    mask = mask.view(concat([bsz, 1, 1, src_len]))

    mask = expand(mask, concat([bsz, 1, tgt_len, src_len]))
    mask = where(mask == 0, float('-inf'), 0.0)
    return mask


def gather_last_token_logits(hidden_states: Tensor, last_token_ids: Tensor,
                             remove_input_padding: bool) -> Tensor:
    '''
    Extract the logits that correspond to the last token from the hidden states.

    That function adds the operations to extract the logits of the last tokens
    in a batch of sequences.

    Depending on whether 'remove_input_padding' is 'True' or 'False', that
    function assumes inputs of different shapes.

    When 'remove_input_padding' is 'True', the 'hidden_states' tensor is
    assumed to be packed. It has a shape '[num_tokens, hidden_dim]' where
    'num_tokens' is the sum of the lengths of the sequences in the batch and
    'hidden_dim' is the hidden dimension. The 'last_tokens_ids' is a 1D tensor
    that encodes the inclusive prefix-sums of the lengths of the sequences in
    the batch.

    When 'remove_input_padding' is 'False', the 'hidden_states' tensor is
    assumed to be padded. It has a shape '[batch_size, max_seqlen, hidden_dim]'
    where 'max_seqlen' is the length of the longest sequence in the batch and
    'hidden_dim' is the hidden dimension.  The 'last_token_ids' is a 1D tensor
    that encodes the length of each sequence in the batch.

    In both cases, that function produces a tensor of shape '[batch_size,
    hidden_size]' where the row at index 'i' corresponds to the logits of the
    last token from the 'i'-th sequence.

    Parameters:
        hidden_states : Tensor
            The hidden states

        last_token_ids : Tensor
            The inclusive prefix-sum of the lengths or the lengths of the
            sequences in the batch.

        remove_input_padding : bool
            Indicate if the hidden_states are packed ('True') or padded
            ('False').

    Returns:
        The tensor created by that sequence of operations.
    '''
    if last_token_ids is None:
        return hidden_states

    if remove_input_padding:
        hidden_states = index_select(hidden_states, 0,
                                     last_token_ids - 1)  # [seq_len, hidden]

        hidden_states = hidden_states.view(
            concat([shape(last_token_ids, 0),
                    shape(hidden_states, 1)]))
    else:
        ndim = last_token_ids.ndim()
        if ndim == 1:
            # only calculate logits for the last token
            # [batch_size, seqlen, hidden_size] -> [batch_size, hidden_size]
            last_token_ids = last_token_ids.view(
                concat([shape(last_token_ids, 0), 1, 1]))
            last_token_ids = expand(
                last_token_ids,
                concat([shape(last_token_ids, 0), 1,
                        shape(hidden_states, 2)]))
            last_token_ids = last_token_ids - 1
            hidden_states = gather(
                hidden_states, dim=1, indices=last_token_ids).view(
                    concat([shape(hidden_states, 0),
                            shape(hidden_states, 2)]))
        elif ndim == 2:  # speculative decoding needs last few token's logits
            # last_token_ids is of shape [batch_size, num_last_tokens]
            # So [batch_size, seqlen, hidden_size] -> [batch_size, num_last_tokens, hidden_size]
            last_token_ids = last_token_ids.view(
                concat([shape(last_token_ids, 0),
                        shape(last_token_ids, 1), 1]))
            last_token_ids = expand(
                last_token_ids,
                concat([
                    shape(last_token_ids, 0),
                    shape(last_token_ids, 1),
                    shape(hidden_states, 2)
                ]))
            hidden_states = gather(hidden_states,
                                   dim=1,
                                   indices=last_token_ids)
    return hidden_states


ACT2FN = {
    'relu': relu,
    'tanh': tanh,
    'gelu': gelu,
    'gelu_new': gelu,
    'gelu_fast': gelu,
    'gelu_pytorch_tanh': gelu,
    'openai-gelu': gelu,
    'geglu': geglu,
    'gegelu': gegelu,
    'silu': silu,
    'softplus': softplus,
    'relu2': squared_relu,
    'squared-relu': squared_relu,
    'swiglu': swiglu,
    'fast-swiglu': swiglu,
    'sigmoid': sigmoid,
    'quick_gelu': quick_gelu,
}

GATED_ACT_2_ACT = {
    'swiglu': 'silu',
    'fast-swiglu': 'silu',
    'geglu': 'gelu',
}


def is_gated_activation(activation):
    '''
    Is a given activation function gated?

    Parameters:
        activation : str
            The name of the activation function.

    Returns:
        True if the function is gated, False otherwise.
    '''
    assert activation in ACT2FN
    return activation in GATED_ACT_2_ACT


def non_gated_version(activation):
    '''
    Given an activation function, get the non-gated version.

    If the activation function is non-gated, it returns the same activation
    function name.

    For example, that function returns 'silu' for 'swiglu' and 'relu' for
    'relu'.

    Parameters:
        activation : str
            The name of the activation function.

    Returns:
        The name of the non-gated activation function.
    '''
    if is_gated_activation(activation):
        return GATED_ACT_2_ACT[activation]
    return activation


def scatter_nd(input: Tensor, mask: Tensor, source: Tensor) -> Tensor:
    '''
    Scatter_nd is a tensor operation that writes or updates values in a tensor based on indices.

    Parameters:
        input: Tensor
            The input tensor to be updated
        mask: Tensor
            A tensor of indices specifying the locations in data to be updated.
        source: Tensor
            A tensor of values to be written or scattered into data.
    Returns:
        New tensor with the same shape as the input tensor data,
        where the values from the source tensor are scattered or written into the output tensor
        at the locations specified by the mask tensor.
    '''
    scatter_layer = default_trtnet().add_scatter(input.trt_tensor,
                                                 mask.trt_tensor,
                                                 source.trt_tensor,
                                                 mode=trt.ScatterMode.ND)
    return _create_tensor(scatter_layer.get_output(0), scatter_layer)


class SideStreamIDType(IntEnum):
    disable = 0
    moe = 1
