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
import gc
import inspect
import json
import linecache
import os
import struct
import trace
import weakref
from contextlib import contextmanager
from dataclasses import asdict
from enum import EnumMeta
from functools import lru_cache, partial, wraps
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import nvtx
from packaging import version

# isort: off
import torch
import tensorrt as trt
# isort: on

from tensorrt_llm_lite.logger import logger

# numpy doesn't know bfloat16, define abstract binary type instead
np_bfloat16 = np.dtype('V2', metadata={"dtype": "bfloat16"})
np_float8 = np.dtype('V1', metadata={"dtype": "float8"})


def torch_to_numpy(x: torch.Tensor):
    assert isinstance(x, torch.Tensor), \
        f'x must be a torch.Tensor object, but got {type(x)}.'
    if x.dtype == torch.bfloat16:
        return x.view(torch.int16).detach().cpu().numpy().view(np_bfloat16)
    elif x.dtype == torch.float8_e4m3fn:
        return x.view(torch.int8).detach().cpu().numpy().view(np_float8)
    else:
        return x.detach().cpu().numpy()


def numpy_to_torch(x):
    if x.dtype == np_bfloat16:
        return torch.from_numpy(x.view(np.int16)).view(torch.bfloat16)
    elif x.dtype == np_float8:
        return torch.from_numpy(x.view(np.int8)).view(torch.float8_e4m3fn)
    else:
        return torch.from_numpy(x)


def numpy_to_dtype(x, dtype: str):
    if str_dtype_to_np(dtype) == x.dtype:
        return x
    if x.dtype not in [np_bfloat16, np_float8
                       ] and dtype not in ['bfloat16', 'fp8']:
        return x.astype(str_dtype_to_np(dtype))
    else:
        return torch_to_numpy(numpy_to_torch(x).to(str_dtype_to_torch(dtype)))


fp32_array = partial(np.array, dtype=np.float32)
fp16_array = partial(np.array, dtype=np.float16)
int32_array = partial(np.array, dtype=np.int32)
int64_array = partial(np.array, dtype=np.int64)
bool_array = partial(np.array, dtype=np.bool_)


def dims_array(x):
    is_int64_dims = True
    try:
        trt.Dims([np.iinfo(np.int64).max])
    except TypeError:
        is_int64_dims = False
    return int64_array(x) if is_int64_dims else int32_array(x)


def bf16_array(x):
    x = torch.tensor(x, dtype=torch.bfloat16)
    x = torch_to_numpy(x)
    return x


def numpy_array(data, trt_dtype):
    # convenient wrapper due to numpy not support bf16 yet
    if trt_dtype == trt.bfloat16:
        return bf16_array(data)
    return np.array(data, trt_dtype_to_np(trt_dtype))


def copy_torch_to_numpy(x: torch.Tensor, ndarray: np.array):
    if x.dtype == torch.bfloat16:
        torch.from_numpy(ndarray.view(np.int16)).copy_(x.view(torch.int16))
    elif x.dtype == torch.float8_e4m3fn:
        torch.from_numpy(ndarray.view(np.int8)).copy_(x.view(torch.int8))
    else:
        torch.from_numpy(ndarray).copy_(x)
    return ndarray


def trt_version():
    return trt.__version__


def trt_gte(major: int, minor: int = 0):
    """
    Check if TRT version is greater than or equal to major.minor
    """
    trt_ver = version.parse(trt_version())
    return trt_ver.major >= major and trt_ver.minor >= minor


def torch_version():
    return torch.__version__


_str_to_np_dict = dict(
    float16=np.float16,
    float32=np.float32,
    int64=np.int64,
    int32=np.int32,
    int8=np.int8,
    bool=np.bool_,
    bfloat16=np_bfloat16,
    fp8=np_float8,
)


def str_dtype_to_np(dtype):
    ret = _str_to_np_dict.get(dtype)
    assert ret is not None, f'Unsupported dtype: {dtype}'
    return ret


_str_to_torch_dtype_dict = dict(
    bfloat16=torch.bfloat16,
    float16=torch.float16,
    float32=torch.float32,
    int64=torch.int64,
    int32=torch.int32,
    int8=torch.int8,
    bool=torch.bool,
    fp8=torch.float8_e4m3fn,
)


def str_dtype_to_torch(dtype):
    ret = _str_to_torch_dtype_dict.get(dtype)
    assert ret is not None, f'Unsupported dtype: {dtype}'
    return ret


_torch_dtype_to_str_dict = {v: k for k, v in _str_to_torch_dtype_dict.items()}


def torch_dtype_to_str(dtype):
    return _torch_dtype_to_str_dict[dtype]


_str_to_trt_dtype_dict = dict(float16=trt.float16,
                              float32=trt.float32,
                              int64=trt.int64,
                              int32=trt.int32,
                              int8=trt.int8,
                              bool=trt.bool,
                              bfloat16=trt.bfloat16,
                              fp8=trt.fp8,
                              nvfp4=trt.fp4)


def str_dtype_to_trt(dtype):
    if dtype == "fp4":
        # Special handling for FP4 since CI's trt version is not recent enough.
        if not hasattr(trt, 'fp4'):
            raise ValueError(
                "fp4 unsupported, trt version needs to be upgraded.")
        return trt.fp4

    ret = _str_to_trt_dtype_dict.get(dtype)
    assert ret is not None, f'Unsupported dtype: {dtype}'
    return ret


_trt_to_str_dtype_dict = {v: k for k, v in _str_to_trt_dtype_dict.items()}


def trt_dtype_to_str(dtype: trt.DataType) -> str:
    assert isinstance(dtype, trt.DataType)
    return _trt_to_str_dtype_dict[dtype]


_np_to_trt_dtype_dict = {
    np.int8: trt.int8,
    np.int32: trt.int32,
    np.int64: trt.int64,
    np.float16: trt.float16,
    np.float32: trt.float32,
    np.bool_: trt.bool,

    # hash of np.dtype('int32') != np.int32
    np.dtype('int8'): trt.int8,
    np.dtype('int32'): trt.int32,
    np.dtype('int64'): trt.int64,
    np.dtype('float16'): trt.float16,
    np.dtype('float32'): trt.float32,
    np.dtype('bool'): trt.bool,
    np_bfloat16: trt.bfloat16,
    np_float8: trt.fp8,
}


def np_dtype_to_trt(dtype):
    ret = _np_to_trt_dtype_dict.get(dtype)
    assert ret is not None, f'Unsupported dtype: {dtype}'
    return ret


_trt_to_np_dtype_dict = {
    trt.int8: np.int8,
    trt.int32: np.int32,
    trt.int64: np.int64,
    trt.float16: np.float16,
    trt.float32: np.float32,
    trt.bool: np.bool_,
    trt.bfloat16: np_bfloat16,
    trt.fp8: np_float8,
}


def trt_dtype_to_np(dtype):
    ret = _trt_to_np_dtype_dict.get(dtype)
    assert ret is not None, f'Unsupported dtype: {dtype}'
    return ret


_torch_to_np_dtype_dict = {
    torch.bool: np.bool_,
    torch.uint8: np.uint8,
    torch.int8: np.int8,
    torch.int16: np.int16,
    torch.int32: np.int32,
    torch.int64: np.int64,
    torch.float16: np.float16,
    torch.bfloat16: np_bfloat16,
    torch.float8_e4m3fn: np_float8,
    torch.float32: np.float32,
    torch.float64: np.float64,
    torch.complex64: np.complex64,
    torch.complex128: np.complex128,
}


def torch_dtype_to_np(dtype):
    ret = _torch_to_np_dtype_dict.get(dtype)
    assert ret is not None, f'Unsupported dtype: {dtype}'
    return ret


_np_to_torch_dtype_dict = {
    np.bool_: torch.bool,
    np.uint8: torch.uint8,
    np.int8: torch.int8,
    np.int16: torch.int16,
    np.int32: torch.int32,
    np.int64: torch.int64,
    np.float16: torch.float16,
    np_bfloat16: torch.bfloat16,
    np_float8: torch.float8_e4m3fn,
    np.float32: torch.float32,
    np.float64: torch.float64,
    np.complex64: torch.complex64,
    np.complex128: torch.complex128,
}


def np_dtype_to_torch(dtype):
    ret = _np_to_torch_dtype_dict.get(dtype)
    assert ret is not None, f'Unsupported dtype: {dtype}'
    return ret


_trt_to_torch_dtype_dict = {
    trt.float16: torch.float16,
    trt.float32: torch.float32,
    trt.int64: torch.int64,
    trt.int32: torch.int32,
    trt.int8: torch.int8,
    trt.bool: torch.bool,
    trt.bfloat16: torch.bfloat16,
    trt.fp8: torch.float8_e4m3fn,
}


def trt_dtype_to_torch(dtype):
    ret = _trt_to_torch_dtype_dict.get(dtype)
    assert ret is not None, f'Unsupported dtype: {dtype}'
    return ret


def is_same_dtype(type_a: Union[str, trt.DataType],
                  type_b: Union[str, trt.DataType]) -> bool:
    if isinstance(type_a, str):
        type_a = str_dtype_to_trt(type_a)

    if isinstance(type_b, str):
        type_b = str_dtype_to_trt(type_b)

    return type_a == type_b


_torch_to_trt_dtype_dict = {
    torch.float16: trt.float16,
    torch.float32: trt.float32,
    torch.int64: trt.int64,
    torch.int32: trt.int32,
    torch.int8: trt.int8,
    torch.float8_e4m3fn: trt.fp8,
    torch.qint8: trt.int8,
    torch.bool: trt.bool,
    torch.bfloat16: trt.bfloat16
}


def torch_dtype_to_trt(dtype):
    ret = _torch_to_trt_dtype_dict.get(dtype)
    assert ret is not None, f'Unsupported dtype: {dtype}'
    return ret


_torch_dtype_to_np_typestr_dict = {
    torch.float16: "<f2",
    torch.float32: "<f4",
    torch.int64: "<i8",
    torch.int32: "<i4",
    torch.int8: "|i1",
    torch.float8_e4m3fn: "|i1",
    torch.qint8: "|u1",
    torch.bool: "|b1",
    torch.bfloat16: "<f2",
}


def torch_dtype_to_np_typestr(dtype):
    ret = _torch_dtype_to_np_typestr_dict.get(dtype)
    assert ret is not None, f'Unsupported dtype: {dtype}'
    return ret


def dim_to_trt_axes(dim):
    """Converts torch dim, or tuple of dims to a tensorrt axes bitmask"""
    if not isinstance(dim, tuple):
        dim = (dim, )

    # create axes bitmask for reduce layer
    axes = 0
    for d in dim:
        axes |= 1 << d

    return axes


def trt_axes_to_dim(axes: int) -> List[int]:
    """Converts tensorrt axes bitmask to dims"""
    dim = []
    for i in range(32):
        if axes & (1 << i):
            dim.append(i)

    return dim


def dim_resolve_negative(dim, ndim):
    if not isinstance(dim, tuple):
        dim = (dim, )
    pos = []
    for d in dim:
        if d < 0:
            d = ndim + d
        pos.append(d)
    return tuple(pos)


def numpy_fp32_to_bf16(src):
    # Numpy doesn't support bfloat16 type
    # Convert float32 to bfloat16 manually and assign with bf16 abstract type
    original_shape = src.shape
    src = src.flatten()
    src = np.ascontiguousarray(src)

    assert src.dtype == np.float32
    dst = np.empty_like(src, dtype=np.uint16)
    for i in range(len(dst)):
        bytes = struct.pack('<f', src[i])
        dst[i] = struct.unpack('<H', struct.pack('BB', bytes[2], bytes[3]))[0]
    return dst.reshape(original_shape).view(np_bfloat16)


_extra_attrs_by_object: Dict[int, Dict[str, Any]] = {}


def get_extra_attr(obj, attr_name):
    if id(obj) not in _extra_attrs_by_object:
        return None
    extra_attrs = _extra_attrs_by_object[id(obj)]
    return extra_attrs.get(attr_name)


def _clean_extra_attrs(obj_id):
    if obj_id in _extra_attrs_by_object:
        del _extra_attrs_by_object[obj_id]


def set_extra_attr(obj, attr_name, value):
    if id(obj) not in _extra_attrs_by_object:
        _extra_attrs_by_object[id(obj)] = {}
        weakref.finalize(obj, _clean_extra_attrs, id(obj))
    _extra_attrs_by_object[id(obj)][attr_name] = value


def has_extra_attr(obj, attr_name):
    if id(obj) not in _extra_attrs_by_object:
        return False
    return attr_name in _extra_attrs_by_object[id(obj)]


def set_obj_attrs(
    obj: torch.Tensor,
    ojb_attrs: Optional[Dict[str, Any]],
):
    """Set attributes on a object.

    This method is used to set attributes on a object. This method
    will not overwrite existing attributes.
    """
    if ojb_attrs is None:
        return
    for key, value in ojb_attrs.items():
        assert not hasattr(
            obj, key), (f"Overwriting existing tensor attribute: {key}")
        setattr(obj, key, value)


def get_init_params(obj, cls=None):
    """
    Get all parameters in object's __init__.
    Use cls's __init__ as filter if cls provided.
    """
    names = None
    if cls is not None:
        names = set(list(inspect.signature(cls.__init__).parameters)[1:])
    return {
        name: getattr(obj, name)
        for name in list(inspect.signature(obj.__class__.__init__).parameters)
        [1:] if names is None or name in names
    }


def release_gc():
    ''' Release memory allocated by PyTorch and Python garbage collector explicitly and immediately.
    This could be used when some states might be kept in memory even after the variables are deleted.
    '''
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


@lru_cache(maxsize=1)
def get_sm_version():
    prop = torch.cuda.get_device_properties(0)
    return prop.major * 10 + prop.minor


@lru_cache(maxsize=1)
def is_sm_100f(sm_version=None):
    if sm_version is None:
        sm_version = get_sm_version()
    return sm_version == 100 or sm_version == 103


def is_trace_enabled(env_var: str):
    value = os.environ.get(env_var, "-1")
    if value == "ALL":
        return True
    try:
        return int(value) == global_mpi_rank()
    except ValueError:
        return False


def trace_func(func):

    @wraps(func)
    def wrapper(*args, **kwargs):

        def globaltrace(frame, why, arg):
            if why == "call":
                code = frame.f_code
                filename = frame.f_globals.get('__file__', None)
                if filename:
                    modulename = trace._modname(filename)
                    if modulename is not None:
                        ignore_it = tracer.ignore.names(filename, modulename)
                        if not ignore_it:
                            print(
                                f"[rank{rank}] --- path: {filename} , funcname: {code.co_name}"
                            )
                            return localtrace
                else:
                    return None

        def localtrace(frame, why, arg):
            if why == "line":
                filename = frame.f_code.co_filename
                lineno = frame.f_lineno
                bname = os.path.basename(filename)
                print(
                    f"[rank{rank}] {bname}:{lineno}: {linecache.getline(filename, lineno)}",
                    end="")
            return localtrace

        ignoredirs = [
            os.path.dirname(package.__file__)
            for package in [os, torch, trace]
        ]
        tracer = trace.Trace(trace=1, count=0, ignoredirs=ignoredirs)
        rank = global_mpi_rank()
        tracer.globaltrace = globaltrace
        tracer.localtrace = localtrace
        result = tracer.runfunc(func, *args, **kwargs)
        return result

    return wrapper


class DictConversion:

    @classmethod
    def from_dict(cls, config: Dict[str, Any]):
        obj = cls()
        fields = obj.__dataclass_fields__
        for key, value in config.items():
            assert hasattr(obj, key), f"cannot find {key} in {obj}"
            field_cls = fields[key].type
            if (isinstance(field_cls, type)
                    and issubclass(field_cls, DictConversion)
                    and isinstance(value, dict)):
                value = field_cls.from_dict(value)
            setattr(obj, key, value)
        return obj

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_json_file(cls, file):
        with open(file) as f:
            return cls.from_dict(json.load(f))

    def set_defaults(self, **kwargs):
        for key, default in kwargs.items():
            value = getattr(self, key)
            if (value is None
                    or (isinstance(value, (list, dict)) and len(value) == 0)):
                setattr(self, key, default)


class BaseEnumMeta(EnumMeta):

    def __contains__(cls, item):
        try:
            cls(item)
        except ValueError:
            return False
        return True


PYTHON_DEFAULT_GC_THRESHOLDS = gc.get_threshold()


@contextmanager
def customized_gc_thresholds(gen0_threshold: Optional[int] = None):
    try:
        if gen0_threshold:
            gc.set_threshold(gen0_threshold)
            logger.debug(
                f'Set Python GC threshold to customized value: {gen0_threshold}'
            )
        yield
    finally:
        if gen0_threshold:
            gc.set_threshold(*PYTHON_DEFAULT_GC_THRESHOLDS)
            logger.debug(
                f'Reset Python GC thresholds to default value: {PYTHON_DEFAULT_GC_THRESHOLDS}'
            )


@contextmanager
def _null_context_manager():
    yield


def nvtx_range(msg: str,
               color: str = "grey",
               domain: str = "TensorRT-LLM",
               category: Optional[str] = None):
    """
    Creates an NVTX range annotation for profiling.

    This function returns a context manager that marks the beginning and end of a
    range in NVIDIA Tools Extension (NVTX) profiling tools like Nsight Systems.

    Args:
        msg (str): The message/name for the NVTX range.
        color (str, optional): The color to use for the range in the profiler. Defaults to "grey".
        domain (str, optional): The domain name for the range. Defaults to "TensorRT-LLM".
        category (str, optional): The category for the range. Defaults to None.

    Returns:
        contextmanager: A context manager that marks the NVTX range.
    """
    return nvtx.annotate(msg, color=color, domain=domain, category=category)


def nvtx_range_debug(msg: str,
                     color: str = "grey",
                     domain: str = "TensorRT-LLM",
                     category: Optional[str] = None):
    """
    Creates an NVTX range annotation for debugging purposes.

    Similar to nvtx_range, but only creates the range if specific environment
    variables are set, making it suitable for debug profiling.

    Args:
        msg (str): The message/name for the NVTX range.
        color (str, optional): The color to use for the range in the profiler. Defaults to "grey".
        domain (str, optional): The domain name for the range. Defaults to "TensorRT-LLM".
        category (str, optional): The category for the range. Defaults to None.

    Returns:
        contextmanager: A context manager that either marks the NVTX range if enabled,
                        or a null context manager that does nothing if disabled.
    """
    if os.getenv("TLLM_LLMAPI_ENABLE_NVTX", "0") == "1" or \
            os.getenv("TLLM_NVTX_DEBUG", "0") == "1":
        return nvtx_range(msg, color=color, domain=domain, category=category)
    else:
        return _null_context_manager()


def nvtx_mark(msg: str,
              color: str = "grey",
              domain: str = "TensorRT-LLM",
              category: Optional[str] = None):
    """
    Creates an NVTX marker for profiling.

    This function places a single marker point in NVIDIA Tools Extension (NVTX)
    profiling tools like Nsight Systems, useful for marking specific events.

    Args:
        msg (str): The message/name for the NVTX marker.
        color (str, optional): The color to use for the marker in the profiler. Defaults to "grey".
        domain (str, optional): The domain name for the marker. Defaults to "TensorRT-LLM".
        category (str, optional): The category for the marker. Defaults to None.
    """
    nvtx.mark(msg, color=color, category=category, domain=domain)


def volume(d: Sequence[int]):
    return np.prod(d)
