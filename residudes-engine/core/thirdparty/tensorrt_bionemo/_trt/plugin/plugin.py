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
import ctypes
import platform
from pathlib import Path

TRT_BNM_PLUGIN_NAMESPACE = 'tensorrt_bionemo'


def plugin_lib_path() -> str:
    project_dir = Path(__file__).parent.parent.parent.absolute()
    dyn_lib = "libnvinfer_plugin_tensorrt_bionemo.so"
    return str(project_dir.joinpath("libs", dyn_lib))


def _load_plugin_lib():
    """Load the TRT-BioNeMo C++ plugin library (IdentitySZ, TriAttn).

    The .so requires GLIBCXX_3.4.32+ and libnvinfer.so.10. Both are preloaded
    from bundled/pip-installed locations before loading the plugin.
    """
    lib_path = plugin_lib_path()
    if not Path(lib_path).exists():
        return None

    # Preload dependencies before the plugin .so
    libs_dir = Path(lib_path).parent
    trt_libs = Path(__import__('tensorrt').__file__).parent.parent / 'tensorrt_libs'
    # libstdc++ with GLIBCXX_3.4.32: bundled in libs/, fallback to conda
    for stdcxx in [libs_dir / "libstdc++.so.6",
                   Path("/opt/conda/lib/libstdc++.so.6")]:
        if stdcxx.exists():
            try:
                ctypes.CDLL(str(stdcxx), mode=ctypes.RTLD_GLOBAL)
                break
            except OSError:
                continue
    # libnvinfer.so.10 from tensorrt pip package
    nvinfer = trt_libs / "libnvinfer.so.10"
    if nvinfer.exists():
        try:
            ctypes.CDLL(str(nvinfer), mode=ctypes.RTLD_GLOBAL)
        except OSError:
            pass

    try:
        handle = ctypes.CDLL(lib_path, mode=ctypes.RTLD_LOCAL)
    except OSError as e:
        import logging
        logging.getLogger(__name__).warning(f"Failed to load plugin .so: {e}")
        return None

    try:
        handle.initTrtBnmPlugins.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        handle.initTrtBnmPlugins.restype = ctypes.c_bool
        handle.initTrtBnmPlugins(None, TRT_BNM_PLUGIN_NAMESPACE.encode('utf-8'))
    except (AttributeError, Exception):
        pass
    return handle


def _load_cuequivariance_trt_plugins():
    """Register cuEquivariance Python TRT plugins.

    Importing cuequivariance_ops_torch.tensorrt triggers @trtp.register()
    decorators that register plugins (identity_2, triangle_attention_mask,
    etc.) under the 'cuequivariance' namespace. These plugins use the
    add_plugin_v3() API and do NOT require any pre-built .so files.
    """
    try:
        import cuequivariance_ops_torch.tensorrt  # noqa: F401
    except (ImportError, Exception):
        pass


def _load_trifast_trt_plugins():
    """Register trifast Python TRT plugins.

    Importing trifast_plugin triggers @trtp.register() decorators that
    register plugins (identity_2, triangle_attention) under the 'trifast'
    namespace. Uses the trifast Triton kernel for triangle attention.
    """
    try:
        import tensorrt_bionemo._trt.plugin.trifast_plugin  # noqa: F401
    except (ImportError, Exception):
        pass
