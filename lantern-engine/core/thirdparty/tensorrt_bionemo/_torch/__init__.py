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
import importlib.util
from pathlib import Path


def _load_cuequivariance_lib():
    """Load cuEquivariance ops library. Silently skipped if unavailable.
    Required only for CUEQUIV triangle attention backend."""
    CUEQUIV_PKG = importlib.util.find_spec("cuequivariance_ops")
    if CUEQUIV_PKG is not None:
        CUEQUIV_PKG_LIB = Path(
            CUEQUIV_PKG.origin).parent.absolute() / "lib" / "libcue_ops.so"
        if CUEQUIV_PKG_LIB.exists():
            try:
                ctypes.CDLL(CUEQUIV_PKG_LIB.as_posix())
            except OSError:
                pass


def _load_kernels_lib():
    """Load TRT-BioNeMo kernels library. Silently skipped if unavailable.
    Required only for CUEQUIV/TRIFAST backends; VANILLA mode works without it."""
    project_dir = Path(__file__).parent.parent.absolute()
    dyn_lib = "libkernels_tensorrt_bionemo.so"
    KERNELS_LIB = project_dir / "libs" / dyn_lib
    if KERNELS_LIB.exists():
        try:
            ctypes.CDLL(KERNELS_LIB.as_posix())
        except OSError:
            pass


__all__ = [
    "_load_cuequivariance_lib",
    "_load_kernels_lib",
]
