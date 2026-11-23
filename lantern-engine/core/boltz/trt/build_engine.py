#!/usr/bin/env python3
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
"""
Python replacement for `trtbnm-build` for Boltz2 engines.

Usage (structure pairformer):
    python build_engine.py \\
        --model boltz-2 --module structure_pairformer \\
        --checkpoint_dir /workspace/pairformer_safetensors \\
        --output_dir /workspace/pairformer_engine \\
        --min_seqlen 4 --max_seqlen 1500 \\
        --weakly_dtype bfloat16

Usage (token transformer):
    python build_engine.py \\
        --model boltz-2 --module token_transformer \\
        --checkpoint_dir /workspace/token_transformer_safetensors \\
        --output_dir /workspace/token_transformer_engine \\
        --min_seqlen 4 --max_seqlen 1500 \\
        --weakly_dtype float32
"""

import sys
import os

_THIRDPARTY = os.path.join(os.path.dirname(__file__), '..', '..', 'thirdparty')
sys.path.insert(0, os.path.abspath(_THIRDPARTY))

from tensorrt_bionemo.registry import register_default_building_modules
register_default_building_modules()

from tensorrt_bionemo.commands.build import main

if __name__ == "__main__":
    main()
