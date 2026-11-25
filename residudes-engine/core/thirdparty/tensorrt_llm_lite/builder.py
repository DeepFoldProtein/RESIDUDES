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
import json
import math
import os
import time
from pathlib import Path
from typing import Optional, Union

import numpy as np
import tensorrt as trt

from ._common import _is_building, serialize_engine
from ._utils import trt_gte
from .logger import logger
from .network import Network
from .plugin import PluginConfig


class ConfigEncoder(json.JSONEncoder):

    def default(self, obj):
        if hasattr(obj, 'model_dump'):
            # Handle Pydantic models (including DecodingBaseConfig and subclasses)
            return obj.model_dump(mode='json')
        else:
            return super().default(obj)


class BuilderConfig(object):

    def __init__(self, **kwargs):
        # intentionally use **kwargs, user should never call this ctor directly,
        # use Builder.create_builder_config() instead
        pass

    def _init(self, trt_builder_config, **kwargs):
        self._trt_builder_config = trt_builder_config
        for key, value in kwargs.items():
            setattr(self, key, value)
        return self

    @property
    def trt_builder_config(self) -> trt.IBuilderConfig:
        return self._trt_builder_config

    def to_dict(self) -> dict:
        '''return a dict with keys
        {
            "builder_config": {
                # all key values set by the _init function
            },
            "plugin_config": {
                # the network plugin_config (if any) attached to this BuilderConfig object
                # inside the Builder.build_engine
            },
            "auto_parallel_config": {
                # the network auto_parallel_config (if any) attached to this BuilderConfig object
                # inside the Builder.build_engine
            }
        }
        '''
        config = {'builder_config': {}}
        for k in self.__dict__.keys():
            if k not in [
                    '_trt_builder_config', 'plugin_config',
                    'auto_parallel_config'
            ]:
                config['builder_config'][k] = self.__getattribute__(k)
        if hasattr(self, 'plugin_config'):
            assert isinstance(self.plugin_config, PluginConfig), \
                f"Found unexpected plugin_config object with type: {type(self.plugin_config)}"
            config['plugin_config'] = self.plugin_config.to_dict()
        return config


class Builder():

    _ALLOWED_PRECISIONS = [
        'float32', 'float16', 'bfloat16', trt.DataType.HALF,
        trt.DataType.FLOAT, trt.DataType.BF16
    ]

    def __init__(self):
        super().__init__()
        self._trt_builder = trt.Builder(logger.trt_logger)
        self.strongly_typed = True

    @property
    def trt_builder(self) -> trt.Builder:
        return self._trt_builder

    def create_network(self) -> Network:
        explicit_batch_flag = 0
        # Explicit batch flag will be deprecated in TRT 10
        if "EXPLICIT_BATCH" in trt.NetworkDefinitionCreationFlag.__members__.keys(
        ):
            explicit_batch_flag = 1 << int(
                trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

        if self.strongly_typed:
            return Network()._init(
                self.trt_builder.create_network(
                    explicit_batch_flag
                    | (1 << int(
                        trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))))
        else:
            return Network()._init(
                self.trt_builder.create_network(explicit_batch_flag))

    def create_builder_config(self,
                              precision: Union[str, trt.DataType],
                              timing_cache: Union[str, Path,
                                                  trt.ITimingCache] = None,
                              tensor_parallel: int = 1,
                              use_refit: bool = False,
                              int8: bool = False,
                              strongly_typed: bool = True,
                              force_num_profiles: Optional[int] = None,
                              profiling_verbosity: str = "layer_names_only",
                              use_strip_plan: bool = False,
                              weight_streaming: bool = False,
                              precision_constraints: Optional[str] = "obey",
                              **kwargs) -> BuilderConfig:
        ''' @brief Create a builder config with given precisions and timing cache
            @param precision: one of allowed precisions, defined in Builder._ALLOWED_PRECISIONS
            @param timing_cache: a timing cache object or a path to a timing cache file
            @param tensor_parallel: number of GPUs used for tensor parallel
            @param kwargs: any other arguments users would like to attach to the config object as attributes
            @param refit: set to accelerate multi-gpu building, build engine for 1 gpu and refit for the others
            @param int8: whether to build with int8 enabled or not. Can't be used together with refit option
            @return: A BuilderConfig object, return None if failed
        '''
        self.strongly_typed = self.strongly_typed and strongly_typed

        fp8 = False
        if not strongly_typed and precision not in self._ALLOWED_PRECISIONS:
            logger.error(
                f"precision should be one of {self._ALLOWED_PRECISIONS}")

        config = self.trt_builder.create_builder_config()
        # residudes-engine: bump the TRT WORKSPACE pool so large fused-tactic plans
        # (tri_mul_in, pair_bias_attn, ...) can be admitted on Blackwell-class
        # GPUs where the activation budget exceeds the default workspace.
        #
        # Resolution order:
        #   1. TRT_WORKSPACE_GB=N (env) — explicit override.
        #   2. Auto: ~95% of the active CUDA device's total memory.
        # The auto path queries torch.cuda; if torch isn't ready the call
        # silently falls through and TRT keeps its default budget.
        import os as _os
        _ws_gb = int(_os.environ.get("TRT_WORKSPACE_GB", "0"))
        if _ws_gb <= 0:
            try:
                import torch as _torch
                if _torch.cuda.is_available():
                    _total = _torch.cuda.get_device_properties(
                        _torch.cuda.current_device()).total_memory
                    _ws_gb = int(_total * 0.95) >> 30
            except Exception:
                pass
        if _ws_gb > 0:
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, _ws_gb << 30)
        if weight_streaming:
            config.set_flag(trt.BuilderFlag.WEIGHT_STREAMING)
        if not self.strongly_typed:
            # fp8 = quant_mode.has_fp8_qdq() or quant_mode.has_fp8_kv_cache()
            if precision == 'float16' or precision == trt.DataType.HALF:
                config.set_flag(trt.BuilderFlag.FP16)
                if precision_constraints == 'obey':
                    config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
            elif precision == 'bfloat16' or precision == trt.DataType.BF16:
                config.set_flag(trt.BuilderFlag.BF16)
                if precision_constraints == 'obey':
                    config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
            if int8:
                config.set_flag(trt.BuilderFlag.INT8)
            if fp8:
                config.set_flag(trt.BuilderFlag.FP8)
                if precision_constraints == 'obey':
                    config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)

        if use_refit:
            config.set_flag(trt.BuilderFlag.REFIT)

        # Use fine-grained refit when strip plan is enabled in TRT10.2+.
        if use_strip_plan:
            config.set_flag(trt.BuilderFlag.REFIT_INDIVIDUAL)

        if use_strip_plan:
            config.set_flag(trt.BuilderFlag.STRIP_PLAN)

        # Set TRT Engine profiling verbosity
        if profiling_verbosity == "detailed":
            config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
        elif profiling_verbosity == "none":
            config.profiling_verbosity = trt.ProfilingVerbosity.NONE
        else:
            config.profiling_verbosity = trt.ProfilingVerbosity.LAYER_NAMES_ONLY

        # set timing cache
        cache = None
        if timing_cache is not None:
            # use given cache
            if isinstance(timing_cache, trt.ITimingCache):
                cache = timing_cache
            # read cache from file
            elif isinstance(timing_cache,
                            (str, Path)) and os.path.exists(timing_cache):
                with open(timing_cache, "rb") as f:
                    cache = config.create_timing_cache(f.read())
            else:
                logger.warning(
                    "Invalid timing cache, using freshly created one")
        if cache is None:
            cache = config.create_timing_cache(b"")
        # When user does not given any existing cache, internally always created one
        # so the cache should never None here
        assert cache is not None and isinstance(cache, trt.ITimingCache)
        config.set_timing_cache(cache, ignore_mismatch=False)

        # set weight sparsity
        weight_sparsity = kwargs.get("weight_sparsity", False)
        if weight_sparsity:
            config.set_flag(trt.BuilderFlag.SPARSE_WEIGHTS)

        # TODO: remove this constraint after trt 10.6 is integrated
        if trt_gte(10, 6):
            # set monitor memory
            monitor_memory = kwargs.get("monitor_memory", False)
            if monitor_memory:
                config.set_flag(trt.BuilderFlag.MONITOR_MEMORY)

        return BuilderConfig()._init(config,
                                     precision=precision,
                                     tensor_parallel=tensor_parallel,
                                     use_refit=use_refit,
                                     int8=int8,
                                     force_num_profiles=force_num_profiles,
                                     strongly_typed=self.strongly_typed,
                                     use_strip_plan=use_strip_plan,
                                     **kwargs)

    def _add_optimization_profile(self, network: Network,
                                  builder_config: BuilderConfig):
        assert isinstance(builder_config, BuilderConfig)
        assert isinstance(network, Network)
        input_tensors = network._inputs
        if len(input_tensors) == 0:
            logger.warning("There are no inputs in the network!")
            return
        num_profiles = len(list(input_tensors.values())[0].profiles)
        force_num_profiles = getattr(builder_config, "force_num_profiles",
                                     None)
        for i in range(num_profiles):
            logger.debug(f'Adding optimization profile {i+1}/{num_profiles}')
            profile = self.trt_builder.create_optimization_profile()
            for input_name in input_tensors.keys():
                if len(input_tensors[input_name].profiles) == 0:
                    continue
                shape_profile = input_tensors[input_name].profiles[i]
                min_shape = [*shape_profile.min]
                opt_shape = [*shape_profile.opt]
                max_shape = [*shape_profile.max]
                if network._auto_parallel_config is not None:
                    io_shards = network._auto_parallel_config["io_shards"]
                    if input_name in io_shards:
                        shards = io_shards[input_name]
                        for dim, shard_num in shards.items():
                            min_shape[dim] = int(
                                math.floor(min_shape[dim] / shard_num))
                            opt_shape[dim] = int(
                                round(opt_shape[dim] / shard_num))
                            max_shape[dim] = int(
                                math.ceil(max_shape[dim] / shard_num))
                profile.set_shape(input_name, min_shape, opt_shape, max_shape)
                logger.debug(
                    f'{input_name}, min: {min_shape}, opt: {opt_shape}, max: {max_shape}, dimension names: {shape_profile.dimension_names}'
                )
            ret = builder_config.trt_builder_config.add_optimization_profile(
                profile)
            logger.debug(f"Added optimization profile: #{ret}")
            if force_num_profiles is not None and (
                    i + 1
            ) == force_num_profiles and force_num_profiles < num_profiles:
                logger.warning(
                    f"Only adding {force_num_profiles} profiles instead of {num_profiles}."
                )
                break
        assert self._validate_named_dimensions(
            network, builder_config
        ), "Validation of the tensor dimension ranges failed, please check the dimension ranges, find the offensive tensor and dimension name in above the error log"

    def _validate_named_dimensions(self, network: Network,
                                   builder_config) -> bool:
        '''
            For each profile, validate that the named dimensions of different input tensors in this profile all have same range.
            TRT will validate the same condition, validate it earlier to make sure the modeling in TensorRT-LLM are correct and
            makes the error msg more user friendly.
        '''
        valid = True
        for profile_idx in range(
                builder_config.trt_builder_config.num_optimization_profiles):
            dimension_to_range = {}
            for input_name, input_tensor in network._inputs.items():
                # it's legal that a Tensor does not have dim_range?
                if len(input_tensor.profiles) != 0:
                    profile = input_tensor.profiles[profile_idx]
                    for dim_idx, dim_name in enumerate(
                            profile.dimension_names):
                        if dim_name not in dimension_to_range:
                            dimension_to_range[dim_name] = []
                        min, opt, max = profile.min[dim_idx], profile.opt[
                            dim_idx], profile.max[dim_idx]
                        dimension_to_range[dim_name].append(
                            (input_name, (min, opt, max)))
            for dim, ranges in dimension_to_range.items():
                unique_ranges = set([r[1] for r in ranges])
                logger.debug(
                    f"Validating dimension:{dim}, ranges for this dim are:{unique_ranges}"
                )
                if len(unique_ranges) != 1:
                    logger.error(
                        f"Found illegal dimension setting for profile {profile_idx}, dimension name is: {dim}"
                    )
                    logger.error(
                        "Offensive tensors which have this dimension are:\n" +
                        "\n".join([f"{r[1]} {dim} {r[0]}" for r in ranges]))
                    valid = False
        return valid

    @_is_building
    def refit_engine(self, network: Network, engine_buffer) -> trt.IHostMemory:
        '''
            @brief: Refit one TensorRT engine using weights from the network,
                user should guarantee that the engine is built with REFIT flag, and the network has the same structure with the engine.
            @param engine_buffer: A serialized TensorRT engine.
            @param network: Network object.
            @return: A serialized TRT engine if refit successfully, None otherwise
        '''
        assert isinstance(network, Network)
        logger.info('Refit TRT engine')
        runtime = trt.Runtime(logger.trt_logger)
        engine = runtime.deserialize_cuda_engine(engine_buffer)

        tik = time.time()

        # Refit engine
        refitter = trt.Refitter(engine, logger.trt_logger)
        if network.named_parameters is not None:
            for name, param in network.named_parameters:
                if param._get_weights(
                ) is None or not refitter.set_named_weights(
                        name, param._get_weights()):
                    logger.error(f'Failed to refit weight: {name}')
                    return None
        else:
            logger.error(
                'Please set named parameters before building multiple engines.'
            )
            return None

        if not refitter.refit_cuda_engine():
            logger.error('Failed to refit engine.')
            return None

        tok = time.time()
        t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
        logger.info(f'Total time of refitting {engine.name}: {t}')
        serialized_engine = engine.serialize()
        return serialized_engine

    @_is_building
    def build_engine(self,
                     network: Network,
                     builder_config: BuilderConfig,
                     managed_weights: dict = None) -> trt.IHostMemory:
        '''
            @brief: Build one TensorRT engine from the network.
            @param network: Network object.
            @param builder_config: BuilderConfig object.
            @return: A serialized TRT engine.
        '''
        assert isinstance(network, Network)
        builder_config.plugin_config = network.plugin_config
        builder_config.auto_parallel_config = network.auto_parallel_config
        if builder_config.trt_builder_config.num_optimization_profiles == 0:
            self._add_optimization_profile(network, builder_config)
        logger.info(
            f"Total optimization profiles added: {builder_config.trt_builder_config.num_optimization_profiles}"
        )
        engine = None

        tik = time.time()
        # Rename weights
        if network.named_parameters is not None:
            managed_parameters = []
            for name, param in network.named_parameters:
                if param.is_managed(network):
                    assert managed_weights is not None, "managed_weights should be provided when enabled"
                    managed_parameters.append(param)
                    param.set_name(name, network)
                    continue
                if param._get_weights(network) is None:
                    if not param.is_buffer:
                        logger.debug(
                            f"Parameter {name} {param.raw_value.shape} {param.raw_value.dtype} was created"
                            " but unused in forward method, so not materialized to TRT network"
                        )
                    continue
                if not param.set_name(name, network):
                    raise RuntimeError(f'Failed to set weight: {name}')
                # This mark_weights_refittable has no side effect when refit_individual is not enabled.
                network.trt_network.mark_weights_refittable(name)

        network._fill_weights()
        tok = time.time()
        t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
        logger.info(
            f'Total time to initialize the weights in network {network.trt_network.name}: {t}'
        )

        # Build engine
        logger.info(f'Build TensorRT engine {network.trt_network.name}')
        tik = time.time()
        engine = self.trt_builder.build_serialized_network(
            network.trt_network, builder_config.trt_builder_config)
        assert engine is not None, 'Engine building failed, please check the error log.'

        tok = time.time()
        t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
        logger.info(f'Total time of building {network.trt_network.name}: {t}')

        if managed_weights is not None and network.named_parameters is not None:
            for param in managed_parameters:
                name = param.name
                value: np.ndarray = param._value
                if value is None:
                    logger.error(f'Failed to get weight: {name}')
                    continue
                if param.need_transpose:
                    # MOE has ndim=3 and uses plugin, no need to transpose
                    value = value.transpose(1, 0)  # WAR for bug 4641821
                managed_weights[name] = value

        return engine

    @staticmethod
    def save_timing_cache(builder_config: BuilderConfig,
                          out_path: str) -> bool:
        '''Serialize timing cache of given builder config to file specified by out_path
            return True if the cache is successfully serialized, False otherwise
        '''
        cache = builder_config.trt_builder_config.get_timing_cache()
        if cache is None:
            logger.warning(
                'No timing cache found in the given builder config, skip saving.'
            )
            return False
        with cache.serialize() as buffer:
            with open(out_path, "wb") as f:
                f.write(buffer)
                f.flush()
                os.fsync(f)
        logger.info(f'Timing cache serialized to {out_path}')
        return True

    @staticmethod
    def save_config(builder_config: BuilderConfig, config_path: str):
        config = builder_config.to_dict()
        to_json_file(config, config_path)
        logger.info(f'Config saved to {config_path}.')


class EngineConfig:

    def __init__(self, pretrained_config: 'PretrainedConfig',
                 build_config: 'BuildConfig', version: str):
        self.pretrained_config = pretrained_config
        self.build_config = build_config
        self.version = version

    @classmethod
    def from_json_file(cls, config_file):
        with open(config_file) as f:
            return cls.from_json_str(f.read())

    @classmethod
    def from_json_str(cls, config_str):
        config = json.loads(config_str)
        return cls(PretrainedConfig.from_dict(config['pretrained_config']),
                   BuildConfig.from_dict(config['build_config']),
                   config['version'])

    def to_dict(self):
        build_config = self.build_config.to_dict()
        build_config.pop('dry_run', None)  # Not an Engine Characteristic
        build_config.pop('visualize_network',
                         None)  # Not an Engine Characteristic
        return {
            'version': self.version,
            'pretrained_config': self.pretrained_config.to_dict(),
            'build_config': build_config,
        }


class Engine:

    def __init__(
        self,
        config: EngineConfig,
        engine: Union[trt.IHostMemory, None],
        managed_weights: dict[str, np.ndarray] = {},
    ):
        self.config = config
        self.engine = engine
        self.managed_weights = managed_weights
        if self.managed_weights is None:
            self.managed_weights = {}
        for name, value in self.managed_weights.items():
            if not value.flags['C_CONTIGUOUS']:
                self.managed_weights[name] = np.ascontiguousarray(value)

    def save(self, engine_dir: str):
        os.makedirs(engine_dir, exist_ok=True)
        if self.config.pretrained_config.mapping.rank == 0:
            config_dict = self.config.to_dict()
            with open(os.path.join(engine_dir, 'config.json'),
                      "w",
                      encoding="utf-8") as f:
                json.dump(config_dict, f, indent=4, cls=ConfigEncoder)
        if self.engine is not None:
            serialize_engine(
                self.engine,
                os.path.join(
                    engine_dir,
                    f'rank{self.config.pretrained_config.mapping.rank}.engine')
            )

    @classmethod
    def from_dir(cls, engine_dir: str, rank: int = 0):
        with open(os.path.join(engine_dir, f'rank{rank}.engine'), 'rb') as f:
            engine_buffer = f.read()

        config = EngineConfig.from_json_file(
            os.path.join(engine_dir, 'config.json'))
        config.pretrained_config.set_rank(rank)
        managed_weights = {}
        return cls(config, engine_buffer, managed_weights)

    @classmethod
    def from_buffer(cls,
                    engine_buffer: Union[trt.IHostMemory, bytes],
                    json_config_str: str,
                    rank: int = 0):
        config = EngineConfig.from_json_str(json_config_str)
        config.pretrained_config.set_rank(rank)
        return cls(config, engine_buffer)
