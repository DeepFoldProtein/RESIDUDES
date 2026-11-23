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
import argparse
from dataclasses import asdict, dataclass, field, fields
from typing import Optional

from .._utils import get_sm_version
from ..logger import logger

DEFAULT_PLUGIN_DTYPE_OPTIONS = [
    "auto", "float16", "float32", "bfloat16", "int32", None
]


def _make_plugin_property(field_name: str, field_type: type):

    def bind(field_name):
        storage_name = f'_{field_name}'

        @property
        def prop(self):
            field_value = getattr(self, storage_name)
            if field_name != 'dtype' and field_value == 'auto':
                return self.dtype
            else:
                return field_value

        @prop.setter
        def prop(self, value):
            if field_type is bool:
                assert isinstance(value, bool), \
                    f"Plugin {field_name} expects {field_type}, got {type(value)}"
            elif field_type in (str, Optional[str]):
                plugin_dtype_options = DEFAULT_PLUGIN_DTYPE_OPTIONS
                assert value in plugin_dtype_options, \
                    f"Plugin {field_name} expects values in {plugin_dtype_options}, got {value}"
            if field_name == 'dtype':
                assert value not in ['auto', None], \
                    "Plugin dtype cannot be auto or None"
            setattr(self, storage_name, value)
            logger.info(f"Set {field_name} to {value}.")

        return prop

    return bind(field_name)


class PluginConfigMeta(type):

    def __new__(cls, name, bases, attrs):
        for storage_name, field_type in attrs['__annotations__'].items():
            assert storage_name.startswith('_')
            field_name = storage_name.lstrip('_')
            attrs[field_name] = _make_plugin_property(field_name, field_type)
        return super().__new__(cls, name, bases, attrs)


@dataclass(slots=True)
class PluginConfig(metaclass=PluginConfigMeta):
    """The config that manages plugin-related options.

    There are two option categories:
    * Plugin options (typically with xxx_plugin naming). These options can be assigned with:
        * "float16"/"bfloat16"/"float32"/"int32", which means the plugin is enabled with the specified precision; (Some plugins only support limited dtype, i.e., gemm_swiglu_plugin and low_latency_gemm_swiglu_plugin only supports fp8 now)
        * "auto", which means the plugin is enabled with the precision of `dtype` field (the `dtype` field must be same to model dtype, i.e., the one in PretrainedConfig);
        * None, which means the plugin is disabled.
    * Other features. These options can be assigned with boolean:
        * True, which means the plugin is enabled;
        * False, which means the plugin is disabled.

    Note: All the fields should use a prefix "_"; PluginConfigMeta will wrap each field as a property.
    This ensures the fields can only be assigned with allowed values.
    """
    _dtype: str = field(default="float16", init=False)

    _multiple_profiles: bool = field(
        default=False,
        init=False,
        metadata={
            "help":
            "Enables multiple TensorRT optimization profiles in the built engines, "
            "will benefits the performance especially when GEMM plugin is disabled, "
            "because more optimization profiles help TensorRT have more chances to select better kernels. "
            "Note: This feature increases engine build time but no other adverse effects are expected."
        })
    _manage_weights: bool = field(
        default=False,
        init=False,
        metadata={
            "help":
            "Enable TensorRT-LLM managed weights to speed up engine building process."
        })

    def update_from_dict(self, config: dict):
        for name in config.keys():
            if hasattr(self, name):
                value_to_be_update = config[name]
                if isinstance(getattr(self, name), bool):
                    if value_to_be_update == "enable":
                        value_to_be_update = True
                    elif value_to_be_update == "disable":
                        value_to_be_update = False
                elif value_to_be_update == "disable":
                    value_to_be_update = None
                setattr(self, name, value_to_be_update)

    @classmethod
    def from_dict(cls, config: dict):
        plugin_config = cls()
        plugin_config.update_from_dict(config)
        return plugin_config

    @classmethod
    def from_arguments(cls, args: argparse.Namespace):
        args = vars(args)
        obj = cls.from_dict(args)

        return obj

    def to_dict(self):
        config = asdict(self)
        # Remove prefix "_" of the storage name
        config = {key.lstrip('_'): value for key, value in config.items()}
        return config

    def to_legacy_setting(self):
        '''Legacy setting means that all of the plugins and features are
        disabled, this is needed for the legacy `build.py` script, which will be
        migrated to the centralized building script `tensorrt_llm/commands/build.py`.

        After the migration is done, this function may or may not be deleted.
        '''
        for field in fields(self):
            # Remove prefix "_" of the storage name
            field_name = field.name.lstrip('_')
            if field_name == 'dtype':
                continue
            if field.type in (str, Optional[str]):
                setattr(self, field_name, None)
            elif field.type == bool:
                setattr(self, field_name, False)

    def validate(self):
        unsupported_plugins = {100: []}
        sm = get_sm_version()
        if sm in unsupported_plugins:
            for plugin in unsupported_plugins[sm]:
                val = getattr(self, plugin, None)
                if val is not None and val != False:
                    raise NotImplementedError(
                        f"{plugin}={val} is not supported on SM {sm}.")


# Only plugin configs in this list will be exposed as `trtllm-build` arguments,
# others are automatically enabled when needed, no need for users to control.
cli_plugin_args = []


def add_plugin_argument(parser: argparse.ArgumentParser):
    plugin_config = PluginConfig()
    for field in fields(plugin_config):
        # Remove prefix "_" of the storage name
        field_name = field.name.lstrip('_')
        if field_name not in cli_plugin_args:
            continue
        if field.metadata and "help" in field.metadata:
            help_message = field.metadata["help"]
        else:
            raise AttributeError(f"Please add help message for {field_name}.")
        if field.type in (str, Optional[str]):
            plugin_dtype_options = DEFAULT_PLUGIN_DTYPE_OPTIONS
            if field_name in PLUGIN_DTYPE_OPTIONS_MAP:
                plugin_dtype_options = PLUGIN_DTYPE_OPTIONS_MAP[field_name]
            if field_name == "gemm_plugin":
                default = field.default
            else:
                default = field.default if field.default else "disable"
            parser.add_argument(
                "--" + field_name,
                type=str,
                default=default,
                choices=[x if x else "disable" for x in plugin_dtype_options],
                help=help_message)
        elif field.type == bool:
            parser.add_argument(
                "--" + field_name,
                type=str,
                default="enable" if field.default else "disable",
                choices=["enable", "disable"],
                help=help_message)
        else:
            parser.add_argument("--" + field_name,
                                type=field.type,
                                default=field.default,
                                help=help_message)
    return parser
