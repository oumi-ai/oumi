# Copyright 2025 - Oumi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import dataclasses
import logging
import re
from collections.abc import Iterator
from io import StringIO
from pathlib import Path
from typing import Any, Optional, TypeVar, Union, cast

from omegaconf import OmegaConf

from oumi.core.configs.params.base_params import BaseParams

T = TypeVar("T", bound="BaseConfig")

_CLI_IGNORED_PREFIXES = ["--local-rank"]


def _filter_ignored_args(arg_list: list[str]) -> list[str]:
    """Filters out ignored CLI arguments."""
    return [
        arg
        for arg in arg_list
        if not any(arg.startswith(prefix) for prefix in _CLI_IGNORED_PREFIXES)
    ]


def _read_config_without_interpolation(config_path: str) -> str:
    """Reads a configuration file without interpolating variables.

    Args:
        config_path: The path to the configuration file.

    Returns:
        str: The stringified configuration.
    """
    with open(config_path) as f:
        stringified_config = f.read()
        pattern = r"(?<!\\)\$\{"  # Matches "${" but not "\${"
        stringified_config = re.sub(pattern, "\\${", stringified_config)
    return stringified_config


@dataclasses.dataclass
class BaseConfig:
    def to_yaml(self, config_path: Union[str, Path, StringIO]) -> None:
        """Saves the configuration to a YAML file."""
        OmegaConf.save(config=self, f=config_path)

    @classmethod
    def from_yaml(
        cls: type[T], config_path: Union[str, Path], ignore_interpolation=True
    ) -> T:
        """Loads a configuration from a YAML file.

        Args:
            config_path: The path to the YAML file.
            ignore_interpolation: If True, then any interpolation variables in the
                configuration file will be escaped.

        Returns:
            BaseConfig: The merged configuration object.
        """
        schema = OmegaConf.structured(cls)
        if ignore_interpolation:
            stringified_config = _read_config_without_interpolation(str(config_path))
            file_config = OmegaConf.create(stringified_config)
        else:
            file_config = OmegaConf.load(config_path)
        config = OmegaConf.to_object(OmegaConf.merge(schema, file_config))
        if not isinstance(config, cls):
            raise TypeError(f"config is not {cls}")
        return cast(T, config)

    @classmethod
    def from_str(cls: type[T], config_str: str) -> T:
        """Loads a configuration from a YAML string.

        Args:
            config_str: The YAML string.

        Returns:
            BaseConfig: The configuration object.
        """
        schema = OmegaConf.structured(cls)
        file_config = OmegaConf.create(config_str)
        config = OmegaConf.to_object(OmegaConf.merge(schema, file_config))
        if not isinstance(config, cls):
            raise TypeError(f"config is not {cls}")
        return cast(T, config)

    @classmethod
    def from_yaml_and_arg_list(
        cls: type[T],
        config_path: Optional[str],
        arg_list: list[str],
        logger: Optional[logging.Logger] = None,
        ignore_interpolation=True,
    ) -> T:
        """Loads a configuration from various sources.

        If both YAML and arguments list are provided, then
        parameters specified in `arg_list` have higher precedence.

        Args:
            config_path: The path to the YAML file.
            arg_list: Command line arguments list.
            logger: (optional) Logger.
            ignore_interpolation: If True, then any interpolation variables in the
                configuration file will be escaped.

        Returns:
            BaseConfig: The merged configuration object.
        """
        # Start with an empty typed config. This forces OmegaConf to validate
        # that all other configs are of this structured type as well.
        all_configs = [OmegaConf.structured(cls)]

        # Override with configuration file if provided.
        if config_path is not None:
            if ignore_interpolation:
                stringified_config = _read_config_without_interpolation(config_path)
                all_configs.append(OmegaConf.create(stringified_config))
            else:
                all_configs.append(cls.from_yaml(config_path))

        # Merge base config and config from yaml.
        try:
            # Merge and validate configs
            config = OmegaConf.merge(*all_configs)
        except Exception:
            if logger:
                configs_str = "\n\n".join([f"{config}" for config in all_configs])
                logger.exception(
                    f"Failed to merge {len(all_configs)} Omega configs:\n{configs_str}"
                )
            raise

        # Override config with CLI arguments, in order. The arguments, aka flag names,
        # are dot-separated arguments, ex. `model.model_name`. This also supports
        # arguments indexing into lists, ex. `tasks[0].num_samples` or
        # `tasks.0.num_samples`. This is because the config is already populated and
        # typed, so the indexing is properly interpreted as a list index as opposed to
        # a dictionary key.
        try:
            # Filter out CLI arguments that should be ignored.
            arg_list = _filter_ignored_args(arg_list)
            # Override with CLI arguments.
            config.merge_with_dotlist(arg_list)
        except Exception:
            if logger:
                logger.exception(
                    f"Failed to merge arglist {arg_list} with Omega config:\n{config}"
                )
            raise

        config = OmegaConf.to_object(config)
        if not isinstance(config, cls):
            raise TypeError(f"config {type(config)} is not {type(cls)}")

        return cast(T, config)

    def finalize_and_validate(self) -> None:
        """Finalizes and validates the top level params objects."""
        for _, attr_value in self:
            if isinstance(attr_value, BaseParams):
                attr_value.finalize_and_validate()

        self.__finalize_and_validate__()

    def __finalize_and_validate__(self) -> None:
        """Finalizes and validates the parameters of this object.

        This method can be overridden by subclasses to implement custom
        validation logic.

        In case of validation errors, this method should raise a `ValueError`
        or other appropriate exception.
        """

    def __iter__(self) -> Iterator[tuple[str, Any]]:
        """Returns an iterator over field names and values.

        Note: for an attribute to be a field, it must be declared in the
        dataclass definition and have a type annotation.
        """
        for param in dataclasses.fields(self):
            yield param.name, getattr(self, param.name)
