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
import inspect
import logging
import re
from collections.abc import Iterator
from enum import Enum
from io import StringIO
from pathlib import Path
from typing import Any, Optional, TypeVar, Union, cast

from omegaconf import OmegaConf

from oumi.core.configs.params.base_params import BaseParams

T = TypeVar("T", bound="BaseConfig")

_CLI_IGNORED_PREFIXES = ["--local-rank"]

# Set of primitive types that OmegaConf can handle directly
_PRIMITIVE_TYPES = {str, int, float, bool, type(None), bytes, Path, Enum}


def _is_primitive_type(value: Any) -> bool:
    """Check if a value is of a primitive type that OmegaConf can handle."""
    return (
        type(value) in _PRIMITIVE_TYPES
        or isinstance(value, Path)
        or isinstance(value, Enum)
    )


def _handle_non_primitives(config: Any, removed_paths: set, path: str = "") -> Any:
    """Recursively process config object to handle non-primitive values.

    Args:
        config: The config object to process
        removed_paths: Set to track paths of removed non-primitive values
        path: The current path in the config (for logging)

    Returns:
        The processed config with non-primitive values removed
    """
    if _is_primitive_type(config):
        return config

    # Try to convert functions to their source code
    if callable(config):
        try:
            # Lambda functions and built-in functions can't have source extracted
            if hasattr(config, "__name__") and config.__name__ == "<lambda>":
                removed_paths.add(path)
                return None

            source = inspect.getsource(config)
            # Only return source if we successfully got it
            return source
        except (TypeError, OSError):
            # Can't get source for lambdas, built-ins, or C extensions
            removed_paths.add(path)
            return None

    if isinstance(config, list):
        return [
            _handle_non_primitives(item, removed_paths, f"{path}[{i}]")
            for i, item in enumerate(config)
        ]

    # Handle dicts and dataclasses.
    if isinstance(config, dict) or hasattr(config, "__dataclass_fields__"):
        result = {}
        if isinstance(config, dict):
            items = config.items()
        else:  # dataclass
            items = (
                (field_name, getattr(config, field_name))
                for field_name in config.__dataclass_fields__
            )
        for key, value in items:
            # Compose path as per type
            current_path = f"{path}.{key}" if path else key
            if _is_primitive_type(value):
                result[key] = value
            else:
                processed_value = _handle_non_primitives(
                    value, removed_paths, current_path
                )
                if processed_value is not None:
                    result[key] = processed_value
                else:
                    removed_paths.add(current_path)
                    result[key] = None
        return result

    # For any other type, remove it and track the path
    removed_paths.add(path)
    return None


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


@dataclasses.dataclass(eq=False)
class BaseConfig:
    def to_yaml(self, config_path: Union[str, Path, StringIO]) -> None:
        """Saves the configuration to a YAML file.

        Non-primitive values are removed and warnings are logged.

        Args:
            config_path: Path to save the config to
        """
        # Convert dataclass fields to a dictionary first
        config_dict = {}
        for field_name, field_value in self:
            config_dict[field_name] = field_value

        # Process non-primitive values before creating OmegaConf structure
        removed_paths = set()
        processed_config = _handle_non_primitives(
            config_dict, removed_paths=removed_paths
        )

        # Log warnings for removed values
        if removed_paths:
            logger = logging.getLogger(__name__)
            logger.warning(
                "The following non-primitive values were removed from the config "
                "as they cannot be saved to YAML:\n"
                + "\n".join(f"- {path}" for path in sorted(removed_paths))
            )

        OmegaConf.save(config=processed_config, f=config_path)

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

    def print_config(self, logger: Optional[logging.Logger] = None) -> None:
        """Prints the configuration in a human-readable format.

        Args:
            logger: Optional logger to use. If None, uses module logger.
        """
        if logger is None:
            logger = logging.getLogger(__name__)

        # Convert dataclass fields to a dictionary first
        config_dict = {}
        for field_name, field_value in self:
            config_dict[field_name] = field_value

        # Process non-primitive values before creating OmegaConf structure
        removed_paths = set()
        processed_config = _handle_non_primitives(
            config_dict, removed_paths=removed_paths
        )

        config_yaml = OmegaConf.to_yaml(processed_config, resolve=True)
        logger.info(f"Configuration:\n{config_yaml}")

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

    def __eq__(self, other: object) -> bool:
        """Custom equality comparison that handles callable objects specially."""
        if not isinstance(other, self.__class__):
            return False

        for field_name, field_value in self:
            other_value = getattr(other, field_name)

            # Special handling for callable objects
            if callable(field_value) and callable(other_value):
                if (
                    hasattr(field_value, "__name__")
                    and hasattr(other_value, "__name__")
                    and field_value.__name__ == "<lambda>"
                    and other_value.__name__ == "<lambda>"
                ):
                    # Consider all lambda functions equal for config comparison purposes
                    continue

                # For regular functions, try to compare by source code
                try:
                    field_source = inspect.getsource(field_value).strip()
                    other_source = inspect.getsource(other_value).strip()
                    if field_source != other_source:
                        return False
                except (TypeError, OSError):
                    # If we can't get source, fall back to identity comparison
                    if field_value != other_value:
                        return False
            elif callable(field_value) or callable(other_value):
                # One is callable, the other is not
                return False
            else:
                # Normal comparison for non-callable values
                if field_value != other_value:
                    return False

        return True
