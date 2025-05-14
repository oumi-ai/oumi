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

import functools
import importlib.util
import os
import sys
from collections import namedtuple
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Optional

from oumi.utils.logging import logger


class RegistryType(Enum):
    CLOUD = auto()
    DATASET = auto()
    METRICS_FUNCTION = auto()
    REWARD_FUNCTION = auto()
    MODEL_CONFIG = auto()
    MODEL = auto()
    JUDGE_CONFIG = auto()
    EVALUATION_FUNCTION = auto()


class RegistryKey(namedtuple("RegistryKey", ["name", "registry_type"])):
    def __new__(cls, name: str, registry_type: RegistryType):
        """Create a new RegistryKey instance.

        Args:
            name: The name of the registry key.
            registry_type: The type of the registry.

        Returns:
            A new RegistryKey instance with lowercase name.
        """
        return super().__new__(cls, name.lower(), registry_type)


def _load_user_requirements(requirements_file: str):
    """Loads user-defined requirements from a file."""
    logger.info(f"Loading user-defined registry from: {requirements_file}")
    logger.info(
        "This value can be set using the OUMI_EXTRA_DEPS_FILE environment variable."
    )
    requirements_path = Path(requirements_file)
    if not requirements_path.exists():
        logger.error(f"OUMI_EXTRA_DEPS_FILE file not found: {requirements_file}")
        raise FileNotFoundError(
            f"OUMI_EXTRA_DEPS_FILE file not found: {requirements_file}"
        )
    with open(requirements_path) as f:
        import_count = 0
        for idx, line in enumerate(f):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            import_count += 1
            import_path = Path(line)
            logger.info(f"Loading user-defined registry module: {import_path}")
            mod_name = import_path.stem
            sys.path.append(str(import_path.parent))
            try:
                importlib.import_module(mod_name)
            except Exception as e:
                logger.error(
                    "Failed to load a user-defined module in "
                    f"OUMI_EXTRA_DEPS_FILE: {line}"
                )
                raise ImportError(f"Failed to load user-defined module: {line}") from e
        logger.info(f"Loaded {import_count} user-defined registry modules.")


def _register_dependencies(cls_function):
    """Decorator to ensure core dependencies are added to the Registry."""

    @functools.wraps(cls_function)
    def wrapper(self, *args, **kwargs):
        if not self._initialized:
            # Immediately set the initialized flag to avoid infinite recursion.
            self._initialized = True
            # Import all core dependencies.
            import oumi.datasets  # noqa: F401
            import oumi.judges  # noqa: F401
            import oumi.launcher  # noqa: F401
            import oumi.models  # noqa: F401

            # Import user-defined dependencies.
            user_req_file = os.environ.get("OUMI_EXTRA_DEPS_FILE", None)
            if user_req_file:
                _load_user_requirements(user_req_file)

        return cls_function(self, *args, **kwargs)

    return wrapper


class Registry:
    _initialized: bool = False

    def __init__(self):
        """Initializes the class Registry."""
        self._registry = dict()

    #
    # Public functions
    #
    @_register_dependencies
    def contains(self, name: str, type: RegistryType) -> bool:
        """Indicates whether a record exists in the registry."""
        return self._contains(RegistryKey(name, type))

    @_register_dependencies
    def clear(self) -> None:
        """Clears the registry."""
        self._registry = dict()

    @_register_dependencies
    def register(self, name: str, type: RegistryType, value: Any) -> None:
        """Registers a new record."""
        registry_key = RegistryKey(name, type)
        if self._contains(registry_key):
            current_value = self.get(name=name, type=type)
            raise ValueError(
                f"Registry: `{name}` of `{type}` "
                f"is already registered as `{current_value}`."
            )
        if type in (
            RegistryType.METRICS_FUNCTION,
            RegistryType.REWARD_FUNCTION,
        ) and not callable(value):
            raise ValueError(f"Registry: `{name}` of `{type}` must be callable.")

        self._registry[registry_key] = value

    @_register_dependencies
    def get(
        self,
        name: str,
        type: RegistryType,
    ) -> Optional[Callable]:
        """Gets a record by name and type."""
        registry_key = RegistryKey(name, type)
        return self._registry.get(registry_key)

    @_register_dependencies
    def get_all(self, type: RegistryType) -> dict:
        """Gets all records of a specific type."""
        return {
            key.name: value
            for key, value in self._registry.items()
            if key.registry_type == type
        }

    #
    # Convenience public function wrappers.
    #
    def get_model(self, name: str) -> Optional[Callable]:
        """Gets a record that corresponds to a registered model."""
        return self.get(name, RegistryType.MODEL)

    def get_model_config(self, name: str) -> Optional[Callable]:
        """Gets a record that corresponds to a registered config."""
        return self.get(name, RegistryType.MODEL_CONFIG)

    def get_metrics_function(self, name: str) -> Optional[Callable]:
        """Gets a record that corresponds to a registered metrics function."""
        return self.get(name, RegistryType.METRICS_FUNCTION)

    def get_reward_function(self, name: str) -> Optional[Callable]:
        """Gets a record that corresponds to a registered rewards function."""
        return self.get(name, RegistryType.REWARD_FUNCTION)

    def get_judge_config(self, name: str) -> Optional[Callable]:
        """Gets a record that corresponds to a registered judge config."""
        return self.get(name, RegistryType.JUDGE_CONFIG)

    def get_evaluation_function(self, name: str) -> Optional[Callable]:
        """Gets a record that corresponds to a registered evaluation function."""
        return self.get(name, RegistryType.EVALUATION_FUNCTION)

    def get_dataset(
        self, name: str, subset: Optional[str] = None
    ) -> Optional[Callable]:
        """Gets a record that corresponds to a registered dataset."""
        if subset:
            # If a subset is provided, first check for subset-specific dataset.
            # If not found, try to get the dataset directly.
            dataset_cls = self.get(f"{name}/{subset}", RegistryType.DATASET)
            if dataset_cls is not None:
                return dataset_cls

        return self.get(name, RegistryType.DATASET)

    #
    # Private functions
    #
    def _contains(self, key: RegistryKey) -> bool:
        """Indicates whether a record already exists in the registry."""
        return key in self._registry

    #
    # Magic methods
    #
    def __getitem__(self, args: tuple[str, RegistryType]) -> Callable:
        """Gets a record by name and type."""
        if not isinstance(args, tuple) or len(args) != 2:
            raise ValueError(
                "Expected a tuple of length 2 with the first element being the name "
                "and the second element being the type."
            )

        name, type = args

        registry_key = RegistryKey(name, type)

        if not self._contains(registry_key):
            raise KeyError(f"Registry: `{name}` of `{type}` does not exist.")
        else:
            return self._registry[registry_key]

    def __repr__(self) -> str:
        """Defines how this class is properly printed."""
        return "\n".join(f"{key}: {value}" for key, value in self._registry.items())


REGISTRY = Registry()


def register(registry_name: str, registry_type: RegistryType) -> Callable:
    """Returns function to register decorated `obj` in the Oumi global registry.

    Args:
        registry_name: The name that the object should be registered with.
        registry_type: The type of object we are registering.

    Returns:
        Decorator function to register the target object.
    """

    def decorator_register(obj):
        """Decorator to register its target `obj`."""
        REGISTRY.register(name=registry_name, type=registry_type, value=obj)
        return obj

    return decorator_register


def register_dataset(registry_name: str, subset: Optional[str] = None) -> Callable:
    """Returns function to register decorated `obj` in the Oumi global registry.

    Args:
        registry_name: The name that the object should be registered with.
        subset: The type of object we are registering.

    Returns:
        Decorator function to register the target object.
    """

    def decorator_register(obj):
        """Decorator to register its target `obj`."""
        full_name = f"{registry_name}/{subset}" if subset else registry_name
        REGISTRY.register(name=full_name, type=RegistryType.DATASET, value=obj)
        return obj

    return decorator_register


def register_cloud_builder(registry_name: str) -> Callable:
    """Returns a function to register decorated builder in the Oumi global registry.

    Use this decorator to register cloud builder functions in the global registry.
    A cloud builder function is a function that accepts no arguments and returns an
    instance of a class that implements the `BaseCloud` interface.

    Args:
        registry_name: The name that the builder should be registered with.

    Returns:
        Decorator function to register the target builder.
    """

    def decorator_register(obj):
        """Decorator to register its target builder."""
        REGISTRY.register(name=registry_name, type=RegistryType.CLOUD, value=obj)
        return obj

    return decorator_register


def register_judge(registry_name: str) -> Callable:
    """Returns a function to register a judge configuration in the Oumi global registry.

    This decorator is used to register judge configuration in the global registry.
    A judge configuration function typically returns a JudgeConfig object that defines
    the parameters and attributes for a specific judge.

    Args:
        registry_name: The name under which the judge configuration should be
            registered.

    Returns:
        Callable: A decorator function that registers the target judge configuration.

    Example:
         .. code-block:: python

            @register_judge("my_custom_judge")
            def my_judge_config() -> JudgeConfig:
                return JudgeConfig(...)
    """

    def decorator_register(obj):
        """Decorator to register its target builder."""
        REGISTRY.register(name=registry_name, type=RegistryType.JUDGE_CONFIG, value=obj)
        return obj

    return decorator_register


def register_evaluation_function(registry_name: str) -> Callable:
    """Returns function to register an evaluation function in the Oumi global registry.

    Args:
        registry_name: The name that the evaluation function should be registered with.

    Returns:
        Decorator function to register the target evaluation function.
    """

    def check_evaluation_function_signature(evaluation_fn):
        if not callable(evaluation_fn):
            raise TypeError(
                f"Registry `{registry_name}` does not correspond to a callable object. "
                "It is required that registered evaluation functions of type "
                f"`{RegistryType.EVALUATION_FUNCTION}` must be callable."
            )

    def decorator_register(obj):
        """Decorator to register its target `obj`."""
        check_evaluation_function_signature(obj)

        REGISTRY.register(
            name=registry_name, type=RegistryType.EVALUATION_FUNCTION, value=obj
        )
        return obj

    return decorator_register
