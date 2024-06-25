from collections import namedtuple
from enum import Enum, auto
from typing import Any, Callable, Optional, Tuple


class RegistryType(Enum):
    MODEL_CONFIG = auto()
    MODEL = auto()


RegistryKey = namedtuple("RegistryKey", ["name", "registry_type"])


class Registry:
    def __init__(self):
        """Initializes the class Registry."""
        self._registry = dict()

    #
    # Public functions
    #
    def contains(self, name: str, type: RegistryType) -> bool:
        """Indicates whether a record exists in the registry."""
        return self._contains(RegistryKey(name, type))

    def clear(self) -> None:
        """Clears the registry."""
        self._registry = dict()

    def register(self, name: str, type: RegistryType, value: Any) -> None:
        """Registers a new record."""
        registry_key = RegistryKey(name, type)
        if self._contains(registry_key):
            current_value = self.get(name=name, type=type)
            raise ValueError(
                f"Registry: `{name}` of `{type}` "
                f"is already registered as `{current_value}`."
            )
        self._registry[registry_key] = value

    def get(
        self,
        name: str,
        type: RegistryType,
    ) -> Optional[Callable]:
        """Gets a record by name and type."""
        registry_key = RegistryKey(name, type)
        return self._registry.get(registry_key)

    #
    # Convenience public function wrappers.
    #
    def get_model(self, name: str) -> Optional[Callable]:
        """Gets a record that corresponds to a registered model."""
        return self.get(name, RegistryType.MODEL)

    def get_model_config(self, name: str) -> Optional[Callable]:
        """Gets a record that corresponds to a registered config."""
        return self.get(name, RegistryType.MODEL_CONFIG)

    #
    # Private functions
    #
    def _contains(self, key: RegistryKey) -> bool:
        """Indicates whether a record already exists in the registry."""
        return key in self._registry

    #
    # Magic methods
    #
    def __getitem__(self, args: Tuple[str, RegistryType]) -> Callable:
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
    """Returns function to register decorated `obj` in the LeMa global registry.

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
