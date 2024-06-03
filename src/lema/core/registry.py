from collections import defaultdict, namedtuple
from enum import Enum
from typing import Any, Callable, Optional


class RegistryType(Enum):
    CLASS = 1
    FUNCTION = 2
    MODEL_CONFIG_CLASS = 3
    MODEL_CLASS = 4


RegistryKey = namedtuple("RegistryKey", ["name", "registry_type"])
RegisteredModel = namedtuple("RegisteredModel", ["model_config", "model_class"])


class Registry:
    def __init__(self):
        """Initialize the class Registry."""
        self._registry = defaultdict(lambda: None)

    def register(self, name: str, type: RegistryType, value: Any) -> None:
        """Register a new record."""
        self._registry[RegistryKey(name, type)] = value

    def lookup(self, name: str, type: RegistryType) -> Optional[Callable]:
        """Lookup a record by name and type."""
        return self._registry[RegistryKey(name, type)]

    def lookup_model(self, name: str) -> Optional[RegisteredModel]:
        """Lookup a record that corresponds to a registered model."""
        model_config = self.lookup(name, RegistryType.MODEL_CONFIG_CLASS)
        model_class = self.lookup(name, RegistryType.MODEL_CLASS)
        if model_config and model_class:
            return RegisteredModel(model_config=model_config, model_class=model_class)
        else:
            return None

    def clear(self) -> None:
        """Clear the registry."""
        self._registry = defaultdict(lambda: None)

    def __repr__(self):
        """Define how this class is properly printed."""
        return "\n".join(f"{key}: {value}" for key, value in self._registry.items())


REGISTRY = Registry()


def register(registry_name: str, registry_type: RegistryType):
    """Register object `obj` in the LeMa global registry."""

    def decorator_register(obj):
        """Decorator to register its target `obj`."""
        REGISTRY.register(name=registry_name, type=registry_type, value=obj)
        return obj

    return decorator_register
