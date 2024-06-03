from collections import defaultdict, namedtuple
from enum import StrEnum
from typing import Callable, Union


class RegistryType(StrEnum):
    CLASS = "class"
    FUNCTION = "function"
    MODEL_CONFIG_CLASS = "model config class"
    MODEL_CLASS = "model class"


RegistryKey = namedtuple("RegistryKey", ["name", "registry_type"])
RegisteredModel = namedtuple("RegisteredModel", ["model_config", "model_class"])


class Registry:
    def __init__(self):
        """Initialize the class Registry."""
        self._registry = defaultdict(lambda: None)

    def register(self, name: str, type: RegistryType, value) -> None:
        """Registrer a new record."""
        registry_key = RegistryKey(name, type)
        self._registry[registry_key] = value

    def lookup(self, name: str, type: RegistryType) -> Union[Callable, None]:
        """Lookup a record by name."""
        registry_key = RegistryKey(name, type)
        return self._registry[registry_key]

    def lookup_model(self, name: str) -> Union[RegisteredModel, None]:
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
        registry_str = ""
        for key in self._registry:
            registry_str += f"{key}: {self._registry[key]}\n"
        return registry_str


REGISTRY = Registry()


def register(registry_name: str, registry_type: RegistryType):
    """Register object `obj` in the LeMa global registry."""

    def decorator_register(obj):
        """Decorator to register its target `obj`."""
        REGISTRY.register(name=registry_name, type=registry_type, value=obj)
        return obj

    return decorator_register
