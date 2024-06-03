from collections import defaultdict, namedtuple
from enum import StrEnum
from typing import Callable, Union

REGISTRY_NAME_ATTR = "registry_name"
REGISTRY_TYPE_ATTR = "registry_type"


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


# Decorator funtion to register a new custom model.
def register_cls(cls: Callable) -> Callable:
    """Register class `cls` in the LeMa global registry."""
    # Ensure that REGISTRY_NAME, REGISTRY_TYPE exist in cls and are valid.
    if not hasattr(cls, REGISTRY_NAME_ATTR):
        raise ValueError(
            f"Class {cls} requires static attribute "
            f"`{REGISTRY_NAME_ATTR}` to be registered"
        )
    if not hasattr(cls, REGISTRY_TYPE_ATTR):
        raise ValueError(
            f"Class {cls} requires static attribute "
            f"`{REGISTRY_TYPE_ATTR}` to be registered"
        )
    registry_name = getattr(cls, REGISTRY_NAME_ATTR)
    registry_type = getattr(cls, REGISTRY_TYPE_ATTR)
    if not isinstance(registry_name, str):
        raise ValueError(
            f"Class {cls} attribute `{REGISTRY_NAME_ATTR}` must be "
            f"of type `str` to be registered"
        )
    if not isinstance(registry_type, RegistryType):
        raise ValueError(
            f"Class {cls} attribute `{REGISTRY_TYPE_ATTR}` must be "
            f"of type `RegistryType` to be registered"
        )

    # Register Class `cls`` in registry.
    REGISTRY.register(name=registry_name, type=registry_type, value=cls)
    return cls


def register_fn(registry_name: str, registry_type: RegistryType):
    """Register function `fn` in the LeMa global registry."""

    def decorator_register(fn):
        """Decorator to register its target `fn`."""
        REGISTRY.register(name=registry_name, type=registry_type, value=fn)
        return fn

    return decorator_register
