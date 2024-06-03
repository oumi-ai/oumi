from collections import defaultdict
from enum import Enum
from typing import Callable, Union

REGISTRY_NAME_ATTR = "registry_name"
REGISTRY_TYPE_ATTR = "registry_type"


class RegistryType(Enum):
    CONFIG_CLASS = 1
    MODEL_CLASS = 2


class RegistryValue:
    def __init__(self, config_class=None, model_class=None):
        """Initialize the class RegistryValue."""
        self.config_class = config_class
        self.model_class = model_class

    def __repr__(self):
        """Define how this class is properly printed."""
        return f"(config_class={self.config_class}, model_class={self.model_class})"


class Registry:
    def __init__(self):
        """Initialize the class Registry."""
        self._registry = defaultdict(lambda: RegistryValue())

    def register(self, name: str, type: RegistryType, value) -> None:
        """Registrer a new record."""
        if type == RegistryType.CONFIG_CLASS:
            self._registry[name].config_class = value
        elif type == RegistryType.MODEL_CLASS:
            self._registry[name].model_class = value
        else:
            raise LookupError(f"No registry handler for type {type}")

    def lookup(self, name: str) -> Union[RegistryValue, None]:
        """Lookup a record by name."""
        if not self._registry[name]:
            return None
        elif self._registry[name].config_class and self._registry[name].model_class:
            return self._registry[name]
        else:  # partial record, ingore for now.
            return None

    def __repr__(self):
        """Define how this class is properly printed."""
        registry_str = ""
        for name in self._registry:
            registry_str += f"{name}: {self._registry[name]}\n"
        return registry_str


REGISTRY = Registry()


# Decorator funtion to register a new custom model.
def register(cls: Callable) -> Callable:
    """Register class `cls` in the LeMa global registry."""
    # Ensure that REGISTRY_NAME, REGISTRY_TYPE exist in cls and are valid.
    if not hasattr(cls, REGISTRY_NAME_ATTR):
        raise AttributeError(
            f"Class {cls} requires static attribute "
            f"`{REGISTRY_NAME_ATTR}` to be registered"
        )
    if not hasattr(cls, REGISTRY_TYPE_ATTR):
        raise AttributeError(
            f"Class {cls} requires static attribute "
            f"`{REGISTRY_TYPE_ATTR}` to be registered"
        )
    registry_name = getattr(cls, REGISTRY_NAME_ATTR)
    registry_type = getattr(cls, REGISTRY_TYPE_ATTR)
    if not isinstance(registry_name, str):
        raise AttributeError(
            f"Class {cls} attribute `{REGISTRY_NAME_ATTR}` must be "
            f"of type `str` to be registered"
        )
    if not isinstance(registry_type, RegistryType):
        raise AttributeError(
            f"Class {cls} attribute `{REGISTRY_TYPE_ATTR}` must be "
            f"of type `RegistryType` to be registered"
        )

    # Register Class `cls`` in registry.
    REGISTRY.register(name=registry_name, type=registry_type, value=cls)
    return cls
