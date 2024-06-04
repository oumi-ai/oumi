from collections import namedtuple
from enum import Enum
from typing import Any, Callable, Optional


class RegistryType(Enum):
    MODEL_CONFIG_CLASS = 1
    MODEL_CLASS = 2


RegistryKey = namedtuple("RegistryKey", ["name", "registry_type"])
RegisteredModel = namedtuple("RegisteredModel", ["model_config", "model_class"])


class Registry:
    def __init__(self):
        """Initialize the class Registry."""
        self._registry = dict()

    def _contains(self, key: RegistryKey) -> bool:
        """Indicates whether a record already exists in the registry."""
        return key in self._registry

    def __repr__(self) -> str:
        """Define how this class is properly printed."""
        return "\n".join(f"{key}: {value}" for key, value in self._registry.items())

    # Public functions.
    def contains(self, name: str, type: RegistryType) -> bool:
        """Indicates whether a record exists in the registry."""
        return self._contains(RegistryKey(name, type))

    def register(self, name: str, type: RegistryType, value: Any) -> None:
        """Register a new record."""
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
        except_if_missing: bool = True,
    ) -> Optional[Callable]:
        """Lookup a record by name and type."""
        registry_key = RegistryKey(name, type)
        if not self._contains(registry_key):
            if except_if_missing:
                raise ValueError(f"Registry: `{name}` of `{type}` does not exist.")
            else:
                return None
        else:
            return self._registry[registry_key]

    def clear(self) -> None:
        """Clear the registry."""
        self._registry = dict()

    # Convinience public function wrappers.
    def get_model(
        self, name: str, except_if_missing: bool = True
    ) -> Optional[RegisteredModel]:
        """Lookup a record that corresponds to a registered model."""
        model_config = self.get(
            name, RegistryType.MODEL_CONFIG_CLASS, except_if_missing
        )
        model_class = self.get(name, RegistryType.MODEL_CLASS, except_if_missing)
        if model_config and model_class:
            return RegisteredModel(model_config=model_config, model_class=model_class)
        else:
            return None


REGISTRY = Registry()


def register(
    registry_name: str, registry_type: RegistryType, registry=REGISTRY
) -> Callable:
    """Register object `obj` in the LeMa global registry.

    Args:
        registry_name: The name that the object should be registered with.
        registry_type: The type of object we are registering.
        registry: The registry to register the object to. This defaults to the
                  LeMa global registry `REGISTRY`.

    Returns:
        object: The original object, after registering it.
    """

    def decorator_register(obj):
        """Decorator to register its target `obj`."""
        registry.register(name=registry_name, type=registry_type, value=obj)
        return obj

    return decorator_register
