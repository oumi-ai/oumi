"""A framework used for registering and accessing objects across Oumi."""

from oumi.core.registry.registry import (
    REGISTRY,
    Registry,
    RegistryType,
    register,
    register_cloud_builder,
    register_dataset,
    register_judge,
)

__all__ = [
    "REGISTRY",
    "Registry",
    "RegistryType",
    "register",
    "register_cloud_builder",
    "register_dataset",
    "register_judge",
]
