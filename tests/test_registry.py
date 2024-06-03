import pytest

from lema.core.registry import REGISTRY, RegistryType, register


def test_registry_basic_usage():
    REGISTRY.clear()

    @register
    class DummyModelConfig:
        registry_name = "learning-machines/dummy"
        registry_type = RegistryType.CONFIG_CLASS

    @register
    class DummyModelClass:
        registry_name = "learning-machines/dummy"
        registry_type = RegistryType.MODEL_CLASS

    custom_model_in_registry = REGISTRY.lookup("learning-machines/dummy")
    config_class = custom_model_in_registry.config_class
    model_class = custom_model_in_registry.model_class
    assert config_class == DummyModelConfig
    assert model_class == DummyModelClass


def test_registry_record_not_present():
    REGISTRY.clear()

    @register
    class DummyModelConfig:
        registry_name = "learning-machines/dummy1"
        registry_type = RegistryType.CONFIG_CLASS

    @register
    class DummyModelClass:
        registry_name = "learning-machines/dummy2"
        registry_type = RegistryType.MODEL_CLASS

    # Non-existent record.
    assert REGISTRY.lookup("learning-machines/dummy") is None

    # Incomplete records.
    assert REGISTRY.lookup("learning-machines/dummy1") is None
    assert REGISTRY.lookup("learning-machines/dummy2") is None


def test_registering_incompatible_class():
    REGISTRY.clear()

    # registering without `registry_type`.
    with pytest.raises(AttributeError):

        @register
        class IncompatibleModelConfig1:
            registry_name = "learning-machines/incompatible"

    # registering without `registry_name`.
    with pytest.raises(AttributeError):

        @register
        class IncompatibleModelConfig2:
            registry_type = RegistryType.CONFIG_CLASS

    # registering with `registry_name` of wrong type (int).
    with pytest.raises(AttributeError):

        @register
        class IncompatibleModelConfig3:
            registry_name = 1
            registry_type = RegistryType.CONFIG_CLASS

    # registering with `registry_type` of wrong type (int).
    with pytest.raises(AttributeError):

        @register
        class IncompatibleModelConfig4:
            registry_name = "learning-machines/incompatible"
            registry_type = 1
