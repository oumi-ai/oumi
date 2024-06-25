import pytest

from lema.core.registry import REGISTRY, RegistryType, register


def test_registry_model_class():
    @register("dummy_class", RegistryType.MODEL)
    class DummyClass:
        pass

    assert REGISTRY.contains("dummy_class", RegistryType.MODEL)
    assert not REGISTRY.contains("dummy_class", RegistryType.MODEL_CONFIG)
    assert REGISTRY.get("dummy_class", RegistryType.MODEL) == DummyClass


def test_registry_model_config_class():
    @register("dummy_config_class", RegistryType.MODEL_CONFIG)
    class DummyConfigClass:
        pass

    assert REGISTRY.contains("dummy_config_class", RegistryType.MODEL_CONFIG)
    assert not REGISTRY.contains("dummy_config_class", RegistryType.MODEL)
    assert (
        REGISTRY.get("dummy_config_class", RegistryType.MODEL_CONFIG)
        == DummyConfigClass
    )


def test_registry_failure_register_class_twice():
    @register("another_dummy_class", RegistryType.MODEL)
    class DummyClass:
        pass

    with pytest.raises(ValueError) as exception_info:

        @register("another_dummy_class", RegistryType.MODEL)
        class AnotherDummyClass:
            pass

    assert "already registered" in str(exception_info.value)


def test_registry_failure_get_unregistered_class():
    assert not REGISTRY.contains("unregistered_class", RegistryType.MODEL)
    assert not REGISTRY.get(name="unregistered_class", type=RegistryType.MODEL)

    with pytest.raises(KeyError) as exception_info:
        REGISTRY["unregistered_class", RegistryType.MODEL]

    assert "does not exist" in str(exception_info.value)


def test_registry_model():
    @register("learning-machines/dummy", RegistryType.MODEL_CONFIG)
    class DummyModelConfig:
        pass

    @register("learning-machines/dummy", RegistryType.MODEL)
    class DummyModelClass:
        pass

    model_class = REGISTRY.get_model("learning-machines/dummy")
    assert model_class
    assert model_class == DummyModelClass

    model_config = REGISTRY.get_model_config("learning-machines/dummy")
    assert model_config
    assert model_config == DummyModelConfig


def test_registry_failure_model_not_present_in_registry():
    @register("learning-machines/dummy1", RegistryType.MODEL_CONFIG)
    class DummyModelConfig:
        pass

    @register("learning-machines/dummy2", RegistryType.MODEL)
    class DummyModelClass:
        pass

    # Non-existent model (without exception).
    assert REGISTRY.get_model(name="learning-machines/yet_another_dummy") is None

    # Non-existent model (with exception).
    with pytest.raises(KeyError) as exception_info:
        REGISTRY["learning-machines/yet_another_dummy", RegistryType.MODEL]

    assert "does not exist" in str(exception_info.value)

    # Incomplete model (without exception).
    assert REGISTRY.get_model(name="learning-machines/dummy1") is None
    assert REGISTRY.get_model_config(name="learning-machines/dummy2") is None


def test_registry_functon():
    # Note: This is ONLY for testing (NOT valid sample code)!
    # We need to support a different `RegistryType` for functions in the future.
    @register("dummy_fn", RegistryType.MODEL)
    def dummy_function():
        pass

    assert REGISTRY.contains("dummy_fn", RegistryType.MODEL)
    assert REGISTRY.get("dummy_fn", RegistryType.MODEL) == dummy_function
