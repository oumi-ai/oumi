import pytest

from lema.core.registry import REGISTRY, RegisteredModel, RegistryType, register


def test_registering_model_class():
    REGISTRY.clear()

    @register("dummy_class", RegistryType.MODEL_CLASS)
    class DummyClass:
        pass

    assert REGISTRY.contains("dummy_class", RegistryType.MODEL_CLASS)
    assert not REGISTRY.contains("dummy_class", RegistryType.MODEL_CONFIG_CLASS)
    assert REGISTRY.get("dummy_class", RegistryType.MODEL_CLASS) == DummyClass


def test_registering_model_config_class():
    REGISTRY.clear()

    @register("dummy_config_class", RegistryType.MODEL_CONFIG_CLASS)
    class DummyConfigClass:
        pass

    assert REGISTRY.contains("dummy_config_class", RegistryType.MODEL_CONFIG_CLASS)
    assert not REGISTRY.contains("dummy_config_class", RegistryType.MODEL_CLASS)
    assert (
        REGISTRY.get("dummy_config_class", RegistryType.MODEL_CONFIG_CLASS)
        == DummyConfigClass
    )


def test_registering_class_twice():
    REGISTRY.clear()

    @register("dummy_class", RegistryType.MODEL_CLASS)
    class DummyClass:
        pass

    with pytest.raises(ValueError) as exception_info:

        @register("dummy_class", RegistryType.MODEL_CLASS)
        class AnotherDummyClass:
            pass

    assert str(exception_info.value) == (
        "Registry: `dummy_class` of `RegistryType.MODEL_CLASS` is already registered "
        "as `<class 'test_registry.test_registering_class_twice.<locals>.DummyClass'>`."
    )


def test_get_unregistered_class():
    REGISTRY.clear()

    assert not REGISTRY.contains("unregistered_class", RegistryType.MODEL_CLASS)
    assert not REGISTRY.get(
        name="unregistered_class",
        type=RegistryType.MODEL_CLASS,
        except_if_missing=False,
    )
    with pytest.raises(ValueError) as exception_info:
        REGISTRY.get(name="unregistered_class", type=RegistryType.MODEL_CLASS)

    assert str(exception_info.value) == (
        "Registry: `unregistered_class` of `RegistryType.MODEL_CLASS` "
        "does not exist."
    )


def test_register_model():
    REGISTRY.clear()

    @register("learning-machines/dummy", RegistryType.MODEL_CONFIG_CLASS)
    class DummyModelConfig:
        pass

    @register("learning-machines/dummy", RegistryType.MODEL_CLASS)
    class DummyModelClass:
        pass

    custom_model_in_registry = REGISTRY.get_model("learning-machines/dummy")
    assert custom_model_in_registry
    assert isinstance(custom_model_in_registry, RegisteredModel)
    model_config = custom_model_in_registry.model_config
    model_class = custom_model_in_registry.model_class
    assert model_config == DummyModelConfig
    assert model_class == DummyModelClass


def test_model_not_present_in_registry():
    REGISTRY.clear()

    @register("learning-machines/dummy1", RegistryType.MODEL_CONFIG_CLASS)
    class DummyModelConfig:
        pass

    @register("learning-machines/dummy2", RegistryType.MODEL_CLASS)
    class DummyModelClass:
        pass

    # Non-existent model (without exception).
    assert (
        REGISTRY.get_model(name="learning-machines/dummy", except_if_missing=False)
        is None
    )

    # Non-existent model (with exception).
    with pytest.raises(ValueError) as exception_info:
        REGISTRY.get_model("learning-machines/dummy")

    assert str(exception_info.value) == (
        "Registry: `learning-machines/dummy` of `RegistryType.MODEL_CONFIG_CLASS` "
        "does not exist."
    )

    # Incomplete model (without exception).
    assert (
        REGISTRY.get_model(name="learning-machines/dummy1", except_if_missing=False)
        is None
    )
    assert (
        REGISTRY.get_model(name="learning-machines/dummy2", except_if_missing=False)
        is None
    )

    # Incomplete model (with exception).
    with pytest.raises(ValueError) as exception_info:
        REGISTRY.get_model("learning-machines/dummy1")

    assert str(exception_info.value) == (
        "Registry: `learning-machines/dummy1` of `RegistryType.MODEL_CLASS` "
        "does not exist."
    )

    with pytest.raises(ValueError) as exception_info:
        REGISTRY.get_model("learning-machines/dummy2")

    assert str(exception_info.value) == (
        "Registry: `learning-machines/dummy2` of `RegistryType.MODEL_CONFIG_CLASS` "
        "does not exist."
    )


def test_registering_functon():
    REGISTRY.clear()

    # Note: This is ONLY for testing (NOT valid sample code)!
    # We need to support a different `RegistryType` for functions in the future.
    @register("dummy_fn", RegistryType.MODEL_CLASS)
    def dummy_function():
        pass

    assert REGISTRY.contains("dummy_fn", RegistryType.MODEL_CLASS)
    assert REGISTRY.get("dummy_fn", RegistryType.MODEL_CLASS) == dummy_function
