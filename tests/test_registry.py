import pytest

from lema.core.registry import REGISTRY, RegistryType, register_cls, register_fn


def test_registry_model_basic():
    REGISTRY.clear()

    @register_cls
    class DummyModelConfig:
        registry_name = "learning-machines/dummy"
        registry_type = RegistryType.MODEL_CONFIG_CLASS

    @register_cls
    class DummyModelClass:
        registry_name = "learning-machines/dummy"
        registry_type = RegistryType.MODEL_CLASS

    custom_model_in_registry = REGISTRY.lookup_model("learning-machines/dummy")
    model_config = custom_model_in_registry.model_config
    model_class = custom_model_in_registry.model_class
    assert model_config == DummyModelConfig
    assert model_class == DummyModelClass


def test_registry_model_not_present():
    REGISTRY.clear()

    @register_cls
    class DummyModelConfig:
        registry_name = "learning-machines/dummy1"
        registry_type = RegistryType.MODEL_CONFIG_CLASS

    @register_cls
    class DummyModelClass:
        registry_name = "learning-machines/dummy2"
        registry_type = RegistryType.MODEL_CLASS

    # Non-existent record.
    assert REGISTRY.lookup_model("learning-machines/dummy") is None

    # Incomplete records.
    assert REGISTRY.lookup_model("learning-machines/dummy1") is None
    assert REGISTRY.lookup_model("learning-machines/dummy2") is None


def test_registering_model_incompatible():
    REGISTRY.clear()

    # registering without `registry_type`.
    with pytest.raises(ValueError) as exception_info:

        @register_cls
        class IncompatibleModelConfig1:
            registry_name = "learning-machines/incompatible"

    assert str(exception_info.value) == (
        "Class <class 'test_registry.test_registering_model_incompatible.<locals>."
        "IncompatibleModelConfig1'> requires static attribute `registry_type` to be "
        "registered"
    )

    # registering without `registry_name`.
    with pytest.raises(ValueError) as exception_info:

        @register_cls
        class IncompatibleModelConfig2:
            registry_type = RegistryType.MODEL_CONFIG_CLASS

    assert str(exception_info.value) == (
        "Class <class 'test_registry.test_registering_model_incompatible.<locals>."
        "IncompatibleModelConfig2'> requires static attribute `registry_name` to be "
        "registered"
    )

    # registering with `registry_name` of wrong type (int).
    with pytest.raises(ValueError) as exception_info:

        @register_cls
        class IncompatibleModelConfig3:
            registry_name = 1
            registry_type = RegistryType.MODEL_CONFIG_CLASS

    assert str(exception_info.value) == (
        "Class <class 'test_registry.test_registering_model_incompatible.<locals>."
        "IncompatibleModelConfig3'> attribute `registry_name` must be of type "
        "`str` to be registered"
    )

    # registering with `registry_type` of wrong type (int).
    with pytest.raises(ValueError) as exception_info:

        @register_cls
        class IncompatibleModelConfig4:
            registry_name = "learning-machines/incompatible"
            registry_type = 1

    assert str(exception_info.value) == (
        "Class <class 'test_registry.test_registering_model_incompatible.<locals>."
        "IncompatibleModelConfig4'> attribute `registry_type` must be of type "
        "`RegistryType` to be registered"
    )


def test_registry_cls_basic():
    REGISTRY.clear()

    @register_cls
    class DummyClass:
        registry_name = "learning-machines/dummy"
        registry_type = RegistryType.CLASS

    assert REGISTRY.lookup("learning-machines/dummy", RegistryType.CLASS) == DummyClass
    assert REGISTRY.lookup("learning-machines/nonExistent", RegistryType.CLASS) is None


def test_registry_fn_basic():
    REGISTRY.clear()

    @register_fn(
        registry_name="learning-machines/dummy", registry_type=RegistryType.FUNCTION
    )
    def dummy_fn():
        pass

    assert REGISTRY.lookup("learning-machines/dummy", RegistryType.FUNCTION) == dummy_fn
    assert REGISTRY.lookup("learning-machines/nonExistent", RegistryType.CLASS) is None
