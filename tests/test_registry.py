from lema.core.registry import REGISTRY, RegistryType, register


def test_registry_model_basic():
    REGISTRY.clear()

    @register("learning-machines/dummy", RegistryType.MODEL_CONFIG_CLASS)
    class DummyModelConfig:
        pass

    @register("learning-machines/dummy", RegistryType.MODEL_CLASS)
    class DummyModelClass:
        pass

    custom_model_in_registry = REGISTRY.lookup_model("learning-machines/dummy")
    assert custom_model_in_registry
    model_config = custom_model_in_registry.model_config
    model_class = custom_model_in_registry.model_class
    assert model_config == DummyModelConfig
    assert model_class == DummyModelClass


def test_registry_model_not_present():
    REGISTRY.clear()

    @register("learning-machines/dummy1", RegistryType.MODEL_CONFIG_CLASS)
    class DummyModelConfig:
        pass

    @register("learning-machines/dummy2", RegistryType.MODEL_CLASS)
    class DummyModelClass:
        pass

    # Non-existent record.
    assert REGISTRY.lookup_model("learning-machines/dummy") is None

    # Incomplete records.
    assert REGISTRY.lookup_model("learning-machines/dummy1") is None
    assert REGISTRY.lookup_model("learning-machines/dummy2") is None


def test_registry_cls_basic():
    REGISTRY.clear()

    @register("learning-machines/dummy", RegistryType.CLASS)
    class DummyClass:
        pass

    assert REGISTRY.lookup("learning-machines/dummy", RegistryType.CLASS) == DummyClass
    assert REGISTRY.lookup("learning-machines/nonExistent", RegistryType.CLASS) is None


def test_registry_fn_basic():
    REGISTRY.clear()

    @register("learning-machines/dummy", RegistryType.FUNCTION)
    def dummy_fn():
        pass

    assert REGISTRY.lookup("learning-machines/dummy", RegistryType.FUNCTION) == dummy_fn
    assert REGISTRY.lookup("learning-machines/nonExistent", RegistryType.CLASS) is None
