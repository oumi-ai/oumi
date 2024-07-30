import pytest

from lema.core.registry import REGISTRY, RegistryType, register, register_dataset


@pytest.fixture(autouse=True)
def cleanup():
    # Clear the registry before each test.
    REGISTRY.clear()


def test_registry_cloud_builder():
    @register("dummy_class", RegistryType.CLOUD)
    class DummyClass:
        pass

    assert REGISTRY.contains("dummy_class", RegistryType.CLOUD)
    assert REGISTRY.get("dummy_class", RegistryType.CLOUD) == DummyClass
    assert not REGISTRY.contains("some_other_class", RegistryType.CLOUD)


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


def test_registry_metrics_function():
    @register("dummy_fn", RegistryType.METRICS_FUNCTION)
    def dummy_function():
        pass

    @register("number2", RegistryType.METRICS_FUNCTION)
    def dummy_function2():
        pass

    assert REGISTRY.contains("dummy_fn", RegistryType.METRICS_FUNCTION)
    assert REGISTRY.get("dummy_fn", RegistryType.METRICS_FUNCTION) == dummy_function

    assert REGISTRY.contains("number2", RegistryType.METRICS_FUNCTION)
    assert REGISTRY.get("number2", RegistryType.METRICS_FUNCTION) == dummy_function2


def test_registry_failure_metrics_function_not_present():
    @register("dummy_fn", RegistryType.METRICS_FUNCTION)
    def dummy_function():
        pass

    @register("number2", RegistryType.METRICS_FUNCTION)
    def dummy_function2():
        pass

    # Non-existent function (without exception).
    assert REGISTRY.get_model(name="dummy") is None

    # Non-existent function (with exception).
    with pytest.raises(KeyError) as exception_info:
        REGISTRY["yet_another_dummy", RegistryType.METRICS_FUNCTION]

    assert "does not exist" in str(exception_info.value)

    # Incomplete function name (without exception).
    assert REGISTRY.get_model(name="dummy_") is None
    assert REGISTRY.get_model_config(name="number") is None


def test_registry_datasets():
    dataset_name = "test_dataset"
    subset_name = "test_subset"

    @register_dataset(dataset_name)
    class TestDataset:
        pass

    @register_dataset(dataset_name, subset=subset_name)
    class TestDatasetSubset:
        pass

    # Dataset class should be registered.
    # If no subset is specified, the main dataset class should be returned.
    dataset_class = REGISTRY.get_dataset(dataset_name)
    assert dataset_class is not None
    assert dataset_class == TestDataset

    # Getting a dataset class for a registered subset
    # If a subset is specificed, we should get the subset class if one was registered
    dataset_subset_class = REGISTRY.get_dataset(dataset_name, subset=subset_name)
    assert dataset_subset_class is not None
    assert dataset_subset_class == TestDatasetSubset

    # Getting a dataset class for a non-registered subset
    # In this case we should get the main class
    dataset_subset_class = REGISTRY.get_dataset(
        dataset_name, subset="non_existent_subset"
    )
    assert dataset_subset_class is not None
    # If a subset is specificed, we should get the subset class if one was registered
    assert dataset_subset_class == TestDataset
