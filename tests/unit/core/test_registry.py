import pytest

from oumi.core.registry import (
    REGISTRY,
    Registry,
    RegistryType,
    register,
    register_dataset,
)


@pytest.fixture(autouse=True)
def cleanup():
    snapshot = Registry()
    for reg_type in RegistryType:
        for key, value in REGISTRY.get_all(reg_type).items():
            snapshot.register(key, reg_type, value)
    # Clear the registry before each test.
    REGISTRY.clear()
    yield
    # Clear the registry after each test.
    REGISTRY.clear()
    # Restore the registry after each test.
    for reg_type in RegistryType:
        for key, value in snapshot.get_all(reg_type).items():
            REGISTRY.register(key, reg_type, value)


def test_registry_cloud_builder():
    class DummyClass:
        pass

    @register("dummy_class", RegistryType.CLOUD)
    def dummy_builder():
        return DummyClass()

    assert REGISTRY.contains("dummy_class", RegistryType.CLOUD)
    assert REGISTRY.get("dummy_class", RegistryType.CLOUD) == dummy_builder
    assert not REGISTRY.contains("some_other_class", RegistryType.CLOUD)


def test_registry_cloud_builder_get_all():
    class DummyClass:
        pass

    @register("dummy_class", RegistryType.CLOUD)
    def dummy_builder():
        return DummyClass()

    class SomeClass:
        pass

    @register("another_one", RegistryType.CLOUD)
    def another_builder():
        return SomeClass()

    class LastClass:
        pass

    @register("finally", RegistryType.CLOUD)
    def last_builder():
        return LastClass()

    all_builders = REGISTRY.get_all(RegistryType.CLOUD).values()
    assert list(all_builders) == [dummy_builder, another_builder, last_builder]


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
    @register("oumi/dummy", RegistryType.MODEL_CONFIG)
    class DummyModelConfig:
        pass

    @register("oumi/dummy", RegistryType.MODEL)
    class DummyModelClass:
        pass

    model_class = REGISTRY.get_model("oumi/dummy")
    assert model_class
    assert model_class == DummyModelClass

    model_config = REGISTRY.get_model_config("oumi/dummy")
    assert model_config
    assert model_config == DummyModelConfig


def test_registry_failure_model_not_present_in_registry():
    @register("oumi/dummy1", RegistryType.MODEL_CONFIG)
    class DummyModelConfig:
        pass

    @register("oumi/dummy2", RegistryType.MODEL)
    class DummyModelClass:
        pass

    # Non-existent model (without exception).
    assert REGISTRY.get_model(name="oumi/yet_another_dummy") is None

    # Non-existent model (with exception).
    with pytest.raises(KeyError) as exception_info:
        REGISTRY["oumi/yet_another_dummy", RegistryType.MODEL]

    assert "does not exist" in str(exception_info.value)

    # Incomplete model (without exception).
    assert REGISTRY.get_model(name="oumi/dummy1") is None
    assert REGISTRY.get_model_config(name="oumi/dummy2") is None


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


@pytest.mark.parametrize(
    "registry_type",
    [
        RegistryType.CLOUD,
        RegistryType.MODEL,
        RegistryType.MODEL_CONFIG,
        RegistryType.METRICS_FUNCTION,
        RegistryType.DATASET,
    ],
)
def test_registry_case_insensitive(registry_type):
    class DummyClass:
        pass

    @register("Dummy_Class", registry_type)
    def dummy_builder():
        return DummyClass()

    # Check if the registry contains the key, regardless of case
    assert REGISTRY.contains("dummy_class", registry_type)
    assert REGISTRY.contains("DUMMY_CLASS", registry_type)
    assert REGISTRY.contains("Dummy_Class", registry_type)

    # Check if we can retrieve the builder using different cases
    assert REGISTRY.get("dummy_class", registry_type) == dummy_builder
    assert REGISTRY.get("DUMMY_CLASS", registry_type) == dummy_builder
    assert REGISTRY.get("Dummy_Class", registry_type) == dummy_builder

    # Check if the registry does not contain the key, regardless of case
    assert not REGISTRY.contains("dummy_class_not_registered", registry_type)
    assert not REGISTRY.contains("DUMMY_CLASS_NOT_REGISTERED", registry_type)
    assert not REGISTRY.contains("Dummy_Class_Not_Registered", registry_type)


def test_registry_case_insensitive_multiple_registrations():
    @register("Test_Class", RegistryType.MODEL)
    class TestClass1:
        pass

    with pytest.raises(ValueError, match="already registered"):

        @register("test_class", RegistryType.MODEL)
        class TestClass2:
            pass


def test_registry_case_insensitive_get_all():
    @register("Class_One", RegistryType.CLOUD)
    def builder_one():
        pass

    @register("CLASS_TWO", RegistryType.CLOUD)
    def builder_two():
        pass

    all_builders = REGISTRY.get_all(RegistryType.CLOUD)
    assert set(all_builders.keys()) == {"class_one", "class_two"}
    assert list(all_builders.values()) == [builder_one, builder_two]
