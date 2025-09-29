import os
import tempfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from omegaconf import OmegaConf

from oumi.core.configs.base_config import BaseConfig, _handle_non_primitives


class TestEnum(Enum):
    VALUE1 = "value1"
    VALUE2 = "value2"


@dataclass(eq=False)
class TestConfig(BaseConfig):
    str_value: str
    int_value: int
    float_value: float
    bool_value: bool
    none_value: Optional[Any]
    bytes_value: bytes
    path_value: Path
    enum_value: TestEnum
    list_value: list[Any]
    dict_value: dict[str, Any]
    func_value: Optional[Any] = None


def test_primitive_types():
    """Test that primitive types are preserved."""
    config = {
        "str": "test",
        "int": 42,
        "float": 3.14,
        "bool": True,
        "none": None,
        "bytes": b"test",
        "path": Path("test/path"),
        "enum": TestEnum.VALUE1,
    }

    removed_paths = set()
    result = _handle_non_primitives(config, removed_paths)

    assert result == config
    assert not removed_paths


def test_nested_lists():
    """Test handling of nested lists with primitive and non-primitive values."""
    config = {"list": ["primitive", {"nested": "value"}, [1, 2, 3], lambda x: x * 2]}

    removed_paths = set()
    result = _handle_non_primitives(config, removed_paths)

    assert result["list"][0] == "primitive"
    assert result["list"][1] == {"nested": "value"}
    assert result["list"][2] == [1, 2, 3]
    assert result["list"][3] is None
    assert "list[3]" in removed_paths


def test_nested_dicts():
    """Test handling of nested dictionaries with primitive and non-primitive values."""
    config = {
        "dict": {
            "primitive": "value",
            "nested": {"func": lambda x: x * 2, "list": [1, 2, 3]},
        }
    }

    removed_paths = set()
    result = _handle_non_primitives(config, removed_paths)

    assert result["dict"]["primitive"] == "value"
    assert result["dict"]["nested"]["list"] == [1, 2, 3]
    assert result["dict"]["nested"]["func"] is None
    assert "dict.nested.func" in removed_paths


def test_function_conversion():
    """Test that functions are converted to their source code when possible."""

    def test_func(x):
        return x * 2

    config = {"func": test_func}

    removed_paths = set()
    result = _handle_non_primitives(config, removed_paths)

    assert isinstance(result["func"], str)
    assert "def test_func" in result["func"]
    assert "return x * 2" in result["func"]
    assert not removed_paths


def test_builtin_function():
    """Test that built-in functions are removed."""
    config = {"func": len}

    removed_paths = set()
    result = _handle_non_primitives(config, removed_paths)

    assert result["func"] is None
    assert "func" in removed_paths


def test_complex_object():
    """Test that complex objects are removed."""

    class ComplexObject:
        def __init__(self):
            self.value = 42

    config = {"obj": ComplexObject()}

    removed_paths = set()
    result = _handle_non_primitives(config, removed_paths)

    assert result["obj"] is None
    assert "obj" in removed_paths


def test_config_serialization():
    """Test config serialization to YAML file."""
    with tempfile.TemporaryDirectory() as folder:
        config = TestConfig(
            str_value="test",
            int_value=42,
            float_value=3.14,
            bool_value=True,
            none_value=None,
            bytes_value=b"test",
            path_value=Path("test/path"),
            enum_value=TestEnum.VALUE1,
            list_value=["primitive", [1, 2, 3]],
            dict_value={"primitive": "value", "nested": {"list": [1, 2, 3]}},
            func_value=lambda x: x * 2,
        )

        filename = os.path.join(folder, "test_config.yaml")
        config.to_yaml(filename)

        assert os.path.exists(filename)

        loaded_config = TestConfig.from_yaml(filename)
        assert loaded_config.str_value == config.str_value
        assert loaded_config.int_value == config.int_value
        assert loaded_config.float_value == config.float_value
        assert loaded_config.bool_value == config.bool_value
        assert loaded_config.none_value == config.none_value
        assert str(loaded_config.bytes_value) == str(config.bytes_value)
        assert loaded_config.path_value == config.path_value
        assert loaded_config.enum_value == config.enum_value
        assert loaded_config.list_value == config.list_value
        assert loaded_config.dict_value == config.dict_value
        assert loaded_config.func_value is None


def test_config_loading_from_str():
    """Test loading config from YAML string."""
    yaml_str = """
        str_value: "test"
        int_value: 42
        float_value: 3.14
        bool_value: true
        none_value: null
        bytes_value: !!binary dGVzdA==
        path_value: "test/path"
        enum_value: "VALUE1"
        list_value: ["primitive", [1, 2, 3]]
        dict_value:
            primitive: "value"
            nested:
                list: [1, 2, 3]
        func_value: "def test_func(x): return x * 2"
    """

    config = TestConfig.from_str(yaml_str)
    assert config.str_value == "test"
    assert config.int_value == 42
    assert config.float_value == 3.14
    assert config.bool_value is True
    assert config.none_value is None
    assert config.bytes_value == b"test"
    assert config.path_value == Path("test/path")
    assert config.enum_value == TestEnum.VALUE1
    assert config.list_value == ["primitive", [1, 2, 3]]
    assert config.dict_value == {"primitive": "value", "nested": {"list": [1, 2, 3]}}


def test_config_equality():
    """Test config equality comparison."""
    config_a = TestConfig(
        str_value="test",
        int_value=42,
        float_value=3.14,
        bool_value=True,
        none_value=None,
        bytes_value=b"test",
        path_value=Path("test/path"),
        enum_value=TestEnum.VALUE1,
        list_value=["primitive"],
        dict_value={"key": "value"},
        func_value=lambda x: x * 2,
    )

    config_b = TestConfig(
        str_value="test",
        int_value=42,
        float_value=3.14,
        bool_value=True,
        none_value=None,
        bytes_value=b"test",
        path_value=Path("test/path"),
        enum_value=TestEnum.VALUE1,
        list_value=["primitive"],
        dict_value={"key": "value"},
        func_value=lambda x: x * 2,
    )

    assert config_a == config_b

    config_b.str_value = "different"
    assert config_a != config_b


def test_config_override():
    """Test config override with CLI arguments."""
    base_config = TestConfig(
        str_value="base",
        int_value=1,
        float_value=1.0,
        bool_value=True,
        none_value=None,
        bytes_value=b"base",
        path_value=Path("base/path"),
        enum_value=TestEnum.VALUE1,
        list_value=["base"],
        dict_value={"key": "base"},
        func_value=lambda x: x,
    )

    override_config = TestConfig(
        str_value="override",
        int_value=2,
        float_value=2.0,
        bool_value=False,
        none_value=None,
        bytes_value=b"override",
        path_value=Path("override/path"),
        enum_value=TestEnum.VALUE2,
        list_value=["override"],
        dict_value={"key": "override"},
        func_value=lambda x: x * 2,
    )

    # Convert configs to dictionaries and process non-primitives before OmegaConf
    base_dict = {}
    for field_name, field_value in base_config:
        base_dict[field_name] = field_value
    removed_paths = set()
    base_processed = _handle_non_primitives(base_dict, removed_paths)

    override_dict = {}
    for field_name, field_value in override_config:
        override_dict[field_name] = field_value
    removed_paths = set()
    override_processed = _handle_non_primitives(override_dict, removed_paths)

    base_omega = OmegaConf.create(base_processed)
    override_omega = OmegaConf.create(override_processed)
    merged_config = OmegaConf.merge(base_omega, override_omega)

    assert merged_config.str_value == "override"
    assert merged_config.int_value == 2
    assert merged_config.float_value == 2.0
    assert merged_config.bool_value is False
    assert str(merged_config.bytes_value) == "b'override'"
    assert str(merged_config.path_value) == "override/path"
    assert merged_config.enum_value == TestEnum.VALUE2
    assert merged_config.list_value == ["override"]
    assert merged_config.dict_value == {"key": "override"}
    assert merged_config.func_value is None


def test_config_from_yaml_and_arg_list():
    """Test loading config from YAML and CLI arguments."""
    with tempfile.TemporaryDirectory() as folder:
        config = TestConfig(
            str_value="base",
            int_value=1,
            float_value=1.0,
            bool_value=True,
            none_value=None,
            bytes_value=b"base",
            path_value=Path("base/path"),
            enum_value=TestEnum.VALUE1,
            list_value=["base"],
            dict_value={"key": "base"},
            func_value=lambda x: x,
        )

        filename = os.path.join(folder, "test_config.yaml")
        config.to_yaml(filename)

        new_config = TestConfig.from_yaml_and_arg_list(
            filename,
            [
                "str_value=override",
                "int_value=2",
                "float_value=2.0",
                "bool_value=false",
                "list_value[0]=override",
                "dict_value.key=override",
            ],
        )

        assert new_config.str_value == "override"
        assert new_config.int_value == 2
        assert new_config.float_value == 2.0
        assert new_config.bool_value is False
        assert new_config.list_value[0] == "override"
        assert new_config.dict_value["key"] == "override"
