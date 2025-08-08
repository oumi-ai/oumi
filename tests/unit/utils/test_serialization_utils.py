import dataclasses
from enum import Enum
from typing import Any, Optional

import torch

from oumi.core.configs.params.model_params import ModelParams
from oumi.core.configs.params.training_params import TrainingParams
from oumi.core.configs.training_config import TrainingConfig
from oumi.utils.serialization_utils import flatten_config


class ConfigTestEnum(Enum):
    VALUE_A = "a"
    VALUE_B = "b"


@dataclasses.dataclass
class SimpleConfig:
    name: str = "test"
    value: int = 42
    enabled: bool = True


@dataclasses.dataclass
class NestedConfig:
    simple: SimpleConfig = dataclasses.field(default_factory=SimpleConfig)
    settings: dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class ComplexConfig:
    name: str = "complex"
    nested: NestedConfig = dataclasses.field(default_factory=NestedConfig)
    items: list[str] = dataclasses.field(default_factory=list)
    optional_value: Optional[int] = None
    enum_value: ConfigTestEnum = ConfigTestEnum.VALUE_A
    torch_dtype: torch.dtype = torch.float32


class TestFlattenConfig:
    def test_flatten_simple_dict(self):
        """Test flattening a simple dictionary."""
        config = {"name": "test", "value": 42, "enabled": True}
        result = flatten_config(config)

        expected = {"name": "test", "value": 42, "enabled": True}
        assert result == expected

    def test_flatten_nested_dict(self):
        """Test flattening a nested dictionary."""
        config = {
            "name": "test",
            "settings": {"debug": True, "level": 2, "nested": {"deep": "value"}},
        }
        result = flatten_config(config)

        expected = {
            "name": "test",
            "settings.debug": True,
            "settings.level": 2,
            "settings.nested.deep": "value",
        }
        assert result == expected

    def test_flatten_simple_dataclass(self):
        """Test flattening a simple dataclass."""
        config = SimpleConfig(name="test_name", value=100, enabled=False)
        result = flatten_config(config)

        expected = {"name": "test_name", "value": 100, "enabled": False}
        assert result == expected

    def test_flatten_nested_dataclass(self):
        """Test flattening nested dataclasses."""
        simple = SimpleConfig(name="nested_test", value=200, enabled=True)
        config = NestedConfig(simple=simple, settings={"key": "value"})
        result = flatten_config(config)

        expected = {
            "simple.name": "nested_test",
            "simple.value": 200,
            "simple.enabled": True,
            "settings.key": "value",
        }
        assert result == expected

    def test_flatten_complex_dataclass(self):
        """Test flattening a complex dataclass with various types."""
        config = ComplexConfig(
            name="complex_test",
            nested=NestedConfig(
                simple=SimpleConfig(name="inner", value=300, enabled=False),
                settings={"mode": "production", "timeout": 30},
            ),
            items=["item1", "item2", "item3"],
            optional_value=42,
            enum_value=ConfigTestEnum.VALUE_B,
            torch_dtype=torch.float16,
        )
        result = flatten_config(config)

        expected = {
            "name": "complex_test",
            "nested.simple.name": "inner",
            "nested.simple.value": 300,
            "nested.simple.enabled": False,
            "nested.settings.mode": "production",
            "nested.settings.timeout": 30,
            "items": "['item1', 'item2', 'item3']",
            "optional_value": 42,
            "enum_value": "ConfigTestEnum.VALUE_B",
            "torch_dtype": "torch.float16",
        }
        assert result == expected

    def test_flatten_with_custom_options(self):
        """Test flattening with custom prefix and separator."""
        config = {"settings": {"debug": True, "level": 2}}

        # Test with prefix
        result = flatten_config({"name": "test"}, prefix="config")
        assert result == {"config.name": "test"}

        # Test with custom separator
        result = flatten_config(config, separator="_")
        assert result == {"settings_debug": True, "settings_level": 2}

        # Test with both prefix and separator
        result = flatten_config({"debug": True}, prefix="app", separator="__")
        assert result == {"app__debug": True}

    def test_flatten_empty_dict(self):
        """Test flattening an empty dictionary."""
        config = {}
        result = flatten_config(config)

        expected = {}
        assert result == expected

    def test_flatten_none_values(self):
        """Test flattening with None values."""
        config = {"name": "test", "value": None, "settings": {"debug": None}}
        result = flatten_config(config)

        expected = {"name": "test", "value": None, "settings.debug": None}
        assert result == expected

    def test_flatten_list_with_simple_values(self):
        """Test flattening lists with simple values."""
        config = {"items": [1, 2, 3], "names": ["alice", "bob"]}
        result = flatten_config(config)

        expected = {"items": "[1, 2, 3]", "names": "['alice', 'bob']"}
        assert result == expected

    def test_flatten_list_with_dicts(self):
        """Test flattening lists containing dictionaries."""
        config = {"users": [{"name": "alice", "age": 30}, {"name": "bob", "age": 25}]}
        result = flatten_config(config)

        expected = {
            "users.0.name": "alice",
            "users.0.age": 30,
            "users.1.name": "bob",
            "users.1.age": 25,
        }
        assert result == expected

    def test_flatten_list_with_dataclasses(self):
        """Test flattening lists containing dataclasses."""
        config = {
            "configs": [
                SimpleConfig(name="config1", value=10, enabled=True),
                SimpleConfig(name="config2", value=20, enabled=False),
            ]
        }
        result = flatten_config(config)

        expected = {
            "configs.0.name": "config1",
            "configs.0.value": 10,
            "configs.0.enabled": True,
            "configs.1.name": "config2",
            "configs.1.value": 20,
            "configs.1.enabled": False,
        }
        assert result == expected

    def test_flatten_tuple(self):
        """Test flattening tuples."""
        config = {"coordinates": (10, 20, 30), "pair": ("x", "y")}
        result = flatten_config(config)

        expected = {"coordinates": "(10, 20, 30)", "pair": "('x', 'y')"}
        assert result == expected

    def test_flatten_non_dict_objects(self):
        """Test flattening non-dict, non-dataclass objects."""
        # Test simple string
        result = flatten_config("simple_string")
        assert result == {"value": "simple_string"}

        # Test with prefix
        result = flatten_config(42, prefix="number")
        assert result == {"number": "42"}

    def test_flatten_special_types(self):
        """Test flattening with special types like torch.dtype."""
        config = {
            "dtype": torch.float32,
            "device": torch.device("cpu"),
            "tensor_shape": torch.Size([2, 3, 4]),
        }
        result = flatten_config(config)

        expected = {
            "dtype": "torch.float32",
            "device": "cpu",
            "tensor_shape": "torch.Size([2, 3, 4])",
        }
        assert result == expected

    def test_flatten_mixed_types(self):
        """Test flattening with mixed types including complex objects."""
        config = {
            "string": "test",
            "integer": 42,
            "float": 3.14,
            "boolean": True,
            "none_value": None,
            "list": [1, 2, 3],
            "dict": {"nested": "value"},
            "enum": ConfigTestEnum.VALUE_A,
            "torch_dtype": torch.float16,
        }
        result = flatten_config(config)

        expected = {
            "string": "test",
            "integer": 42,
            "float": 3.14,
            "boolean": True,
            "none_value": None,
            "list": "[1, 2, 3]",
            "dict.nested": "value",
            "enum": "ConfigTestEnum.VALUE_A",
            "torch_dtype": "torch.float16",
        }
        assert result == expected

    def test_flatten_deeply_nested_structure(self):
        """Test flattening deeply nested structures."""
        config = {"level1": {"level2": {"level3": {"level4": {"value": "deep_value"}}}}}
        result = flatten_config(config)

        expected = {"level1.level2.level3.level4.value": "deep_value"}
        assert result == expected

    def test_flatten_empty_containers(self):
        """Test flattening empty containers."""
        config = {"empty_list": [], "empty_dict": {}}
        result = flatten_config(config)

        assert result == {"empty_list": "[]"}

    def test_flatten_preserves_basic_types(self):
        """Test that basic types are preserved without conversion."""
        config = {
            "string": "test",
            "int": 42,
            "float": 3.14,
            "bool": True,
            "none": None,
        }
        result = flatten_config(config)

        # Check that basic types are preserved
        assert isinstance(result["string"], str)
        assert isinstance(result["int"], int)
        assert isinstance(result["float"], float)
        assert isinstance(result["bool"], bool)
        assert result["none"] is None

    def test_flatten_training_config(self):
        """Test flattening with actual oumi TrainingConfig."""
        config = TrainingConfig(
            model=ModelParams(model_name="microsoft/DialoGPT-medium"),
            training=TrainingParams(
                output_dir="./outputs",
                learning_rate=3e-5,
                num_train_epochs=3,
                per_device_train_batch_size=2,
            ),
        )
        result = flatten_config(config)

        # Should have flattened all nested parameters
        assert "model.model_name" in result
        assert "training.output_dir" in result
        assert "training.learning_rate" in result
        assert "training.num_train_epochs" in result
        assert len(result) > 10  # Should have many flattened parameters

    def test_flatten_enum_values(self):
        """Test that enum values are extracted correctly."""
        config = {"enum_a": ConfigTestEnum.VALUE_A, "enum_b": ConfigTestEnum.VALUE_B}
        result = flatten_config(config)

        assert result["enum_a"] == "ConfigTestEnum.VALUE_A"
        assert result["enum_b"] == "ConfigTestEnum.VALUE_B"
