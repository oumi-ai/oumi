# Copyright 2025 - Oumi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for cache utilities."""

from oumi.utils.cache_utils import dict_cache, dict_lru_cache, make_hashable


class TestMakeHashable:
    """Tests for make_hashable function."""

    def test_simple_types(self):
        """Test with simple hashable types."""
        assert make_hashable("string") == "string"
        assert make_hashable(123) == 123
        assert make_hashable(True) is True
        assert make_hashable(None) is None

    def test_dict_conversion(self):
        """Test dictionary conversion to frozenset."""
        d = {"a": 1, "b": 2}
        result = make_hashable(d)
        expected = frozenset([("a", 1), ("b", 2)])
        assert result == expected

    def test_nested_dict(self):
        """Test nested dictionary conversion."""
        d = {"outer": {"inner": "value"}}
        result = make_hashable(d)
        expected = frozenset([("outer", frozenset([("inner", "value")]))])
        assert result == expected

    def test_list_conversion(self):
        """Test list conversion to tuple."""
        lst = [1, 2, {"key": "value"}]
        result = make_hashable(lst)
        expected = (1, 2, frozenset([("key", "value")]))
        assert result == expected

    def test_set_conversion(self):
        """Test set conversion to frozenset."""
        s = {1, 2, 3}
        result = make_hashable(s)
        expected = frozenset([1, 2, 3])
        assert result == expected

    def test_complex_nested_structure(self):
        """Test complex nested structure."""
        obj = {"list": [1, {"nested": "dict"}], "set": {2, 3}, "simple": "value"}
        result = make_hashable(obj)

        # Check that result is hashable
        hash(result)  # Should not raise an error

        # Check structure
        assert isinstance(result, frozenset)


class TestDictCache:
    """Tests for dict_cache decorator."""

    def test_basic_caching(self):
        """Test basic caching functionality."""
        call_count = 0

        @dict_cache
        def test_func(x):
            nonlocal call_count
            call_count += 1
            return x * 2

        # First call
        result1 = test_func(5)
        assert result1 == 10
        assert call_count == 1

        # Second call with same args should use cache
        result2 = test_func(5)
        assert result2 == 10
        assert call_count == 1  # Should not increment

        # Different args should call function again
        result3 = test_func(6)
        assert result3 == 12
        assert call_count == 2

    def test_dict_arguments(self):
        """Test caching with dictionary arguments."""
        call_count = 0

        @dict_cache
        def process_config(name, config):
            nonlocal call_count
            call_count += 1
            return f"{name}:{len(config)}"

        config = {"key1": "value1", "key2": "value2"}

        # First call
        result1 = process_config("test", config)
        assert result1 == "test:2"
        assert call_count == 1

        # Same call should use cache
        result2 = process_config("test", config)
        assert result2 == "test:2"
        assert call_count == 1

        # Different dict content should call function
        config2 = {"key1": "value1", "key3": "value3"}
        result3 = process_config("test", config2)
        assert result3 == "test:2"
        assert call_count == 2

    def test_nested_dict_arguments(self):
        """Test caching with nested dictionary arguments."""
        call_count = 0

        @dict_cache
        def process_nested(config):
            nonlocal call_count
            call_count += 1
            return "processed"

        config1 = {"rope_scaling": {"type": "yarn", "factor": 4.0}}
        config2 = {"rope_scaling": {"type": "yarn", "factor": 4.0}}  # Same content
        config3 = {"rope_scaling": {"type": "yarn", "factor": 5.0}}  # Different content

        # First call
        result1 = process_nested(config1)
        assert result1 == "processed"
        assert call_count == 1

        # Same nested dict should use cache
        result2 = process_nested(config2)
        assert result2 == "processed"
        assert call_count == 1

        # Different nested dict should call function
        result3 = process_nested(config3)
        assert result3 == "processed"
        assert call_count == 2

    def test_kwargs_caching(self):
        """Test caching with keyword arguments including dicts."""
        call_count = 0

        @dict_cache
        def func_with_kwargs(name, config=None, **kwargs):
            nonlocal call_count
            call_count += 1
            return f"{name}:{len(config or {})}:{len(kwargs)}"

        # Test with dict in both regular and kwargs
        func_with_kwargs("test", config={"a": 1}, extra={"b": 2})
        assert call_count == 1

        # Same args should use cache
        func_with_kwargs("test", config={"a": 1}, extra={"b": 2})
        assert call_count == 1

        # Different kwargs should call function
        func_with_kwargs("test", config={"a": 1}, extra={"b": 3})
        assert call_count == 2

    def test_cache_control_methods(self):
        """Test cache control methods."""

        @dict_cache
        def test_func(x):
            return x * 2

        # Test that methods exist
        assert hasattr(test_func, "cache_clear")
        assert hasattr(test_func, "cache_info")

        # Test cache_clear
        test_func(1)
        test_func.cache_clear()

        # Test cache_info (basic check)
        info = test_func.cache_info()
        assert isinstance(info, str)
        assert "CacheInfo" in info


class TestDictLRUCache:
    """Tests for dict_lru_cache decorator."""

    def test_basic_lru_caching(self):
        """Test basic LRU caching functionality."""
        call_count = 0

        @dict_lru_cache(maxsize=2)
        def test_func(x):
            nonlocal call_count
            call_count += 1
            return x * 2

        # First calls
        assert test_func(1) == 2
        assert test_func(2) == 4
        assert call_count == 2

        # Should use cache
        assert test_func(1) == 2
        assert call_count == 2

    def test_lru_with_dicts(self):
        """Test LRU caching with dictionary arguments."""
        call_count = 0

        @dict_lru_cache(maxsize=2)
        def process_config(config):
            nonlocal call_count
            call_count += 1
            return len(config)

        config1 = {"a": 1}
        config2 = {"b": 2}

        # First calls
        assert process_config(config1) == 1
        assert process_config(config2) == 1
        assert call_count == 2

        # Should use cache
        assert process_config(config1) == 1
        assert call_count == 2

    def test_cache_control_methods_lru(self):
        """Test LRU cache control methods."""

        @dict_lru_cache(maxsize=2)
        def test_func(x):
            return x * 2

        # Test that methods exist
        assert hasattr(test_func, "cache_clear")
        assert hasattr(test_func, "cache_info")

        # Basic functionality test
        test_func(1)
        test_func.cache_clear()
        info = test_func.cache_info()
        assert "CacheInfo" in str(info)


def test_real_world_scenario():
    """Test with a real-world scenario similar to find_model_hf_config."""
    call_count = 0

    @dict_cache
    def find_config(model_name, *, trust_remote_code=True, revision=None, **kwargs):
        nonlocal call_count
        call_count += 1
        # Simulate processing kwargs
        rope_scaling = kwargs.get("rope_scaling", {})
        return f"config_for_{model_name}_with_{len(rope_scaling)}_rope_params"

    # Test with nested dict in kwargs
    kwargs1 = {
        "rope_scaling": {
            "type": "yarn",
            "factor": 4.0,
            "original_max_position_embeddings": 32768,
        }
    }

    # First call
    result1 = find_config("qwen", trust_remote_code=True, **kwargs1)
    expected = "config_for_qwen_with_3_rope_params"
    assert result1 == expected
    assert call_count == 1

    # Same call should use cache
    result2 = find_config("qwen", trust_remote_code=True, **kwargs1)
    assert result2 == expected
    assert call_count == 1  # Should not increment

    # Different model should call function
    result3 = find_config("llama", trust_remote_code=True, **kwargs1)
    assert result3 == "config_for_llama_with_3_rope_params"
    assert call_count == 2
