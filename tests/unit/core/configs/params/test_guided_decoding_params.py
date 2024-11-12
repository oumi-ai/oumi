from typing import Dict

import pytest
from pydantic import BaseModel

from oumi.core.configs.params.guided_decoding_params import GuidedDecodingParams


class TestGuidedDecodingParams:
    def test_init_empty(self):
        """Test initializing with no parameters."""
        params = GuidedDecodingParams()
        assert params.json is None
        assert params.regex is None
        assert params.choice is None
        assert params.grammar is None
        assert params.json_object is None
        assert params.backend is None
        assert params.whitespace_pattern is None

    def test_init_with_json_dict(self):
        """Test initializing with a JSON schema dict."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "number"}
            }
        }
        params = GuidedDecodingParams(json=schema)
        assert params.json == schema
        assert isinstance(params.json, Dict)

    def test_init_with_pydantic_model(self):
        """Test initializing with a Pydantic model."""
        class Person(BaseModel):
            name: str
            age: int

        params = GuidedDecodingParams(json=Person)
        assert params.json is not None
        assert "properties" in params.json
        assert "name" in params.json["properties"]
        assert "age" in params.json["properties"]

    def test_init_with_regex(self):
        """Test initializing with a regex pattern."""
        pattern = r"\d{3}-\d{2}-\d{4}"
        params = GuidedDecodingParams(regex=pattern)
        assert params.regex == pattern

    def test_init_with_choices(self):
        """Test initializing with choices."""
        choices = ["yes", "no", "maybe"]
        params = GuidedDecodingParams(choice=choices)
        assert params.choice == choices

    def test_init_with_grammar(self):
        """Test initializing with a grammar."""
        grammar = "start = 'hello' 'world'"
        params = GuidedDecodingParams(grammar=grammar)
        assert params.grammar == grammar

    def test_init_with_json_object(self):
        """Test initializing with json_object flag."""
        params = GuidedDecodingParams(json_object=True)
        assert params.json_object is True

    def test_init_with_backend(self):
        """Test initializing with a specific backend."""
        params = GuidedDecodingParams(backend="custom_backend")
        assert params.backend == "custom_backend"

    def test_init_with_whitespace_pattern(self):
        """Test initializing with a whitespace pattern."""
        pattern = r"\s+"
        params = GuidedDecodingParams(whitespace_pattern=pattern)
        assert params.whitespace_pattern == pattern 