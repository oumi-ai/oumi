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

"""Tests for format converters."""

import pytest

from oumi.core.converters.format_converters import (
    auto_detect_converter,
    convert_alpaca,
    convert_conversations,
    convert_langchain,
    convert_langfuse,
    convert_opentelemetry,
    convert_oumi,
    convert_sharegpt,
    create_alpaca_converter,
    get_converter,
)
from oumi.core.registry import REGISTRY, RegistryType
from oumi.core.types.conversation import Role


class TestConverterRegistry:
    """Tests for converter registry integration."""

    def test_converters_are_registered(self):
        """Test that all converters are registered in the registry."""
        expected_converters = [
            "oumi",
            "alpaca",
            "sharegpt",
            "conversations",
            "langfuse",
            "opentelemetry",
            "langchain",
        ]
        for name in expected_converters:
            converter = REGISTRY.get_converter(name)
            assert converter is not None, f"Converter '{name}' not registered"
            assert callable(converter), f"Converter '{name}' is not callable"

    def test_get_all_converters(self):
        """Test that get_all returns all registered converters."""
        all_converters = REGISTRY.get_all(RegistryType.CONVERTER)
        assert len(all_converters) >= 7, "Expected at least 7 converters"
        assert "oumi" in all_converters
        assert "alpaca" in all_converters

    def test_get_converter_function(self):
        """Test the get_converter convenience function."""
        converter = get_converter("alpaca")
        assert callable(converter)

        example = {"instruction": "test", "input": "", "output": "result"}
        conv = converter(example)
        assert len(conv.messages) == 2

    def test_get_converter_unknown_raises(self):
        """Test that get_converter raises for unknown converter."""
        with pytest.raises(ValueError, match="Unknown converter"):
            get_converter("nonexistent_converter")


class TestConvertOumi:
    """Tests for convert_oumi converter."""

    def test_basic_conversation(self):
        """Test basic oumi format conversion."""
        example = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ]
        }
        conv = convert_oumi(example)
        assert len(conv.messages) == 2
        assert conv.messages[0].role == Role.USER
        assert conv.messages[0].content == "Hello"
        assert conv.messages[1].role == Role.ASSISTANT
        assert conv.messages[1].content == "Hi there!"

    def test_with_system_message(self):
        """Test oumi format with system message."""
        example = {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi!"},
            ]
        }
        conv = convert_oumi(example)
        assert len(conv.messages) == 3
        assert conv.messages[0].role == Role.SYSTEM

    def test_missing_messages_key_raises(self):
        """Test that missing messages key raises ValueError."""
        example = {"content": "Hello"}
        with pytest.raises(ValueError, match="requires 'messages' key"):
            convert_oumi(example)


class TestConvertAlpaca:
    """Tests for convert_alpaca converter."""

    def test_basic_conversion(self):
        """Test basic alpaca format conversion."""
        example = {
            "instruction": "Translate to French",
            "input": "Hello",
            "output": "Bonjour",
        }
        conv = convert_alpaca(example)
        assert len(conv.messages) == 2
        assert conv.messages[0].role == Role.USER
        assert "Translate to French" in conv.messages[0].content
        assert "Hello" in conv.messages[0].content
        assert conv.messages[1].role == Role.ASSISTANT
        assert conv.messages[1].content == "Bonjour"

    def test_empty_input(self):
        """Test alpaca format with empty input field."""
        example = {
            "instruction": "Say hello",
            "input": "",
            "output": "Hello!",
        }
        conv = convert_alpaca(example)
        assert len(conv.messages) == 2
        # With empty input, should just be instruction
        assert conv.messages[0].content == "Say hello"

    def test_missing_keys_raises(self):
        """Test that missing required keys raises ValueError."""
        example = {"instruction": "Do something"}
        with pytest.raises(ValueError, match="Missing:.*output"):
            convert_alpaca(example)


class TestConvertShareGPT:
    """Tests for convert_sharegpt converter."""

    def test_basic_conversion(self):
        """Test basic ShareGPT format conversion."""
        example = {
            "conversations": [
                {"from": "human", "value": "What is AI?"},
                {"from": "gpt", "value": "AI stands for Artificial Intelligence."},
            ]
        }
        conv = convert_sharegpt(example)
        assert len(conv.messages) == 2
        assert conv.messages[0].role == Role.USER
        assert conv.messages[0].content == "What is AI?"
        assert conv.messages[1].role == Role.ASSISTANT

    def test_with_system_role(self):
        """Test ShareGPT format with system message."""
        example = {
            "conversations": [
                {"from": "system", "value": "You are helpful."},
                {"from": "human", "value": "Hi"},
                {"from": "gpt", "value": "Hello!"},
            ]
        }
        conv = convert_sharegpt(example)
        assert len(conv.messages) == 3
        assert conv.messages[0].role == Role.SYSTEM

    def test_unknown_role_raises(self):
        """Test that unknown role raises ValueError."""
        example = {
            "conversations": [
                {"from": "unknown_role", "value": "Hello"},
            ]
        }
        with pytest.raises(ValueError, match="Unknown ShareGPT role"):
            convert_sharegpt(example)

    def test_missing_conversations_key_raises(self):
        """Test that missing conversations key raises ValueError."""
        example = {"messages": []}
        with pytest.raises(ValueError, match="requires 'conversations' key"):
            convert_sharegpt(example)


class TestConvertConversations:
    """Tests for convert_conversations converter."""

    def test_nested_conversation_format(self):
        """Test nested conversation format conversion."""
        example = {
            "conversation": {
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi!"},
                ]
            }
        }
        conv = convert_conversations(example)
        assert len(conv.messages) == 2
        assert conv.messages[0].role == Role.USER

    def test_missing_conversation_key_raises(self):
        """Test that missing conversation key raises ValueError."""
        example = {"messages": []}
        with pytest.raises(ValueError, match="requires 'conversation' key"):
            convert_conversations(example)


class TestConvertLangfuse:
    """Tests for convert_langfuse converter."""

    def test_input_output_format(self):
        """Test Langfuse input/output format."""
        example = {"input": "What is 2+2?", "output": "4"}
        conv = convert_langfuse(example)
        assert len(conv.messages) == 2
        assert conv.messages[0].role == Role.USER
        assert conv.messages[0].content == "What is 2+2?"
        assert conv.messages[1].content == "4"

    def test_prompt_completion_format(self):
        """Test Langfuse prompt/completion format."""
        example = {"prompt": "Hello", "completion": "Hi there!"}
        conv = convert_langfuse(example)
        assert len(conv.messages) == 2
        assert conv.messages[0].content == "Hello"

    def test_input_only(self):
        """Test Langfuse with input only (no output)."""
        example = {"input": "Hello", "output": None}
        conv = convert_langfuse(example)
        assert len(conv.messages) == 1

    def test_missing_keys_raises(self):
        """Test that missing required keys raises ValueError."""
        example = {"data": "something"}
        with pytest.raises(ValueError, match="requires 'input'/'output'"):
            convert_langfuse(example)


class TestConvertOpenTelemetry:
    """Tests for convert_opentelemetry converter."""

    def test_flat_format(self):
        """Test OpenTelemetry flat attribute format."""
        example = {
            "gen_ai.prompt": "Hello",
            "gen_ai.completion": "Hi there!",
        }
        conv = convert_opentelemetry(example)
        assert len(conv.messages) == 2
        assert conv.messages[0].content == "Hello"
        assert conv.messages[1].content == "Hi there!"

    def test_nested_attributes_format(self):
        """Test OpenTelemetry nested attributes format."""
        example = {
            "attributes": {
                "gen_ai.prompt": "What is AI?",
                "gen_ai.completion": "AI is...",
            }
        }
        conv = convert_opentelemetry(example)
        assert len(conv.messages) == 2
        assert conv.messages[0].content == "What is AI?"

    def test_prompt_only(self):
        """Test OpenTelemetry with prompt only."""
        example = {"gen_ai.prompt": "Hello", "gen_ai.completion": None}
        conv = convert_opentelemetry(example)
        assert len(conv.messages) == 1

    def test_missing_prompt_raises(self):
        """Test that missing gen_ai.prompt raises ValueError."""
        example = {"gen_ai.completion": "response"}
        with pytest.raises(ValueError, match="requires 'gen_ai.prompt'"):
            convert_opentelemetry(example)


class TestConvertLangChain:
    """Tests for convert_langchain converter."""

    def test_nested_inputs_outputs_format(self):
        """Test LangChain nested inputs/outputs format."""
        example = {
            "inputs": {"input": "What is 2+2?"},
            "outputs": {"output": "4"},
        }
        conv = convert_langchain(example)
        assert len(conv.messages) == 2
        assert conv.messages[0].content == "What is 2+2?"
        assert conv.messages[1].content == "4"

    def test_question_answer_keys(self):
        """Test LangChain with question/answer keys."""
        example = {
            "inputs": {"question": "What is AI?"},
            "outputs": {"answer": "Artificial Intelligence"},
        }
        conv = convert_langchain(example)
        assert len(conv.messages) == 2
        assert conv.messages[0].content == "What is AI?"
        assert conv.messages[1].content == "Artificial Intelligence"

    def test_flat_format(self):
        """Test LangChain flat input/output format."""
        example = {"input": "Hello", "output": "Hi!"}
        conv = convert_langchain(example)
        assert len(conv.messages) == 2

    def test_missing_keys_raises(self):
        """Test that missing required keys raises ValueError."""
        example = {"data": "something"}
        with pytest.raises(ValueError, match="requires 'inputs'/'outputs'"):
            convert_langchain(example)


class TestCreateAlpacaConverter:
    """Tests for create_alpaca_converter factory."""

    def test_without_system_prompt(self):
        """Test factory creates converter without system prompt."""
        converter = create_alpaca_converter(include_system_prompt=False)
        example = {
            "instruction": "Say hello",
            "input": "",
            "output": "Hello!",
        }
        conv = converter(example)
        assert len(conv.messages) == 2
        assert conv.messages[0].role == Role.USER

    def test_with_system_prompt(self):
        """Test factory creates converter with system prompt."""
        converter = create_alpaca_converter(include_system_prompt=True)
        example = {
            "instruction": "Say hello",
            "input": "",
            "output": "Hello!",
        }
        conv = converter(example)
        assert len(conv.messages) == 3
        assert conv.messages[0].role == Role.SYSTEM
        assert "instruction" in conv.messages[0].content.lower()

    def test_custom_system_prompts(self):
        """Test factory with custom system prompts."""
        converter = create_alpaca_converter(
            include_system_prompt=True,
            system_prompt_with_context="Custom with context",
            system_prompt_without_context="Custom without context",
        )
        # Without input
        conv = converter({
            "instruction": "Do something",
            "input": "",
            "output": "Done",
        })
        assert conv.messages[0].content == "Custom without context"

        # With input
        conv = converter({
            "instruction": "Do something",
            "input": "some context",
            "output": "Done",
        })
        assert conv.messages[0].content == "Custom with context"


class TestAutoDetectConverter:
    """Tests for auto_detect_converter function."""

    def test_detects_oumi_format(self):
        """Test detection of oumi format."""
        example = {
            "messages": [
                {"role": "user", "content": "Hello"},
            ]
        }
        assert auto_detect_converter(example) == "oumi"

    def test_detects_alpaca_format(self):
        """Test detection of alpaca format."""
        example = {
            "instruction": "Do something",
            "input": "context",
            "output": "result",
        }
        assert auto_detect_converter(example) == "alpaca"

    def test_detects_sharegpt_format(self):
        """Test detection of ShareGPT format."""
        example = {
            "conversations": [
                {"from": "human", "value": "Hi"},
                {"from": "gpt", "value": "Hello"},
            ]
        }
        assert auto_detect_converter(example) == "sharegpt"

    def test_detects_conversations_format(self):
        """Test detection of nested conversations format."""
        example = {
            "conversation": {
                "messages": [
                    {"role": "user", "content": "Hi"},
                ]
            }
        }
        assert auto_detect_converter(example) == "conversations"

    def test_detects_langfuse_format(self):
        """Test detection of Langfuse format."""
        example = {"input": "Hello", "output": "Hi"}
        assert auto_detect_converter(example) == "langfuse"

    def test_detects_opentelemetry_format(self):
        """Test detection of OpenTelemetry format."""
        example = {"gen_ai.prompt": "Hello", "gen_ai.completion": "Hi"}
        assert auto_detect_converter(example) == "opentelemetry"

    def test_detects_langchain_format(self):
        """Test detection of LangChain format."""
        example = {"inputs": {"input": "Hello"}, "outputs": {"output": "Hi"}}
        assert auto_detect_converter(example) == "langchain"

    def test_unknown_format_raises(self):
        """Test that unknown format raises ValueError."""
        example = {"random_key": "random_value"}
        with pytest.raises(ValueError, match="Unable to auto-detect format"):
            auto_detect_converter(example)
