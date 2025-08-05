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

"""Tests for the LengthAnalyzer."""

from unittest.mock import Mock

from oumi.core.analyze.length_analyzer import LengthAnalyzer
from oumi.core.types.conversation import Conversation, Message, Role


def _single_message_conversation(text):
    return Conversation(messages=[Message(role=Role.USER, content=text)])


def test_char_count():
    """Test character count functionality."""
    analyzer = LengthAnalyzer(
        char_count=True, word_count=False, sentence_count=False, token_count=False
    )
    conv = _single_message_conversation("Hello, world!")
    result = analyzer.analyze_sample(conv)
    assert result.messages[0].analyzer_metrics["char_count"] == 13
    # Only char_count should be present
    assert len(result.messages[0].analyzer_metrics) == 1


def test_word_count():
    """Test word count functionality."""
    analyzer = LengthAnalyzer(
        char_count=False, word_count=True, sentence_count=False, token_count=False
    )
    conv = _single_message_conversation("Hello world! This is a test.")
    result = analyzer.analyze_sample(conv)
    assert result.messages[0].analyzer_metrics["word_count"] == 6
    # Only word_count should be present
    assert len(result.messages[0].analyzer_metrics) == 1


def test_sentence_count():
    """Test sentence count functionality."""
    analyzer = LengthAnalyzer(
        char_count=False, word_count=False, sentence_count=True, token_count=False
    )
    conv = _single_message_conversation("Hello world! This is a test. How are you?")
    result = analyzer.analyze_sample(conv)
    assert result.messages[0].analyzer_metrics["sentence_count"] == 3
    # Only sentence_count should be present
    assert len(result.messages[0].analyzer_metrics) == 1


def test_analyzer_instantiation():
    """Test analyzer can be instantiated with different parameter combinations."""
    # Test with defaults
    analyzer = LengthAnalyzer()
    conv = _single_message_conversation("Hello, world!")
    result = analyzer.analyze_sample(conv)
    assert result.messages[0].analyzer_metrics["char_count"] == 13
    assert result.messages[0].analyzer_metrics["word_count"] == 2
    assert result.messages[0].analyzer_metrics["sentence_count"] == 1
    assert "token_count" not in result.messages[0].analyzer_metrics

    # Test with custom parameters
    analyzer = LengthAnalyzer(
        char_count=True, word_count=False, sentence_count=True, token_count=False
    )
    conv = _single_message_conversation("Hello, world!")
    result = analyzer.analyze_sample(conv)
    assert result.messages[0].analyzer_metrics["char_count"] == 13
    assert "word_count" not in result.messages[0].analyzer_metrics
    assert result.messages[0].analyzer_metrics["sentence_count"] == 1
    assert "token_count" not in result.messages[0].analyzer_metrics

    # Test with partial parameters (some defaults, some overridden)
    analyzer = LengthAnalyzer(char_count=False, word_count=True)
    conv = _single_message_conversation("Hello, world!")
    result = analyzer.analyze_sample(conv)
    assert "char_count" not in result.messages[0].analyzer_metrics
    assert result.messages[0].analyzer_metrics["word_count"] == 2
    assert result.messages[0].analyzer_metrics["sentence_count"] == 1  # Default True
    assert "token_count" not in result.messages[0].analyzer_metrics  # Default False


def test_token_count():
    """Test token count functionality."""
    # Test token count with tokenizer and dataset (default: includes special tokens)
    mock_tokenizer = Mock()
    mock_tokenizer.encode.return_value = [0, 1, 2, 3, 4, 5, 2]  # 7 tokens

    # Create a mock dataset
    mock_dataset = Mock()
    mock_dataset.tokenize.return_value = {
        "input_ids": [0, 1, 2, 3, 4, 5, 2, 3, 4]
    }  # 9 tokens for conversation
    mock_dataset.text_col = "text"

    analyzer = LengthAnalyzer(
        char_count=False,
        word_count=False,
        sentence_count=False,
        token_count=True,
        tokenizer=mock_tokenizer,
        dataset=mock_dataset,
    )
    conv = _single_message_conversation("Hello, world!")
    result = analyzer.analyze_sample(conv, tokenizer=mock_tokenizer)
    assert result.messages[0].analyzer_metrics["token_count"] == 7
    # analyze_sample calls tokenizer once for message, dataset.tokenize once for
    # conversation
    assert mock_tokenizer.encode.call_count == 1  # Only for message-level
    assert mock_dataset.tokenize.call_count == 1  # For conversation-level
    # Check that it was called with the message text
    mock_tokenizer.encode.assert_called_with("Hello, world!", add_special_tokens=True)
    # Check that dataset.tokenize was called with the conversation
    mock_dataset.tokenize.assert_called_with(conv, tokenize=True)

    # Test without special tokens (explicitly set to False)
    mock_tokenizer_no_special = Mock()
    mock_tokenizer_no_special.encode.return_value = [1, 2, 3, 4, 5]  # 5 tokens

    mock_dataset_no_special = Mock()
    mock_dataset_no_special.tokenize.return_value = {
        "input_ids": [1, 2, 3, 4, 5, 6]
    }  # 6 tokens for conversation
    mock_dataset_no_special.text_col = "text"

    analyzer_no_special = LengthAnalyzer(
        char_count=False,
        word_count=False,
        sentence_count=False,
        token_count=True,
        tokenizer=mock_tokenizer_no_special,
        include_special_tokens=False,
        dataset=mock_dataset_no_special,
    )
    conv = _single_message_conversation("Hello, world!")
    result = analyzer_no_special.analyze_sample(
        conv, tokenizer=mock_tokenizer_no_special
    )
    assert result.messages[0].analyzer_metrics["token_count"] == 5
    # analyze_sample calls tokenizer once for message, dataset.tokenize once for
    # conversation
    assert mock_tokenizer_no_special.encode.call_count == 1  # Only for message-level
    assert mock_dataset_no_special.tokenize.call_count == 1  # For conversation-level
    # Check that it was called with the message text
    mock_tokenizer_no_special.encode.assert_called_with(
        "Hello, world!", add_special_tokens=False
    )
    # Check that dataset.tokenize was called with the conversation
    mock_dataset_no_special.tokenize.assert_called_with(conv, tokenize=True)


def test_conversation_level_token_count():
    """Test that conversation-level token count is computed correctly with dataset."""
    # This test would have caught the bug where dataset wasn't passed to LengthAnalyzer
    mock_tokenizer = Mock()
    mock_tokenizer.encode.return_value = [0, 1, 2, 3, 4, 5]  # 6 tokens for message

    # Create a mock dataset
    mock_dataset = Mock()
    mock_dataset.tokenize.return_value = {
        "input_ids": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    }  # 10 tokens for conversation
    mock_dataset.text_col = "text"

    # Create analyzer with dataset
    analyzer = LengthAnalyzer(
        char_count=False,
        word_count=False,
        sentence_count=False,
        token_count=True,
        tokenizer=mock_tokenizer,
        dataset=mock_dataset,
    )

    # Create a conversation with multiple messages
    conv = Conversation(
        messages=[
            Message(role=Role.USER, content="Hello, how are you?"),
            Message(role=Role.ASSISTANT, content="I am doing well, thank you!"),
        ]
    )

    # Analyze the conversation
    result = analyzer.analyze_sample(conv)

    # Check that conversation-level token count is computed
    assert "token_count" in result.conversation.analyzer_metrics
    assert result.conversation.analyzer_metrics["token_count"] == 10

    # Verify that dataset.tokenize was called for conversation-level token count
    assert mock_dataset.tokenize.call_count == 1
    mock_dataset.tokenize.assert_called_with(conv, tokenize=True)

    # Verify that tokenizer.encode was called for each message
    assert mock_tokenizer.encode.call_count == 2  # Once for each message


def test_conversation_level_token_count_without_dataset():
    """Test that conversation-level token count is not computed when dataset is
    missing."""
    # This test ensures that without a dataset, conversation-level token count is not
    # computed
    mock_tokenizer = Mock()
    mock_tokenizer.encode.return_value = [0, 1, 2, 3, 4, 5]  # 6 tokens for message

    # Create analyzer WITHOUT dataset
    analyzer = LengthAnalyzer(
        char_count=False,
        word_count=False,
        sentence_count=False,
        token_count=True,
        tokenizer=mock_tokenizer,
        # No dataset parameter
    )

    # Create a conversation with multiple messages
    conv = Conversation(
        messages=[
            Message(role=Role.USER, content="Hello, how are you?"),
            Message(role=Role.ASSISTANT, content="I am doing well, thank you!"),
        ]
    )

    # Analyze the conversation
    result = analyzer.analyze_sample(conv)

    # Check that conversation-level token count is NOT computed
    assert "token_count" not in result.conversation.analyzer_metrics

    # Verify that tokenizer.encode was called for each message only
    assert mock_tokenizer.encode.call_count == 2  # Once for each message


def test_conversation_level_metrics_aggregation():
    """Test that conversation-level metrics are correctly aggregated from message-level
    metrics."""
    # Test that char, word, and sentence counts are aggregated from message-level
    # results
    mock_tokenizer = Mock()
    mock_tokenizer.encode.return_value = [0, 1, 2, 3, 4, 5]  # 6 tokens for message

    # Create a mock dataset
    mock_dataset = Mock()
    mock_dataset.tokenize.return_value = {
        "input_ids": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    }  # 10 tokens for conversation
    mock_dataset.text_col = "text"

    # Create analyzer with all metrics enabled
    analyzer = LengthAnalyzer(
        char_count=True,
        word_count=True,
        sentence_count=True,
        token_count=True,
        tokenizer=mock_tokenizer,
        dataset=mock_dataset,
    )

    # Create a conversation with multiple messages
    conv = Conversation(
        messages=[
            Message(
                role=Role.USER, content="Hello, how are you?"
            ),  # 18 chars, 4 words, 1 sentence
            Message(
                role=Role.ASSISTANT, content="I am doing well, thank you!"
            ),  # 26 chars, 6 words, 1 sentence
        ]
    )

    # Analyze the conversation
    result = analyzer.analyze_sample(conv)

    # Check conversation-level metrics are aggregated correctly
    assert result.conversation.analyzer_metrics["char_count"] == 46  # 19 + 27
    assert result.conversation.analyzer_metrics["word_count"] == 10  # 4 + 6
    assert result.conversation.analyzer_metrics["sentence_count"] == 2  # 1 + 1
    assert (
        result.conversation.analyzer_metrics["token_count"] == 10
    )  # From dataset.tokenize

    # Verify that dataset.tokenize was called for conversation-level token count
    assert mock_dataset.tokenize.call_count == 1
    mock_dataset.tokenize.assert_called_with(conv, tokenize=True)
