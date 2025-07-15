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

import tempfile

import pandas as pd
import pytest

from oumi.core.analyze.dataset_analyzer import DatasetAnalyzer
from oumi.core.configs import DatasetAnalyzeConfig


@pytest.fixture
def test_jsonl_content():
    """Test JSONL content for analysis results."""
    return """
    {"dataset_name": "test", "total_conversations": 2, "total_messages": 4}
    {"conversation_id": "conv_0", "message_index": 0, "role": "user",
    "text_content": "Hello", "length_word_count": 1}
    {"conversation_id": "conv_0", "message_index": 1, "role": "assistant",
    "text_content": "Hi there!", "length_word_count": 2}
    {"conversation_id": "conv_1", "message_index": 0, "role": "user",
    "text_content": "How are you?", "length_word_count": 3}
    {"conversation_id": "conv_1", "message_index": 1, "role": "assistant",
    "text_content": "I'm good, thanks!", "length_word_count": 4}"""


@pytest.fixture
def analyzer_with_test_data(test_jsonl_content):
    """Create an analyzer with test data in a temporary directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create analyzer config with a real dataset name
        config = DatasetAnalyzeConfig(
            # will not actually load the dataset, just need the config
            dataset_name="tatsu-lab/alpaca",
            split="train",
            sample_count=2,
            output_path=temp_dir,
            analyzers=[],
        )

        # Create analyzer
        analyzer = DatasetAnalyzer(config)

        # Create test JSONL file using the analyzer's output path
        with open(analyzer._output_path, "w", encoding="utf-8") as f:
            f.write(test_jsonl_content)

        yield analyzer


def test_load_jsonl_results(analyzer_with_test_data):
    """Test loading JSONL results into DataFrame."""
    analyzer = analyzer_with_test_data
    results_df = analyzer.load_jsonl_results()  # Use default path

    # Verify DataFrame
    assert isinstance(results_df, pd.DataFrame)
    assert len(results_df) == 4  # 4 messages
    assert list(results_df.columns) == [
        "conversation_id",
        "message_index",
        "role",
        "text_content",
        "length_word_count",
    ]

    # Verify data
    user_messages = results_df[results_df["role"] == "user"]
    assert len(user_messages) == 2
    assert user_messages.iloc[0]["text_content"] == "Hello"
    assert user_messages.iloc[1]["text_content"] == "How are you?"

    # Test caching behavior
    # Second call should use cache
    results_df2 = analyzer.load_jsonl_results()  # Use default path
    assert results_df2 is results_df  # Should be the same object (cached)


def test_query_results(analyzer_with_test_data):
    """Test pandas querying of results."""
    analyzer = analyzer_with_test_data

    # Test query for user messages
    user_results = analyzer.query_results("role == 'user'")
    assert len(user_results) == 2
    assert all(user_results["role"] == "user")

    # Test filtering by word count
    long_messages = analyzer.query_results("length_word_count > 2")
    assert len(long_messages) == 2  # messages with 3 and 4 words

    # Test filtering by conversation
    conv_results = analyzer.query_results("conversation_id == 'conv_0'")
    assert len(conv_results) == 2

    # Test complex query
    complex_results = analyzer.query_results("role == 'user' and length_word_count > 1")
    assert len(complex_results) == 1  # only "How are you?" has > 1 word

    # Test clear cache
    analyzer.clear_cache()
    # After clearing cache, should load from file again
    df_after_clear = analyzer.load_jsonl_results()
    assert df_after_clear is not complex_results  # Should be different object
