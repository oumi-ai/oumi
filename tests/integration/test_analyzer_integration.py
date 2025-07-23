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

"""Integration test for DatasetAnalyzer with length analyzer."""


# Import the length analyzer to register it

from oumi.core.analyze.dataset_analyzer import DatasetAnalyzer
from oumi.core.configs import AnalyzeConfig, SampleAnalyzerParams


def test_analyzer_integration():
    """Test DatasetAnalyzer integration with length analyzer."""
    # Create length analyzer configuration
    length_analyzer = SampleAnalyzerParams(
        id="length",
        config={
            "char_count": True,
            "word_count": True,
            "sentence_count": True,
            "token_count": False,
        },
    )

    # Create configuration using simple fields
    config = AnalyzeConfig(
        dataset_name="test_dataset",
        split="train",
        sample_count=2,  # Limit analysis to 2 conversations
        output_path="./test_results",
        analyzers=[length_analyzer],
    )

    # Create mock dataset with test conversations
    from unittest.mock import patch

    from tests.unit.core.analyze.test_dataset_analyzer import (
        MockConversation,
        MockDataset,
        MockMessage,
    )

    conversations = [
        MockConversation(
            "conv_1",
            [
                MockMessage("Hello, how are you?", "user", "msg_1_0"),
                MockMessage("I'm doing well, thank you!", "assistant", "msg_1_1"),
            ],
        ),
        MockConversation(
            "conv_2",
            [
                MockMessage("What is 2+2?", "user", "msg_2_0"),
                MockMessage("2+2 equals 4.", "assistant", "msg_2_1"),
            ],
        ),
    ]

    mock_dataset = MockDataset(conversations)

    # Run analysis with mocked dataset
    with patch(
        "oumi.core.analyze.dataset_analyzer.load_dataset_from_config"
    ) as mock_load:
        mock_load.return_value = mock_dataset

        analyzer = DatasetAnalyzer(config)

        # Run analysis
        print("Running analysis...")
        results = analyzer.analyze_dataset()

        # Verify results structure
        assert results.dataset_name == "test_dataset"
        assert results.total_conversations == 2
        assert results.conversations_analyzed == 2
        assert results.total_messages == 4

        # Verify length analyzer metrics are present
        messages = results.messages
        assert len(messages) == 4

        # Check first message has length metrics
        first_message = messages[0]
        assert "length_char_count" in first_message.analyzer_metrics
        assert "length_word_count" in first_message.analyzer_metrics
        assert "length_sentence_count" in first_message.analyzer_metrics

        # Verify specific metric values
        assert (
            first_message.analyzer_metrics["length_char_count"] == 19
        )  # "Hello, how are you?"
        assert first_message.analyzer_metrics["length_word_count"] == 4
        assert first_message.analyzer_metrics["length_sentence_count"] == 1

        print("✅ Integration test passed!")
        print(
            f"Analyzed {results.total_messages} messages from "
            f"{results.conversations_analyzed} conversations"
        )
        print(f"Sample metrics from first message: {first_message.analyzer_metrics}")


if __name__ == "__main__":
    test_analyzer_integration()
