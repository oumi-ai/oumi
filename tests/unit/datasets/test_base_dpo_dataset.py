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

from unittest.mock import MagicMock, patch

import pytest

from oumi.core.datasets.base_dpo_dataset import BaseDpoDataset
from oumi.core.tokenizers.base_tokenizer import BaseTokenizer


@pytest.fixture
def mock_tokenizer() -> MagicMock:
    """Create a mock tokenizer for testing."""
    mock = MagicMock(spec=BaseTokenizer)
    mock.eos_token_id = 2

    def mock_call(text, add_special_tokens=True):
        # Simple mock tokenization: return token IDs based on text length
        return {"input_ids": list(range(len(text)))}

    mock.side_effect = mock_call
    mock.__call__ = mock_call

    def mock_apply_chat_template(messages, tokenize=True):
        # Simple mock: concatenate all message contents
        result = ""
        for msg in messages:
            if isinstance(msg.get("content"), str):
                result += msg["content"]
        return result

    mock.apply_chat_template = mock_apply_chat_template

    return mock


class ConcreteDpoDataset(BaseDpoDataset):
    """Concrete implementation of BaseDpoDataset for testing."""

    default_dataset = "test_dpo_dataset"

    def __init__(self, data, **kwargs):
        self._test_data = data
        super().__init__(**kwargs)

    def _load_data(self):
        import pandas as pd

        return pd.DataFrame(self._test_data)


@pytest.fixture
def sample_dpo_data():
    """Sample DPO data for testing."""
    return [
        {
            "prompt": "What is 2+2?",
            "chosen": [{"role": "assistant", "content": "4"}],
            "rejected": [{"role": "assistant", "content": "5"}],
        },
    ]


class TestBaseDpoDatasetColumnNames:
    """Tests for DPO dataset column name handling based on TRL version."""

    def test_column_names_trl_pre_029(self, mock_tokenizer, sample_dpo_data):
        """Test that pre-0.29 TRL column names are used when TRL < 0.29."""
        with patch(
            "oumi.core.datasets.base_dpo_dataset.is_trl_v0_29_or_later",
            return_value=False,
        ):
            dataset = ConcreteDpoDataset(
                data=sample_dpo_data,
                tokenizer=mock_tokenizer,
            )

            result = dataset[0]

            # Pre-0.29 TRL uses *_input_ids suffix
            assert "prompt_input_ids" in result
            assert "chosen_input_ids" in result
            assert "rejected_input_ids" in result
            # New column names should NOT be present
            assert "prompt_ids" not in result
            assert "chosen_ids" not in result
            assert "rejected_ids" not in result

    def test_column_names_trl_029_or_later(self, mock_tokenizer, sample_dpo_data):
        """Test that 0.29+ TRL column names are used when TRL >= 0.29."""
        with patch(
            "oumi.core.datasets.base_dpo_dataset.is_trl_v0_29_or_later",
            return_value=True,
        ):
            dataset = ConcreteDpoDataset(
                data=sample_dpo_data,
                tokenizer=mock_tokenizer,
            )

            result = dataset[0]

            # TRL 0.29+ uses shorter *_ids suffix
            assert "prompt_ids" in result
            assert "chosen_ids" in result
            assert "rejected_ids" in result
            # Old column names should NOT be present
            assert "prompt_input_ids" not in result
            assert "chosen_input_ids" not in result
            assert "rejected_input_ids" not in result

    def test_column_values_are_lists(self, mock_tokenizer, sample_dpo_data):
        """Test that column values are lists of token IDs."""
        with patch(
            "oumi.core.datasets.base_dpo_dataset.is_trl_v0_29_or_later",
            return_value=True,
        ):
            dataset = ConcreteDpoDataset(
                data=sample_dpo_data,
                tokenizer=mock_tokenizer,
            )

            result = dataset[0]

            assert isinstance(result["prompt_ids"], list)
            assert isinstance(result["chosen_ids"], list)
            assert isinstance(result["rejected_ids"], list)

    def test_eos_token_appended_to_chosen_and_rejected(
        self, mock_tokenizer, sample_dpo_data
    ):
        """Test that EOS token is appended to chosen and rejected sequences."""
        with patch(
            "oumi.core.datasets.base_dpo_dataset.is_trl_v0_29_or_later",
            return_value=True,
        ):
            dataset = ConcreteDpoDataset(
                data=sample_dpo_data,
                tokenizer=mock_tokenizer,
            )

            result = dataset[0]

            # EOS token should be at the end of chosen and rejected
            assert result["chosen_ids"][-1] == mock_tokenizer.eos_token_id
            assert result["rejected_ids"][-1] == mock_tokenizer.eos_token_id
