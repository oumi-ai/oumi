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

from unittest.mock import MagicMock

import pytest

from oumi.core.tokenizers.special_tokens import get_default_special_tokens


def _tokenizer(name_or_path: str) -> MagicMock:
    tokenizer = MagicMock()
    tokenizer.name_or_path = name_or_path
    return tokenizer


@pytest.mark.parametrize(
    "name_or_path,expected_pad_token",
    [
        ("meta-llama/Llama-3.1-8B-Instruct", "<|finetune_right_pad_id|>"),
        ("nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16", "<unk>"),
        # Lookup is case-insensitive.
        ("nvidia/nvidia-nemotron-3-nano-4b-bf16", "<unk>"),
    ],
)
def test_known_tokenizers_have_a_pad_token(name_or_path, expected_pad_token):
    special_tokens = get_default_special_tokens(_tokenizer(name_or_path))

    assert special_tokens.pad_token == expected_pad_token


def test_unknown_tokenizer_returns_empty_config():
    special_tokens = get_default_special_tokens(_tokenizer("some/unknown-model"))

    assert special_tokens.pad_token is None


def test_missing_tokenizer_returns_empty_config():
    special_tokens = get_default_special_tokens(None)

    assert special_tokens.pad_token is None
