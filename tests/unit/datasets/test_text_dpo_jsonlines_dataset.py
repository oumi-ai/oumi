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

from oumi.core.tokenizers.base_tokenizer import BaseTokenizer
from oumi.datasets.preference_tuning.dpo_jsonlines import TextDpoJsonlinesDataset

ROWS = [
    {
        "prompt": "<user>hi<model>",
        "chosen": "answer<end>",
        "rejected": "CALL search<end>",
    },
    {"prompt": "<user>yo<model>", "chosen": "reply", "rejected": "CALL again"},
]


@pytest.fixture
def mock_tokenizer() -> MagicMock:
    mock = MagicMock(spec=BaseTokenizer)
    mock.eos_token_id = 9

    def tokenize(text, add_special_tokens=True):
        # ord-per-char keeps assertions readable; "<end>" tails map to ...9
        ids = [ord(c) % 256 for c in text]
        if text.endswith("<end>"):
            ids[-1] = 9
        return {"input_ids": ids}

    mock.side_effect = tokenize
    mock.__call__ = tokenize
    return mock


def test_rows_tokenize_without_chat_template(mock_tokenizer):
    ds = TextDpoJsonlinesDataset(data=list(ROWS), tokenizer=mock_tokenizer)
    out = ds.transform(ROWS[0])
    key = "prompt_ids" if "prompt_ids" in out else "prompt_input_ids"
    assert out[key] == [ord(c) % 256 for c in ROWS[0]["prompt"]]
    mock_tokenizer.apply_chat_template.assert_not_called()


def test_eos_appended_once(mock_tokenizer):
    ds = TextDpoJsonlinesDataset(data=list(ROWS), tokenizer=mock_tokenizer)
    ends_with_eos = ds.transform(ROWS[0])
    missing_eos = ds.transform(ROWS[1])
    for out in (ends_with_eos, missing_eos):
        chosen = out.get("chosen_ids", out.get("chosen_input_ids"))
        rejected = out.get("rejected_ids", out.get("rejected_input_ids"))
        assert chosen is not None and rejected is not None
        assert chosen[-1] == 9 and chosen[-2] != 9
        assert rejected[-1] == 9 and rejected[-2] != 9


def test_non_string_row_rejected(mock_tokenizer):
    ds = TextDpoJsonlinesDataset(data=list(ROWS), tokenizer=mock_tokenizer)
    with pytest.raises(ValueError, match="pre-rendered"):
        ds.transform(
            {
                "prompt": [{"role": "user", "content": "hi"}],
                "chosen": "a",
                "rejected": "b",
            }
        )


def test_loads_from_jsonl(tmp_path, mock_tokenizer):
    import json

    path = tmp_path / "pairs.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in ROWS))
    ds = TextDpoJsonlinesDataset(dataset_path=str(path), tokenizer=mock_tokenizer)
    assert len(ds._data) == 2
