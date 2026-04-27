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

import functools

import pytest
import transformers

from oumi.builders.collators import resolve_collator_templates
from tests.markers import requires_hf_token


def _normalize(s: str) -> str:
    """Normalize sentencepiece ▁ (U+2581) to space for cross-version comparison."""
    return s.replace("\u2581", " ")


@functools.cache
def _load_tokenizer(
    model_name: str, trust_remote_code: bool = False
) -> transformers.PreTrainedTokenizerBase:
    return transformers.AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=trust_remote_code
    )


# -- Public models (no HF token required) ------------------------------------


@pytest.mark.parametrize(
    "model_name,trust_remote_code,expected_response,expected_eot",
    [
        pytest.param(
            "Qwen/Qwen2.5-0.5B-Instruct",
            False,
            "<|im_start|>assistant",
            "<|im_end|>\n",
            id="qwen2.5-chatml",
        ),
        pytest.param(
            "Qwen/Qwen3-0.6B",
            False,
            "<|im_start|>assistant",
            "<|im_end|>\n",
            id="qwen3-chatml-think",
        ),
        pytest.param(
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            False,
            "<｜Assistant｜>",
            "<｜end▁of▁sentence｜>",
            id="deepseek-r1-qwen",
        ),
        pytest.param(
            "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
            False,
            "<｜Assistant｜>",
            "<｜end▁of▁sentence｜>",
            id="deepseek-r1-llama",
        ),
        pytest.param(
            "allenai/Olmo-3-7B-Instruct",
            True,
            "\n<|im_start|>assistant",
            "<|im_end|>",
            id="olmo3",
        ),
        pytest.param(
            "HuggingFaceTB/SmolLM2-135M-Instruct",
            False,
            "<|im_start|>assistant",
            "<|im_end|>\n",
            id="smollm2",
        ),
        pytest.param(
            "HuggingFaceTB/SmolLM3-3B",
            False,
            "<|im_start|>assistant",
            "<|im_end|>\n",
            id="smollm3",
        ),
        pytest.param(
            "Qwen/Qwen3.5-0.8B",
            True,
            "<|im_start|>assistant",
            "<|im_end|>\n",
            id="qwen3.5",
        ),
        pytest.param(
            "Qwen/Qwen3.6-35B-A3B",
            True,
            "<|im_start|>assistant",
            "<|im_end|>\n",
            id="qwen3.6",
        ),
        pytest.param(
            "MiniMaxAI/MiniMax-M2.5",
            True,
            "]~b]ai",
            "[e~[\n",
            id="minimax-m2.5",
        ),
    ],
)
def test_template_detection_public(
    model_name, trust_remote_code, expected_response, expected_eot
):
    tokenizer = _load_tokenizer(model_name, trust_remote_code)
    response_template, end_of_turn_template = resolve_collator_templates(tokenizer)
    assert _normalize(response_template) == _normalize(expected_response)
    assert _normalize(end_of_turn_template) == _normalize(expected_eot)
    assert response_template.strip()
    assert end_of_turn_template.strip()
    assert "<think>" not in response_template


# -- Gated models (require HF token) -----------------------------------------


@pytest.mark.parametrize(
    "model_name,trust_remote_code,expected_response,expected_eot",
    [
        pytest.param(
            "meta-llama/Llama-3.2-1B-Instruct",
            False,
            "<|start_header_id|>assistant<|end_header_id|>",
            "<|eot_id|>",
            id="llama3",
        ),
        pytest.param(
            "google/gemma-3-4b-it",
            False,
            "<start_of_turn>model",
            "<end_of_turn>\n",
            id="gemma3",
        ),
        pytest.param(
            "microsoft/Phi-4-reasoning-plus",
            True,
            "<|im_start|>assistant<|im_sep|>",
            "<|im_end|>",
            id="phi4-reasoning",
        ),
        pytest.param(
            "openai/gpt-oss-20b",
            True,
            "<|start|>assistant<|channel|>final<|message|>",
            "<|end|>",
            id="gpt-oss",
        ),
        pytest.param(
            "mistralai/Mistral-7B-Instruct-v0.3",
            False,
            "[/INST] ",
            "</s>",
            id="mistral",
        ),
        pytest.param(
            "Qwen/Qwen3-Next-80B-A3B-Instruct",
            True,
            "<|im_start|>assistant",
            "<|im_end|>\n",
            id="qwen3-next",
        ),
    ],
)
@requires_hf_token()
def test_template_detection_gated(
    model_name, trust_remote_code, expected_response, expected_eot
):
    tokenizer = _load_tokenizer(model_name, trust_remote_code)
    response_template, end_of_turn_template = resolve_collator_templates(tokenizer)
    assert _normalize(response_template) == _normalize(expected_response)
    assert _normalize(end_of_turn_template) == _normalize(expected_eot)
    assert response_template.strip()
    assert end_of_turn_template.strip()
    assert "<think>" not in response_template


# -- Models requiring newer transformers --------------------------------------
# These tokenizers need transformers >= 5.3.
# Tests skip gracefully if the tokenizer cannot be loaded.


@pytest.mark.parametrize(
    "model_name,trust_remote_code,expected_response,expected_eot",
    [
        pytest.param(
            "google/gemma-4-E2B-it",
            True,
            "<|turn>model",
            "<turn|>\n",
            id="gemma4",
        ),
        pytest.param(
            "mistralai/Mistral-Small-4-119B-2603",
            True,
            "[/INST]",
            "</s>",
            id="mistral-small-4",
        ),
    ],
)
@requires_hf_token()
def test_template_detection_newer_transformers(
    model_name, trust_remote_code, expected_response, expected_eot
):
    try:
        tokenizer = _load_tokenizer(model_name, trust_remote_code)
    except (AttributeError, ValueError, KeyError) as e:
        pytest.skip(
            f"Tokenizer for {model_name} not loadable with "
            f"transformers {transformers.__version__}: {e}"
        )
    response_template, end_of_turn_template = resolve_collator_templates(tokenizer)
    assert _normalize(response_template) == _normalize(expected_response)
    assert _normalize(end_of_turn_template) == _normalize(expected_eot)
    assert response_template.strip()
    assert end_of_turn_template.strip()
    assert "<think>" not in response_template
