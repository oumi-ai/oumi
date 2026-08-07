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
import json
from typing import Any

import pytest
import transformers

from oumi.builders import build_data_collator
from oumi.builders.collators import (
    resolve_collator_templates,
    resolve_tool_response_template,
)
from oumi.core.constants import LABEL_IGNORE_INDEX
from oumi.core.types import Conversation, Message, Role
from oumi.core.types.tool_call import (
    FunctionCall,
    FunctionDefinition,
    JSONSchema,
    ToolCall,
    ToolDefinition,
)
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


# -- Tool results nested inside assistant turns --------------------------------
# gemma-4 and GLM-4.5 render tool results *inside* the model turn, so span masking
# unmasks environment output unless the tool-result bracket is subtracted.


def _tool_conversation() -> Conversation:
    """Multi-turn conversation with parallel tool calls and interleaved results."""

    def _call(call_id: str, name: str, **arguments) -> ToolCall:
        return ToolCall(
            id=call_id,
            function=FunctionCall(name=name, arguments=json.dumps(arguments)),
        )

    def _tool(name: str, **properties) -> ToolDefinition:
        return ToolDefinition(
            function=FunctionDefinition(
                name=name,
                description=f"{name} description",
                parameters=JSONSchema(
                    type="object",
                    properties={k: JSONSchema(type="string") for k in properties},
                    required=list(properties),
                ),
            )
        )

    return Conversation(
        tools=[_tool("get_weather", city=""), _tool("search_flights", origin="")],
        messages=[
            Message(role=Role.SYSTEM, content=_SYSTEM_TEXT),
            Message(role=Role.USER, content=_USER_TEXT_1),
            Message(
                role=Role.ASSISTANT,
                tool_calls=[
                    _call("weathr001", "get_weather", city="Paris"),
                    _call("flight001", "search_flights", origin="Boston"),
                ],
            ),
            Message(role=Role.TOOL, tool_call_id="weathr001", content=_TOOL_RESULT_1),
            Message(role=Role.TOOL, tool_call_id="flight001", content=_TOOL_RESULT_2),
            Message(role=Role.ASSISTANT, content=_ASSISTANT_TEXT_1),
            Message(role=Role.USER, content=_USER_TEXT_2),
            Message(
                role=Role.ASSISTANT,
                tool_calls=[_call("weathr002", "get_weather", city="Tokyo")],
            ),
            Message(role=Role.TOOL, tool_call_id="weathr002", content=_TOOL_RESULT_3),
            Message(role=Role.ASSISTANT, content=_ASSISTANT_TEXT_2),
        ],
    )


_SYSTEM_TEXT = "SYSTEM_PROMPT_TEXT"
_USER_TEXT_1 = "USER_ASKS_WEATHER_AND_FLIGHTS"
_USER_TEXT_2 = "USER_ASKS_TOKYO"
_ASSISTANT_TEXT_1 = "ASSISTANT_ANSWERS_PARIS"
_ASSISTANT_TEXT_2 = "ASSISTANT_ANSWERS_TOKYO"
_TOOL_RESULT_1 = "TOOL_RESULT_PARIS_WEATHER"
_TOOL_RESULT_2 = "TOOL_RESULT_BOSTON_FLIGHTS"
_TOOL_RESULT_3 = "TOOL_RESULT_TOKYO_WEATHER"


def _template_inputs(tokenizer, conversation: Conversation):
    """Messages/tools shaped for templates that require mapping arguments."""
    data = conversation.to_dict()
    messages = []
    for message in data["messages"]:
        message = dict(message)
        if message.get("tool_calls"):
            message["content"] = message.get("content") or ""
            message["tool_calls"] = [
                {
                    **call,
                    "function": {
                        **call["function"],
                        "arguments": json.loads(call["function"]["arguments"]),
                    },
                }
                for call in message["tool_calls"]
            ]
        messages.append(message)
    return messages, data.get("tools")


def _trained_text(tokenizer, conversation: Conversation) -> str:
    """Decoded text of every token the collator leaves in the loss."""
    response_template, end_of_turn_template = resolve_collator_templates(tokenizer)
    bracket = resolve_tool_response_template(
        tokenizer, response_template, end_of_turn_template
    )
    collator_kwargs: dict[str, Any] = {}
    if bracket is not None:
        collator_kwargs = {
            "tool_response_template": bracket[0],
            "end_of_tool_response_template": bracket[1],
        }

    messages, tools = _template_inputs(tokenizer, conversation)
    encoded = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=True,
        return_dict=True,
        add_generation_prompt=False,
    )
    input_ids = list(encoded["input_ids"])

    collator = build_data_collator(
        "text_completions_only_with_padding",
        tokenizer=tokenizer,
        max_length=None,
        response_template=response_template,
        end_of_turn_template=end_of_turn_template,
        train_target="all_assistant_turns",
        **collator_kwargs,
    )
    batch = collator([{"input_ids": input_ids, "attention_mask": [1] * len(input_ids)}])
    labels = batch["labels"][0].tolist()
    return tokenizer.decode(
        [
            token
            for token, label in zip(input_ids, labels)
            if label != LABEL_IGNORE_INDEX
        ],
        skip_special_tokens=False,
    )


def _masked_bracket_regions_are_complete(tokenizer, conversation: Conversation) -> bool:
    """Every token between a tool-result bracket pair must be masked.

    Works on token ids, so it catches a partially-masked payload that a decoded
    substring assertion would miss (drop only the first token and the remainder no
    longer matches the sentinel). Returns True when the template doesn't nest.
    """
    response_template, end_of_turn_template = resolve_collator_templates(tokenizer)
    bracket = resolve_tool_response_template(
        tokenizer, response_template, end_of_turn_template
    )
    if bracket is None:
        return True
    open_ids, close_ids = bracket

    messages, tools = _template_inputs(tokenizer, conversation)
    encoded = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=True,
        return_dict=True,
        add_generation_prompt=False,
    )
    input_ids = list(encoded["input_ids"])
    collator = build_data_collator(
        "text_completions_only_with_padding",
        tokenizer=tokenizer,
        max_length=None,
        response_template=response_template,
        end_of_turn_template=end_of_turn_template,
        train_target="all_assistant_turns",
        tool_response_template=open_ids,
        end_of_tool_response_template=close_ids,
    )
    labels = collator(
        [{"input_ids": input_ids, "attention_mask": [1] * len(input_ids)}]
    )["labels"][0].tolist()

    starts = [
        i
        for i in range(len(input_ids) - len(open_ids) + 1)
        if input_ids[i : i + len(open_ids)] == open_ids
    ]
    assert starts, "bracket was resolved but never occurs in the tokenized conversation"
    for start in starts:
        closes = [
            i
            for i in range(start, len(input_ids) - len(close_ids) + 1)
            if input_ids[i : i + len(close_ids)] == close_ids
        ]
        end = closes[0] + len(close_ids) if closes else len(input_ids)
        if any(label != LABEL_IGNORE_INDEX for label in labels[start:end]):
            return False
    return True


@pytest.mark.parametrize(
    "model_name,trust_remote_code",
    [
        pytest.param("google/gemma-4-E2B-it", False, id="gemma-4-nested"),
        pytest.param("zai-org/GLM-4.5", False, id="glm-4.5-nested"),
        pytest.param("Qwen/Qwen3-0.6B", False, id="qwen3-separate-turn"),
    ],
)
@requires_hf_token()
def test_tool_results_are_excluded_from_the_loss(model_name, trust_remote_code):
    try:
        tokenizer = _load_tokenizer(model_name, trust_remote_code)
    except Exception as e:
        pytest.skip(f"Could not load tokenizer {model_name}: {e}")

    trained = _trained_text(tokenizer, _tool_conversation())

    # Environment output must never be trained on: the harness emits it, not the model.
    for tool_result in (_TOOL_RESULT_1, _TOOL_RESULT_2, _TOOL_RESULT_3):
        assert tool_result not in trained, (
            f"{model_name} trains on tool result {tool_result!r}"
        )

    # Prompt text is context, not a target.
    for prompt_text in (_SYSTEM_TEXT, _USER_TEXT_1, _USER_TEXT_2):
        assert prompt_text not in trained, (
            f"{model_name} trains on prompt text {prompt_text!r}"
        )

    # The model's own turns must survive, including the answer that gemma-4 renders
    # after the tool results inside the same turn.
    for assistant_text in (_ASSISTANT_TEXT_1, _ASSISTANT_TEXT_2):
        assert assistant_text in trained, (
            f"{model_name} dropped assistant text {assistant_text!r} from the loss"
        )

    # Token-level check: the assertions above compare decoded substrings, which a
    # partially-masked payload would slip past.
    assert _masked_bracket_regions_are_complete(tokenizer, _tool_conversation()), (
        f"{model_name} leaves part of a bracketed tool result in the loss"
    )
