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

"""Multi-turn tool-calling rollout for TRL's GRPO trainer."""

from __future__ import annotations

import json
import re
from functools import cache
from typing import Any

import torch
from trl.chat_template_utils import (
    get_training_chat_template,
    is_chat_template_prefix_preserving,
)

from oumi.core.configs.environment_config import EnvironmentConfig
from oumi.core.registry import RegistryType, register
from oumi.core.synthesis.tool_router import ToolRouter
from oumi.core.tokenizers.utils import detect_chat_template_tool_format
from oumi.utils.logging import logger

TOOL_AGENT_ROLLOUT = "tool_agent"

_TOOL_CALL_PATTERN = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


@cache
def _load_router(environment_config_path: str) -> ToolRouter:
    """Build the process-wide router every rollout clones."""
    env_config = EnvironmentConfig.from_yaml(environment_config_path)
    env_config.finalize_and_validate()
    return ToolRouter.from_environment_config(env_config)


def _as_messages(prompt: Any) -> list[dict[str, Any]]:
    """Normalize a TRL prompt into chat messages."""
    messages = (
        [{"role": "user", "content": prompt}]
        if isinstance(prompt, str)
        else [dict(message) for message in prompt]
    )
    for message in messages:
        if not isinstance(message.get("content"), str):
            raise ValueError(
                "The tool_agent rollout only supports text prompts; "
                f"got {type(message.get('content')).__name__} content."
            )
    return messages


def _parse_tool_calls(text: str) -> list[tuple[str, dict[str, Any]]]:
    """Extract `<tool_call>{"name": ..., "arguments": ...}</tool_call>` blocks."""
    calls = []
    for match in _TOOL_CALL_PATTERN.finditer(text):
        try:
            payload = json.loads(match.group(1))
        except json.JSONDecodeError:
            logger.warning(f"Tool call is not valid JSON: {match.group(1)!r}")
            continue
        name = payload.get("name")
        arguments = payload.get("arguments")
        if isinstance(name, str) and isinstance(arguments, dict):
            calls.append((name, arguments))
        else:
            logger.warning(f"Tool call is missing a name or arguments: {payload!r}")
    return calls


def _render(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    chat_template_kwargs: dict[str, Any],
    *,
    add_generation_prompt: bool,
    chat_template: str | None = None,
) -> list[int]:
    extra = {"chat_template": chat_template} if chat_template else {}
    return tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
        return_dict=False,
        **extra,
        **chat_template_kwargs,
    )


def _tool_suffix_ids(
    tokenizer: Any,
    tools: list[dict[str, Any]],
    tool_name: str,
    tool_messages: list[dict[str, Any]],
    chat_template_kwargs: dict[str, Any],
    chat_template: str | None,
) -> list[int]:
    """Tokenize the tool messages plus the next generation prompt.

    Rendered against a dummy assistant turn so the suffix starts right after
    the turn's end token and composes with the model's own tokens.
    """
    tool_format = detect_chat_template_tool_format(tokenizer)
    dummy = [
        {"role": "user", "content": "dummy"},
        {
            "role": "assistant",
            "content": "" if tool_format.content == "empty" else None,
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": {} if tool_format.arguments == "mapping" else "{}",
                    },
                }
            ],
        },
    ]
    prefix = _render(
        tokenizer,
        dummy,
        tools,
        chat_template_kwargs,
        add_generation_prompt=False,
        chat_template=chat_template,
    )
    full = _render(
        tokenizer,
        dummy + tool_messages,
        tools,
        chat_template_kwargs,
        add_generation_prompt=True,
        chat_template=chat_template,
    )

    eos_positions = [
        index
        for index, token_id in enumerate(prefix)
        if token_id == tokenizer.eos_token_id
    ]
    if eos_positions:
        prefix = prefix[: eos_positions[-1] + 1]
    if full[: len(prefix)] != prefix:
        raise ValueError(
            "The chat template is not prefix-preserving, so tool results cannot "
            "be appended to the generated turn."
        )
    return full[len(prefix) :]


def _generate(
    model: Any,
    tokenizer: Any,
    sequences: list[list[int]],
    args: Any,
    *,
    max_new_tokens: int,
) -> list[list[int]]:
    """Sample one continuation per sequence, batched with left padding."""
    device = next(model.parameters()).device
    pad_token_id = tokenizer.pad_token_id
    longest = max(len(ids) for ids in sequences)
    input_ids = torch.full(
        (len(sequences), longest), pad_token_id, dtype=torch.long, device=device
    )
    attention_mask = torch.zeros_like(input_ids)
    for row, ids in enumerate(sequences):
        input_ids[row, longest - len(ids) :] = torch.tensor(
            ids, dtype=torch.long, device=device
        )
        attention_mask[row, longest - len(ids) :] = 1

    # Gradient checkpointing corrupts generation in train mode, so generate in
    # eval mode and restore the mode the trainer left the model in.
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                pad_token_id=pad_token_id,
            )
    finally:
        model.train(was_training)

    eos_token_id = tokenizer.eos_token_id
    continuations = []
    for row in output[:, longest:].tolist():
        if eos_token_id in row:
            row = row[: row.index(eos_token_id) + 1]
        else:
            row = [token for token in row if token != pad_token_id]
        continuations.append(row)
    return continuations


def _execute_calls(
    router: ToolRouter, calls: list[tuple[str, dict[str, Any]]]
) -> list[dict[str, Any]]:
    """Run each call and turn a failing tool into an observation."""
    messages = []
    for index, (name, arguments) in enumerate(calls):
        try:
            output = router.route_batch([(name, arguments)])[0].output
        except Exception as error:  # A tool failure is an observation, not a crash.
            logger.warning(f"Tool '{name}' call failed: {error}")
            output = {"error": str(error)}
        content = output if isinstance(output, str) else json.dumps(output)
        messages.append(
            {
                "role": "tool",
                "name": name,
                "tool_call_id": f"call_{index}",
                "content": content,
            }
        )
    return messages


def _tool_definitions(router: ToolRouter) -> list[dict[str, Any]]:
    return [tool.model_dump(mode="json") for tool in router.tool_specs]


def _run_one(
    model: Any,
    tokenizer: Any,
    router: ToolRouter,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    chat_template_kwargs: dict[str, Any],
    args: Any,
    max_tool_turns: int,
    chat_template: str | None,
) -> tuple[list[int], list[int], list[int]]:
    """Roll out one sample: generate, run tools, and generate again."""
    prompt_ids = _render(
        tokenizer,
        messages,
        tools,
        chat_template_kwargs,
        add_generation_prompt=True,
        chat_template=chat_template,
    )
    completion_ids: list[int] = []
    env_mask: list[int] = []

    for turn in range(max_tool_turns + 1):
        budget = args.max_completion_length - len(completion_ids)
        if budget <= 0:
            break
        new_ids = _generate(
            model,
            tokenizer,
            [prompt_ids + completion_ids],
            args,
            max_new_tokens=budget,
        )[0]
        completion_ids.extend(new_ids)
        env_mask.extend([1] * len(new_ids))

        # A truncated turn has no end token, so the tool result cannot follow it.
        if not new_ids or new_ids[-1] != tokenizer.eos_token_id:
            break
        calls = _parse_tool_calls(tokenizer.decode(new_ids, skip_special_tokens=True))
        if not calls or turn == max_tool_turns:
            break

        tool_messages = _execute_calls(router, calls)
        suffix_ids = _tool_suffix_ids(
            tokenizer,
            tools,
            calls[0][0],
            tool_messages,
            chat_template_kwargs,
            chat_template,
        )
        completion_ids.extend(suffix_ids)
        env_mask.extend([0] * len(suffix_ids))

    return prompt_ids, completion_ids, env_mask


@register(TOOL_AGENT_ROLLOUT, RegistryType.ROLLOUT_FUNCTION)
def tool_agent_rollout(
    prompts: list[Any],
    trainer: Any,
    *,
    environment_config_path: str,
    max_tool_turns: int = 4,
) -> dict[str, Any]:
    """Generate GRPO completions with multi-turn Oumi tool calls.

    Each prompt gets its own `ToolRouter` clone, so stateful environments stay
    isolated across samples. Tool result tokens are marked `0` in `env_mask`
    and therefore excluded from the policy loss.
    """
    router_template = _load_router(environment_config_path)
    tokenizer = trainer.processing_class
    model = trainer.model
    tools = _tool_definitions(router_template)
    chat_template_kwargs = dict(getattr(trainer, "chat_template_kwargs", None) or {})
    # Some templates render an earlier turn differently once a later one exists.
    # TRL ships prefix-preserving replacements for those; use the same one it uses.
    chat_template = (
        None
        if is_chat_template_prefix_preserving(tokenizer)
        else get_training_chat_template(tokenizer)
    )

    prompt_ids, completion_ids, env_masks = [], [], []
    for prompt in prompts:
        router = router_template.for_sample()
        try:
            one_prompt, one_completion, one_mask = _run_one(
                model,
                tokenizer,
                router,
                _as_messages(prompt),
                tools,
                chat_template_kwargs,
                trainer.args,
                max_tool_turns,
                chat_template,
            )
        finally:
            router.close()
        prompt_ids.append(one_prompt)
        completion_ids.append(one_completion)
        env_masks.append(one_mask)

    return {
        "prompt_ids": prompt_ids,
        "completion_ids": completion_ids,
        "logprobs": None,
        "env_mask": env_masks,
    }
