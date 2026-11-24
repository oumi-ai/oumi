"""oumi's tool agent loop on verl 0.9.1, against scripted generations.

Run by hand in a verl 0.9.1 environment whose Hugging Face cache holds the
tokenizer below. No GPU is needed.
"""

import asyncio
import json
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("verl", minversion="0.9.1")

from omegaconf import OmegaConf
from transformers import AutoConfig, AutoTokenizer
from verl.experimental.agent_loop.agent_loop import (  # pyright: ignore[reportMissingImports]
    DictConfigWrap,
    ToolListWrap,
)
from verl.tools.base_tool import BaseTool  # pyright: ignore[reportMissingImports]
from verl.tools.schemas import (  # pyright: ignore[reportMissingImports]
    OpenAIFunctionToolSchema,
    ToolResponse,
)
from verl.utils.dataset.rl_dataset import (  # pyright: ignore[reportMissingImports]
    RLHFDataset,
)
from verl.workers.rollout.replica import (  # pyright: ignore[reportMissingImports]
    TokenOutput,
)

from oumi_verl_ext.tool_agent_loop import OumiToolAgentLoop
from oumi_verl_ext.tool_episode import RENDERED_TOOLS_KEY, TRANSCRIPT_KEY

QWEN3 = "Qwen/Qwen3-1.7B"
PROMPT = [{"role": "user", "content": "How many pens are in stock?"}]


def _tool(name: str) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"The {name} tool.",
            "parameters": {
                "type": "object",
                "properties": {"item": {"type": "string"}},
                "required": ["item"],
            },
        },
    }


def _call(name: str, arguments: dict[str, Any]) -> str:
    call = json.dumps({"name": name, "arguments": arguments})
    return f"<tool_call>\n{call}\n</tool_call>"


def _turn(*calls: tuple[str, dict[str, Any]], text: str = "") -> str:
    return text + "".join(_call(*call) for call in calls) + "<|im_end|>"


PEN = ("get_stock", {"item": "pen"})
ANSWER = _turn(text="Answer: 3")


class _StockTool(BaseTool):
    def __init__(self):
        super().__init__(
            {"type": "native"},
            OpenAIFunctionToolSchema.model_validate(_tool("get_stock")),
        )

    async def execute(self, instance_id, parameters, **kwargs):
        return ToolResponse(text=f"{parameters['item']}: 3"), 0.0, {}


class _ScriptedServer:
    def __init__(self, tokenizer, turns):
        self.turns = [
            tokenizer.encode(turn, add_special_tokens=False) for turn in turns
        ]

    async def generate(self, request_id, *, prompt_ids, sampling_params, **kwargs):
        return TokenOutput(token_ids=self.turns.pop(0))


def _episode(
    turns: list[str], *, response_length: int = 1024, **multi_turn: Any
) -> SimpleNamespace:
    """Runs one episode; verl builds its loops inside the running event loop."""
    tokenizer = AutoTokenizer.from_pretrained(QWEN3)
    config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {
                    "prompt_length": 4096,
                    "response_length": response_length,
                    "full_determinism": False,
                    "multi_turn": {
                        "format": "hermes",
                        "max_assistant_turns": None,
                        "max_user_turns": None,
                        "max_parallel_calls": 1,
                        "max_tool_response_length": 4096,
                        "tool_response_truncate_side": "middle",
                        **multi_turn,
                    },
                }
            }
        }
    )

    async def run():
        loop = OumiToolAgentLoop(
            trainer_config=DictConfigWrap(config),
            server_manager=_ScriptedServer(tokenizer, turns),
            tokenizer=tokenizer,
            processor=None,
            dataset_cls=RLHFDataset,
            data_config=DictConfigWrap(OmegaConf.create({})),
            hf_model_type=AutoConfig.from_pretrained(QWEN3).model_type,
            tools=ToolListWrap([_StockTool()]),
        )
        # tools_kwargs is None as parquet reads it back for a row without it.
        return await loop.run({}, raw_prompt=PROMPT, extra_info={}, tools_kwargs=None)

    output = asyncio.run(run())
    return SimpleNamespace(
        output=output,
        fields=output.extra_fields,
        transcript=json.loads(output.extra_fields[TRANSCRIPT_KEY]),
        tokenizer=tokenizer,
    )


def test_an_episode_gives_the_judge_its_transcript_and_tools():
    episode = _episode([_turn(PEN), ANSWER])

    assert [message["role"] for message in episode.transcript] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert episode.transcript[1]["tool_calls"]
    assert episode.transcript[2]["content"] == "pen: 3"
    assert episode.transcript[3]["content"] == "Answer: 3"
    assert json.loads(episode.fields[RENDERED_TOOLS_KEY]) == [_tool("get_stock")]


def test_a_budget_cut_turn_keeps_only_its_tokens_inside_the_response():
    probe = _episode([_turn(PEN), ANSWER])
    first_turn_and_tool = len(probe.output.response_ids) - len(
        probe.tokenizer.encode(ANSWER, add_special_tokens=False)
    )
    long_answer = _turn(text="Answer: " + "three " * 50)

    episode = _episode(
        [_turn(PEN), long_answer], response_length=first_turn_and_tool + 10
    )

    kept = episode.output.response_ids[first_turn_and_tool:]
    assert len(kept) == 10
    assert episode.transcript[-1] == {
        "role": "assistant",
        "content": episode.tokenizer.decode(kept),
    }


def test_a_turn_the_assistant_turn_cap_stopped_reaches_the_judge_as_text():
    episode = _episode([_turn(PEN)], max_assistant_turns=1)

    assert episode.transcript == [
        *PROMPT,
        {"role": "assistant", "content": _call(*PEN)},
    ]
