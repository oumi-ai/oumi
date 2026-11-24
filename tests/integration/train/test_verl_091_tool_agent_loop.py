"""oumi's tool agent loop on verl 0.9.1, against scripted generations.

Run by hand in a verl 0.9.1 environment whose Hugging Face cache holds the
tokenizers below. No GPU is needed.
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
from verl.experimental.agent_loop.single_turn_agent_loop import (  # pyright: ignore[reportMissingImports]
    SingleTurnAgentLoop,
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
from oumi_verl_ext.tool_episode import (
    RENDERED_TOOLS_KEY,
    TOOLS_JSON_KEY,
    TRANSCRIPT_KEY,
)

QWEN3 = "Qwen/Qwen3-1.7B"
QWEN3_5 = "Qwen/Qwen3.5-4B"
GEMMA4 = "google/gemma-4-E2B-it"
PROMPT = [{"role": "user", "content": "How many pens are in stock?"}]


def _tool(name: str, **properties: Any) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"The {name} tool.",
            "parameters": {
                "type": "object",
                "properties": properties or {"item": {"type": "string"}},
                "required": list(properties or ["item"]),
            },
        },
    }


def _call(parser: str, name: str, arguments: Any) -> str:
    if parser == "hermes":
        call = json.dumps({"name": name, "arguments": arguments})
        return f"<tool_call>\n{call}\n</tool_call>"
    parameters = "".join(
        f"<parameter={key}>\n{value}\n</parameter>\n"
        for key, value in arguments.items()
    )
    return f"<tool_call>\n<function={name}>\n{parameters}</function>\n</tool_call>"


def _turn(*calls: tuple[str, Any], text: str = "", parser: str = "hermes") -> str:
    return text + "".join(_call(parser, *call) for call in calls) + "<|im_end|>"


PEN = ("get_stock", {"item": "pen"})
ANSWER = _turn(text="Answer: 3")


class _StockTool(BaseTool):
    def __init__(self, name: str = "get_stock", *, ends_episode: bool = False):
        super().__init__(
            {"type": "native", "ends_episode": ends_episode},
            OpenAIFunctionToolSchema.model_validate(_tool(name)),
        )
        self.ran: list[Any] = []

    async def execute(self, instance_id, parameters, **kwargs):
        self.ran.append(parameters)
        return ToolResponse(text=f"{parameters['item']}: 3"), 0.0, {}


class _ScriptedServer:
    def __init__(self, tokenizer, turns, extra_fields):
        self.turns = [
            tokenizer.encode(turn, add_special_tokens=False) for turn in turns
        ]
        self.extra_fields = extra_fields
        self.generated = 0

    async def generate(self, request_id, *, prompt_ids, sampling_params, **kwargs):
        self.generated += 1
        return TokenOutput(
            token_ids=self.turns.pop(0), extra_fields=dict(self.extra_fields)
        )


def _episode(
    turns: list[str],
    *,
    model: str = QWEN3,
    parser: str = "hermes",
    run_tools: tuple = (),
    row_tools: list[dict[str, Any]] | None = None,
    response_length: int = 1024,
    server_extra_fields: dict[str, Any] | None = None,
    loop_class: type = OumiToolAgentLoop,
    **multi_turn: Any,
) -> SimpleNamespace:
    """Runs one episode; verl builds its loops inside the running event loop."""
    tokenizer = AutoTokenizer.from_pretrained(model)
    server = _ScriptedServer(tokenizer, turns, server_extra_fields or {})
    config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {
                    "prompt_length": 4096,
                    "response_length": response_length,
                    "full_determinism": False,
                    "multi_turn": {
                        "format": parser,
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
    row = {"raw_prompt": PROMPT, "extra_info": {}, "tools_kwargs": {}}
    if row_tools is not None:
        row[TOOLS_JSON_KEY] = json.dumps(row_tools)

    async def run():
        loop = loop_class(
            trainer_config=DictConfigWrap(config),
            server_manager=server,
            tokenizer=tokenizer,
            processor=None,
            dataset_cls=RLHFDataset,
            data_config=DictConfigWrap(OmegaConf.create({})),
            hf_model_type=AutoConfig.from_pretrained(model).model_type,
            tools=ToolListWrap(list(run_tools)),
        )
        return await loop.run({}, **row)

    output = asyncio.run(run())
    return SimpleNamespace(
        output=output,
        fields=output.extra_fields,
        transcript=json.loads(output.extra_fields.get(TRANSCRIPT_KEY, "[]")),
        server=server,
        tokenizer=tokenizer,
    )


def _mask_runs(episode: SimpleNamespace) -> list[tuple[int, str]]:
    runs: list[tuple[int, list[int]]] = []
    for token, mask in zip(
        episode.output.response_ids, episode.output.response_mask, strict=True
    ):
        if runs and runs[-1][0] == mask:
            runs[-1][1].append(token)
        else:
            runs.append((mask, [token]))
    return [(mask, episode.tokenizer.decode(ids)) for mask, ids in runs]


@pytest.mark.parametrize(
    ("model", "parser"), [(QWEN3, "hermes"), (QWEN3_5, "qwen3_coder")]
)
def test_one_turn_episode_matches_the_single_turn_loop(model, parser):
    turns = [_turn(PEN, parser=parser)]
    tool = _episode(
        turns,
        model=model,
        parser=parser,
        row_tools=[_tool("get_stock")],
        max_assistant_turns=1,
    )
    single = _episode(turns, model=model, parser=parser, loop_class=SingleTurnAgentLoop)

    assert tool.output.response_ids == single.output.response_ids
    assert tool.output.response_mask == single.output.response_mask
    assert tool.output.prompt_ids == tool.tokenizer.apply_chat_template(
        PROMPT,
        tools=[_tool("get_stock")],
        add_generation_prompt=True,
        tokenize=True,
        return_dict=False,
    )


def test_tool_turns_never_enter_the_loss():
    episode = _episode([_turn(PEN), ANSWER], run_tools=(_StockTool(),))

    runs = _mask_runs(episode)

    assert [mask for mask, _ in runs] == [1, 0, 1]
    assert (runs[0][1], runs[2][1]) == (_turn(PEN), ANSWER)
    assert "pen: 3" in runs[1][1]
    assert runs[1][1].endswith("<|im_start|>assistant\n")


@pytest.mark.parametrize(
    ("caps", "turns", "ran", "generated", "dropped"),
    [
        ({"max_assistant_turns": 2}, [_turn(PEN), _turn(PEN), ANSWER], 2, 2, 0),
        ({}, [_turn(PEN), _turn(PEN), ANSWER], 2, 3, 0),
        ({"max_user_turns": 1}, [_turn(PEN), ANSWER], 1, 2, 0),
        ({"max_user_turns": 1}, [_turn(PEN), _turn(PEN), ANSWER], 1, 2, 1),
    ],
    ids=[
        "max_assistant_turns_2",
        "uncapped",
        "max_user_turns_1_answers",
        "max_user_turns_1_calls",
    ],
)
def test_last_turn_calls_run_before_the_cap_ends_the_episode(
    caps, turns, ran, generated, dropped
):
    tool = _StockTool()

    episode = _episode(turns, run_tools=(tool,), **caps)

    assert len(tool.ran) == ran
    assert episode.server.generated == generated
    last = episode.transcript[-1]
    assert last["role"] == "assistant"
    assert bool(last.get("tool_calls")) == (turns[generated - 1] != ANSWER)
    assert episode.fields["oumi_tool_calls_dropped"] == dropped


@pytest.mark.parametrize(
    "ending_tool", ["schema_only", "ends_episode", "ending_beside_a_lookup"]
)
def test_calls_that_end_the_episode(ending_tool):
    lookup, submit = _StockTool(), _StockTool("submit", ends_episode=True)
    setup = {
        "schema_only": ([_turn(PEN)], {"row_tools": [_tool("get_stock")]}),
        "ends_episode": (
            [_turn(("submit", {"item": "3"}))],
            {"run_tools": (lookup, submit)},
        ),
        "ending_beside_a_lookup": (
            [_turn(("submit", {"item": "3"}), PEN)],
            {"run_tools": (lookup, submit), "max_parallel_calls": 2},
        ),
    }
    turns, kwargs = setup[ending_tool]

    episode = _episode([*turns, ANSWER], **kwargs)

    assert episode.server.generated == 1
    assert lookup.ran == submit.ran == []
    assert episode.transcript[-1]["tool_calls"]
    dropped = {"schema_only": 0, "ends_episode": 0, "ending_beside_a_lookup": 1}
    assert episode.fields["oumi_tool_calls_dropped"] == dropped[ending_tool]


def test_dropped_and_malformed_calls_are_counted():
    tool = _StockTool()
    turn = _turn(PEN, PEN, PEN, text="<tool_call>\nnot json\n</tool_call>")

    episode = _episode([turn, ANSWER], run_tools=(tool,))

    assert len(tool.ran) == 1
    assert len(episode.transcript[1]["tool_calls"]) == 1
    assert episode.fields["oumi_tool_calls_dropped"] == 2
    assert episode.fields["oumi_tool_calls_malformed"] == 1


def test_the_reward_gets_the_transcript_and_the_server_s_extra_fields():
    episode = _episode(
        [_turn(PEN), ANSWER],
        run_tools=(_StockTool(),),
        server_extra_fields={"min_global_steps": 3, "max_global_steps": 3},
    )

    assert episode.transcript[-1]["content"] == "Answer: 3"
    assert json.loads(episode.fields[RENDERED_TOOLS_KEY]) == [_tool("get_stock")]
    assert episode.fields["min_global_steps"] == 3
    assert "turn_scores" in episode.fields


@pytest.mark.parametrize("budget_runs_out", ["mid_turn", "after_tool_turn"])
def test_a_budget_cut_episode_gives_a_judgeable_transcript(budget_runs_out):
    probe = _episode([_turn(PEN), ANSWER], run_tools=(_StockTool(),))
    first_turn_and_tool = len(probe.output.response_ids) - len(
        probe.tokenizer.encode(ANSWER, add_special_tokens=False)
    )
    long_answer = _turn(text="Answer: " + "three " * 50)
    response_length = {
        "mid_turn": first_turn_and_tool + 10,
        "after_tool_turn": first_turn_and_tool - 1,
    }[budget_runs_out]

    episode = _episode(
        [_turn(PEN), long_answer],
        run_tools=(_StockTool(),),
        response_length=response_length,
    )

    last = episode.transcript[-1]
    assert last["role"] == "assistant"
    if budget_runs_out == "mid_turn":
        kept = episode.output.response_ids[first_turn_and_tool:]
        assert len(kept) == 10
        assert last["content"] == episode.tokenizer.decode(kept)
    else:
        assert last["tool_calls"]


def test_model_mistakes_become_tool_messages():
    # hermes re-serializes arguments, so a non-object is the invalid one it passes on.
    turn = _turn(("lookup", {"item": "pen"}), ("get_stock", "pen"))

    episode = _episode([turn, ANSWER], run_tools=(_StockTool(),), max_parallel_calls=2)

    tool_messages = [m["content"] for m in episode.transcript if m["role"] == "tool"]
    assert tool_messages[0].startswith("Unknown function 'lookup'")
    assert tool_messages[1].startswith("Error executing tool 'get_stock'")
    assert episode.transcript[-1]["content"] == "Answer: 3"


def test_run_tools_render_their_full_schema():
    full = _tool(
        "get_stock",
        item={"type": "string", "default": "pen"},
        sizes={"type": "array", "items": {"type": "integer"}},
    )

    episode = _episode([ANSWER], run_tools=(_StockTool(),), row_tools=[full])

    prompt = episode.tokenizer.decode(episode.output.prompt_ids)
    assert json.dumps(full["function"]["parameters"]) in prompt
    verl_dump = json.dumps(OpenAIFunctionToolSchema.model_validate(full).model_dump())
    assert "default" not in verl_dump and "items" not in verl_dump


def test_qwen3_coder_parses_calls_to_row_tools():
    episode = _episode(
        [_turn(PEN, parser="qwen3_coder")],
        model=QWEN3_5,
        parser="qwen3_coder",
        row_tools=[_tool("get_stock")],
    )

    (call,) = episode.transcript[-1]["tool_calls"]
    assert call["function"] == {"name": "get_stock", "arguments": {"item": "pen"}}
    assert episode.fields["oumi_tool_calls"] == 1


@pytest.mark.xfail(
    strict=True,
    reason="verl 0.9.1's gemma4 parser reads an array argument flat, so the "
    "call runs with a mangled argument and isn't counted as malformed.",
)
def test_gemma4_array_arguments_parse_or_count_as_malformed():
    tool = _StockTool("get_items")
    turn = '<|tool_call>call:get_items{item:[<|"|>pen<|"|>,<|"|>ink<|"|>]}<tool_call|>'

    episode = _episode(
        [turn, "Answer: 3<turn|>"], model=GEMMA4, parser="gemma4", run_tools=(tool,)
    )

    ran_as_written = tool.ran == [{"item": ["pen", "ink"]}]
    counted = not tool.ran and episode.fields["oumi_tool_calls_malformed"] == 1
    assert ran_as_written or counted
