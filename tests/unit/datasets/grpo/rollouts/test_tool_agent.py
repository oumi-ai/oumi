from types import SimpleNamespace

import pytest
import torch

import oumi.datasets.grpo.rollouts.tool_agent as tool_agent
from oumi.core.types.tool_call import ToolResult

_EOS = 99
_TOOL_CALL = '<tool_call>{"name": "get_price", "arguments": {"item": "x"}}</tool_call>'


class _FakeTokenizer:
    def __init__(self) -> None:
        self.pad_token_id = 0
        self.eos_token_id = _EOS

    def decode(self, ids, skip_special_tokens=True):
        if 5 in ids:
            return _TOOL_CALL
        return "The price is 9."


class _FakeToolSpec:
    def model_dump(self, mode):
        return {
            "type": "function",
            "function": {
                "name": "get_price",
                "description": "Return the price of an item.",
                "parameters": {"type": "object"},
            },
        }


class _FakeRouter:
    def __init__(self, outputs=None, error=None) -> None:
        self.tool_specs = [_FakeToolSpec()]
        self.outputs: list = outputs if outputs is not None else [{"price": 9}]
        self.error = error
        self.closed = 0
        self.call_count = 0

    def route_batch(self, calls):
        self.call_count += 1
        if self.error is not None:
            raise self.error
        return [ToolResult(output=output) for output in self.outputs]  # type: ignore[arg-type]


class _SampleRouter(_FakeRouter):
    def close(self) -> None:
        self.closed += 1


class _TemplateRouter(_FakeRouter):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.samples: list[_SampleRouter] = []

    def for_sample(self):
        sample = _SampleRouter(outputs=self.outputs, error=self.error)
        self.samples.append(sample)
        return sample


def _install(monkeypatch, generations, suffix_ids=(90, 91)):
    router = _TemplateRouter()
    monkeypatch.setattr(tool_agent, "_load_router", lambda path: router)
    monkeypatch.setattr(tool_agent, "_render", lambda *args, **kwargs: [1, 2])
    monkeypatch.setattr(
        tool_agent,
        "_tool_suffix_ids",
        lambda *args, **kwargs: list(suffix_ids),
    )
    monkeypatch.setattr(
        tool_agent, "is_chat_template_prefix_preserving", lambda tokenizer: True
    )
    calls = iter(generations)

    def fake_generate(model, tokenizer, sequences, args, *, max_new_tokens):
        return [next(calls)]

    monkeypatch.setattr(tool_agent, "_generate", fake_generate)
    return router


def _trainer(tokenizer=None):
    tokenizer = tokenizer or _FakeTokenizer()
    return SimpleNamespace(
        processing_class=tokenizer,
        model=object(),
        chat_template_kwargs={"enable_thinking": False},
        args=SimpleNamespace(
            temperature=1.0, top_p=1.0, top_k=10, max_completion_length=100
        ),
    )


def test_parse_tool_calls_reads_each_block():
    calls = tool_agent._parse_tool_calls(
        '<tool_call>{"name": "a", "arguments": {"x": 1}}</tool_call>'
        'noise<tool_call>{"name": "b", "arguments": {}}</tool_call>'
    )

    assert calls == [("a", {"x": 1}), ("b", {})]


def test_parse_tool_calls_skips_invalid_json():
    assert tool_agent._parse_tool_calls("<tool_call>{not json}</tool_call>") == []


def test_as_messages_wraps_a_string_prompt():
    assert tool_agent._as_messages("hi") == [{"role": "user", "content": "hi"}]


def test_as_messages_rejects_non_text_content():
    with pytest.raises(ValueError, match="only supports text prompts"):
        tool_agent._as_messages(
            [{"role": "user", "content": [{"type": "image", "image": "x"}]}]
        )


def test_execute_calls_marks_tool_results():
    router = _FakeRouter()
    messages = tool_agent._execute_calls(router, [("get_price", {"item": "x"})])  # type: ignore[arg-type]  # type: ignore[arg-type]

    assert messages == [
        {
            "role": "tool",
            "name": "get_price",
            "tool_call_id": "call_0",
            "content": '{"price": 9}',
        }
    ]


def test_execute_calls_turns_a_failure_into_an_observation():
    router = _FakeRouter(error=ValueError("no such item"))

    messages = tool_agent._execute_calls(router, [("get_price", {"item": "x"})])  # type: ignore[arg-type]

    assert messages[0]["content"] == '{"error": "no such item"}'


def test_rollout_appends_tool_results_and_masks_their_tokens(monkeypatch):
    router = _install(monkeypatch, [[5, _EOS], [7, _EOS]])

    result = tool_agent.tool_agent_rollout(
        [[{"role": "user", "content": "hi"}]],
        _trainer(),
        environment_config_path="e",
    )

    assert result["prompt_ids"] == [[1, 2]]
    assert result["completion_ids"] == [[5, _EOS, 90, 91, 7, _EOS]]
    assert result["env_mask"] == [[1, 1, 0, 0, 1, 1]]
    assert result["logprobs"] is None
    assert len(router.samples) == 1
    assert router.samples[0].closed == 1


def test_rollout_stops_after_max_tool_turns(monkeypatch):
    router = _install(monkeypatch, [[5, _EOS], [5, _EOS]])

    result = tool_agent.tool_agent_rollout(
        [[{"role": "user", "content": "hi"}]],
        _trainer(),
        environment_config_path="e",
        max_tool_turns=1,
    )

    # Turn 0 runs the tool; turn 1 is the last allowed turn, so its call is dropped.
    assert result["completion_ids"] == [[5, _EOS, 90, 91, 5, _EOS]]
    assert result["env_mask"] == [[1, 1, 0, 0, 1, 1]]
    assert router.samples[0].call_count == 1


def test_rollout_closes_the_router_when_generation_fails(monkeypatch):
    router = _install(monkeypatch, [])

    def failing_generate(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(tool_agent, "_generate", failing_generate)

    with pytest.raises(RuntimeError, match="boom"):
        tool_agent.tool_agent_rollout(
            [[{"role": "user", "content": "hi"}]],
            _trainer(),
            environment_config_path="e",
        )

    assert router.samples[0].closed == 1


def test_rollout_gives_each_prompt_its_own_router(monkeypatch):
    router = _install(monkeypatch, [[7, _EOS], [7, _EOS]])

    tool_agent.tool_agent_rollout(
        [[{"role": "user", "content": "a"}], [{"role": "user", "content": "b"}]],
        _trainer(),
        environment_config_path="e",
    )

    assert len(router.samples) == 2


def test_generate_trims_at_eos_and_restores_train_mode():
    class _Model:
        def __init__(self) -> None:
            self.training = True
            self.seen_training: list[bool] = []

        def parameters(self):
            yield torch.zeros(1)

        def eval(self):
            self.training = False

        def train(self, mode=True):
            self.training = mode

        def generate(self, **kwargs):
            self.seen_training.append(self.training)
            return torch.tensor([[1, 2, _EOS, 50, 50]])

    args = SimpleNamespace(temperature=1.0, top_p=1.0, top_k=10)
    model = _Model()

    continuations = tool_agent._generate(
        model, _FakeTokenizer(), [[1, 2]], args, max_new_tokens=8
    )

    assert continuations == [[_EOS]]
    assert model.seen_training == [False]
    assert model.training is True
