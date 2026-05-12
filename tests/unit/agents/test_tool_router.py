# Copyright 2025 - Oumi
# Licensed under the Apache License, Version 2.0
"""Contract tests for ``ToolRouter``."""

import pytest

import oumi.environments  # noqa: F401  populates env registry
from oumi.agents.exceptions import InvalidToolArgumentsError, UnknownToolError
from oumi.agents.tool_router import ToolRouter
from oumi.core.configs.environment_config import EnvironmentConfig
from oumi.core.configs.params.environment_params import EnvironmentParams
from oumi.core.configs.params.tool_params import ToolParams


def _tool(tool_id: str, *, parameters: dict | None = None) -> ToolParams:
    return ToolParams(
        id=tool_id,
        name=tool_id,
        description=f"Tool {tool_id}.",
        parameters=parameters or {"type": "object", "properties": {}},
    )


def _env(env_id: str, tools: list[ToolParams]) -> EnvironmentParams:
    return EnvironmentParams(
        id=env_id,
        name=env_id,
        description=f"Env {env_id}.",
        env_type="synthetic",
        tools=tools,
        env_kwargs={"system_prompt": "."},
    )


def test_router_flattens_tools_across_envs():
    config = EnvironmentConfig(
        environments=[
            _env("env_a", [_tool("alpha"), _tool("beta")]),
            _env("env_b", [_tool("gamma")]),
        ]
    )
    router = ToolRouter.from_environment_config(config)

    spec_names = {s.function.name for s in router.tool_specs}
    assert spec_names == {"alpha", "beta", "gamma"}
    assert router.route("alpha")[0] == "env_a"
    assert router.route("beta")[0] == "env_a"
    assert router.route("gamma")[0] == "env_b"


def test_router_rejects_unknown_tool_name():
    """SLM hallucinates a tool that doesn't exist — pre-gate must catch it."""
    config = EnvironmentConfig(environments=[_env("e", [_tool("known")])])
    router = ToolRouter.from_environment_config(config)
    with pytest.raises(UnknownToolError, match="not registered"):
        router.route("hallucinated")


def test_router_parses_valid_arguments():
    config = EnvironmentConfig(
        environments=[
            _env(
                "e",
                [
                    _tool(
                        "echo",
                        parameters={
                            "type": "object",
                            "properties": {"x": {"type": "integer"}},
                            "required": ["x"],
                        },
                    )
                ],
            )
        ]
    )
    router = ToolRouter.from_environment_config(config)
    _, tool = router.route("echo")
    assert router.parse_and_validate_arguments(tool, '{"x": 7}') == {"x": 7}


def test_router_rejects_malformed_json():
    config = EnvironmentConfig(environments=[_env("e", [_tool("echo")])])
    router = ToolRouter.from_environment_config(config)
    _, tool = router.route("echo")
    with pytest.raises(InvalidToolArgumentsError, match="not valid JSON"):
        router.parse_and_validate_arguments(tool, "{not json")


def test_router_rejects_non_object_arguments():
    """OpenAI wire format requires JSON object args; reject arrays / scalars."""
    config = EnvironmentConfig(environments=[_env("e", [_tool("echo")])])
    router = ToolRouter.from_environment_config(config)
    _, tool = router.route("echo")
    with pytest.raises(InvalidToolArgumentsError, match="must be a JSON object"):
        router.parse_and_validate_arguments(tool, "[1, 2, 3]")


def test_router_rejects_schema_mismatch():
    """Well-formed JSON, wrong shape — schema validation must fire."""
    config = EnvironmentConfig(
        environments=[
            _env(
                "e",
                [
                    _tool(
                        "echo",
                        parameters={
                            "type": "object",
                            "properties": {"x": {"type": "integer"}},
                            "required": ["x"],
                        },
                    )
                ],
            )
        ]
    )
    router = ToolRouter.from_environment_config(config)
    _, tool = router.route("echo")
    with pytest.raises(InvalidToolArgumentsError, match="schema validation"):
        router.parse_and_validate_arguments(tool, '{"x": "not-an-int"}')


def test_router_treats_empty_arguments_string_as_empty_object():
    """Anthropic and OpenAI both emit ``""`` for parameterless tool calls.
    Schema with no required fields should accept that."""
    config = EnvironmentConfig(environments=[_env("e", [_tool("ping")])])
    router = ToolRouter.from_environment_config(config)
    _, tool = router.route("ping")
    assert router.parse_and_validate_arguments(tool, "") == {}
