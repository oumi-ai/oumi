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

import inspect
import sqlite3
from pathlib import Path
from typing import Any

import pytest
from transformers.utils import get_json_schema

from oumi.builders.grpo_environments import (
    OumiEnvironmentAdapter,
    build_environment_factory,
)
from oumi.core.types.tool_call import ToolResult


def read_note(arguments: dict, context: sqlite3.Connection) -> ToolResult:
    """Read a note by id; local so the test doesn't depend on an example."""
    row = context.execute(
        "SELECT body FROM notes WHERE id = ?", (arguments["note_id"],)
    ).fetchone()
    return ToolResult(output={"body": row[0] if row else None})


def write_note(arguments: dict, context: sqlite3.Connection) -> ToolResult:
    """Overwrite a note's body (uncommitted; rolled back at episode end)."""
    context.execute(
        "UPDATE notes SET body = ? WHERE id = ?",
        (arguments["body"], arguments["note_id"]),
    )
    return ToolResult(output={"ok": True})


@pytest.fixture
def env_config_path(tmp_path: Path) -> str:
    """A one-tool lookup environment, in the shape the synthesizer already uses."""
    path = tmp_path / "environment.yaml"
    path.write_text(
        """
environments:
  - id: store
    name: store
    description: A small store price catalog.
    env_type: lookup
    tools:
      - id: get_price
        name: get_price
        description: Return the price, in USD, of an item sold by the store.
        parameters:
          type: object
          properties:
            item:
              type: string
              description: The item name, e.g. "apple".
          required:
            - item
    env_kwargs:
      lookup_table:
        get_price:
          - input: { item: mug }
            output: { item: mug, price_usd: 9 }
"""
    )
    return str(path)


@pytest.fixture
def stateful_env_config_path(tmp_path: Path) -> str:
    """A database env: ``requires_isolation()`` is true, so it is rebuilt per step.

    ``ToolRouter.from_environment_config`` closes such envs after building them
    and expects ``for_sample()`` to rebuild them, so this is the case that
    catches an adapter dispatching on the parent router.
    """
    path = tmp_path / "stateful_environment.yaml"
    path.write_text(
        f"""
environments:
  - id: notes
    name: notes
    description: A tiny notes table.
    env_type: database
    env_kwargs:
      schema_sql: "CREATE TABLE notes (id INTEGER PRIMARY KEY, body TEXT);"
      seed_sql: "INSERT INTO notes (id, body) VALUES (1, 'original');"
    tools:
      - id: read_note
        name: read_note
        description: Read a note by id.
        parameters:
          type: object
          properties:
            note_id:
              type: integer
          required:
            - note_id
        executor: {__name__}.read_note
        read_only: true
      - id: write_note
        name: write_note
        description: Overwrite a note's body.
        parameters:
          type: object
          properties:
            note_id:
              type: integer
            body:
              type: string
          required:
            - note_id
            - body
        executor: {__name__}.write_note
        read_only: false
"""
    )
    return str(path)


def _build(config_path: str) -> Any:
    """Build an adapter; its tools are installed on a subclass at runtime."""
    return build_environment_factory(config_path)()


def test_tools_are_exposed_as_bound_methods(env_config_path: str):
    """TRL discovers tools with ``inspect.ismethod``, so they must be bound."""
    env = _build(env_config_path)

    assert inspect.ismethod(env.get_price)


def test_tool_schema_is_advertised_with_its_arguments(env_config_path: str):
    """TRL renders the tool schema from the callable's signature and docstring."""
    env = _build(env_config_path)

    schema = get_json_schema(env.get_price)["function"]

    assert schema["name"] == "get_price"
    assert schema["description"].startswith("Return the price")
    assert schema["parameters"]["properties"]["item"]["type"] == "string"
    assert schema["parameters"]["required"] == ["item"]


def test_tool_call_returns_the_environment_output(env_config_path: str):
    env = _build(env_config_path)

    assert env.get_price(item="mug") == '{"item": "mug", "price_usd": 9}'


def test_each_factory_call_gets_its_own_environment(env_config_path: str):
    """TRL builds one environment per generation slot; state must not be shared."""
    factory = build_environment_factory(env_config_path)

    first, second = factory(), factory()

    assert first._router is not second._router


def test_reset_accepts_the_dataset_row(env_config_path: str):
    """TRL calls ``reset(**row)``, so it must tolerate arbitrary columns."""
    env = _build(env_config_path)

    assert env.reset(prompt="hi", answer=9, extra=[1, 2]) is None


def test_reset_rebuilds_the_rollout_router(env_config_path: str):
    """TRL reuses one adapter across steps, so ``reset()`` must rebuild state.

    ``ToolRouter.for_sample()`` shares stateless envs but returns a new router,
    which is what lets a stateful env start each step from its config.
    """
    env = _build(env_config_path)
    before = env._router

    env.reset()

    assert env._router is not before
    assert env.get_price(item="mug") == '{"item": "mug", "price_usd": 9}'


def test_stateful_env_works_before_any_reset(stateful_env_config_path: str):
    """``from_environment_config`` closes isolation envs; the adapter must rebuild.

    Without a per-sample router the first tool call hits a closed session, so
    this fails with ``Cannot operate on a closed database``.
    """
    env = _build(stateful_env_config_path)

    assert env.read_note(note_id=1) == '{"body": "original"}'


def test_reset_rebuilds_isolated_environment_state(stateful_env_config_path: str):
    """A stateful env must start clean on every step, not carry mutations over."""
    env = _build(stateful_env_config_path)

    assert env.read_note(note_id=1) == '{"body": "original"}'
    env.write_note(note_id=1, body="MUTATED")
    assert env.read_note(note_id=1) == '{"body": "MUTATED"}'

    env.reset()

    assert env.read_note(note_id=1) == '{"body": "original"}'


def test_tools_do_not_leak_onto_the_base_class(env_config_path: str):
    """Tool methods live on a per-config subclass, not on the shared base."""
    env = _build(env_config_path)

    assert hasattr(env, "get_price")
    assert not hasattr(OumiEnvironmentAdapter, "get_price")
