# Copyright 2025 - Oumi
# Licensed under the Apache License, Version 2.0
"""End-to-end test of ``oumi agent`` against the EHR DB SQLite fixture."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import sqlalchemy

from oumi.agents.agent_session import AgentSession
from oumi.agents.tool_router import ToolRouter
from oumi.builders.environments import build_environment
from oumi.core.configs.agent_harness_config import AgentHarnessConfig
from oumi.core.types.conversation import Conversation, Message, Role
from oumi.core.types.tool_call import FunctionCall, ToolCall

CONFIG_PATH = (
    Path(__file__).resolve().parents[3]
    / "configs"
    / "examples"
    / "agent"
    / "ehr_db_agent.yaml"
)
SCHEMA_DIR = (
    Path(__file__).resolve().parents[3] / "src" / "oumi" / "examples" / "ehr_db"
)


def _exec_sql_file(conn, path: Path) -> None:
    raw = path.read_text()
    lines = [line for line in raw.splitlines() if not line.lstrip().startswith("--")]
    body = "\n".join(lines)
    for stmt in body.split(";"):
        if stmt.strip():
            conn.execute(sqlalchemy.text(stmt))


@pytest.fixture
def seeded_db(tmp_path):
    db_file = tmp_path / "ehr_e2e.db"
    engine = sqlalchemy.create_engine(
        f"sqlite:///{db_file}", isolation_level="AUTOCOMMIT", future=True
    )
    with engine.connect() as conn:
        _exec_sql_file(conn, SCHEMA_DIR / "schema.sql")
        _exec_sql_file(conn, SCHEMA_DIR / "seed.sql")
    engine.dispose()
    return db_file


@pytest.fixture
def harness_config(seeded_db) -> AgentHarnessConfig:
    """Load the shipped EHR agent YAML and point it at the per-test sqlite."""
    cfg: AgentHarnessConfig = AgentHarnessConfig.from_yaml(str(CONFIG_PATH))
    env_params = cfg.environment.environments[0]
    assert env_params.env_kwargs is not None
    env_params.env_kwargs["connection"]["database"] = str(seeded_db)
    return cfg


def _tool_call(call_id: str, name: str, arguments: dict[str, Any]) -> ToolCall:
    return ToolCall(
        id=call_id,
        function=FunctionCall(name=name, arguments=json.dumps(arguments)),
    )


class ScriptedEngine:
    """Replays a fixed sequence of assistant messages."""

    def __init__(self, scripted: list[Message]) -> None:
        self._scripted = list(scripted)
        self.idx = 0

    def infer(self, conversations: list[Conversation]) -> list[Conversation]:
        msg = self._scripted[self.idx]
        self.idx += 1
        out: list[Conversation] = []
        for conv in conversations:
            new = conv.model_copy(deep=True)
            new.messages.append(msg)
            out.append(new)
        return out


def _build_session_from_config(
    config: AgentHarnessConfig, engine: ScriptedEngine
) -> AgentSession:
    """Build a session with the real env builder but a scripted engine."""
    config.finalize_and_validate()
    envs = {
        env_params.id: build_environment(env_params)
        for env_params in config.environment.environments
    }
    router = ToolRouter.from_environment_config(config.environment)
    return AgentSession(
        engine=engine,  # type: ignore[arg-type]  scripted fake quacks the same shape
        envs=envs,
        router=router,
        system_prompt=config.system_prompt,
    )


def test_agent_lists_then_fetches_a_real_patient(harness_config):
    """Two-tool happy path: ``list_patients`` then ``get_patient``."""
    engine = ScriptedEngine(
        [
            Message(
                role=Role.ASSISTANT,
                content=None,
                tool_calls=[_tool_call("c1", "list_patients", {})],
            ),
            Message(
                role=Role.ASSISTANT,
                content=None,
                tool_calls=[_tool_call("c2", "get_patient", {"patient_id": "P001"})],
            ),
            Message(role=Role.ASSISTANT, content="Pulled patient P001."),
        ]
    )

    with _build_session_from_config(harness_config, engine) as session:
        reply = session.send("Show me the first patient's full record.")

    assert reply == "Pulled patient P001."
    tool_msgs = [m for m in session.conversation.messages if m.role == Role.TOOL]
    assert len(tool_msgs) == 2
    listing = json.loads(str(tool_msgs[0].content))
    assert "patients" in listing
    assert any(p["patient_id"] == "P001" for p in listing["patients"])
    detail = json.loads(str(tool_msgs[1].content))
    assert detail["status"] == "ok"
    assert detail["patient"]["patient_id"] == "P001"


def test_agent_persists_writes_through_real_db(harness_config, seeded_db):
    """``add_diagnosis`` must persist; verify by re-opening the SQLite file."""
    engine = ScriptedEngine(
        [
            Message(
                role=Role.ASSISTANT,
                content=None,
                tool_calls=[
                    _tool_call(
                        "c1",
                        "add_diagnosis",
                        {
                            "patient_id": "P001",
                            "code": "E11.9",
                            "description": "Type 2 diabetes",
                            "date": "2026-05-09",
                        },
                    )
                ],
            ),
            Message(role=Role.ASSISTANT, content="Diagnosis added."),
        ]
    )

    with _build_session_from_config(harness_config, engine) as session:
        session.send("Add an E11.9 diagnosis to P001.")

    engine_check = sqlalchemy.create_engine(
        f"sqlite:///{seeded_db}", isolation_level="AUTOCOMMIT", future=True
    )
    with engine_check.connect() as conn:
        rows = (
            conn.execute(
                sqlalchemy.text("SELECT code FROM diagnoses WHERE patient_id = 'P001'")
            )
            .scalars()
            .all()
        )
    engine_check.dispose()
    assert "E11.9" in set(rows)


def test_agent_recovers_from_db_constraint_error(harness_config):
    """Duplicate diagnosis surfaces as a structured tool error, not a crash."""
    engine = ScriptedEngine(
        [
            Message(
                role=Role.ASSISTANT,
                content=None,
                tool_calls=[
                    _tool_call(
                        "c1",
                        "add_diagnosis",
                        {
                            "patient_id": "P001",
                            "code": "I10",
                            "description": "Hypertension",
                            "date": "2026-05-09",
                        },
                    )
                ],
            ),
            Message(
                role=Role.ASSISTANT,
                content=None,
                tool_calls=[
                    _tool_call(
                        "c2",
                        "add_diagnosis",
                        {
                            "patient_id": "P001",
                            "code": "I10",
                            "description": "Hypertension",
                            "date": "2026-05-10",
                        },
                    )
                ],
            ),
            Message(role=Role.ASSISTANT, content="Skipping; already on file."),
        ]
    )

    with _build_session_from_config(harness_config, engine) as session:
        reply = session.send("Add I10 hypertension diagnosis to P001.")

    assert reply == "Skipping; already on file."
    tool_msgs = [m for m in session.conversation.messages if m.role == Role.TOOL]
    second = json.loads(str(tool_msgs[1].content))
    assert second["status"] == "error"
    assert second["error"] == "duplicate_diagnosis"
