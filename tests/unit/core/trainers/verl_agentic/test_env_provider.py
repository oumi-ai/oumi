import gc
import sqlite3
from types import SimpleNamespace
from typing import Any

from oumi.core.configs.environment_config import EnvironmentConfig
from oumi.core.configs.params.environment_params import EnvironmentParams
from oumi.core.trainers.verl_agentic.env_provider import get_or_build_router

BASE = EnvironmentConfig(
    environments=[
        EnvironmentParams(
            id="db",
            env_type="database",
            env_kwargs={"schema_sql": "CREATE TABLE t(x);"},
            tools=[
                {
                    "id": "run_sql",
                    "name": "run_sql",
                    "description": "run sql",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                    "executor": "oumi.environments.examples.nl2sql.run_sql",
                    "read_only": True,
                }
            ],
        )
    ]
)


class _FakeAgentData(SimpleNamespace):
    """Subclass so instances support weakref.finalize (bare SimpleNamespace doesn't)."""

    request_id: str
    tools_kwargs: dict[str, Any]


def _agent_data(request_id: str):
    return _FakeAgentData(
        request_id=request_id,
        tools_kwargs={
            "run_sql": {
                "create_kwargs": {
                    "schema_sql": "CREATE TABLE t(x INTEGER);",
                    "seed_sql": "INSERT INTO t VALUES (1),(2);",
                }
            }
        },
    )


def test_same_request_id_returns_same_router():
    ad = _agent_data("same-1")
    r1 = get_or_build_router(ad, BASE)
    r2 = get_or_build_router(ad, BASE)
    assert r1 is r2


def test_router_closed_and_evicted_when_agent_data_is_collected(monkeypatch):
    request_id = "collected-1"
    ad = _agent_data(request_id)
    router = get_or_build_router(ad, BASE)
    original_close = router.close
    close_calls = 0

    def close_spy():
        nonlocal close_calls
        close_calls += 1
        original_close()

    monkeypatch.setattr(router, "close", close_spy)

    del ad
    gc.collect()

    assert close_calls == 1

    replacement_ad = _agent_data(request_id)
    replacement_router = get_or_build_router(replacement_ad, BASE)
    assert replacement_router is not router

    del replacement_ad
    gc.collect()


def test_router_routes_run_sql():
    ad = _agent_data("routes-1")
    router = get_or_build_router(ad, BASE)
    out = router.route_batch([("run_sql", {"query": "SELECT count(*) FROM t"})])[0]
    assert out.output == {"columns": ["count(*)"], "rows": [[2]]}


def test_different_request_id_gets_isolated_router():
    ad1 = _agent_data("iso-a")
    ad2 = _agent_data("iso-b")
    ad2.tools_kwargs["run_sql"]["create_kwargs"]["seed_sql"] = (
        "INSERT INTO t VALUES (1),(2),(3);"
    )
    r1 = get_or_build_router(ad1, BASE)
    r2 = get_or_build_router(ad2, BASE)
    assert r1 is not r2
    c1 = r1.route_batch([("run_sql", {"query": "SELECT count(*) FROM t"})])[0].output
    c2 = r2.route_batch([("run_sql", {"query": "SELECT count(*) FROM t"})])[0].output
    assert c1 == {"columns": ["count(*)"], "rows": [[2]]}
    assert c2 == {"columns": ["count(*)"], "rows": [[3]]}


def test_db_path_create_kwargs_overrides_placeholder_schema(tmp_path):
    # create_kwargs must replace env_kwargs to avoid setting both database sources.
    db = tmp_path / "shared.sqlite"
    conn = sqlite3.connect(db)
    conn.executescript("CREATE TABLE t(x INTEGER); INSERT INTO t VALUES (9);")
    conn.commit()
    conn.close()
    ad = _FakeAgentData(
        request_id="dbpath-1",
        tools_kwargs={"run_sql": {"create_kwargs": {"db_path": str(db)}}},
    )
    router = get_or_build_router(ad, BASE)
    out = router.route_batch([("run_sql", {"query": "SELECT count(*) FROM t"})])[0]
    assert out.output == {"columns": ["count(*)"], "rows": [[1]]}


def _db_env_spec(env_id: str, tool_id: str) -> EnvironmentParams:
    return EnvironmentParams(
        id=env_id,
        env_type="database",
        env_kwargs={
            "schema_sql": "CREATE TABLE t(x INTEGER);",
            "seed_sql": "INSERT INTO t VALUES (7);",
        },
        tools=[
            {
                "id": tool_id,
                "name": tool_id,
                "description": "run sql",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
                "executor": "oumi.environments.examples.nl2sql.run_sql",
                "read_only": True,
            }
        ],
    )


def test_create_kwargs_scoped_to_owning_env_only():
    # Only db_a's tool carries create_kwargs; db_b must keep its own configuration.
    cfg = EnvironmentConfig(
        environments=[
            _db_env_spec("db_a", "q_a"),
            _db_env_spec("db_b", "q_b"),
        ]
    )
    ad = _FakeAgentData(
        request_id="scope-1",
        tools_kwargs={
            "q_a": {
                "create_kwargs": {
                    "schema_sql": "CREATE TABLE t(x INTEGER);",
                    "seed_sql": "INSERT INTO t VALUES (1),(2),(3),(4),(5);",
                }
            }
        },
    )
    router = get_or_build_router(ad, cfg)
    a = router.route_batch([("q_a", {"query": "SELECT count(*) FROM t"})])[0].output
    b = router.route_batch([("q_b", {"query": "SELECT count(*) FROM t"})])[0].output
    assert a == {"columns": ["count(*)"], "rows": [[5]]}
    assert b == {"columns": ["count(*)"], "rows": [[1]]}
