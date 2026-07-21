import sqlite3
from types import SimpleNamespace

from oumi.core.configs.environment_config import EnvironmentConfig
from oumi.core.trainers.verl_agentic.env_provider import get_or_build_router

BASE = EnvironmentConfig(
    environments=[  # type: ignore[list-item]
        {
            "id": "db",
            "env_type": "database",
            "env_kwargs": {
                "schema_sql": "CREATE TABLE t(x);"
            },  # overridden per rollout
            "tools": [
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
        }
    ]
)


class _FakeAgentData(SimpleNamespace):
    """Subclass so instances support weakref.finalize (bare SimpleNamespace doesn't)."""


def _agent_data(request_id: str):
    # request_id is the per-rollout key the provider registers the router under.
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
    assert r1 is r2  # one env per rollout, shared across tool calls


def test_router_routes_run_sql():
    ad = _agent_data("routes-1")
    router = get_or_build_router(ad, BASE)
    out = router.route_batch([("run_sql", {"query": "SELECT count(*) FROM t"})])[0]
    assert out.output == {"columns": ["count(*)"], "rows": [[2]]}


def test_different_request_id_gets_isolated_router():
    ad1 = _agent_data("iso-a")
    ad2 = _agent_data("iso-b")
    # Each rollout builds its own env from its own create_kwargs; rollout 2 seeds
    # an extra row so the two DBs are provably independent.
    ad2.tools_kwargs["run_sql"]["create_kwargs"]["seed_sql"] = (
        "INSERT INTO t VALUES (1),(2),(3);"
    )
    r1 = get_or_build_router(ad1, BASE)
    r2 = get_or_build_router(ad2, BASE)
    assert r1 is not r2
    c1 = r1.route_batch([("run_sql", {"query": "SELECT count(*) FROM t"})])[0].output
    c2 = r2.route_batch([("run_sql", {"query": "SELECT count(*) FROM t"})])[0].output
    assert c1 == {"columns": ["count(*)"], "rows": [[2]]}
    assert c2 == {"columns": ["count(*)"], "rows": [[3]]}  # isolated: own seed_sql


def test_db_path_create_kwargs_overrides_placeholder_schema(tmp_path):
    # A db_path rollout against a schema_sql-placeholder config must not leave
    # both keys set — that would fail "exactly one of db_path/schema_sql" on
    # every tool call (create_kwargs replaces, rather than merges into, env_kwargs).
    db = tmp_path / "shared.sqlite"
    conn = sqlite3.connect(db)
    conn.executescript("CREATE TABLE t(x INTEGER); INSERT INTO t VALUES (9);")
    conn.commit()
    conn.close()
    ad = _FakeAgentData(
        request_id="dbpath-1",
        tools_kwargs={"run_sql": {"create_kwargs": {"db_path": str(db)}}},
    )
    router = get_or_build_router(ad, BASE)  # must not raise
    out = router.route_batch([("run_sql", {"query": "SELECT count(*) FROM t"})])[0]
    assert out.output == {"columns": ["count(*)"], "rows": [[1]]}


def _db_env_spec(env_id: str, tool_id: str) -> dict:
    # placeholder seeds exactly 1 row, so an un-overwritten env is distinguishable.
    return {
        "id": env_id,
        "env_type": "database",
        "env_kwargs": {
            "schema_sql": "CREATE TABLE t(x INTEGER);",
            "seed_sql": "INSERT INTO t VALUES (7);",
        },
        "tools": [
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
    }


def test_create_kwargs_scoped_to_owning_env_only():
    # Two DB envs; only db_a's tool carries create_kwargs. db_b must keep its own
    # placeholder (1 row), not receive db_a's spec (the multi-env overwrite guard).
    cfg = EnvironmentConfig(
        environments=[  # type: ignore[list-item]
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
    assert a == {"columns": ["count(*)"], "rows": [[5]]}  # db_a used create_kwargs
    assert b == {"columns": ["count(*)"], "rows": [[1]]}  # db_b kept its placeholder
