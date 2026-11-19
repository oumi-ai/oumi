import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from oumi.core.synthesis import tool_router
from oumi.core.synthesis.tool_router import ToolRouter
from oumi.core.types.tool_call import ToolResult
from oumi.verl_grpo.tool_routers import RolloutRouters

COUNT_QUERY = {"query": "SELECT count(*) FROM t"}


class _Overlap:
    lock = threading.Lock()
    active = 0
    peak = 0


def slow_sql(arguments: dict, context: sqlite3.Connection) -> ToolResult:
    with _Overlap.lock:
        _Overlap.active += 1
        _Overlap.peak = max(_Overlap.peak, _Overlap.active)
    time.sleep(0.05)
    with _Overlap.lock:
        _Overlap.active -= 1
    cursor = context.execute(arguments["query"])
    return ToolResult(output={"rows": [list(row) for row in cursor.fetchall()]})


def _seeded(rows: int, tool_id: str = "run_sql") -> dict:
    values = ",".join(f"({i})" for i in range(rows))
    return {
        tool_id: {
            "create_kwargs": {
                "schema_sql": "CREATE TABLE t(x INTEGER);",
                "seed_sql": f"INSERT INTO t VALUES {values};",
            }
        }
    }


def _count(session, tool_id: str = "run_sql") -> int:
    return session.route([(tool_id, COUNT_QUERY)])[0].output["rows"][0][0]


@pytest.fixture
def routers():
    routers = RolloutRouters()
    yield routers
    routers.close_all()


def test_same_request_id_returns_the_same_router(routers, env_config):
    config = env_config()
    first = routers.get_or_build("r", env_config=config, tools_kwargs=_seeded(2))
    assert (
        routers.get_or_build("r", env_config=config, tools_kwargs=_seeded(2)) is first
    )
    assert routers.live_count == 1


def test_different_request_ids_are_isolated(routers, env_config):
    config = env_config()
    two = routers.get_or_build("a", env_config=config, tools_kwargs=_seeded(2))
    three = routers.get_or_build("b", env_config=config, tools_kwargs=_seeded(3))
    assert (_count(two), _count(three)) == (2, 3)


def test_create_kwargs_scoped_to_their_owning_env(routers, env_config, sql_env):
    config = env_config(sql_env("db_a", "q_a"), sql_env("db_b", "q_b"))
    session = routers.get_or_build(
        "r", env_config=config, tools_kwargs=_seeded(5, tool_id="q_a")
    )
    assert (_count(session, "q_a"), _count(session, "q_b")) == (5, 1)


def test_db_path_create_kwargs_override_the_placeholder_schema(
    routers, tmp_path, env_config
):
    db = tmp_path / "shared.sqlite"
    with sqlite3.connect(db) as connection:
        connection.executescript("CREATE TABLE t(x INTEGER); INSERT INTO t VALUES (9);")
    connection.close()
    session = routers.get_or_build(
        "r",
        env_config=env_config(),
        tools_kwargs={"run_sql": {"create_kwargs": {"db_path": str(db)}}},
    )
    assert _count(session) == 1


def test_unknown_tool_id_in_create_kwargs_is_refused(routers, env_config):
    with pytest.raises(ValueError, match="unknown tool IDs: \\['nope'\\]"):
        routers.get_or_build(
            "r", env_config=env_config(), tools_kwargs=_seeded(1, "nope")
        )
    assert routers.live_count == 0


def test_conflicting_create_kwargs_are_refused(routers, env_config, sql_env):
    config = env_config(sql_env("db", "run_sql", "run_sql_2"))
    with pytest.raises(ValueError, match="conflicting create_kwargs"):
        routers.get_or_build(
            "r", env_config=config, tools_kwargs=_seeded(2) | _seeded(3, "run_sql_2")
        )


def test_two_env_configs_in_one_rollout_are_refused(routers, env_config):
    routers.get_or_build("r", env_config=env_config(), tools_kwargs={})
    with pytest.raises(ValueError, match="must share a single environment config"):
        routers.get_or_build(
            "r", env_config=env_config(name="other.yaml"), tools_kwargs={}
        )


def test_each_rollout_builds_its_envs_once(routers, monkeypatch, env_config):
    builds = []
    original_build = tool_router.build_environment

    def counting_build(env_params):
        builds.append(env_params.id)
        return original_build(env_params)

    monkeypatch.setattr(tool_router, "build_environment", counting_build)
    config = env_config()
    for request_id in ("a", "a", "b"):
        routers.get_or_build(request_id, env_config=config, tools_kwargs=_seeded(1))
    # The parent template once, then one clone per rollout.
    assert builds == ["db", "db", "db"]


def test_close_closes_and_evicts(routers, monkeypatch, env_config):
    closed = []
    original_close = ToolRouter.close

    def close_spy(router):
        closed.append(router)
        original_close(router)

    monkeypatch.setattr(ToolRouter, "close", close_spy)
    config = env_config()
    session = routers.get_or_build("r", env_config=config, tools_kwargs={})

    assert routers.close("r") is True
    assert len(closed) == 1
    assert routers.live_count == 0
    assert routers.close("r") is False
    assert routers.get_or_build("r", env_config=config, tools_kwargs={}) is not session


def test_close_all_returns_the_count_and_empties(routers, env_config):
    config = env_config()
    for request_id in ("a", "b", "c"):
        routers.get_or_build(request_id, env_config=config, tools_kwargs={})
    assert routers.close_all() == 3
    assert routers.live_count == 0
    assert routers.close_all() == 0


def test_no_router_is_built_after_close_all(routers, env_config):
    config = env_config()
    routers.close_all()
    with pytest.raises(RuntimeError, match="closed"):
        routers.get_or_build("late", env_config=config, tools_kwargs={})
    assert routers.live_count == 0


def test_check_tool_names_the_known_tools(routers, env_config):
    config = env_config()
    routers.check_tool(config, "run_sql")
    with pytest.raises(ValueError, match="Known tools: \\['run_sql'\\]"):
        routers.check_tool(config, "run_sqll")


def test_router_calls_and_close_run_on_one_thread_per_rollout(routers, env_config):
    # SQLite connections are bound to the thread that opened them.
    session = routers.get_or_build(
        "r", env_config=env_config(), tools_kwargs=_seeded(2)
    )
    counts = []
    for _ in range(2):
        with ThreadPoolExecutor(max_workers=1) as caller:
            counts.append(caller.submit(_count, session).result())
    with ThreadPoolExecutor(max_workers=1) as closer:
        assert closer.submit(routers.close, "r").result() is True
    assert counts == [2, 2]


def test_concurrent_calls_on_one_rollout_are_serialized(routers, env_config, sql_env):
    config = env_config(sql_env("db", "slow_sql", executor=slow_sql))
    session = routers.get_or_build("r", env_config=config, tools_kwargs={})
    with ThreadPoolExecutor(max_workers=4) as callers:
        counts = list(callers.map(lambda _: _count(session, "slow_sql"), range(4)))
    assert counts == [1, 1, 1, 1]
    assert _Overlap.peak == 1
