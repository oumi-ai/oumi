import asyncio
from unittest.mock import patch

import pytest
from aiohttp import web

from oumi.core.trainers.verl_agentic.user_sim_provider import (
    generate_reply,
    user_sim_engine,
)
from oumi.core.types.conversation import Conversation, Message, Role


def _write_config(
    tmp_path, engine, api_url="http://localhost:8000/v1/chat/completions"
):
    path = tmp_path / "user_sim.yaml"
    engine_line = f"engine: {engine}\n" if engine else ""
    path.write_text(
        f"model:\n  model_name: m\n{engine_line}"
        f"remote_params:\n  api_url: {api_url}\n"
        "generation:\n  max_new_tokens: 7\n"
    )
    return str(path)


@pytest.mark.parametrize("engine", [None, "NATIVE", "VLLM", "LLAMACPP"])
def test_non_remote_engine_rejected_before_build(tmp_path, engine):
    with patch("oumi.builders.inference_engines.build_inference_engine") as build:
        with pytest.raises(ValueError, match="remote inference engine"):
            user_sim_engine(_write_config(tmp_path, engine))
    build.assert_not_called()


def test_engine_built_once_with_generation_params(tmp_path):
    path = _write_config(tmp_path, "REMOTE_VLLM")

    engine = user_sim_engine(path)

    assert user_sim_engine(path) is engine
    assert engine._generation_params.max_new_tokens == 7


@pytest.mark.timeout(30, method="thread")
def test_concurrent_replies_share_one_engine_on_one_event_loop(tmp_path):
    async def chat(request):
        await asyncio.sleep(0.01)
        return web.json_response(
            {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}
        )

    async def main():
        app = web.Application()
        app.router.add_post("/v1/chat/completions", chat)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]  # pyright: ignore
        path = _write_config(
            tmp_path, "REMOTE", f"http://127.0.0.1:{port}/v1/chat/completions"
        )
        conversation = Conversation(messages=[Message(role=Role.USER, content="hi")])
        try:
            return await asyncio.gather(
                *(generate_reply(path, conversation) for _ in range(64))
            )
        finally:
            await runner.cleanup()

    assert asyncio.run(main()) == ["ok"] * 64
