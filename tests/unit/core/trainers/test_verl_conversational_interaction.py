import asyncio
import sys
import types

import pytest


@pytest.fixture
def interaction_cls(monkeypatch):
    base_mod = types.ModuleType("verl.interactions.base")

    class _BaseInteraction:
        def __init__(self, config):
            self.config = config
            self.name = config.get("name", "interaction_agent")

    setattr(base_mod, "BaseInteraction", _BaseInteraction)
    monkeypatch.setitem(sys.modules, "verl", types.ModuleType("verl"))
    monkeypatch.setitem(
        sys.modules, "verl.interactions", types.ModuleType("verl.interactions")
    )
    monkeypatch.setitem(sys.modules, "verl.interactions.base", base_mod)

    import importlib

    import oumi.core.trainers.verl_conversational_interaction as mod

    importlib.reload(mod)

    fake_config = types.SimpleNamespace(engine=None, model=None, remote_params=None)
    monkeypatch.setattr(
        mod.InferenceConfig, "from_yaml", staticmethod(lambda p: fake_config)
    )
    monkeypatch.setattr(mod, "build_inference_engine", lambda **kw: object())
    return mod.OumiVerlInteraction


def test_lifecycle_delegates_and_pops(interaction_cls, monkeypatch):
    inter = interaction_cls(
        {"name": "oumi_conversation", "user_sim_inference": "x.yaml", "max_turns": 4}
    )
    monkeypatch.setattr(inter, "_infer_one", lambda conv: "hello there")

    async def run():
        iid = await inter.start_interaction(
            "id1", user_persona="You are Jane.", goal="refund"
        )
        assert iid == "id1"
        assert inter._state["id1"].persona == "You are Jane."
        assert inter._state["id1"].goal == "refund"
        done, text, score, extra = await inter.generate_response(
            "id1", [{"role": "assistant", "content": "hi"}]
        )
        assert (done, text, score, extra) == (False, "hello there", 0.0, {})
        assert inter._state["id1"].turn_idx == 1
        await inter.finalize_interaction("id1")
        assert "id1" not in inter._state

    asyncio.run(run())
