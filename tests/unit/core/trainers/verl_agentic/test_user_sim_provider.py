from unittest.mock import MagicMock, patch

import pytest

from oumi.builders.inference_engines import ENGINE_MAP
from oumi.core.configs.inference_engine_type import InferenceEngineType
from oumi.core.inference.base_inference_engine import BaseInferenceEngine
from oumi.core.trainers.verl_agentic.user_sim_provider import (
    _IN_PROCESS_ENGINES,
    user_sim_engine,
)
from oumi.inference.remote_inference_engine import RemoteInferenceEngine

_BUILD = "oumi.core.trainers.verl_agentic.user_sim_provider.build_inference_engine"


def _write_config(tmp_path, engine: str) -> str:
    path = tmp_path / f"user_sim_{engine.lower()}.yaml"
    path.write_text(
        f"model:\n  model_name: 'gpt-4o-mini'\nengine: {engine}\n"
        "remote_params:\n  api_url: 'http://localhost:8000/v1/chat/completions'\n"
    )
    return str(path)


@pytest.mark.parametrize("engine", ["NATIVE", "VLLM", "LLAMACPP"])
def test_in_process_engine_rejected_before_build(tmp_path, engine):
    config_path = _write_config(tmp_path, engine)
    with patch(_BUILD) as build:
        with pytest.raises(ValueError, match="remote inference engine"):
            user_sim_engine(config_path)
    build.assert_not_called()


def test_missing_engine_rejected(tmp_path):
    path = tmp_path / "no_engine.yaml"
    path.write_text("model:\n  model_name: 'gpt-4o-mini'\n")
    with pytest.raises(ValueError, match="No inference engine"):
        user_sim_engine(str(path))


def test_remote_engine_built_once(tmp_path):
    config_path = _write_config(tmp_path, "REMOTE_VLLM")
    with patch(_BUILD) as build:
        build.return_value = MagicMock(spec=RemoteInferenceEngine)
        first = user_sim_engine(config_path)
        second = user_sim_engine(config_path)
    assert first is second
    assert build.call_count == 1


def test_non_remote_engine_rejected(tmp_path):
    """A newly-added engine type that isn't remote-backed must not slip through."""
    config_path = _write_config(tmp_path, "REMOTE")
    with patch(_BUILD) as build:
        build.return_value = MagicMock(spec=BaseInferenceEngine)
        with pytest.raises(ValueError, match="not remote-backed"):
            user_sim_engine(config_path)


@pytest.mark.parametrize("engine_type,engine_cls", sorted(ENGINE_MAP.items(), key=str))
def test_every_engine_is_classified(engine_type, engine_cls):
    """A new engine type must be either in-process or remote-backed, never neither."""
    assert (engine_type in _IN_PROCESS_ENGINES) ^ issubclass(
        engine_cls, RemoteInferenceEngine
    ), f"{engine_type} is classified neither in-process nor remote"


def test_in_process_set_matches_expected_members():
    assert _IN_PROCESS_ENGINES == {
        InferenceEngineType.NATIVE,
        InferenceEngineType.VLLM,
        InferenceEngineType.LLAMACPP,
    }
