import pytest

from oumi.core.configs import ModelParams, RemoteParams
from oumi.inference.together_inference_engine import TogetherInferenceEngine


@pytest.fixture
def together_engine():
    return TogetherInferenceEngine(
        model_params=ModelParams(model_name="together-model"),
        remote_params=RemoteParams(api_key="test_api_key", api_url="<placeholder>"),
    )


def test_together_init_with_custom_params():
    """Test initialization with custom parameters."""
    model_params = ModelParams(model_name="together-model")
    remote_params = RemoteParams(
        api_url="custom-url",
        api_key="custom-key",
    )
    engine = TogetherInferenceEngine(
        model_params=model_params,
        remote_params=remote_params,
    )
    assert engine._model_params.model_name == "together-model"
    assert engine._remote_params.api_url == "custom-url"
    assert engine._remote_params.api_key == "custom-key"


def test_together_init_default_params():
    """Test initialization with default parameters."""
    model_params = ModelParams(model_name="together-model")
    engine = TogetherInferenceEngine(model_params)
    assert engine._model_params.model_name == "together-model"
    assert (
        engine._remote_params.api_url == "https://api.together.xyz/v1/chat/completions"
    )
    assert engine._remote_params.api_key_env_varname == "TOGETHER_API_KEY"


def test_get_batch_status_maps_progress(together_engine):
    """Together's progress percentage flows through to BatchInfo."""
    from unittest.mock import AsyncMock, MagicMock, patch

    mock_resp = AsyncMock()
    mock_resp.status = 200
    mock_resp.json = AsyncMock(return_value={
        "id": "batch-123",
        "status": "IN_PROGRESS",
        "progress": 50.0,
    })
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=None)

    mock_session = MagicMock()
    mock_session.get = MagicMock(return_value=mock_resp)
    mock_session.__aenter__ = AsyncMock(return_value=(mock_session, {}))
    mock_session.__aexit__ = AsyncMock(return_value=None)

    with patch.object(together_engine, "_create_session", return_value=mock_session):
        result = together_engine.get_batch_status("batch-123")

    assert result.completion_percentage == 50.0
    assert result.total_requests == 100
    assert result.completed_requests == 50


def test_get_batch_status_zero_progress(together_engine):
    """Zero progress yields zero completion percentage."""
    from unittest.mock import AsyncMock, MagicMock, patch

    mock_resp = AsyncMock()
    mock_resp.status = 200
    mock_resp.json = AsyncMock(return_value={
        "id": "batch-123",
        "status": "VALIDATING",
    })
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=None)

    mock_session = MagicMock()
    mock_session.get = MagicMock(return_value=mock_resp)
    mock_session.__aenter__ = AsyncMock(return_value=(mock_session, {}))
    mock_session.__aexit__ = AsyncMock(return_value=None)

    with patch.object(together_engine, "_create_session", return_value=mock_session):
        result = together_engine.get_batch_status("batch-123")

    assert result.completion_percentage == 0.0
