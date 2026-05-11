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

import pytest

from oumi.builders.inference_engines import build_inference_engine
from oumi.core.configs import InferenceEngineType, ModelParams, RemoteParams
from oumi.inference import OumiPlatformInferenceEngine


@pytest.fixture(autouse=True)
def _clear_platform_env(monkeypatch):
    for k in ("OUMI_API_URL", "OUMI_API_KEY", "OUMI_CREDENTIALS_FILE"):
        monkeypatch.delenv(k, raising=False)


def test_engine_type_value():
    assert InferenceEngineType.OUMI_PLATFORM.value == "OUMI_PLATFORM"


def test_default_base_url_when_env_unset():
    engine = OumiPlatformInferenceEngine(
        model_params=ModelParams(model_name="any"),
    )
    assert engine.base_url == (
        "https://api.oumi.ai/inference/v1/chat/completions"
    )


def test_base_url_from_env(monkeypatch):
    monkeypatch.setenv("OUMI_API_URL", "https://custom.platform.test/")
    engine = OumiPlatformInferenceEngine(
        model_params=ModelParams(model_name="any"),
    )
    assert engine.base_url == (
        "https://custom.platform.test/inference/v1/chat/completions"
    )


def test_base_url_from_credentials_file(monkeypatch, tmp_path):
    """If env is unset, fall back to api_url in the credentials file."""
    creds = tmp_path / "credentials.json"
    creds.write_text(
        '{"api_url": "https://from-file.test", "api_key": "k"}'
    )
    monkeypatch.setenv("OUMI_CREDENTIALS_FILE", str(creds))

    engine = OumiPlatformInferenceEngine(
        model_params=ModelParams(model_name="any"),
    )
    assert engine.base_url == (
        "https://from-file.test/inference/v1/chat/completions"
    )


def test_api_key_env_varname():
    engine = OumiPlatformInferenceEngine(
        model_params=ModelParams(model_name="any"),
    )
    assert engine.api_key_env_varname == "OUMI_API_KEY"


def test_build_inference_engine_returns_platform_engine():
    engine = build_inference_engine(
        engine_type=InferenceEngineType.OUMI_PLATFORM,
        model_params=ModelParams(model_name="any"),
        remote_params=RemoteParams(),
    )
    assert isinstance(engine, OumiPlatformInferenceEngine)


def test_credentials_file_error_does_not_crash_init(monkeypatch, tmp_path):
    """A malformed credentials file should not block engine construction.

    The engine only consults the file to *default* the URL; the user can
    always override via RemoteParams.api_url.
    """
    creds = tmp_path / "credentials.json"
    creds.write_text("not-json")
    monkeypatch.setenv("OUMI_CREDENTIALS_FILE", str(creds))

    engine = OumiPlatformInferenceEngine(
        model_params=ModelParams(model_name="any"),
    )
    # Falls back to the public default.
    assert engine.base_url == (
        "https://api.oumi.ai/inference/v1/chat/completions"
    )
