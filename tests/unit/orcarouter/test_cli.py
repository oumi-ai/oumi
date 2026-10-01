# Copyright 2026 - Oumi
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

"""Tests for the `oumi orcarouter` CLI."""

import json

from typer.testing import CliRunner

import oumi.cli.orcarouter as orcarouter_cli
from oumi.cli.main import get_app
from oumi.orcarouter.credentials import (
    CredentialSource,
    CredentialStore,
    OrcaRouterSession,
)

runner = CliRunner()

_FAKE_KEY = "sk-orca-77777777777777777777"


def test_root_cli_registers_the_orcarouter_panel():
    result = runner.invoke(get_app(), ["--help"])
    assert result.exit_code == 0
    assert "orcarouter" in result.stdout


def test_orcarouter_login_offers_both_sign_in_paths():
    result = runner.invoke(get_app(), ["orcarouter", "login", "--help"])
    assert result.exit_code == 0
    # API key
    assert "--api-key" in result.stdout
    # Browser login
    assert "--oob" in result.stdout


def test_login_with_an_api_key_stores_a_masked_credential(
    tmp_path, monkeypatch, capsys
):
    store = CredentialStore(directory=tmp_path)
    monkeypatch.setattr(
        orcarouter_cli, "OrcaRouterSession", lambda: OrcaRouterSession(store=store)
    )

    result = runner.invoke(get_app(), ["orcarouter", "login", "--api-key", _FAKE_KEY])

    assert result.exit_code == 0
    assert _FAKE_KEY not in result.stdout
    assert "sk-orca-…" in result.stdout

    stored = store.load()
    assert stored is not None
    assert stored.key == _FAKE_KEY
    assert stored.source is CredentialSource.API_KEY


def test_login_rejects_a_bad_key_without_storing_anything(tmp_path, monkeypatch):
    store = CredentialStore(directory=tmp_path)
    monkeypatch.setattr(
        orcarouter_cli, "OrcaRouterSession", lambda: OrcaRouterSession(store=store)
    )
    result = runner.invoke(
        get_app(), ["orcarouter", "login", "--api-key", "not-an-orcarouter-key"]
    )
    assert result.exit_code == 1
    assert store.load() is None


def test_status_and_logout_round_trip(tmp_path, monkeypatch):
    store = CredentialStore(directory=tmp_path)
    monkeypatch.setattr(
        orcarouter_cli, "OrcaRouterSession", lambda: OrcaRouterSession(store=store)
    )
    OrcaRouterSession(store=store).connect_with_api_key(_FAKE_KEY)

    result = runner.invoke(get_app(), ["orcarouter", "status"])
    assert result.exit_code == 0
    assert "api_key" in result.stdout
    assert _FAKE_KEY not in result.stdout

    result = runner.invoke(get_app(), ["orcarouter", "logout"])
    assert result.exit_code == 0
    assert store.load() is None


def test_models_command_binds_filtered_options_to_the_selector(tmp_path, monkeypatch):
    """The CLI lists exactly the models the selector would offer."""
    store = CredentialStore(directory=tmp_path)
    OrcaRouterSession(store=store).connect_with_api_key(_FAKE_KEY)
    monkeypatch.setattr(
        orcarouter_cli, "OrcaRouterSession", lambda: OrcaRouterSession(store=store)
    )
    monkeypatch.setattr(
        orcarouter_cli,
        "resolve_api_base_url",
        lambda *a, **k: "https://api.orcarouter.ai",
    )

    captured = {}

    def fake_selector(api_base_url, api_key, *, capability, current_model, **kwargs):
        from oumi.orcarouter.catalog import (
            CatalogResult,
            filter_models,
            verified_seed_models,
        )

        captured["capability"] = capability
        captured["api_key"] = api_key
        captured["current_model"] = current_model
        # The real filter derives the required modality from the key itself.
        models = filter_models(verified_seed_models(), capability)
        catalog = CatalogResult(models=models, source="live", degraded=False)
        from oumi.orcarouter.catalog import ModelSelector

        return ModelSelector(options=models, selected=None, catalog=catalog)

    monkeypatch.setattr("oumi.orcarouter.catalog.resolve_model_selector", fake_selector)

    result = runner.invoke(
        get_app(),
        [
            "orcarouter",
            "models",
            "--capability",
            "multimodal:image",
            "--current-model",
            "deepseek/deepseek-v4-pro",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.stdout
    # The root command prints its banner first, so isolate the JSON document.
    payload = json.loads(
        result.stdout[result.stdout.index("{") : result.stdout.rindex("}") + 1]
    )
    assert payload["source"] == "live"
    assert payload["degraded"] is False
    assert captured["api_key"] == _FAKE_KEY
    assert captured["capability"] == "multimodal:image"
    # Only models that declare image input remain, and the text-only selection
    # was handed to the selector for revalidation.
    ids = [model["id"] for model in payload["models"]]
    assert ids == [
        "anthropic/claude-opus-4.8",
        "google/gemini-3.5-flash",
        "openai/gpt-5.5",
    ]
    assert "deepseek/deepseek-v4-pro" not in ids
    assert captured["current_model"] == "deepseek/deepseek-v4-pro"


def test_models_command_refuses_an_unknown_capability(tmp_path, monkeypatch):
    store = CredentialStore(directory=tmp_path)
    OrcaRouterSession(store=store).connect_with_api_key(_FAKE_KEY)
    monkeypatch.setattr(
        orcarouter_cli, "OrcaRouterSession", lambda: OrcaRouterSession(store=store)
    )
    result = runner.invoke(
        get_app(), ["orcarouter", "models", "--capability", "telepathy"]
    )
    assert result.exit_code == 2


def test_models_command_reports_a_credential_that_needs_reauth(tmp_path, monkeypatch):
    store = CredentialStore(directory=tmp_path)
    session = OrcaRouterSession(store=store)
    credential = session.connect_with_api_key(_FAKE_KEY)
    session.mark_needs_reauth(credential)
    monkeypatch.setattr(
        orcarouter_cli, "OrcaRouterSession", lambda: OrcaRouterSession(store=store)
    )
    monkeypatch.setattr(
        orcarouter_cli,
        "resolve_api_base_url",
        lambda *a, **k: "https://api.orcarouter.ai",
    )
    monkeypatch.delenv("ORCAROUTER_API_KEY", raising=False)
    result = runner.invoke(get_app(), ["orcarouter", "models"])
    assert result.exit_code == 1


def test_models_command_marks_the_credential_on_a_401(tmp_path, monkeypatch):
    """A rejected key is not silently retried; it needs reauthorization."""
    from oumi.orcarouter.catalog import OrcaRouterAuthError
    from oumi.orcarouter.credentials import OrcaRouterCredentialError

    store = CredentialStore(directory=tmp_path)
    session = OrcaRouterSession(store=store)
    credential = session.connect_with_api_key(_FAKE_KEY)
    monkeypatch.setattr(
        orcarouter_cli, "OrcaRouterSession", lambda: OrcaRouterSession(store=store)
    )

    def unauthorized(*args, **kwargs):
        raise OrcaRouterAuthError("rejected")

    monkeypatch.setattr("oumi.orcarouter.catalog.discover_models", unauthorized)
    monkeypatch.setattr(
        orcarouter_cli,
        "resolve_api_base_url",
        lambda *a, **k: "https://api.orcarouter.ai",
    )

    result = runner.invoke(get_app(), ["orcarouter", "models"])

    assert result.exit_code == 1
    stored = store.load()
    assert stored is not None
    assert stored.needs_reauth is True
    assert stored.generation == credential.generation
    # The secret itself was kept, not deleted.
    assert stored.key == _FAKE_KEY
    assert OrcaRouterCredentialError is not None
