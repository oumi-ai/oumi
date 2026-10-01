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

"""Tests for the OrcaRouter credential seam and its two adapters."""

import json
import logging
import os
import stat
from pathlib import Path

import pytest

from oumi.core.configs import ModelParams, RemoteParams
from oumi.inference.orcarouter_inference_engine import OrcaRouterInferenceEngine
from oumi.orcarouter.credentials import (
    API_KEY_PREFIX,
    ORCAROUTER_API_KEY_ENV_VAR,
    CredentialResult,
    CredentialSource,
    CredentialStore,
    OrcaRouterApiKeyAdapter,
    OrcaRouterCredentialError,
    OrcaRouterSession,
    mask_api_key,
    resolve_api_base_url,
    resolve_auth_base_url,
)
from oumi.orcarouter.login import OAuthPkceAdapter
from tests.unit.orcarouter.test_login import make_pkce_credential

_FAKE_KEY = "sk-orca-00000000000000000000"


def test_mask_api_key_never_returns_the_secret():
    masked = mask_api_key(_FAKE_KEY)
    assert _FAKE_KEY not in masked
    assert masked.endswith("0000")
    assert mask_api_key(None) == "<unset>"
    assert mask_api_key("short") == "sk-orca-…"


def test_credential_repr_hides_the_key():
    credential = CredentialResult(key=_FAKE_KEY, source=CredentialSource.API_KEY)
    assert _FAKE_KEY not in repr(credential)
    assert _FAKE_KEY not in str(credential)
    assert _FAKE_KEY not in f"{credential}"


def test_store_round_trips_and_restricts_permissions(store: CredentialStore):
    credential = OrcaRouterApiKeyAdapter(_FAKE_KEY, store=store, environ={}).acquire()

    assert store.path.exists()
    mode = stat.S_IMODE(os.stat(store.path).st_mode)
    assert mode == 0o600

    loaded = store.load()
    assert loaded is not None
    assert loaded.key == _FAKE_KEY
    assert loaded.source is CredentialSource.API_KEY
    assert loaded.generation == credential.generation


def test_store_clear_removes_the_secret(store: CredentialStore):
    OrcaRouterApiKeyAdapter(_FAKE_KEY, store=store, environ={}).acquire()
    assert store.clear() is True
    assert store.load() is None
    assert store.clear() is False


def test_store_tolerates_a_corrupt_file(store: CredentialStore):
    store.path.parent.mkdir(parents=True, exist_ok=True)
    store.path.write_text("{not json", encoding="utf-8")
    assert store.load() is None

    store.path.write_text("[1, 2, 3]", encoding="utf-8")
    assert store.load() is None

    store.path.write_text(json.dumps({"source": "api_key"}), encoding="utf-8")
    assert store.load() is None


def test_api_key_adapter_prefers_the_explicit_key(store: CredentialStore):
    adapter = OrcaRouterApiKeyAdapter(
        _FAKE_KEY,
        store=store,
        environ={ORCAROUTER_API_KEY_ENV_VAR: "sk-orca-88888888888888888888"},
    )
    assert adapter.resolve_key() == _FAKE_KEY


def test_api_key_adapter_reads_the_environment(store: CredentialStore):
    adapter = OrcaRouterApiKeyAdapter(
        store=store, environ={ORCAROUTER_API_KEY_ENV_VAR: _FAKE_KEY}
    )
    credential = adapter.acquire()
    assert credential.key == _FAKE_KEY
    assert credential.source is CredentialSource.API_KEY


def test_api_key_adapter_rejects_a_missing_key(store: CredentialStore):
    adapter = OrcaRouterApiKeyAdapter(store=store, environ={})
    with pytest.raises(OrcaRouterCredentialError, match="No OrcaRouter API key"):
        adapter.acquire()


def test_api_key_adapter_rejects_an_implausible_key(store: CredentialStore):
    adapter = OrcaRouterApiKeyAdapter("not-a-key", store=store, environ={})
    with pytest.raises(OrcaRouterCredentialError, match="sk-orca-"):
        adapter.acquire()


def test_both_adapters_produce_the_same_credential_shape_and_route_identically(
    store: CredentialStore,
):
    """The two sign-in paths are interchangeable to everything downstream."""
    api_key_credential = OrcaRouterApiKeyAdapter(
        _FAKE_KEY, store=store, environ={}
    ).acquire()
    pkce_credential = make_pkce_credential(_FAKE_KEY)

    for credential in (api_key_credential, pkce_credential):
        assert isinstance(credential, CredentialResult)
        assert credential.key.startswith("sk-orca-")
        # Downstream inference cannot tell the two apart.
        remote_params = RemoteParams(api_key=credential.key)
        engine = OrcaRouterInferenceEngine(
            ModelParams(model_name="openai/gpt-5.5"), remote_params=remote_params
        )
        headers = engine._get_request_headers(engine._remote_params)
        assert headers["Authorization"] == f"Bearer {_FAKE_KEY}"

    assert api_key_credential.source is CredentialSource.API_KEY
    assert pkce_credential.source is CredentialSource.OAUTH_PKCE


def test_mark_needs_reauth_updates_only_the_rejected_generation(
    store: CredentialStore,
):
    session = OrcaRouterSession(store=store)
    stale = session.connect_with_api_key(_FAKE_KEY)

    # The user reauthorizes; the new credential has a new generation.
    fresh = CredentialResult(
        key="sk-orca-11111111111111111111", source=CredentialSource.API_KEY
    )
    session.store.save(fresh)
    assert fresh.generation != stale.generation

    # A late 401 from the superseded credential must not disable the new one.
    assert session.mark_needs_reauth(stale) is False
    active = session.active_credential()
    assert active is not None
    assert active.key == "sk-orca-11111111111111111111"
    assert active.needs_reauth is False

    # The exact rejected generation does flip the flag, and reauth clears it.
    assert session.mark_needs_reauth(fresh) is True
    assert session.mark_needs_reauth(fresh) is False
    assert session.active_credential() is None
    assert session.current_credential() is not None

    session.store.note_reauthenticated(fresh)
    assert session.active_credential() is not None


def test_mark_needs_reauth_without_a_stored_credential_is_a_no_op(
    store: CredentialStore,
):
    session = OrcaRouterSession(store=store)
    orphan = CredentialResult(key=_FAKE_KEY, source=CredentialSource.OAUTH_PKCE)
    assert session.mark_needs_reauth(orphan) is False


def test_engine_reuses_the_stored_login_without_minting_another_key(
    store: CredentialStore, monkeypatch: pytest.MonkeyPatch
):
    """A second run reuses the stored credential instead of re-authorizing."""
    session = OrcaRouterSession(store=store)
    session.connect_with_api_key(_FAKE_KEY)

    monkeypatch.delenv(ORCAROUTER_API_KEY_ENV_VAR, raising=False)
    monkeypatch.setattr(
        "oumi.inference.orcarouter_inference_engine.OrcaRouterSession",
        lambda: OrcaRouterSession(store=store),
    )

    engine = OrcaRouterInferenceEngine(ModelParams(model_name="openai/gpt-5.5"))
    assert engine._get_api_key(engine._remote_params) == _FAKE_KEY


def test_engine_ignores_a_credential_that_needs_reauthentication(
    store: CredentialStore, monkeypatch: pytest.MonkeyPatch
):
    session = OrcaRouterSession(store=store)
    credential = session.connect_with_api_key(_FAKE_KEY)
    session.mark_needs_reauth(credential)

    monkeypatch.delenv(ORCAROUTER_API_KEY_ENV_VAR, raising=False)
    monkeypatch.setattr(
        "oumi.inference.orcarouter_inference_engine.OrcaRouterSession",
        lambda: OrcaRouterSession(store=store),
    )

    engine = OrcaRouterInferenceEngine(ModelParams(model_name="openai/gpt-5.5"))
    assert engine._get_api_key(engine._remote_params) is None


def test_the_environment_variable_wins_over_the_stored_credential(
    store: CredentialStore, monkeypatch: pytest.MonkeyPatch
):
    OrcaRouterSession(store=store).connect_with_api_key("sk-orca-22222222222222222222")
    monkeypatch.setenv(ORCAROUTER_API_KEY_ENV_VAR, "sk-orca-33333333333333333333")
    monkeypatch.setattr(
        "oumi.inference.orcarouter_inference_engine.OrcaRouterSession",
        lambda: OrcaRouterSession(store=store),
    )

    engine = OrcaRouterInferenceEngine(ModelParams(model_name="openai/gpt-5.5"))
    # The stored key was not copied in, so the documented env var is what
    # resolution falls through to.
    assert engine._remote_params.api_key is None
    assert engine._get_api_key(engine._remote_params) == "sk-orca-33333333333333333333"


def test_credential_seam_yields_one_kind_of_result_from_both_adapters(
    fake_auth, tmp_path: Path
):
    """Both adapters implement the seam and produce the same artifact shape."""
    from oumi.orcarouter.credentials import CredentialAdapter

    store = CredentialStore(directory=tmp_path)
    api_key_adapter = OrcaRouterApiKeyAdapter(_FAKE_KEY, store=store, environ={})
    assert isinstance(api_key_adapter, CredentialAdapter)
    assert api_key_adapter.source is CredentialSource.API_KEY
    api_key_result = api_key_adapter.acquire()

    pkce_adapter = OrcaRouterSession(store=store)
    holder = {}

    pkce_result = pkce_adapter.connect_with_oauth(
        flow="oob",
        open_browser=False,
        auth_base_url=fake_auth.auth_base_url,
        on_authorize_url=lambda url: holder.update(url=url),
        read_code=lambda: fake_auth.mint_code(
            fake_auth.code_challenge_from_authorize_url(holder["url"])
        ),
    )

    assert type(api_key_result) is type(pkce_result)
    assert api_key_result.key.startswith(API_KEY_PREFIX)
    assert pkce_result.key.startswith(API_KEY_PREFIX)
    assert api_key_result.source is not pkce_result.source
    # The concrete PKCE adapter also satisfies the seam.

    concrete = OAuthPkceAdapter(
        flow="oob", open_browser=False, auth_base_url=fake_auth.auth_base_url
    )
    assert isinstance(concrete, CredentialAdapter)


def test_secrets_never_reach_logs_or_errors(
    store: CredentialStore, caplog: pytest.LogCaptureFixture
):
    caplog.set_level(logging.DEBUG)
    with pytest.raises(OrcaRouterCredentialError) as excinfo:
        OrcaRouterApiKeyAdapter("wrong-prefix-value", store=store, environ={}).acquire()
    session = OrcaRouterSession(store=store)
    session.connect_with_api_key(_FAKE_KEY)
    credential = session.current_credential()
    assert credential is not None
    session.mark_needs_reauth(credential)

    combined = caplog.text + str(excinfo.value)
    assert _FAKE_KEY not in combined
    assert "wrong-prefix-value" not in combined


def test_origins_are_configurable_and_never_derived_from_each_other(
    monkeypatch: pytest.MonkeyPatch,
):
    for var in ("ORCA_AUTH_BASE_URL", "ORCA_API_BASE_URL", "ORCA_BASE_URL"):
        monkeypatch.delenv(var, raising=False)

    # Public defaults: two different origins.
    assert resolve_auth_base_url() == "https://www.orcarouter.ai"
    assert resolve_api_base_url() == "https://api.orcarouter.ai"

    # A shared self-hosted base applies to both.
    monkeypatch.setenv("ORCA_BASE_URL", "https://gateway.internal.example")
    assert resolve_auth_base_url() == "https://gateway.internal.example"
    assert resolve_api_base_url() == "https://gateway.internal.example"

    # Explicit overrides take precedence over the shared base, independently.
    monkeypatch.setenv("ORCA_AUTH_BASE_URL", "https://auth.internal.example")
    monkeypatch.setenv("ORCA_API_BASE_URL", "https://api.internal.example")
    assert resolve_auth_base_url() == "https://auth.internal.example"
    assert resolve_api_base_url() == "https://api.internal.example"


def test_plain_http_is_only_allowed_for_loopback(monkeypatch: pytest.MonkeyPatch):
    assert resolve_api_base_url("http://127.0.0.1:8080") == "http://127.0.0.1:8080"
    assert resolve_auth_base_url("http://localhost:8080") == "http://localhost:8080"
    with pytest.raises(OrcaRouterCredentialError, match="https"):
        resolve_api_base_url("http://gateway.example.com")
    with pytest.raises(OrcaRouterCredentialError):
        resolve_auth_base_url("ftp://gateway.example.com")
