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

"""Tests for the OrcaRouter OAuth 2.0 + PKCE login.

Every code, verifier, and key here is a fixture. The fake authorization server
lives on loopback and implements the documented authentication contract, so the
tests exercise the real adapter: authorize URL -> callback/OOB -> exchange ->
persist.
"""

import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from oumi.orcarouter.credentials import (
    API_KEY_PREFIX,
    CredentialSource,
    CredentialStore,
    OrcaRouterCredentialError,
    OrcaRouterSession,
)
from oumi.orcarouter.login import (
    DEFAULT_CALLBACK_PATH,
    PkceExchangeError,
    build_authorize_url,
    code_challenge_for,
    exchange_code,
    generate_state,
    generate_verifier,
    new_pkce_authorization,
    parse_granted_scope,
    run_oob_authorization,
    run_redirect_authorization,
)


def make_pkce_credential(key: str = "sk-orca-66666666666666666666"):
    """Builds a PKCE credential the way the exchange does, without the network."""
    from oumi.orcarouter.credentials import CredentialResult

    return CredentialResult(
        key=key,
        source=CredentialSource.OAUTH_PKCE,
        scope="api",
        account_id="test-user-1",
    )


# --- PKCE primitives ---------------------------------------------------------


def test_verifier_and_state_are_fresh_cryptographic_randomness():
    verifiers = {generate_verifier() for _ in range(64)}
    states = {generate_state() for _ in range(64)}
    assert len(verifiers) == 64
    assert len(states) == 64
    for verifier in verifiers:
        assert "=" not in verifier
        assert "+" not in verifier and "/" not in verifier
        assert len(verifier) >= 43


def test_challenge_is_unpadded_base64url_sha256_of_the_verifier():
    verifier = "fixed-verifier-for-the-known-answer-test"
    challenge = code_challenge_for(verifier)
    assert challenge == "hir5lifyQVcu9A9qBx5ERSlfcUM1Pb7GT1uWaY-CEsg"
    assert "=" not in challenge


def test_authorize_url_targets_the_auth_origin_with_s256(fake_auth):
    authorization = new_pkce_authorization(
        auth_base_url=fake_auth.auth_base_url,
        callback_url="oob",
        app_name="Oumi",
    )
    assert authorization.authorize_url.startswith(f"{fake_auth.auth_base_url}/auth?")

    query = dict(
        pair.split("=", 1)
        for pair in authorization.authorize_url.split("?", 1)[1].split("&")
    )
    assert query["code_challenge_method"] == "S256"
    assert query["callback_url"] == "oob"
    assert query["scope"] == "api"
    assert query["app_name"] == "Oumi"
    # The verifier must never ride on the URL, and the challenge must match it.
    assert authorization.verifier not in authorization.authorize_url
    assert code_challenge_for(authorization.verifier) == authorization.challenge


def test_authorize_url_never_targets_the_inference_origin(fake_auth):
    url = build_authorize_url(
        "https://www.orcarouter.ai",
        callback_url="oob",
        challenge="abc",
        state="st",
    )
    assert url.startswith("https://www.orcarouter.ai/auth?")
    assert "api.orcarouter.ai" not in url


def test_pkce_authorization_repr_hides_the_verifier(fake_auth):
    authorization = new_pkce_authorization(
        auth_base_url=fake_auth.auth_base_url, callback_url="oob"
    )
    assert authorization.verifier not in repr(authorization)
    assert authorization.verifier not in str(authorization)


def test_scope_normalization():
    assert parse_granted_scope("api") == "api"
    assert parse_granted_scope(["api", "connector"]) == "api connector"
    assert parse_granted_scope(None) == ""


# --- Exchange ----------------------------------------------------------------


def test_exchange_uses_the_correct_path_and_body(fake_auth):
    authorization = new_pkce_authorization(
        auth_base_url=fake_auth.auth_base_url, callback_url="oob"
    )
    code = fake_auth.mint_code(authorization.challenge)

    credential = exchange_code(fake_auth.auth_base_url, code, authorization.verifier)

    assert len(fake_auth.exchange_requests) == 1
    request = fake_auth.exchange_requests[0]
    assert request["path"] == "/api/v1/auth/keys"
    assert request["body"] == {
        "code": code,
        "code_verifier": authorization.verifier,
        "code_challenge_method": "S256",
    }
    assert credential.key == fake_auth.returned_key
    assert credential.source is CredentialSource.OAUTH_PKCE
    assert credential.scope == "api"
    assert credential.account_id == fake_auth.returned_user_id
    assert credential.generation


def test_exchange_never_calls_the_inference_origin(fake_auth):
    authorization = new_pkce_authorization(
        auth_base_url=fake_auth.auth_base_url, callback_url="oob"
    )
    code = fake_auth.mint_code(authorization.challenge)
    exchange_code(fake_auth.auth_base_url, code, authorization.verifier)

    assert all(path == "/api/v1/auth/keys" for path in fake_auth.seen_paths), (
        fake_auth.seen_paths
    )
    assert not any("api.orcarouter.ai/v1/auth" in path for path in fake_auth.seen_paths)

    # The documented wrong path is refused before any request is made.
    with pytest.raises(OrcaRouterCredentialError, match="inference origin"):
        exchange_code("https://api.orcarouter.ai", "code", "verifier")


def test_exchange_rejects_a_reused_code(fake_auth):
    authorization = new_pkce_authorization(
        auth_base_url=fake_auth.auth_base_url, callback_url="oob"
    )
    code = fake_auth.mint_code(authorization.challenge)
    exchange_code(fake_auth.auth_base_url, code, authorization.verifier)

    with pytest.raises(PkceExchangeError, match="403"):
        exchange_code(fake_auth.auth_base_url, code, authorization.verifier)


def test_exchange_rejects_an_unknown_or_expired_code(fake_auth):
    authorization = new_pkce_authorization(
        auth_base_url=fake_auth.auth_base_url, callback_url="oob"
    )
    with pytest.raises(PkceExchangeError, match="403"):
        exchange_code(fake_auth.auth_base_url, "stale-code", authorization.verifier)


def test_exchange_rejects_a_mismatched_verifier(fake_auth):
    authorization = new_pkce_authorization(
        auth_base_url=fake_auth.auth_base_url, callback_url="oob"
    )
    code = fake_auth.mint_code(authorization.challenge)
    with pytest.raises(PkceExchangeError, match="403"):
        exchange_code(fake_auth.auth_base_url, code, "not-the-verifier")


def test_exchange_reports_a_scope_downgrade(fake_auth):
    fake_auth.returned_scope = "connector"
    authorization = new_pkce_authorization(
        auth_base_url=fake_auth.auth_base_url, callback_url="oob"
    )
    code = fake_auth.mint_code(authorization.challenge)
    with pytest.raises(PkceExchangeError, match="narrower scope"):
        exchange_code(fake_auth.auth_base_url, code, authorization.verifier)


def test_exchange_reports_a_missing_scope(fake_auth):
    fake_auth.returned_scope = ""
    authorization = new_pkce_authorization(
        auth_base_url=fake_auth.auth_base_url, callback_url="oob"
    )
    code = fake_auth.mint_code(authorization.challenge)
    with pytest.raises(PkceExchangeError, match="which scope was granted"):
        exchange_code(fake_auth.auth_base_url, code, authorization.verifier)


def test_exchange_reports_rate_limiting(fake_auth):
    fake_auth.forced_status = 429
    authorization = new_pkce_authorization(
        auth_base_url=fake_auth.auth_base_url, callback_url="oob"
    )
    code = fake_auth.mint_code(authorization.challenge)
    with pytest.raises(PkceExchangeError, match="429"):
        exchange_code(fake_auth.auth_base_url, code, authorization.verifier)


def test_exchange_reports_denial_and_method_downgrade(fake_auth):
    fake_auth.forced_status = 400
    authorization = new_pkce_authorization(
        auth_base_url=fake_auth.auth_base_url, callback_url="oob"
    )
    code = fake_auth.mint_code(authorization.challenge)
    with pytest.raises(PkceExchangeError, match="400"):
        exchange_code(fake_auth.auth_base_url, code, authorization.verifier)


def test_exchange_reports_a_network_failure_without_leaking_the_verifier():
    authorization = new_pkce_authorization(callback_url="oob")

    def unreachable(request, timeout):
        raise urllib.error.URLError("connection refused")

    with pytest.raises(PkceExchangeError) as excinfo:
        exchange_code(
            "https://www.orcarouter.ai",
            "some-code",
            authorization.verifier,
            transport=unreachable,
        )
    assert authorization.verifier not in str(excinfo.value)
    assert "Could not reach" in str(excinfo.value)


def test_exchange_never_echoes_the_verifier_in_a_provider_error(fake_auth):
    authorization = new_pkce_authorization(
        auth_base_url=fake_auth.auth_base_url, callback_url="oob"
    )
    code = fake_auth.mint_code(authorization.challenge)
    with pytest.raises(PkceExchangeError) as excinfo:
        exchange_code(fake_auth.auth_base_url, code, "wrong-verifier-secret-value")
    assert "wrong-verifier-secret-value" not in str(excinfo.value)


def test_exchange_rejects_a_non_key_response():
    authorization = new_pkce_authorization(callback_url="oob")

    def fake_transport(request, timeout):
        return b'{"user_id": "1", "scope": "api"}'

    with pytest.raises(PkceExchangeError, match="without a usable API key"):
        exchange_code(
            "https://www.orcarouter.ai",
            "code",
            authorization.verifier,
            transport=fake_transport,
        )


# --- Flow B: out-of-band -----------------------------------------------------


def test_oob_flow_runs_end_to_end_and_returns_a_plain_key(fake_auth):
    holder = {}

    def collect_code():
        # The client has already opened the authorize URL by now; the consent
        # screen would have shown the code, so we mint one for that challenge.
        return fake_auth.mint_code(
            fake_auth.code_challenge_from_authorize_url(holder["url"])
        )

    result = run_oob_authorization(
        auth_base_url=fake_auth.auth_base_url,
        app_name="Oumi",
        open_browser=False,
        on_authorize_url=lambda url: holder.update(url=url),
        read_code=collect_code,
    )

    assert holder["url"].startswith(f"{fake_auth.auth_base_url}/auth?")
    assert fake_auth.callback_url_from_authorize_url(holder["url"]) == "oob"
    assert result.key.startswith(API_KEY_PREFIX)
    assert result.source is CredentialSource.OAUTH_PKCE


def test_oob_flow_without_a_code_stores_nothing(fake_auth):
    with pytest.raises(OrcaRouterCredentialError, match="No authorization code"):
        run_oob_authorization(
            auth_base_url=fake_auth.auth_base_url,
            open_browser=False,
            on_authorize_url=lambda url: None,
            read_code=lambda: "   ",
        )


def test_oob_flow_rejects_a_cancelled_prompt(fake_auth):
    def cancelled():
        raise OrcaRouterCredentialError("cancelled by the user")

    with pytest.raises(OrcaRouterCredentialError):
        run_oob_authorization(
            auth_base_url=fake_auth.auth_base_url,
            open_browser=False,
            on_authorize_url=lambda url: None,
            read_code=cancelled,
        )


# --- Flow A: loopback redirect ----------------------------------------------


def _drive_redirect(fake_auth, authorization_url_holder, decide):
    """A driver that answers the loopback callback with ``decide``'s query."""

    def on_url(url):
        authorization_url_holder["url"] = url
        callback = fake_auth.callback_url_from_authorize_url(url)
        state = fake_auth.state_from_authorize_url(url)
        challenge = fake_auth.code_challenge_from_authorize_url(url)
        query = decide(state, challenge)

        def deliver():
            with urllib.request.urlopen(f"{callback}?{query}") as response:
                response.read()

        threading.Thread(target=deliver, daemon=True).start()

    return on_url


def test_redirect_flow_exchanges_the_callback_code(fake_auth):
    holder = {}

    def decide(state, challenge):
        code = fake_auth.mint_code(challenge)
        return f"code={code}&state={state}"

    credential = run_redirect_authorization(
        auth_base_url=fake_auth.auth_base_url,
        open_browser=False,
        timeout_seconds=10,
        on_authorize_url=_drive_redirect(fake_auth, holder, decide),
    )

    assert credential.key.startswith(API_KEY_PREFIX)
    callback = fake_auth.callback_url_from_authorize_url(holder["url"])
    assert callback.startswith("http://127.0.0.1:")
    assert callback.endswith(DEFAULT_CALLBACK_PATH)
    assert fake_auth.exchange_requests[0]["path"] == "/api/v1/auth/keys"


def test_redirect_flow_rejects_a_state_mismatch(fake_auth):
    def decide(state, challenge):
        code = fake_auth.mint_code(challenge)
        return f"code={code}&state=someone-elses-state"

    with pytest.raises(OrcaRouterCredentialError, match="did not match the state"):
        run_redirect_authorization(
            auth_base_url=fake_auth.auth_base_url,
            open_browser=False,
            timeout_seconds=10,
            on_authorize_url=_drive_redirect(fake_auth, {}, decide),
        )
    assert fake_auth.exchange_requests == []


def test_redirect_flow_reports_denial_and_releases_the_listener(fake_auth):
    def decide(state, challenge):
        return f"error=access_denied&state={state}"

    with pytest.raises(OrcaRouterCredentialError, match="access_denied"):
        run_redirect_authorization(
            auth_base_url=fake_auth.auth_base_url,
            open_browser=False,
            timeout_seconds=10,
            on_authorize_url=_drive_redirect(fake_auth, {}, decide),
        )
    assert fake_auth.exchange_requests == []


def test_redirect_flow_times_out_cleanly(fake_auth):
    with pytest.raises(OrcaRouterCredentialError, match="Timed out"):
        run_redirect_authorization(
            auth_base_url=fake_auth.auth_base_url,
            open_browser=False,
            timeout_seconds=0.3,
            on_authorize_url=lambda url: None,
        )


def test_redirect_flow_can_accept_a_code_shown_on_the_consent_screen(fake_auth):
    """The user may pick "show me a code" even though a callback_url was sent."""
    holder = {}

    def on_url(url):
        holder["challenge"] = fake_auth.code_challenge_from_authorize_url(url)

    credential = run_redirect_authorization(
        auth_base_url=fake_auth.auth_base_url,
        open_browser=False,
        timeout_seconds=0.3,
        on_authorize_url=on_url,
        prompt_for_code=lambda: fake_auth.mint_code(holder["challenge"]),
    )
    assert credential.key.startswith(API_KEY_PREFIX)


# --- Session integration -----------------------------------------------------


def test_session_connect_with_oauth_persists_a_reusable_credential(
    fake_auth, tmp_path: Path
):
    store = CredentialStore(directory=tmp_path)
    session = OrcaRouterSession(store=store)
    holder = {}

    def decide(state, challenge):
        return f"code={fake_auth.mint_code(challenge)}&state={state}"

    credential = session.connect_with_oauth(
        flow="redirect",
        open_browser=False,
        timeout_seconds=10,
        auth_base_url=fake_auth.auth_base_url,
        on_authorize_url=_drive_redirect(fake_auth, holder, decide),
    )

    assert credential.source is CredentialSource.OAUTH_PKCE
    stored = store.load()
    assert stored is not None
    assert stored.key == credential.key
    assert stored.needs_reauth is False

    # A second run reuses it and mints nothing new.
    assert session.active_credential().key == credential.key  # type: ignore[union-attr]
    assert len(fake_auth.exchange_requests) == 1
