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

"""OAuth 2.0 + PKCE login for OrcaRouter.

Two delivery flows are implemented, both with ``S256``:

* **Flow A -- loopback redirect** (default): a browser is opened at
  ``{auth_base}/auth`` and the one-time code is delivered to a listener bound to
  ``127.0.0.1``. This is the normal path for a developer running ``oumi`` on
  their own machine.
* **Flow B -- out-of-band code**: ``callback_url=oob`` makes the consent screen
  display the code, which the user pastes back. Used on hosts that cannot accept
  an inbound connection.

The flow yields an ordinary OrcaRouter API key. OrcaRouter does not issue refresh
tokens and there is no refresh grant, so the key is stored and reused until the
user revokes it.

No client secret is involved and no redirect URI has to be pre-registered: PKCE
binds the authorization code to this process, so an intercepted code cannot be
redeemed by anyone else. The verifier never leaves the process, is never logged,
and is never placed in a URL.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import threading
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from collections.abc import Callable
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from oumi.orcarouter.credentials import (
    API_KEY_PREFIX,
    AUTHORIZE_PATH,
    EXCHANGE_PATH,
    CredentialResult,
    CredentialSource,
    OrcaRouterCredentialError,
    auth_endpoint_url,
    resolve_api_base_url,
    resolve_auth_base_url,
)
from oumi.utils.logging import logger

DEFAULT_CALLBACK_PATH = "/callback"
# Loopback path the authorization code is delivered to.

DEFAULT_SCOPE = "api"
# The scope this client asks for: inference access. The relay never needs more.

DEFAULT_TIMEOUT_SECONDS = 300.0
# How long to wait for the user to finish in the browser.

EXCHANGE_TIMEOUT_SECONDS = 30.0
# Per-request timeout for the code exchange.

BROWSER_LABEL = "Oumi"
# Label shown on the consent screen. It is a claim by this client, nothing more.

_CLOSE_TAB_HTML = (
    "<!doctype html><html><head><meta charset='utf-8'>"
    "<title>OrcaRouter</title></head><body style='font-family: sans-serif; "
    "margin: 4rem auto; max-width: 32rem;'>"
    "<h2>Oumi is connected.</h2>"
    "<p>You can close this tab and return to your terminal.</p>"
    "</body></html>"
)

_FAILURE_HTML = (
    "<!doctype html><html><head><meta charset='utf-8'>"
    "<title>OrcaRouter</title></head><body style='font-family: sans-serif; "
    "margin: 4rem auto; max-width: 32rem;'>"
    "<h2>Authorization did not complete.</h2>"
    "<p>Return to your terminal for details, then try again.</p>"
    "</body></html>"
)


class PkceExchangeError(OrcaRouterCredentialError):
    """Raised when the authorization code cannot be exchanged for a key."""


@dataclass
class PkceAuthorization:
    """The per-attempt PKCE material.

    The verifier is generated fresh from a cryptographic RNG for every attempt
    and must never be logged, printed, or placed in a URL. Only the challenge
    travels on the authorize URL.
    """

    verifier: str
    challenge: str
    state: str
    authorize_url: str

    def __repr__(self) -> str:
        """Returns a representation that deliberately omits the verifier."""
        return (
            f"PkceAuthorization(verifier=<redacted>, challenge={self.challenge!r}, "
            f"state={self.state!r}, authorize_url={self.authorize_url!r})"
        )

    __str__ = __repr__


def _b64url(raw: bytes) -> str:
    """Returns unpadded base64url, as required for ``code_challenge``."""
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def generate_verifier() -> str:
    """Returns a fresh high-entropy PKCE verifier from the cryptographic RNG."""
    return _b64url(secrets.token_bytes(32))


def code_challenge_for(verifier: str) -> str:
    """Returns ``base64url(sha256(verifier))`` without padding."""
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    return _b64url(digest)


def generate_state() -> str:
    """Returns a fresh opaque CSRF token from the cryptographic RNG."""
    return _b64url(secrets.token_bytes(16))


def build_authorize_url(
    auth_base_url: str,
    *,
    callback_url: str,
    challenge: str,
    state: str,
    app_name: str = BROWSER_LABEL,
    scope: str = DEFAULT_SCOPE,
    login_hint: str | None = None,
    workspace_hint: str | None = None,
    prompt: str | None = None,
) -> str:
    """Builds the consent-screen URL.

    ``code_challenge_method`` is always ``S256``: even when a real
    ``callback_url`` is supplied, the user may choose "Show me a code" on the
    consent screen, which puts the code in human hands.

    Args:
        auth_base_url: The authentication origin.
        callback_url: Where the code is delivered, or the literal ``oob``.
        challenge: ``base64url(sha256(verifier))``.
        state: Opaque CSRF token, echoed back verbatim.
        app_name: Label shown on the consent screen.
        scope: ``api`` for inference access.
        login_hint: Optional email pre-fill.
        workspace_hint: Optional workspace pre-selection.
        prompt: Optional ``consent`` to force re-approval.

    Returns:
        str: The absolute authorize URL.
    """
    query = {
        "callback_url": callback_url,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "state": state,
        "app_name": app_name,
        "scope": scope,
    }
    if login_hint:
        query["login_hint"] = login_hint
    if workspace_hint:
        query["workspace_hint"] = workspace_hint
    if prompt:
        query["prompt"] = prompt
    base = auth_endpoint_url(auth_base_url, AUTHORIZE_PATH)
    return f"{base}?{urllib.parse.urlencode(query)}"


def new_pkce_authorization(
    *,
    auth_base_url: str | None = None,
    callback_url: str,
    app_name: str = BROWSER_LABEL,
    scope: str = DEFAULT_SCOPE,
) -> PkceAuthorization:
    """Creates fresh PKCE material and the authorize URL for one attempt."""
    verifier = generate_verifier()
    challenge = code_challenge_for(verifier)
    state = generate_state()
    url = build_authorize_url(
        resolve_auth_base_url(auth_base_url),
        callback_url=callback_url,
        challenge=challenge,
        state=state,
        app_name=app_name,
        scope=scope,
    )
    return PkceAuthorization(
        verifier=verifier, challenge=challenge, state=state, authorize_url=url
    )


def parse_granted_scope(raw_scope: object) -> str:
    """Normalizes a granted ``scope`` value into a canonical token set string."""
    if isinstance(raw_scope, str):
        tokens = raw_scope.replace(",", " ").split()
    elif isinstance(raw_scope, list):
        tokens = [t for t in raw_scope if isinstance(t, str)]
    else:
        tokens = []
    return " ".join(sorted({token for token in tokens if token}))


def _scrub(message: str, *secrets_to_remove: str | None) -> str:
    """Removes any occurrence of a secret from a message before it is surfaced."""
    scrubbed = message
    for secret in secrets_to_remove:
        if secret:
            scrubbed = scrubbed.replace(secret, "<redacted>")
    return scrubbed


def exchange_code(
    auth_base_url: str,
    code: str,
    code_verifier: str,
    *,
    code_challenge_method: str = "S256",
    timeout_seconds: float = EXCHANGE_TIMEOUT_SECONDS,
    transport: Callable[[urllib.request.Request, float], object] | None = None,
    api_base_url: str | None = None,
) -> CredentialResult:
    """Exchanges an authorization code for a durable OrcaRouter API key.

    The request goes to the *authentication* origin's ``/api/v1/auth/keys`` --
    never to the inference origin's ``/v1``.

    Args:
        auth_base_url: The authentication origin.
        code: The one-time authorization code.
        code_verifier: The PKCE verifier generated for this attempt.
        code_challenge_method: Must stay ``S256``.
        timeout_seconds: Request timeout.
        transport: Optional transport override, used by tests.
        api_base_url: Optional inference origin, carried for diagnostics only.

    Returns:
        CredentialResult: The newly issued credential with its granted scope.

    Raises:
        PkceExchangeError: On denial, expiry, reuse, scope downgrade, rate
            limiting, or any transport failure. Messages never contain the
            verifier or the returned key.
    """
    url = auth_endpoint_url(resolve_auth_base_url(auth_base_url), EXCHANGE_PATH)
    body = json.dumps(
        {
            "code": code,
            "code_verifier": code_verifier,
            "code_challenge_method": code_challenge_method,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )

    opener = transport if transport is not None else _default_transport
    try:
        raw = opener(request, timeout_seconds)
    except urllib.error.HTTPError as e:
        raise PkceExchangeError(_describe_http_error(e, code_verifier)) from None
    except urllib.error.URLError as e:
        reason = _scrub(str(getattr(e, "reason", e)), code_verifier)
        raise PkceExchangeError(
            "Could not reach the OrcaRouter authentication endpoint "
            f"({type(e).__name__}). Check your network connection and try again. "
            f"Details: {reason}"
        ) from None
    except TimeoutError:
        raise PkceExchangeError(
            "The OrcaRouter authentication endpoint timed out while exchanging "
            "the authorization code. Start the login again."
        ) from None

    payload = _decode_exchange_payload(raw, code_verifier)
    key = payload.get("key")
    if not isinstance(key, str) or not key.startswith(API_KEY_PREFIX):
        raise PkceExchangeError(
            "OrcaRouter returned a successful response without a usable API key."
        )

    granted_scope = parse_granted_scope(payload.get("scope"))
    _require_sufficient_scope(granted_scope)

    account_id = payload.get("user_id")
    return CredentialResult(
        key=key,
        source=CredentialSource.OAUTH_PKCE,
        scope=granted_scope or DEFAULT_SCOPE,
        account_id=str(account_id) if account_id is not None else None,
    )


def _default_transport(request: urllib.request.Request, timeout: float) -> bytes:
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        return response.read()


def _describe_http_error(error: urllib.error.HTTPError, code_verifier: str) -> str:
    """Returns an actionable, secret-free description of a failed exchange."""
    detail = ""
    try:
        raw = error.read()
        parsed = json.loads(raw.decode("utf-8", errors="replace"))
        if isinstance(parsed, dict):
            candidate = parsed.get("error_description") or parsed.get("error")
            if isinstance(candidate, str):
                detail = candidate
    except Exception:
        detail = ""

    detail = _scrub(detail, code_verifier)
    if error.code == 400:
        base = (
            "OrcaRouter rejected the code exchange (HTTP 400). The authorization "
            "request and the exchange disagreed, or an unsupported "
            "code_challenge_method was used. Start the login again."
        )
    elif error.code == 403:
        base = (
            "OrcaRouter rejected the authorization code (HTTP 403). The code was "
            "unknown, already used, or expired, or the verifier did not match. "
            "Start the login again."
        )
    elif error.code == 429:
        base = (
            "OrcaRouter rate-limited the login (HTTP 429). Each account may issue "
            "at most 10 keys per 24 hours; reuse the stored key, or wait before "
            "retrying."
        )
    else:
        base = f"OrcaRouter rejected the code exchange (HTTP {error.code})."
    return f"{base} Details: {detail}" if detail else base


def _decode_exchange_payload(raw: object, code_verifier: str) -> dict:
    if isinstance(raw, (bytes, bytearray)):
        text = bytes(raw).decode("utf-8", errors="replace")
    elif isinstance(raw, str):
        text = raw
    else:
        raise PkceExchangeError(
            "OrcaRouter returned an unexpected response while exchanging the code."
        )
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        raise PkceExchangeError(
            "OrcaRouter returned a response that is not JSON while exchanging "
            "the authorization code."
        ) from None
    if not isinstance(parsed, dict):
        raise PkceExchangeError(
            "OrcaRouter returned an unexpected response shape while exchanging "
            "the authorization code."
        )
    if parsed.get("error"):
        message = parsed.get("error_description") or parsed.get("error")
        raise PkceExchangeError(
            _scrub(f"OrcaRouter refused the code exchange: {message}", code_verifier)
        )
    return parsed


def _require_sufficient_scope(granted_scope: str) -> None:
    """Refuses to proceed when the granted scope cannot serve inference.

    The value read back is what was *granted*, not what was requested.
    """
    tokens = set(granted_scope.split())
    if DEFAULT_SCOPE in tokens:
        return
    if not tokens:
        raise PkceExchangeError(
            "OrcaRouter did not report which scope was granted, so this client "
            "cannot confirm it may make inference requests. Start the login again."
        )
    raise PkceExchangeError(
        "OrcaRouter granted a narrower scope than this client needs "
        f"(granted: {granted_scope!r}). Re-authorize with the 'api' scope, or "
        "generate an API key in the console instead."
    )


def _open_browser(url: str, open_browser: bool) -> None:
    if not open_browser:
        return
    try:
        if not webbrowser.open(url):
            logger.warning(
                "Could not open a browser automatically. Open the URL above to "
                "continue."
            )
    except Exception as e:  # pragma: no cover - platform dependent
        logger.warning(f"Could not open a browser automatically ({type(e).__name__}).")


class _CallbackOutcome:
    """Thread-safe holder for the single callback the listener will handle."""

    def __init__(self, expected_state: str) -> None:
        self._expected_state = expected_state
        self._event = threading.Event()
        self._code: str | None = None
        self._error: str | None = None
        self._delivered = False

    def deliver(self, query: dict[str, list[str]]) -> None:
        """Records the first callback only; later requests are ignored."""
        if self._delivered:
            return
        self._delivered = True

        state = (query.get("state") or [""])[0]
        if not hmac.compare_digest(state, self._expected_state):
            self._error = (
                "The authorization response did not match the state this client "
                "sent. The login was discarded; start it again."
            )
            self._event.set()
            return

        error = (query.get("error") or [""])[0]
        if error:
            self._error = (
                f"OrcaRouter did not authorize this client (error: {error}). "
                "No credential was stored."
            )
            self._event.set()
            return

        self._code = (query.get("code") or [""])[0] or None
        if self._code is None:
            self._error = (
                "OrcaRouter returned no authorization code. The login was "
                "discarded; start it again."
            )
        self._event.set()

    def wait(self, timeout_seconds: float) -> None:
        """Blocks until a callback arrives or the deadline passes."""
        if not self._event.wait(timeout_seconds):
            raise OrcaRouterCredentialError(
                "Timed out waiting for the OrcaRouter authorization. Nothing was "
                "stored; start the login again when you are ready."
            )

    def result(self) -> str:
        """Returns the authorization code, or raises with the recorded reason."""
        if self._error:
            raise OrcaRouterCredentialError(self._error)
        if not self._code:
            raise OrcaRouterCredentialError(
                "OrcaRouter did not return an authorization code."
            )
        return self._code

    @property
    def delivered(self) -> bool:
        """True when a callback was handled."""
        return self._delivered


class _UnboundHandler(BaseHTTPRequestHandler):
    """Placeholder handler used only while the loopback port is being reserved."""

    def do_GET(self) -> None:  # noqa: N802 - stdlib naming
        self.send_response(503)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, format: str, *args) -> None:
        return


class OAuthPkceAdapter:
    """The PKCE side of the credential seam.

    It acquires a credential by running the OAuth 2.0 + PKCE login and returns
    the same :class:`~oumi.orcarouter.credentials.CredentialResult` shape as the
    API-key adapter, so inference and model discovery never branch on how the
    key was obtained.
    """

    def __init__(
        self,
        *,
        flow: str = "redirect",
        auth_base_url: str | None = None,
        api_base_url: str | None = None,
        app_name: str = BROWSER_LABEL,
        open_browser: bool = True,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        scope: str = DEFAULT_SCOPE,
        on_authorize_url: Callable[[str], None] | None = None,
        prompt_for_code: Callable[[], str] | None = None,
        read_code: Callable[[], str] | None = None,
        transport: Callable[[urllib.request.Request, float], object] | None = None,
    ) -> None:
        """Initializes the adapter.

        Args:
            flow: ``"redirect"`` for the loopback flow, ``"oob"`` for the
                out-of-band code flow.
            auth_base_url: Authentication-origin override.
            api_base_url: Inference-origin override (diagnostics only).
            app_name: Label shown on the consent screen.
            open_browser: Whether to open the consent screen automatically.
            timeout_seconds: How long to wait for the redirect.
            scope: Requested scope; the granted scope is read back regardless.
            on_authorize_url: Called with the authorize URL before it is opened.
            prompt_for_code: Loopback fallback when the user is shown a code.
            read_code: Out-of-band code collector. Defaults to ``input()``.
            transport: Optional transport override, used by tests.
        """
        self._flow = flow
        self._auth_base_url = auth_base_url
        self._api_base_url = api_base_url
        self._app_name = app_name
        self._open_browser = open_browser
        self._timeout_seconds = timeout_seconds
        self._scope = scope
        self._on_authorize_url = on_authorize_url
        self._prompt_for_code = prompt_for_code
        self._read_code = read_code
        self._transport = transport

    @property
    def source(self) -> CredentialSource:
        """Returns the credential source for this adapter."""
        return CredentialSource.OAUTH_PKCE

    def acquire(self) -> CredentialResult:
        """Runs the PKCE login and returns the resulting credential."""
        if self._flow == "oob":
            return run_oob_authorization(
                auth_base_url=self._auth_base_url,
                api_base_url=self._api_base_url,
                app_name=self._app_name,
                open_browser=self._open_browser,
                timeout_seconds=self._timeout_seconds,
                scope=self._scope,
                transport=self._transport,
                on_authorize_url=self._on_authorize_url,
                read_code=self._read_code,
            )
        return run_redirect_authorization(
            auth_base_url=self._auth_base_url,
            api_base_url=self._api_base_url,
            app_name=self._app_name,
            open_browser=self._open_browser,
            timeout_seconds=self._timeout_seconds,
            scope=self._scope,
            transport=self._transport,
            on_authorize_url=self._on_authorize_url,
            prompt_for_code=self._prompt_for_code,
        )


def _make_handler(outcome: _CallbackOutcome, path: str):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self) -> None:  # noqa: N802 - stdlib naming
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path != path:
                self.send_response(404)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            outcome.deliver(urllib.parse.parse_qs(parsed.query))
            body = _CLOSE_TAB_HTML if outcome._code else _FAILURE_HTML
            encoded = body.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, format: str, *args) -> None:
            """Suppresses the default stderr logging, which would echo the URL."""
            return

    return Handler


def run_redirect_authorization(
    *,
    auth_base_url: str | None = None,
    api_base_url: str | None = None,
    app_name: str = BROWSER_LABEL,
    open_browser: bool = True,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    scope: str = DEFAULT_SCOPE,
    transport: Callable[[urllib.request.Request, float], object] | None = None,
    on_authorize_url: Callable[[str], None] | None = None,
    prompt_for_code: Callable[[], str] | None = None,
) -> CredentialResult:
    """Runs Flow A: loopback redirect with PKCE (S256).

    The listener is bound before the browser opens, so the port is known and
    nothing races. ``state`` is compared in constant time before the code is
    touched. Every terminal path releases the listener.

    Args:
        auth_base_url: Authentication origin override.
        api_base_url: Inference origin override (recorded for diagnostics).
        app_name: Label shown on the consent screen.
        open_browser: Whether to open the consent screen automatically.
        timeout_seconds: How long to wait for the redirect.
        scope: Requested scope; the granted scope is read back regardless.
        transport: Optional transport override for the exchange, used by tests.
        on_authorize_url: Called with the authorize URL before it is opened.
        prompt_for_code: Optional fallback used when the consent screen shows a
            code instead of redirecting.

    Returns:
        CredentialResult: The newly issued credential.
    """
    authorization = new_pkce_authorization(
        auth_base_url=auth_base_url, callback_url="", app_name=app_name, scope=scope
    )

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _UnboundHandler)
    port = httpd.server_address[1]
    callback_url = f"http://127.0.0.1:{port}{DEFAULT_CALLBACK_PATH}"
    outcome = _CallbackOutcome(authorization.state)
    httpd.RequestHandlerClass = _make_handler(outcome, DEFAULT_CALLBACK_PATH)

    authorize_url = build_authorize_url(
        resolve_auth_base_url(auth_base_url),
        callback_url=callback_url,
        challenge=authorization.challenge,
        state=authorization.state,
        app_name=app_name,
        scope=scope,
    )

    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        logger.info(
            "Opening your browser to authorize Oumi with OrcaRouter. If it does "
            f"not open, visit this URL:\n{authorize_url}"
        )
        if on_authorize_url is not None:
            on_authorize_url(authorize_url)
        _open_browser(authorize_url, open_browser)

        try:
            outcome.wait(timeout_seconds)
        except OrcaRouterCredentialError:
            if prompt_for_code is None:
                raise
            # The consent screen may have shown a code instead of redirecting.
            pasted = prompt_for_code().strip()
            if not pasted:
                raise
            return exchange_code(
                resolve_auth_base_url(auth_base_url),
                pasted,
                authorization.verifier,
                transport=transport,
                api_base_url=resolve_api_base_url(api_base_url),
            )

        code = outcome.result()
    finally:
        httpd.shutdown()
        httpd.server_close()

    return exchange_code(
        resolve_auth_base_url(auth_base_url),
        code,
        authorization.verifier,
        timeout_seconds=min(timeout_seconds, EXCHANGE_TIMEOUT_SECONDS),
        transport=transport,
        api_base_url=resolve_api_base_url(api_base_url),
    )


def run_oob_authorization(
    *,
    auth_base_url: str | None = None,
    api_base_url: str | None = None,
    app_name: str = BROWSER_LABEL,
    open_browser: bool = True,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    scope: str = DEFAULT_SCOPE,
    transport: Callable[[urllib.request.Request, float], object] | None = None,
    on_authorize_url: Callable[[str], None] | None = None,
    read_code: Callable[[], str] | None = None,
) -> CredentialResult:
    """Runs Flow B: out-of-band code with PKCE (S256, mandatory here).

    The code is displayed on the consent screen and pasted back by the user.
    Because a human handles the code, ``S256`` is required rather than merely
    advisable -- under ``plain`` the challenge would *be* the verifier, which
    rode on the authorize URL through browser history and request logs.

    Args:
        auth_base_url: Authentication origin override.
        api_base_url: Inference origin override (recorded for diagnostics).
        app_name: Label shown on the consent screen.
        open_browser: Whether to open the consent screen automatically.
        timeout_seconds: Unused for the paste path; kept for a symmetric API.
        scope: Requested scope; the granted scope is read back regardless.
        transport: Optional transport override for the exchange, used by tests.
        on_authorize_url: Called with the authorize URL before it is opened.
        read_code: Overrides how the code is collected. Defaults to ``input()``.

    Returns:
        CredentialResult: The newly issued credential.
    """
    del timeout_seconds  # Flow B waits on the user, not on a socket.
    authorization = new_pkce_authorization(
        auth_base_url=auth_base_url, callback_url="oob", app_name=app_name, scope=scope
    )

    logger.info(
        "Open this URL, approve access, then paste the code OrcaRouter shows "
        f"you:\n{authorization.authorize_url}"
    )
    if on_authorize_url is not None:
        on_authorize_url(authorization.authorize_url)
    _open_browser(authorization.authorize_url, open_browser)

    collector = read_code if read_code is not None else _default_code_reader
    code = collector().strip()
    if not code:
        raise OrcaRouterCredentialError(
            "No authorization code was entered, so no credential was stored."
        )

    return exchange_code(
        resolve_auth_base_url(auth_base_url),
        code,
        authorization.verifier,
        transport=transport,
        api_base_url=resolve_api_base_url(api_base_url),
    )


def _default_code_reader() -> str:
    try:
        return input("OrcaRouter code: ")
    except (EOFError, KeyboardInterrupt) as e:
        raise OrcaRouterCredentialError(
            "The OrcaRouter login was cancelled, so no credential was stored."
        ) from e


def authorization_expiry_hint() -> str:
    """Returns a short note about the authorization code lifetime."""
    return (
        "OrcaRouter authorization codes are single-use and expire about 10 "
        "minutes after the consent screen mints them."
    )
