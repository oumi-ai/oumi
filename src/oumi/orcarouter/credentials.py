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

"""Credential acquisition, storage, redaction, and lifecycle for OrcaRouter.

Two user-facing entry points produce the *same* artifact -- an ordinary OrcaRouter
API key (``sk-orca-...``):

* :class:`OrcaRouterApiKeyAdapter` for a key the user already holds;
* :class:`OAuthPkceAdapter` (see :mod:`oumi.orcarouter.login`) for a key minted by
  an OAuth 2.0 + PKCE authorization against the OrcaRouter account.

The rest of the integration consumes :class:`CredentialResult` and never inspects
``source``, so inference, model discovery, and every AI entry point share one code
path regardless of how the key was obtained.

A key minted by PKCE is durable: OrcaRouter does not issue refresh tokens, so the
stored key is reused until the user revokes it. An upstream ``401`` is terminal:
exactly the account/generation that issued the rejected request is marked
``needs_reauth`` and re-authorization is required. Nothing here ever refreshes,
and the old secret is never deleted before a replacement has been stored.
"""

from __future__ import annotations

import json
import os
import stat
import time
import uuid
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Protocol, runtime_checkable
from urllib.parse import urlparse

from oumi.utils.logging import logger

DEFAULT_AUTH_BASE_URL = "https://www.orcarouter.ai"
# Public OrcaRouter authentication origin (consent screen and code exchange).

DEFAULT_API_BASE_URL = "https://api.orcarouter.ai"
# Public OrcaRouter inference origin. The OpenAI-compatible API lives under ``/v1``.

ORCAROUTER_BASE_URL_ENV_VAR = "ORCA_BASE_URL"
# Shared self-hosted fallback used when the explicit overrides are unset.

AUTH_BASE_URL_ENV_VAR = "ORCA_AUTH_BASE_URL"
# Explicit authentication-origin override. Takes precedence over the shared base.

API_BASE_URL_ENV_VAR = "ORCA_API_BASE_URL"
# Explicit inference-origin override. Takes precedence over the shared base.

ORCAROUTER_API_KEY_ENV_VAR = "ORCAROUTER_API_KEY"
# Environment variable holding a user-supplied OrcaRouter API key.

AUTHORIZE_PATH = "/auth"
# Consent-screen path on the authentication origin. It is not an API.

EXCHANGE_PATH = "/api/v1/auth/keys"
# Code-exchange path on the authentication origin.
#
# Deliberately *not* ``/v1/auth/keys`` on the inference origin: the relay is at
# ``/v1`` and the auth endpoints are not, so ``api.orcarouter.ai/v1/auth/keys``
# does not exist.

KEYS_CONSOLE_URL = "https://www.orcarouter.ai/console/authorized-apps"
# Where a user reviews and revokes the keys issued to this application.

API_KEY_PREFIX = "sk-orca-"

_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "[::1]"})

_STORE_FILE_NAME = "orcarouter_credentials.json"
_DEFAULT_OUMI_DIR = "~/.oumi"


class OrcaRouterCredentialError(Exception):
    """Raised when an OrcaRouter credential cannot be obtained or used."""


class OrcaRouterAuthError(OrcaRouterCredentialError):
    """Raised when OrcaRouter rejects a credential (HTTP 401/403)."""


def mask_api_key(api_key: str | None) -> str:
    """Returns a redacted rendering of ``api_key`` that is safe to log.

    The full secret is never returned, never written to an error message, and
    never included in a snapshot.
    """
    if not api_key:
        return "<unset>"
    if len(api_key) <= len(API_KEY_PREFIX) + 4:
        return API_KEY_PREFIX + "…"
    return f"{API_KEY_PREFIX}…{api_key[-4:]}"


def _require_supported_origin(origin: str, env_var: str) -> str:
    """Validates and normalizes an origin read from configuration.

    Remote origins must use HTTPS; plain HTTP is permitted only for loopback so
    that local development against a self-hosted gateway keeps working.
    """
    normalized = origin.strip().rstrip("/")
    if not normalized:
        raise OrcaRouterCredentialError(
            f"{env_var} is set but empty; expected an absolute URL."
        )

    parsed = urlparse(normalized)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise OrcaRouterCredentialError(
            f"{env_var} must be an absolute http(s) URL, got a value that is not."
        )
    if parsed.scheme == "http":
        hostname = (parsed.hostname or "").lower()
        if hostname not in _LOOPBACK_HOSTS:
            raise OrcaRouterCredentialError(
                f"{env_var} must use https for non-loopback origins "
                f"(the configured host is not a loopback address)."
            )
    return normalized


def resolve_auth_base_url(
    explicit: str | None = None,
    *,
    environ: dict[str, str] | None = None,
) -> str:
    """Resolves the authentication origin.

    Precedence: an explicit value (config or CLI), then ``ORCA_AUTH_BASE_URL``,
    then the shared ``ORCA_BASE_URL`` for single-origin self-hosted deployments,
    then the public default. The inference origin is never derived from this one.
    """
    env = os.environ if environ is None else environ
    for value, source in (
        (explicit, "the configured authentication base URL"),
        (env.get(AUTH_BASE_URL_ENV_VAR), AUTH_BASE_URL_ENV_VAR),
        (env.get(ORCAROUTER_BASE_URL_ENV_VAR), ORCAROUTER_BASE_URL_ENV_VAR),
    ):
        if value:
            return _require_supported_origin(value, source)
    return DEFAULT_AUTH_BASE_URL


def resolve_api_base_url(
    explicit: str | None = None,
    *,
    environ: dict[str, str] | None = None,
) -> str:
    """Resolves the inference origin (the OpenAI-compatible API lives under ``/v1``).

    Precedence mirrors :func:`resolve_auth_base_url`. This value is never derived
    from the authentication origin by rewriting its hostname or appending ``/v1``.
    """
    env = os.environ if environ is None else environ
    for value, source in (
        (explicit, "the configured API base URL"),
        (env.get(API_BASE_URL_ENV_VAR), API_BASE_URL_ENV_VAR),
        (env.get(ORCAROUTER_BASE_URL_ENV_VAR), ORCAROUTER_BASE_URL_ENV_VAR),
    ):
        if value:
            return _require_supported_origin(value, source)
    return DEFAULT_API_BASE_URL


def api_v1_base_url(api_base_url: str) -> str:
    """Returns the OpenAI-compatible ``/v1`` root for an inference origin."""
    return f"{api_base_url.rstrip('/')}/v1"


def auth_endpoint_url(auth_base_url: str, path: str) -> str:
    """Joins the authentication origin with an auth path.

    Guards against the single most common integration mistake: pointing an auth
    call at the inference origin (``api.orcarouter.ai/v1/auth/keys``).
    """
    if "api.orcarouter.ai" in auth_base_url and path.startswith("/api/v1/auth"):
        raise OrcaRouterCredentialError(
            "Refusing to send an authentication request to the inference origin. "
            "The auth endpoints live on the authentication origin "
            f"({DEFAULT_AUTH_BASE_URL}), not under the inference relay's /v1."
        )
    return f"{auth_base_url.rstrip('/')}{path}"


class CredentialSource(str, Enum):
    """How a credential was obtained. Downstream code must not branch on this."""

    API_KEY = "api_key"
    """The user pasted an existing ``sk-orca-...`` key."""

    OAUTH_PKCE = "oauth_pkce"
    """An OAuth 2.0 + PKCE authorization issued a new key."""


@dataclass
class CredentialResult:
    """A usable OrcaRouter credential.

    This is the single shape produced by both adapters. ``key`` is a plain
    OrcaRouter API key in both cases; ``generation`` identifies the credential
    instance so a late failure can only ever affect the generation it belongs to.
    """

    key: str
    source: CredentialSource
    scope: str = "api"
    account_id: str | None = None
    generation: str = field(default_factory=lambda: uuid.uuid4().hex)
    obtained_at: float = field(default_factory=time.time)
    needs_reauth: bool = False

    def __repr__(self) -> str:
        """Returns a redacted representation; the key must never be interpolated."""
        return (
            f"CredentialResult(key={mask_api_key(self.key)!r}, "
            f"source={self.source.value!r}, scope={self.scope!r}, "
            f"account_id={self.account_id!r}, generation={self.generation!r}, "
            f"needs_reauth={self.needs_reauth!r})"
        )

    __str__ = __repr__

    def to_dict(self) -> dict:
        """Serializes the credential for the project's credential file."""
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> CredentialResult:
        """Rebuilds a credential from a stored payload, discarding unknown keys."""
        key = payload.get("key")
        if not isinstance(key, str) or not key:
            raise OrcaRouterCredentialError(
                "The stored OrcaRouter credential has no usable key; "
                "re-run the OrcaRouter login."
            )
        raw_source = payload.get("source", CredentialSource.API_KEY.value)
        try:
            source = CredentialSource(raw_source)
        except ValueError:
            source = CredentialSource.API_KEY
        generation = payload.get("generation")
        return cls(
            key=key,
            source=source,
            scope=str(payload.get("scope") or "api"),
            account_id=payload.get("account_id"),
            generation=(
                generation
                if isinstance(generation, str) and generation
                else uuid.uuid4().hex
            ),
            obtained_at=float(payload.get("obtained_at") or time.time()),
            needs_reauth=bool(payload.get("needs_reauth", False)),
        )


@runtime_checkable
class CredentialAdapter(Protocol):
    """The seam both authentication entry points implement.

    Implementations acquire one credential and hand back a
    :class:`CredentialResult`. Nothing downstream may depend on which adapter ran.
    """

    @property
    def source(self) -> CredentialSource:
        """The :class:`CredentialSource` this adapter produces."""
        ...

    def acquire(self) -> CredentialResult:
        """Obtains a credential or raises :class:`OrcaRouterCredentialError`."""
        ...


class CredentialStore:
    """Persists one credential in the project's existing ``~/.oumi`` directory.

    OrcaRouter keys are durable, so the credential is written once and reused
    across runs. The file is created with owner-only permissions. Overwriting is
    a single atomic replace, and :meth:`clear` is the only path that removes a
    stored secret -- a failed login never deletes the previous credential.
    """

    def __init__(self, directory: str | Path | None = None) -> None:
        """Initializes the store.

        Args:
            directory: Overrides the ``~/.oumi`` home the project already uses
                for its own state. Mostly useful in tests.
        """
        if directory is None:
            directory = os.environ.get("OUMI_DIR") or _DEFAULT_OUMI_DIR
        self._directory = Path(directory).expanduser()

    @property
    def path(self) -> Path:
        """Returns the path of the credential file."""
        return self._directory / _STORE_FILE_NAME

    def load(self) -> CredentialResult | None:
        """Reads the stored credential, or returns None when there is none."""
        path = self.path
        try:
            raw = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        except OSError as e:
            logger.warning(f"Could not read the stored OrcaRouter credential: {e}")
            return None

        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning(
                "The stored OrcaRouter credential is not valid JSON; "
                "re-run the OrcaRouter login."
            )
            return None

        if not isinstance(payload, dict):
            logger.warning(
                "The stored OrcaRouter credential has an unexpected shape; "
                "re-run the OrcaRouter login."
            )
            return None

        try:
            return CredentialResult.from_dict(payload)
        except OrcaRouterCredentialError as e:
            logger.warning(str(e))
            return None

    def save(self, credential: CredentialResult) -> CredentialResult:
        """Stores ``credential``, replacing any previous value atomically.

        Args:
            credential: The credential to persist.

        Returns:
            CredentialResult: The stored credential, for convenience.
        """
        self._directory.mkdir(parents=True, exist_ok=True)
        try:
            self._directory.chmod(stat.S_IRWXU)
        except OSError:
            logger.debug("Could not restrict permissions on the credential directory.")

        temporary_path = self.path.with_suffix(".tmp")
        temporary_path.write_text(
            json.dumps(credential.to_dict(), indent=2), encoding="utf-8"
        )
        temporary_path.chmod(stat.S_IRUSR | stat.S_IWUSR)
        temporary_path.replace(self.path)
        return credential

    def clear(self) -> bool:
        """Removes the stored credential.

        Returns:
            bool: True when a stored credential was removed.
        """
        try:
            self.path.unlink()
            return True
        except FileNotFoundError:
            return False

    def note_reauthenticated(self, credential: CredentialResult) -> None:
        """Persists a credential that just replaced a rejected one."""
        self.save(replace(credential, needs_reauth=False))


class OrcaRouterApiKeyAdapter:
    """Credential adapter for a user-supplied OrcaRouter API key.

    Accepts a literal key or the name of an environment variable holding one.
    Only a lightweight prefix check is performed: an ``sk-orca-`` prefix is not
    proof that a credential is valid, and OrcaRouter exposes no non-billing
    validation request, so validity is established by the first real request.
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        env_varname: str | None = ORCAROUTER_API_KEY_ENV_VAR,
        store: CredentialStore | None = None,
        environ: dict[str, str] | None = None,
    ) -> None:
        """Initializes the adapter.

        Args:
            api_key: An explicit key. Takes precedence over the environment.
            env_varname: Environment variable to read the key from.
            store: Credential store to persist into. Defaults to the project store.
            environ: Environment mapping to read from. Defaults to ``os.environ``.
        """
        self._api_key = api_key
        self._env_varname = env_varname
        self._store = store if store is not None else CredentialStore()
        self._environ = os.environ if environ is None else environ

    @property
    def source(self) -> CredentialSource:
        """Returns the credential source for this adapter."""
        return CredentialSource.API_KEY

    def resolve_key(self) -> str | None:
        """Returns the configured key without persisting it, or None."""
        if self._api_key:
            return self._api_key
        if self._env_varname:
            return self._environ.get(self._env_varname)
        return None

    def acquire(self) -> CredentialResult:
        """Builds, persists, and returns a credential from the supplied key.

        Raises:
            OrcaRouterCredentialError: If no key is configured, or the value is
                not a plausible OrcaRouter key.
        """
        key = self.resolve_key()
        if not key:
            raise OrcaRouterCredentialError(
                "No OrcaRouter API key was provided. Pass --api-key, or set the "
                f"{ORCAROUTER_API_KEY_ENV_VAR} environment variable, or run the "
                "OrcaRouter browser login instead."
            )
        if not key.startswith(API_KEY_PREFIX):
            raise OrcaRouterCredentialError(
                "The supplied OrcaRouter API key does not start with "
                f"{API_KEY_PREFIX!r}. Copy a key from {KEYS_CONSOLE_URL}."
            )
        credential = CredentialResult(key=key, source=CredentialSource.API_KEY)
        return self._store.save(credential)


class OrcaRouterSession:
    """Coordinates credential acquisition, storage, and terminal reauthentication.

    This is the only object that decides whether a stored credential is still
    usable. It is generation-safe: :meth:`mark_needs_reauth` changes state only
    when the caller passes the exact ``generation`` that made the rejected
    request, so a late failure from a superseded credential can never disable a
    freshly authorized one.
    """

    def __init__(self, store: CredentialStore | None = None) -> None:
        """Initializes the session.

        Args:
            store: Credential store to use. Defaults to the project store.
        """
        self._store = store if store is not None else CredentialStore()

    @property
    def store(self) -> CredentialStore:
        """Returns the underlying credential store."""
        return self._store

    def api_key_adapter(
        self,
        api_key: str | None = None,
        *,
        env_varname: str | None = ORCAROUTER_API_KEY_ENV_VAR,
    ) -> OrcaRouterApiKeyAdapter:
        """Returns the API-key adapter bound to this session's store."""
        return OrcaRouterApiKeyAdapter(
            api_key, env_varname=env_varname, store=self._store
        )

    def current_credential(self) -> CredentialResult | None:
        """Returns the stored credential, or None when the user is not signed in."""
        return self._store.load()

    def active_credential(self) -> CredentialResult | None:
        """Returns the stored credential when it is usable.

        A credential marked ``needs_reauth`` is deliberately unusable: the user
        must complete a new login before requests resume.
        """
        credential = self._store.load()
        if credential is None or credential.needs_reauth:
            return None
        return credential

    def connect_with_api_key(
        self, api_key: str | None = None, *, env_varname: str | None = None
    ) -> CredentialResult:
        """Signs in with an existing API key and persists the credential."""
        adapter = self.api_key_adapter(api_key, env_varname=env_varname)
        return adapter.acquire()

    def connect_with_oauth(
        self,
        *,
        flow: str = "redirect",
        open_browser: bool = True,
        app_name: str = "Oumi",
        timeout_seconds: float = 300.0,
        **kwargs: object,
    ) -> CredentialResult:
        """Signs in with OAuth 2.0 + PKCE and persists the resulting key.

        Args:
            flow: ``"redirect"`` for the loopback flow, ``"oob"`` for the
                out-of-band code flow used on hosts that cannot listen.
            open_browser: Whether to open the consent screen automatically.
            app_name: Label shown on the consent screen.
            timeout_seconds: How long to wait for the user.
            **kwargs: Additional arguments forwarded to the PKCE adapter, such as
                an authentication-origin override.

        Returns:
            CredentialResult: The persisted credential.
        """
        # Imported lazily so a plain API-key user never pays for the login stack.
        from oumi.orcarouter.login import OAuthPkceAdapter

        adapter = OAuthPkceAdapter(
            flow=flow,
            app_name=app_name,
            open_browser=open_browser,
            timeout_seconds=timeout_seconds,
            **kwargs,  # type: ignore[arg-type]
        )
        credential = adapter.acquire()
        return self._store.save(credential)

    def mark_needs_reauth(self, credential: CredentialResult) -> bool:
        """Marks exactly ``credential``'s generation as requiring reauthentication.

        Args:
            credential: The credential that made the rejected request.

        Returns:
            bool: True when the stored credential was updated.
        """
        stored = self._store.load()
        if stored is None:
            return False
        if stored.generation != credential.generation:
            # A superseded credential failed late. Leave the newer one alone.
            logger.debug(
                "Ignoring an authentication failure from a superseded OrcaRouter "
                "credential generation."
            )
            return False
        if stored.needs_reauth:
            return False
        self._store.save(replace(stored, needs_reauth=True))
        logger.warning(
            "OrcaRouter rejected the stored credential; re-run the OrcaRouter "
            f"login. Manage authorized apps at {KEYS_CONSOLE_URL}."
        )
        return True

    def clear(self) -> bool:
        """Removes the stored credential."""
        return self._store.clear()
