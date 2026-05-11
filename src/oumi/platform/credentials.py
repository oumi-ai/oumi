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

"""Credential resolution for the Oumi Enterprise platform client.

Resolution order, highest precedence first:

1. Explicit arguments passed to :class:`oumi.platform.Client`.
2. Environment variables: ``OUMI_API_URL``, ``OUMI_API_KEY``,
   ``OUMI_PROJECT_ID``.
3. The credentials file at ``$OUMI_CREDENTIALS_FILE`` if set, otherwise
   ``$XDG_CONFIG_HOME/oumi/credentials.json`` if set, otherwise
   ``~/.config/oumi/credentials.json``.

The credentials file is JSON with the following shape::

    {
      "api_url": "https://api.oumi.ai",
      "api_key": "oumi_...",
      "project_id": "..."
    }
"""

import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path

from oumi.platform.exceptions import PlatformError

_USER_RW = stat.S_IRUSR | stat.S_IWUSR

_DEFAULT_API_URL = "https://api.oumi.ai"

_ENV_API_URL = "OUMI_API_URL"
_ENV_API_KEY = "OUMI_API_KEY"
_ENV_PROJECT_ID = "OUMI_PROJECT_ID"
_ENV_CREDENTIALS_FILE = "OUMI_CREDENTIALS_FILE"


class CredentialsNotFoundError(PlatformError):
    """No API key could be resolved from env vars or the credentials file."""


@dataclass(frozen=True)
class Credentials:
    """Resolved credentials for the Oumi Enterprise platform.

    Attributes:
        api_url: Base URL of the platform API, e.g. ``https://api.oumi.ai``.
            No trailing slash.
        api_key: The API key to send as ``X-API-Key`` and ``Authorization:
            Bearer``.
        project_id: Default project id, used when a per-call ``project_id`` is
            not provided. May be ``None``.
    """

    api_url: str
    api_key: str
    project_id: str | None = None


def default_credentials_path() -> Path:
    """Return the default credentials file path.

    Honors ``OUMI_CREDENTIALS_FILE`` and ``XDG_CONFIG_HOME`` if set; otherwise
    falls back to ``~/.config/oumi/credentials.json``.
    """
    explicit = os.environ.get(_ENV_CREDENTIALS_FILE)
    if explicit:
        return Path(explicit).expanduser()
    xdg = os.environ.get("XDG_CONFIG_HOME")
    root = Path(xdg).expanduser() if xdg else Path.home() / ".config"
    return root / "oumi" / "credentials.json"


def load_credentials(
    *,
    api_url: str | None = None,
    api_key: str | None = None,
    project_id: str | None = None,
    credentials_path: Path | None = None,
) -> Credentials:
    """Resolve credentials from explicit args, env vars, then the credentials file.

    Args:
        api_url: Optional explicit API URL; takes precedence over env and file.
        api_key: Optional explicit API key; takes precedence over env and file.
        project_id: Optional explicit project id; takes precedence over env and
            file. ``None`` here means "fall back", not "no project".
        credentials_path: Override the path of the credentials file. Mostly
            useful for tests.

    Returns:
        A populated :class:`Credentials` instance.

    Raises:
        CredentialsNotFoundError: If no API key could be resolved.
    """
    path = credentials_path or default_credentials_path()
    file_data = _read_credentials_file(path)

    resolved_url = (
        api_url
        or os.environ.get(_ENV_API_URL)
        or file_data.get("api_url")
        or _DEFAULT_API_URL
    )
    resolved_key = (
        api_key or os.environ.get(_ENV_API_KEY) or file_data.get("api_key")
    )
    resolved_project = (
        project_id
        or os.environ.get(_ENV_PROJECT_ID)
        or file_data.get("project_id")
    )

    if not resolved_key:
        raise CredentialsNotFoundError(
            "No Oumi Enterprise API key found. Set the OUMI_API_KEY environment "
            "variable or run `oumi platform login` to create a credentials file "
            f"at {path}."
        )

    return Credentials(
        api_url=resolved_url.rstrip("/"),
        api_key=resolved_key,
        project_id=resolved_project,
    )


def save_credentials(
    credentials: Credentials,
    *,
    credentials_path: Path | None = None,
) -> Path:
    """Persist credentials to disk with user-only (0600) permissions.

    The parent directory is created if it does not exist.

    Args:
        credentials: The credentials to write.
        credentials_path: Override the destination path. Mostly useful for tests.

    Returns:
        The path the credentials were written to.
    """
    path = credentials_path or default_credentials_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "api_url": credentials.api_url,
        "api_key": credentials.api_key,
    }
    if credentials.project_id is not None:
        payload["project_id"] = credentials.project_id

    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.chmod(_USER_RW)
    tmp.replace(path)
    return path


def _read_credentials_file(path: Path) -> dict[str, str]:
    """Read the credentials file at ``path``, returning ``{}`` if absent.

    Returns ``{}`` for any of: missing file, empty file, non-object JSON.
    A malformed-but-present file raises :class:`PlatformError` so a typo
    surfaces instead of being silently ignored.
    """
    if not path.exists():
        return {}
    try:
        raw = path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise PlatformError(
            f"Could not read Oumi credentials file at {path}: {exc}"
        ) from exc
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise PlatformError(
            f"Oumi credentials file at {path} is not valid JSON: {exc}"
        ) from exc
    if not isinstance(data, dict):
        raise PlatformError(
            f"Oumi credentials file at {path} must be a JSON object."
        )
    return {k: v for k, v in data.items() if isinstance(v, str)}
