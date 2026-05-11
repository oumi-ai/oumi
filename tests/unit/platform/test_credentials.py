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

import json
import os
import stat
from pathlib import Path

import pytest

from oumi.platform import (
    Credentials,
    CredentialsNotFoundError,
    load_credentials,
    save_credentials,
)
from oumi.platform.credentials import default_credentials_path
from oumi.platform.exceptions import PlatformError

_ENV_VARS = (
    "OUMI_API_URL",
    "OUMI_API_KEY",
    "OUMI_PROJECT_ID",
    "OUMI_CREDENTIALS_FILE",
    "XDG_CONFIG_HOME",
)


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    for v in _ENV_VARS:
        monkeypatch.delenv(v, raising=False)


def test_load_credentials_from_env(monkeypatch, tmp_path):
    monkeypatch.setenv("OUMI_API_URL", "https://api.example.com/")
    monkeypatch.setenv("OUMI_API_KEY", "key-abc")
    monkeypatch.setenv("OUMI_PROJECT_ID", "proj-1")

    creds = load_credentials(credentials_path=tmp_path / "missing.json")

    assert creds.api_url == "https://api.example.com"  # trailing slash stripped
    assert creds.api_key == "key-abc"
    assert creds.project_id == "proj-1"


def test_load_credentials_from_file(tmp_path):
    path = tmp_path / "credentials.json"
    path.write_text(
        json.dumps(
            {
                "api_url": "https://api.platform.example",
                "api_key": "file-key",
                "project_id": "file-proj",
            }
        )
    )

    creds = load_credentials(credentials_path=path)

    assert creds.api_url == "https://api.platform.example"
    assert creds.api_key == "file-key"
    assert creds.project_id == "file-proj"


def test_explicit_args_beat_env_and_file(monkeypatch, tmp_path):
    path = tmp_path / "credentials.json"
    path.write_text(json.dumps({"api_url": "https://file", "api_key": "file"}))
    monkeypatch.setenv("OUMI_API_KEY", "env-key")

    creds = load_credentials(
        api_url="https://explicit.example",
        api_key="explicit",
        project_id="explicit-proj",
        credentials_path=path,
    )

    assert creds.api_url == "https://explicit.example"
    assert creds.api_key == "explicit"
    assert creds.project_id == "explicit-proj"


def test_env_beats_file(monkeypatch, tmp_path):
    path = tmp_path / "credentials.json"
    path.write_text(
        json.dumps({"api_url": "https://file", "api_key": "file-key"})
    )
    monkeypatch.setenv("OUMI_API_KEY", "env-key")

    creds = load_credentials(credentials_path=path)

    assert creds.api_key == "env-key"
    assert creds.api_url == "https://file"


def test_default_api_url_when_unset(monkeypatch, tmp_path):
    monkeypatch.setenv("OUMI_API_KEY", "k")

    creds = load_credentials(credentials_path=tmp_path / "missing.json")

    assert creds.api_url == "https://api.oumi.ai"


def test_missing_api_key_raises(tmp_path):
    with pytest.raises(CredentialsNotFoundError):
        load_credentials(credentials_path=tmp_path / "missing.json")


def test_malformed_credentials_file_raises(tmp_path):
    path = tmp_path / "credentials.json"
    path.write_text("not-json{")
    with pytest.raises(PlatformError):
        load_credentials(credentials_path=path)


def test_non_object_credentials_file_raises(tmp_path):
    path = tmp_path / "credentials.json"
    path.write_text('"just a string"')
    with pytest.raises(PlatformError):
        load_credentials(credentials_path=path)


def test_empty_credentials_file_is_ignored(monkeypatch, tmp_path):
    path = tmp_path / "credentials.json"
    path.write_text("")
    monkeypatch.setenv("OUMI_API_KEY", "k")

    creds = load_credentials(credentials_path=path)

    assert creds.api_key == "k"


def test_save_credentials_writes_user_only_perms(tmp_path):
    path = tmp_path / "nested" / "credentials.json"
    creds = Credentials(
        api_url="https://api.example",
        api_key="secret",
        project_id="p",
    )

    written = save_credentials(creds, credentials_path=path)

    assert written == path
    data = json.loads(path.read_text())
    assert data == {
        "api_url": "https://api.example",
        "api_key": "secret",
        "project_id": "p",
    }
    mode = stat.S_IMODE(path.stat().st_mode)
    assert mode & 0o077 == 0  # group/other have no permissions


def test_save_credentials_omits_null_project(tmp_path):
    path = tmp_path / "credentials.json"
    save_credentials(
        Credentials(api_url="https://api.example", api_key="k"),
        credentials_path=path,
    )
    data = json.loads(path.read_text())
    assert "project_id" not in data


def test_round_trip_save_then_load(tmp_path):
    path = tmp_path / "credentials.json"
    save_credentials(
        Credentials(
            api_url="https://api.example/",
            api_key="abc",
            project_id="p1",
        ),
        credentials_path=path,
    )

    creds = load_credentials(credentials_path=path)

    assert creds == Credentials(
        api_url="https://api.example",
        api_key="abc",
        project_id="p1",
    )


def test_default_credentials_path_respects_xdg(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    assert default_credentials_path() == tmp_path / "oumi" / "credentials.json"


def test_default_credentials_path_respects_explicit_env(monkeypatch, tmp_path):
    explicit = tmp_path / "elsewhere.json"
    monkeypatch.setenv("OUMI_CREDENTIALS_FILE", str(explicit))
    assert default_credentials_path() == explicit


def test_default_credentials_path_falls_back_to_home(monkeypatch):
    # No XDG, no explicit env var: falls back to ~/.config/oumi/credentials.json
    expected = Path.home() / ".config" / "oumi" / "credentials.json"
    assert default_credentials_path() == expected


def test_read_only_unreadable_file_raises(tmp_path):
    path = tmp_path / "credentials.json"
    path.write_text("{}")
    os.chmod(path, 0)
    try:
        # Re-read denied permission; should surface as PlatformError, not silent {}.
        # On some platforms (notably when running as root), chmod 0 does not
        # block reads — skip in that case rather than asserting nonsense.
        try:
            path.read_text()
        except OSError:
            with pytest.raises(PlatformError):
                load_credentials(credentials_path=path)
        else:
            pytest.skip("File still readable despite chmod 0 (running as root?)")
    finally:
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
