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

"""Shared fixtures for the OrcaRouter tests.

``FakeAuthServer`` is a loopback stand-in for the OrcaRouter authentication
origin. It implements the documented contract: ``POST /api/v1/auth/keys``
verifies ``base64url(sha256(code_verifier))`` against the challenge that was
minted for the code, enforces single use and the S256 method, and returns an
ordinary ``sk-orca-...`` key. Every code and key here is a test fixture, and no
traffic leaves loopback.
"""

import base64
import hashlib
import json
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from oumi.orcarouter.credentials import CredentialStore

EXCHANGE_PATH = "/api/v1/auth/keys"
AUTHORIZE_PATH = "/auth"


@pytest.fixture
def store(tmp_path: Path) -> CredentialStore:
    """A credential store rooted in a temporary directory."""
    return CredentialStore(directory=tmp_path)


def _b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _sha256_b64url(value: str) -> str:
    return _b64url(hashlib.sha256(value.encode("ascii")).digest())


class FakeAuthServer:
    """A loopback stand-in for the OrcaRouter authentication origin."""

    def __init__(self) -> None:
        self._codes: dict[str, dict] = {}
        self._lock = threading.Lock()
        self.exchange_requests: list[dict] = []
        self.authorize_requests: list[str] = []
        self.seen_paths: list[str] = []
        self.forced_status: int | None = None
        self.returned_scope = "api"
        self.returned_key = "sk-orca-66666666666666666666"
        self.returned_user_id = "test-user-1"

    def __enter__(self) -> "FakeAuthServer":
        server = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def _respond(self, status: int, payload: dict) -> None:
                encoded = json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

            def do_GET(self) -> None:  # noqa: N802 - stdlib naming
                parsed = urllib.parse.urlparse(self.path)
                server.seen_paths.append(parsed.path)
                if parsed.path == AUTHORIZE_PATH:
                    server.authorize_requests.append(self.path)
                    body = b"<html>consent</html>"
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                self._respond(404, {"error": "not_found"})

            def do_POST(self) -> None:  # noqa: N802 - stdlib naming
                parsed = urllib.parse.urlparse(self.path)
                server.seen_paths.append(parsed.path)
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length)
                try:
                    body = json.loads(raw.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    body = {}

                if parsed.path != EXCHANGE_PATH:
                    self._respond(404, {"error": "not_found"})
                    return

                server.exchange_requests.append(
                    {"path": parsed.path, "body": body, "raw": raw.decode("utf-8")}
                )

                if server.forced_status is not None:
                    self._respond(server.forced_status, {"error": "forced"})
                    return

                code = body.get("code")
                verifier = body.get("code_verifier")
                method = body.get("code_challenge_method")

                with server._lock:
                    record = server._codes.get(code) if isinstance(code, str) else None
                    if record is None or record["used"]:
                        self._respond(
                            403,
                            {
                                "error": "invalid_grant",
                                "error_description": "code unknown, used, or expired",
                            },
                        )
                        return
                    if method != "S256":
                        self._respond(
                            400,
                            {
                                "error": "invalid_request",
                                "error_description": "unsupported challenge method",
                            },
                        )
                        return
                    if (
                        not isinstance(verifier, str)
                        or _sha256_b64url(verifier) != record["challenge"]
                    ):
                        self._respond(
                            403,
                            {
                                "error": "invalid_grant",
                                "error_description": "verifier does not match",
                            },
                        )
                        return
                    record["used"] = True

                self._respond(
                    200,
                    {
                        "key": server.returned_key,
                        "user_id": server.returned_user_id,
                        "scope": server.returned_scope,
                    },
                )

            def log_message(self, format: str, *args) -> None:
                return

        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc_info) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()

    @property
    def auth_base_url(self) -> str:
        """The loopback origin this fake server listens on."""
        return f"http://127.0.0.1:{self._httpd.server_address[1]}"

    def mint_code(self, challenge: str) -> str:
        """Mints a single-use authorization code bound to ``challenge``."""
        code = f"test-code-{len(self._codes) + 1}-{challenge[:8]}"
        with self._lock:
            self._codes[code] = {"challenge": challenge, "used": False}
        return code

    def code_challenge_from_authorize_url(self, url: str) -> str:
        """Extracts the S256 challenge the client sent on the authorize URL."""
        query = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
        assert query["code_challenge_method"] == ["S256"]
        return query["code_challenge"][0]

    def state_from_authorize_url(self, url: str) -> str:
        """Extracts the state the client sent on the authorize URL."""
        return urllib.parse.parse_qs(urllib.parse.urlparse(url).query)["state"][0]

    def callback_url_from_authorize_url(self, url: str) -> str:
        """Extracts the callback_url the client asked for."""
        return urllib.parse.parse_qs(urllib.parse.urlparse(url).query)["callback_url"][
            0
        ]


@pytest.fixture
def fake_auth():
    """A loopback stand-in for the OrcaRouter authentication origin."""
    with FakeAuthServer() as server:
        yield server
