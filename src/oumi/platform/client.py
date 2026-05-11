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

"""HTTP client for the Oumi Enterprise platform.

The :class:`Client` wraps :mod:`httpx` and exposes project-scoped resource
clients for the subset of the platform API that local oumi code needs:
datasets, models, judges/evaluators, recipes, deployments, and operations.

Construction:

* ``Client()`` — load credentials from env/credentials file.
* ``Client(api_url=..., api_key=..., project_id=...)`` — explicit args win over
  env/file resolution.
* :func:`get_default_client` — process-wide singleton for callers that don't
  need their own instance (CLI commands, MCP server tools).

All resource methods raise :class:`PlatformAuthError` on 401/403,
:class:`PlatformAPIError` on other non-2xx responses, and propagate
:class:`httpx.RequestError` on transport failures.
"""

import logging
import time
from pathlib import Path
from typing import Any

import httpx

from oumi.platform.credentials import Credentials, load_credentials
from oumi.platform.exceptions import (
    PlatformAPIError,
    PlatformAuthError,
    PlatformError,
    PlatformOperationError,
)

_DEFAULT_TIMEOUT_SECONDS = 30.0
_OPERATION_POLL_INITIAL_SECONDS = 1.0
_OPERATION_POLL_MAX_SECONDS = 10.0
_OPERATION_TERMINAL_STATUSES = frozenset(
    {"completed", "failed", "cancelled"}
)
_OPERATION_FAILURE_STATUSES = frozenset({"failed", "cancelled"})
_USER_AGENT = "oumi-platform-client/0"

logger = logging.getLogger(__name__)


class Client:
    """High-level client for the Oumi Enterprise platform.

    Resource collections are exposed as attributes:
    :attr:`datasets`, :attr:`models`, :attr:`evaluators`, :attr:`recipes`,
    :attr:`deployments`, :attr:`operations`.

    The client is safe to share across threads; the underlying
    :class:`httpx.Client` does its own connection pooling.

    Example:
        >>> client = Client()
        >>> ds = client.datasets.get("123")
        >>> client.operations.wait(op_id)
    """

    def __init__(
        self,
        *,
        api_url: str | None = None,
        api_key: str | None = None,
        project_id: str | None = None,
        credentials: Credentials | None = None,
        timeout: float = _DEFAULT_TIMEOUT_SECONDS,
        http_client: httpx.Client | None = None,
    ):
        """Initialize a platform client.

        Args:
            api_url: Optional explicit base URL. If omitted, resolved from
                ``OUMI_API_URL`` or the credentials file.
            api_key: Optional explicit API key. If omitted, resolved from
                ``OUMI_API_KEY`` or the credentials file.
            project_id: Optional default project id for project-scoped calls.
            credentials: Pre-resolved :class:`Credentials`. If provided, the
                ``api_url`` / ``api_key`` / ``project_id`` args must not be
                set; mixing them is a programming error.
            timeout: Per-request timeout in seconds. Defaults to 30s.
            http_client: Optional pre-configured :class:`httpx.Client`. Used
                primarily by tests; production callers should pass credentials
                instead.
        """
        if credentials is not None:
            if any(v is not None for v in (api_url, api_key, project_id)):
                raise ValueError(
                    "Pass either `credentials` or individual args, not both."
                )
            self._credentials = credentials
        else:
            self._credentials = load_credentials(
                api_url=api_url,
                api_key=api_key,
                project_id=project_id,
            )

        if http_client is None:
            http_client = httpx.Client(
                base_url=self._credentials.api_url,
                timeout=timeout,
                headers=self._default_headers(),
            )
            self._owns_http_client = True
        else:
            self._owns_http_client = False
        self._http = http_client

        self.datasets = _DatasetsClient(self)
        self.models = _ModelsClient(self)
        self.evaluators = _EvaluatorsClient(self)
        self.recipes = _RecipesClient(self)
        self.deployments = _DeploymentsClient(self)
        self.operations = _OperationsClient(self)

    @property
    def credentials(self) -> Credentials:
        """The resolved credentials this client is using."""
        return self._credentials

    @property
    def default_project_id(self) -> str | None:
        """The default project id used when a call omits one."""
        return self._credentials.project_id

    def close(self) -> None:
        """Close the underlying HTTP client if owned by this instance."""
        if self._owns_http_client:
            self._http.close()

    def __enter__(self) -> "Client":
        """Enter context manager; the underlying HTTP client is reused."""
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        """Exit context manager; closes the HTTP client if owned."""
        self.close()

    def request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: Any = None,
        expect_json: bool = True,
    ) -> Any:
        """Issue a request and return the decoded body.

        Args:
            method: HTTP method (``"GET"``, ``"POST"``, etc.).
            path: Path relative to the API base URL, starting with ``/``.
            params: Query string params. ``None``-valued entries are dropped.
            json_body: Request body to JSON-encode.
            expect_json: When ``True`` (default), parse the response body as
                JSON and return it. When ``False``, return the raw
                :class:`httpx.Response`.
        """
        clean_params = (
            {k: v for k, v in params.items() if v is not None}
            if params
            else None
        )
        try:
            response = self._http.request(
                method,
                path,
                params=clean_params,
                json=json_body,
            )
        except httpx.RequestError as exc:
            raise PlatformError(
                f"Network error talking to {self._credentials.api_url}: {exc}"
            ) from exc

        if response.status_code in (401, 403):
            raise PlatformAuthError(
                "Oumi Enterprise rejected the request — check your API key. "
                f"({response.status_code} on {method} {path})"
            )
        if response.status_code >= 400:
            body: Any = None
            try:
                body = response.json()
            except ValueError:
                body = response.text
            raise PlatformAPIError(
                f"{method} {path} returned {response.status_code}",
                status_code=response.status_code,
                response_body=body,
            )

        if not expect_json:
            return response
        if not response.content:
            return None
        return response.json()

    def _stream_to_file(self, url: str, destination: Path) -> None:
        """Stream the contents of ``url`` to ``destination``.

        Reuses this client's underlying connection pool; the URL may be on a
        different host (e.g. presigned cloud storage URLs).
        """
        try:
            with self._http.stream(
                "GET",
                url,
                timeout=None,
                headers={"Authorization": "", "X-API-Key": ""},
            ) as response:
                response.raise_for_status()
                with destination.open("wb") as fh:
                    for chunk in response.iter_bytes():
                        fh.write(chunk)
        except httpx.HTTPError as exc:
            raise PlatformError(
                f"Failed to download {url} to {destination}: {exc}"
            ) from exc

    def _default_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._credentials.api_key}",
            "X-API-Key": self._credentials.api_key,
            "User-Agent": _USER_AGENT,
            "Accept": "application/json",
        }

    def _resolve_project_id(self, project_id: str | None) -> str:
        resolved = project_id or self._credentials.project_id
        if not resolved:
            raise PlatformError(
                "No project id given and no default project configured. Pass "
                "`project_id=...` or set OUMI_PROJECT_ID."
            )
        return resolved


class _ResourceClient:
    """Shared base for resource-collection clients."""

    def __init__(self, client: Client):
        self._client = client

    def _project_path(self, project_id: str | None, suffix: str) -> str:
        pid = self._client._resolve_project_id(project_id)
        return f"/v1/projects/{pid}{suffix}"


class _DatasetsClient(_ResourceClient):
    """Datasets endpoints under ``/v1/projects/{}/datasets``."""

    def list(
        self,
        *,
        project_id: str | None = None,
        page_size: int | None = None,
        page_token: str | None = None,
    ) -> dict[str, Any]:
        """List datasets for a project."""
        return self._client.request(
            "GET",
            self._project_path(project_id, "/datasets"),
            params={"pageSize": page_size, "pageToken": page_token},
        )

    def get(
        self,
        dataset_id: str,
        *,
        project_id: str | None = None,
    ) -> dict[str, Any]:
        """Fetch a single dataset by id."""
        return self._client.request(
            "GET",
            self._project_path(project_id, f"/datasets/{dataset_id}"),
        )

    def list_versions(
        self,
        dataset_id: str,
        *,
        project_id: str | None = None,
    ) -> dict[str, Any]:
        """List the versions of a dataset."""
        return self._client.request(
            "GET",
            self._project_path(
                project_id, f"/datasets/{dataset_id}/versions"
            ),
        )

    def download(
        self,
        dataset_id: str,
        destination: str | Path,
        *,
        project_id: str | None = None,
    ) -> Path:
        """Download a dataset to ``destination``.

        Calls the platform's ``:download`` endpoint, which returns a presigned
        URL, then streams the content to disk.
        """
        info = self._client.request(
            "GET",
            self._project_path(project_id, f"/datasets/{dataset_id}:download"),
        )
        url = info.get("url") if isinstance(info, dict) else None
        if not url:
            raise PlatformError(
                f"Dataset download response did not include a URL: {info!r}"
            )
        dest = Path(destination)
        dest.parent.mkdir(parents=True, exist_ok=True)
        self._client._stream_to_file(url, dest)
        return dest


class _ModelsClient(_ResourceClient):
    """Models endpoints under ``/v1/projects/{}/models``."""

    def list(
        self,
        *,
        project_id: str | None = None,
        page_size: int | None = None,
        page_token: str | None = None,
    ) -> dict[str, Any]:
        return self._client.request(
            "GET",
            self._project_path(project_id, "/models"),
            params={"pageSize": page_size, "pageToken": page_token},
        )

    def get(
        self,
        model_id: str,
        *,
        project_id: str | None = None,
    ) -> dict[str, Any]:
        return self._client.request(
            "GET",
            self._project_path(project_id, f"/models/{model_id}"),
        )

    def list_versions(
        self,
        model_id: str,
        *,
        project_id: str | None = None,
    ) -> dict[str, Any]:
        return self._client.request(
            "GET",
            self._project_path(project_id, f"/models/{model_id}/versions"),
        )

    def download(
        self,
        model_id: str,
        destination: str | Path,
        *,
        version_id: str | None = None,
        project_id: str | None = None,
    ) -> Path:
        """Download a model checkpoint to ``destination`` (a directory).

        When ``version_id`` is omitted, the latest version is downloaded.
        """
        if version_id is not None:
            path = self._project_path(
                project_id,
                f"/models/{model_id}/versions/{version_id}:download",
            )
        else:
            path = self._project_path(
                project_id, f"/models/{model_id}:download"
            )
        info = self._client.request("GET", path)
        urls = _extract_download_urls(info)
        if not urls:
            raise PlatformError(
                f"Model download response did not include any URLs: {info!r}"
            )
        dest_dir = Path(destination)
        dest_dir.mkdir(parents=True, exist_ok=True)
        for name, url in urls:
            target = dest_dir / name
            target.parent.mkdir(parents=True, exist_ok=True)
            self._client._stream_to_file(url, target)
        return dest_dir


class _EvaluatorsClient(_ResourceClient):
    """Evaluators (and judges) endpoints under ``/v1/projects/{}/evaluators``.

    Judges are evaluators with ``evaluatorType == "judge"``. There is no
    separate ``/judges`` route at the platform; clients filter by type.
    """

    def list(
        self,
        *,
        evaluator_type: str | None = None,
        project_id: str | None = None,
        page_size: int | None = None,
        page_token: str | None = None,
    ) -> dict[str, Any]:
        return self._client.request(
            "GET",
            self._project_path(project_id, "/evaluators"),
            params={
                "evaluatorType": evaluator_type,
                "pageSize": page_size,
                "pageToken": page_token,
            },
        )

    def get(
        self,
        evaluator_id: str,
        *,
        project_id: str | None = None,
    ) -> dict[str, Any]:
        return self._client.request(
            "GET",
            self._project_path(project_id, f"/evaluators/{evaluator_id}"),
        )

    def supported_models(
        self,
        *,
        project_id: str | None = None,
    ) -> dict[str, Any]:
        return self._client.request(
            "GET",
            self._project_path(project_id, "/evaluators:supported_models"),
        )


class _RecipesClient(_ResourceClient):
    """Recipes endpoints under ``/v1/projects/{}/recipes``."""

    def list(
        self,
        *,
        project_id: str | None = None,
        page_size: int | None = None,
        page_token: str | None = None,
    ) -> dict[str, Any]:
        return self._client.request(
            "GET",
            self._project_path(project_id, "/recipes"),
            params={"pageSize": page_size, "pageToken": page_token},
        )

    def get(
        self,
        recipe_id: str,
        *,
        project_id: str | None = None,
    ) -> dict[str, Any]:
        return self._client.request(
            "GET",
            self._project_path(project_id, f"/recipes/{recipe_id}"),
        )

    def list_versions(
        self,
        recipe_id: str,
        *,
        project_id: str | None = None,
    ) -> dict[str, Any]:
        return self._client.request(
            "GET",
            self._project_path(project_id, f"/recipes/{recipe_id}/versions"),
        )


class _DeploymentsClient(_ResourceClient):
    """Deployments endpoints under ``/v1/projects/{}/deployments``."""

    def list(
        self,
        *,
        project_id: str | None = None,
        page_size: int | None = None,
        page_token: str | None = None,
    ) -> dict[str, Any]:
        return self._client.request(
            "GET",
            self._project_path(project_id, "/deployments"),
            params={"pageSize": page_size, "pageToken": page_token},
        )

    def get(
        self,
        deployment_id: str,
        *,
        project_id: str | None = None,
    ) -> dict[str, Any]:
        return self._client.request(
            "GET",
            self._project_path(project_id, f"/deployments/{deployment_id}"),
        )


class _OperationsClient(_ResourceClient):
    """Operations endpoints under ``/v1/projects/{}/operations``."""

    def get(
        self,
        operation_id: str | int,
        *,
        project_id: str | None = None,
    ) -> dict[str, Any]:
        return self._client.request(
            "GET",
            self._project_path(project_id, f"/operations/{operation_id}"),
        )

    def stop(
        self,
        operation_id: str | int,
        *,
        project_id: str | None = None,
    ) -> dict[str, Any]:
        return self._client.request(
            "POST",
            self._project_path(
                project_id, f"/operations/{operation_id}:stop"
            ),
        )

    def wait(
        self,
        operation_id: str | int,
        *,
        project_id: str | None = None,
        timeout: float | None = None,
        poll_interval: float = _OPERATION_POLL_INITIAL_SECONDS,
        max_poll_interval: float = _OPERATION_POLL_MAX_SECONDS,
        raise_on_failure: bool = True,
        sleep_fn=time.sleep,
        time_fn=time.monotonic,
    ) -> dict[str, Any]:
        """Poll an operation until it reaches a terminal status.

        Backoff: ``poll_interval`` doubles after each non-terminal poll, capped
        at ``max_poll_interval``.

        Args:
            operation_id: The id returned by the platform when the job was
                submitted.
            project_id: Override the default project id.
            timeout: Maximum seconds to wait. ``None`` waits indefinitely.
            poll_interval: Initial sleep between polls (seconds).
            max_poll_interval: Cap on the sleep between polls (seconds).
            raise_on_failure: When ``True`` (default), raise
                :class:`PlatformOperationError` if the operation terminates in
                ``failed`` or ``cancelled``. When ``False``, return the final
                operation payload regardless.
            sleep_fn: Sleep function. Override for tests.
            time_fn: Monotonic time function. Override for tests.

        Returns:
            The final operation payload.

        Raises:
            PlatformOperationError: When the operation failed or was cancelled
                and ``raise_on_failure`` is ``True``.
            TimeoutError: When ``timeout`` is exceeded before a terminal status.
        """
        deadline = None if timeout is None else time_fn() + timeout
        interval = poll_interval
        while True:
            op = self.get(operation_id, project_id=project_id)
            status = str(op.get("status", "")).lower()
            done = bool(op.get("done")) or status in _OPERATION_TERMINAL_STATUSES
            if done:
                if raise_on_failure and status in _OPERATION_FAILURE_STATUSES:
                    raise PlatformOperationError(
                        f"Operation {operation_id} ended with status {status!r}: "
                        f"{_summarize_error(op)}",
                        operation_id=operation_id,
                        status=status,
                        operation=op,
                    )
                return op
            if deadline is not None and time_fn() >= deadline:
                raise TimeoutError(
                    f"Operation {operation_id} did not finish within {timeout}s "
                    f"(last status: {status!r})."
                )
            sleep_fn(interval)
            interval = min(interval * 2, max_poll_interval)


_default_client: Client | None = None


def get_default_client() -> Client:
    """Return a process-wide singleton :class:`Client`.

    The first call resolves credentials from env vars and the credentials
    file. Subsequent calls return the same instance. Use this from CLI
    commands and MCP tools where you don't need a per-call client.
    """
    global _default_client
    if _default_client is None:
        _default_client = Client()
    return _default_client


def _extract_download_urls(info: Any) -> list[tuple[str, str]]:
    """Normalize a download-response payload into ``(filename, url)`` pairs.

    Supports both single-file shapes (``{"url": "..."}``) and multi-file
    shapes (``{"files": [{"name": "...", "url": "..."}]}``).
    """
    if not isinstance(info, dict):
        return []
    files = info.get("files")
    if isinstance(files, list):
        out: list[tuple[str, str]] = []
        for entry in files:
            if not isinstance(entry, dict):
                continue
            url = entry.get("url")
            name = entry.get("name") or entry.get("filename")
            if isinstance(url, str) and isinstance(name, str):
                out.append((name, url))
        return out
    url = info.get("url")
    if isinstance(url, str):
        name = info.get("name") or info.get("filename") or "download"
        return [(name, url)]
    return []


def _summarize_error(op: dict[str, Any]) -> str:
    """Return a short, log-friendly description of an operation's error field."""
    error = op.get("error")
    if isinstance(error, dict):
        msg = error.get("message") or error.get("detail")
        if msg:
            return str(msg)
        return str(error)
    if error:
        return str(error)
    return "no error message provided"
