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

"""Resolution of ``oumi://`` URIs to local paths and structured payloads.

A platform user can write ``dataset_name: oumi://datasets/abc@v3`` in a config
file and have the URI transparently resolve to a local path on first use.

URI grammar::

    oumi://<kind>/<resource_id>[@<version>]

    kind        := "datasets" | "models" | "judges" | "evaluators" | "recipes"
    resource_id := arbitrary string (the platform's id for the resource)
    version     := the platform's version id or version name

Resolved outputs:

* ``datasets`` → :class:`pathlib.Path` to a downloaded file in the cache.
* ``models``   → :class:`pathlib.Path` to a directory containing the model
  artifacts in the cache.
* ``judges`` and ``evaluators`` → the raw evaluator payload (a ``dict``) as
  returned by the platform. Caller is responsible for mapping that to
  :class:`oumi.core.configs.judge_config.JudgeConfig`; that mapping lives in
  the judge builder, not here.
* ``recipes`` → the raw recipe payload (a ``dict``).

Caching: resolved artifacts are stored under ``$OUMI_CACHE_DIR/platform`` or
``~/.cache/oumi/platform`` keyed by ``(kind, project_id, resource_id,
version)``. Set ``force_refresh=True`` to bypass the cache. Cache entries are
treated as immutable when a concrete version is pinned; ``@latest`` (the
default) is re-resolved each call so a moving target stays fresh.
"""

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from oumi.platform.client import Client, get_default_client
from oumi.platform.exceptions import PlatformError

_URI_SCHEME = "oumi"
_SUPPORTED_KINDS = frozenset(
    {"datasets", "models", "judges", "evaluators", "recipes"}
)
_LATEST = "latest"
_URI_PATTERN = re.compile(
    r"^oumi://(?P<kind>[a-z]+)/(?P<id>[^/@\s]+)(?:@(?P<version>[^/\s]+))?$"
)


@dataclass(frozen=True)
class ParsedURI:
    """A parsed ``oumi://`` URI.

    Attributes:
        kind: The resource kind: ``datasets``, ``models``, ``judges``,
            ``evaluators``, or ``recipes``.
        resource_id: The platform's id for the resource.
        version: The version id or name, or ``None`` if unspecified.
    """

    kind: str
    resource_id: str
    version: str | None

    @property
    def version_or_latest(self) -> str:
        """Return the version, or the literal ``"latest"`` if unspecified."""
        return self.version or _LATEST


def is_oumi_uri(value: object) -> bool:
    """Return ``True`` if ``value`` looks like an ``oumi://`` URI string."""
    return isinstance(value, str) and value.startswith(f"{_URI_SCHEME}://")


def parse_uri(uri: str) -> ParsedURI:
    """Parse an ``oumi://`` URI and validate its grammar.

    Args:
        uri: The URI string.

    Returns:
        The parsed components.

    Raises:
        ValueError: If the URI is not a well-formed ``oumi://`` reference
            (wrong scheme, unsupported kind, empty id, etc.).
    """
    if not isinstance(uri, str):
        raise ValueError(f"oumi URI must be a string, got {type(uri).__name__}")
    match = _URI_PATTERN.match(uri)
    if not match:
        raise ValueError(
            f"Not a well-formed oumi:// URI: {uri!r}. Expected "
            "oumi://<kind>/<id>[@<version>]."
        )
    kind = match.group("kind")
    if kind not in _SUPPORTED_KINDS:
        raise ValueError(
            f"Unsupported oumi:// kind {kind!r} in {uri!r}. "
            f"Supported kinds: {sorted(_SUPPORTED_KINDS)}."
        )
    return ParsedURI(
        kind=kind,
        resource_id=match.group("id"),
        version=match.group("version"),
    )


def default_cache_dir() -> Path:
    """Return the root directory for cached platform artifacts.

    Honors ``OUMI_CACHE_DIR`` (used as ``<dir>/platform``), otherwise falls
    back to ``$XDG_CACHE_HOME/oumi/platform`` or
    ``~/.cache/oumi/platform``.
    """
    explicit = os.environ.get("OUMI_CACHE_DIR")
    if explicit:
        return Path(explicit).expanduser() / "platform"
    xdg = os.environ.get("XDG_CACHE_HOME")
    root = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
    return root / "oumi" / "platform"


def resolve(
    uri: str,
    *,
    client: Client | None = None,
    cache_dir: Path | None = None,
    force_refresh: bool = False,
) -> Path | dict[str, Any]:
    """Resolve any ``oumi://`` URI to a local path or structured payload.

    Args:
        uri: The URI to resolve.
        client: Optional platform :class:`Client`. Defaults to
            :func:`get_default_client`.
        cache_dir: Override the cache root. Defaults to
            :func:`default_cache_dir`.
        force_refresh: When ``True``, ignore any cached artifact and re-fetch.

    Returns:
        For ``datasets`` and ``models``, a :class:`pathlib.Path` pointing
        into the cache. For ``judges``, ``evaluators``, and ``recipes``, a
        ``dict`` with the platform's response payload.

    Raises:
        ValueError: If ``uri`` is not a valid ``oumi://`` URI.
        PlatformError: If the platform returns an error or the response shape
            is unexpected.
    """
    parsed = parse_uri(uri)
    if parsed.kind == "datasets":
        return resolve_dataset(
            parsed,
            client=client,
            cache_dir=cache_dir,
            force_refresh=force_refresh,
        )
    if parsed.kind == "models":
        return resolve_model(
            parsed,
            client=client,
            cache_dir=cache_dir,
            force_refresh=force_refresh,
        )
    if parsed.kind in ("judges", "evaluators"):
        return resolve_evaluator(parsed, client=client)
    return resolve_recipe(parsed, client=client)


def resolve_dataset(
    parsed: ParsedURI,
    *,
    client: Client | None = None,
    cache_dir: Path | None = None,
    force_refresh: bool = False,
) -> Path:
    """Resolve a ``oumi://datasets/...`` URI to a local file path."""
    if parsed.kind != "datasets":
        raise ValueError(
            f"Expected a datasets URI, got kind={parsed.kind!r}"
        )
    resolved_client = client or get_default_client()
    project_id = resolved_client._resolve_project_id(None)
    cache_root = cache_dir or default_cache_dir()
    version = _resolve_dataset_version(resolved_client, parsed)
    target = (
        cache_root
        / "datasets"
        / project_id
        / parsed.resource_id
        / version
        / "data.jsonl"
    )
    if target.exists() and not force_refresh:
        return target
    resolved_client.datasets.download(parsed.resource_id, target)
    return target


def resolve_model(
    parsed: ParsedURI,
    *,
    client: Client | None = None,
    cache_dir: Path | None = None,
    force_refresh: bool = False,
) -> Path:
    """Resolve a ``oumi://models/...`` URI to a local directory path."""
    if parsed.kind != "models":
        raise ValueError(
            f"Expected a models URI, got kind={parsed.kind!r}"
        )
    resolved_client = client or get_default_client()
    project_id = resolved_client._resolve_project_id(None)
    cache_root = cache_dir or default_cache_dir()
    version = _resolve_model_version(resolved_client, parsed)
    target = (
        cache_root / "models" / project_id / parsed.resource_id / version
    )
    if target.exists() and any(target.iterdir()) and not force_refresh:
        return target
    resolved_client.models.download(
        parsed.resource_id,
        target,
        version_id=parsed.version,
    )
    return target


def resolve_evaluator(
    parsed: ParsedURI,
    *,
    client: Client | None = None,
) -> dict[str, Any]:
    """Resolve a ``oumi://judges/...`` or ``oumi://evaluators/...`` URI.

    Returns the raw evaluator payload. The judges/evaluators distinction is
    purely cosmetic on the URI side; both map to the platform's evaluator
    endpoint. Callers that need a :class:`JudgeConfig` should map the
    returned dict through the judge builder.
    """
    if parsed.kind not in ("judges", "evaluators"):
        raise ValueError(
            f"Expected a judges/evaluators URI, got kind={parsed.kind!r}"
        )
    resolved_client = client or get_default_client()
    payload = resolved_client.evaluators.get(parsed.resource_id)
    if parsed.kind == "judges":
        actual_type = payload.get("evaluatorType")
        if actual_type and actual_type != "judge":
            raise PlatformError(
                f"oumi://judges/{parsed.resource_id} resolved to evaluator of "
                f"type {actual_type!r}, not 'judge'."
            )
    return payload


def resolve_recipe(
    parsed: ParsedURI,
    *,
    client: Client | None = None,
) -> dict[str, Any]:
    """Resolve a ``oumi://recipes/...`` URI to the recipe's payload."""
    if parsed.kind != "recipes":
        raise ValueError(
            f"Expected a recipes URI, got kind={parsed.kind!r}"
        )
    resolved_client = client or get_default_client()
    return resolved_client.recipes.get(parsed.resource_id)


def push_back_dataset(
    output_uri: str,
    local_path: Path,
    *,
    client: Client | None = None,
) -> dict[str, Any]:
    """Upload ``local_path`` to the platform as a new dataset.

    Called by ``oumi synth`` / ``oumi judge`` when the user's output target
    is an ``oumi://datasets/...`` URI. The URI's ``resource_id`` is used as
    the new dataset's display name.

    Args:
        output_uri: The ``oumi://datasets/...`` URI from the user's config.
        local_path: Path to the file produced by the local run.
        client: Override the platform client.

    Returns:
        The :upload response from the platform.

    Raises:
        ValueError: If ``output_uri`` is not an ``oumi://datasets/...`` URI.
        PlatformError: If the platform rejects the upload.
    """
    parsed = parse_uri(output_uri)
    if parsed.kind != "datasets":
        raise ValueError(
            f"Cannot push-back to {output_uri!r}: kind must be 'datasets'."
        )
    resolved_client = client or get_default_client()
    return resolved_client.datasets.upload(
        local_path, display_name=parsed.resource_id
    )


def _resolve_dataset_version(
    client: Client, parsed: ParsedURI
) -> str:
    """Return a cache-key version label for a dataset URI.

    If the URI pins a version, use it verbatim. Otherwise call the platform
    to get the dataset's current ``version`` field so the cache key tracks
    the moving "latest" target correctly.
    """
    if parsed.version is not None:
        return parsed.version
    info = client.datasets.get(parsed.resource_id)
    return _stringify_version(info)


def _resolve_model_version(
    client: Client, parsed: ParsedURI
) -> str:
    """Return a cache-key version label for a model URI."""
    if parsed.version is not None:
        return parsed.version
    info = client.models.get(parsed.resource_id)
    return _stringify_version(info)


def _stringify_version(info: object) -> str:
    """Pick a stable version label from a resource payload.

    Prefers ``versionName``, falls back to ``version``, then ``"latest"`` if
    the payload omits both (unusual but tolerable for cache-key purposes).
    """
    if isinstance(info, dict):
        name = info.get("versionName")
        if isinstance(name, str) and name:
            return name
        num = info.get("version")
        if num is not None:
            return str(num)
    # Without server-supplied version info, fall back to a literal "latest"
    # cache slot. This is only reached if the platform omits version fields.
    return _LATEST
