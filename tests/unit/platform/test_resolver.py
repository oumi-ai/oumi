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

from collections.abc import Callable
from pathlib import Path

import httpx
import pytest

from oumi.platform import (
    Client,
    Credentials,
    ParsedURI,
    PlatformError,
    default_cache_dir,
    is_oumi_uri,
    parse_uri,
    resolve,
    resolve_dataset,
    resolve_evaluator,
    resolve_recipe,
)

_TEST_API_URL = "https://platform.test"
_TEST_CREDS = Credentials(
    api_url=_TEST_API_URL,
    api_key="test-key",
    project_id="proj-1",
)


def _make_client(
    handler: Callable[[httpx.Request], httpx.Response],
) -> Client:
    transport = httpx.MockTransport(handler)
    http = httpx.Client(
        base_url=_TEST_API_URL,
        transport=transport,
        headers={
            "Authorization": f"Bearer {_TEST_CREDS.api_key}",
            "X-API-Key": _TEST_CREDS.api_key,
        },
    )
    return Client(credentials=_TEST_CREDS, http_client=http)


# ---------------------------------------------------------------- parse_uri ---


@pytest.mark.parametrize(
    "uri,kind,resource_id,version",
    [
        ("oumi://datasets/abc", "datasets", "abc", None),
        ("oumi://datasets/abc@v3", "datasets", "abc", "v3"),
        ("oumi://models/m1@7", "models", "m1", "7"),
        ("oumi://judges/quality", "judges", "quality", None),
        ("oumi://evaluators/qual@v1", "evaluators", "qual", "v1"),
        ("oumi://recipes/sft@latest", "recipes", "sft", "latest"),
    ],
)
def test_parse_valid_uris(uri, kind, resource_id, version):
    parsed = parse_uri(uri)
    assert parsed == ParsedURI(
        kind=kind, resource_id=resource_id, version=version
    )


@pytest.mark.parametrize(
    "bad_uri",
    [
        "datasets/abc",  # no scheme
        "http://datasets/abc",  # wrong scheme
        "oumi://datasets/",  # empty id
        "oumi://datasets",  # missing /id
        "oumi://nope/abc",  # unsupported kind
        "oumi://datasets/abc/extra",  # extra path segment
        "oumi://Datasets/abc",  # kind must be lowercase
        "",
    ],
)
def test_parse_invalid_uris(bad_uri):
    with pytest.raises(ValueError):
        parse_uri(bad_uri)


def test_parse_uri_rejects_non_string():
    with pytest.raises(ValueError):
        parse_uri(123)  # type: ignore[arg-type]


def test_version_or_latest():
    assert parse_uri("oumi://datasets/x").version_or_latest == "latest"
    assert parse_uri("oumi://datasets/x@v2").version_or_latest == "v2"


def test_is_oumi_uri():
    assert is_oumi_uri("oumi://datasets/abc")
    assert not is_oumi_uri("oumi:/datasets/abc")  # missing slash
    assert not is_oumi_uri("https://example.com")
    assert not is_oumi_uri(None)
    assert not is_oumi_uri(123)


# ----------------------------------------------------------- cache_dir env ---


def test_default_cache_dir_respects_oumi_env(monkeypatch, tmp_path):
    monkeypatch.setenv("OUMI_CACHE_DIR", str(tmp_path))
    assert default_cache_dir() == tmp_path / "platform"


def test_default_cache_dir_respects_xdg(monkeypatch, tmp_path):
    monkeypatch.delenv("OUMI_CACHE_DIR", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    assert default_cache_dir() == tmp_path / "oumi" / "platform"


def test_default_cache_dir_falls_back_to_home(monkeypatch):
    monkeypatch.delenv("OUMI_CACHE_DIR", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    assert default_cache_dir() == Path.home() / ".cache" / "oumi" / "platform"


# ------------------------------------------------------------- datasets ---


def test_resolve_dataset_downloads_and_caches(tmp_path):
    """First resolve downloads, second resolve hits the cache."""
    presigned = "https://storage.test/signed/dataset.jsonl"
    download_calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "storage.test":
            download_calls.append(str(request.url))
            return httpx.Response(200, content=b'{"row":1}\n')
        path = request.url.path
        if path.endswith("/datasets/abc"):
            return httpx.Response(
                200,
                json={
                    "id": 1,
                    "displayName": "abc",
                    "schemaType": "conversation",
                    "version": 4,
                    "versionName": "v4",
                },
            )
        if path.endswith("/datasets/abc:download"):
            return httpx.Response(200, json={"url": presigned})
        raise AssertionError(f"Unexpected URL: {request.url}")

    client = _make_client(handler)
    uri = "oumi://datasets/abc"

    first = resolve(uri, client=client, cache_dir=tmp_path)
    second = resolve(uri, client=client, cache_dir=tmp_path)

    assert isinstance(first, Path)
    assert first == second
    assert first == tmp_path / "datasets" / "proj-1" / "abc" / "v4" / "data.jsonl"
    assert first.read_text() == '{"row":1}\n'
    assert len(download_calls) == 1  # second call served from cache


def test_resolve_dataset_pinned_version_skips_metadata_call(tmp_path):
    """Pinning @version means the resolver never has to call GET /datasets/<id>."""
    presigned = "https://storage.test/dataset.jsonl"
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        if request.url.host == "storage.test":
            return httpx.Response(200, content=b'{"row":1}\n')
        if request.url.path.endswith("/datasets/abc:download"):
            return httpx.Response(200, json={"url": presigned})
        raise AssertionError(f"Unexpected URL: {request.url}")

    client = _make_client(handler)

    target = resolve(
        "oumi://datasets/abc@v7",
        client=client,
        cache_dir=tmp_path,
    )

    assert isinstance(target, Path)
    assert target.parent.name == "v7"
    assert all(":download" in c or "storage.test" not in c for c in calls)
    metadata_hits = [c for c in calls if c.endswith("/datasets/abc")]
    assert metadata_hits == []  # no version probe


def test_resolve_dataset_force_refresh_redownloads(tmp_path):
    presigned = "https://storage.test/dataset.jsonl"
    bodies = iter([b"first", b"second"])

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "storage.test":
            return httpx.Response(200, content=next(bodies))
        if request.url.path.endswith("/datasets/abc:download"):
            return httpx.Response(200, json={"url": presigned})
        raise AssertionError(f"Unexpected URL: {request.url}")

    client = _make_client(handler)
    uri = "oumi://datasets/abc@v1"

    first = resolve_dataset(parse_uri(uri), client=client, cache_dir=tmp_path)
    refreshed = resolve_dataset(
        parse_uri(uri),
        client=client,
        cache_dir=tmp_path,
        force_refresh=True,
    )

    assert isinstance(first, Path) and isinstance(refreshed, Path)
    assert first == refreshed
    assert refreshed.read_bytes() == b"second"


# --------------------------------------------------------------- models ---


def test_resolve_model_downloads_files(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "storage.test":
            return httpx.Response(
                200, content=request.url.path.encode()
            )
        path = request.url.path
        if path.endswith("/models/m1"):
            return httpx.Response(
                200, json={"version": 2, "versionName": "v2"}
            )
        if path.endswith("/models/m1:download"):
            return httpx.Response(
                200,
                json={
                    "files": [
                        {
                            "name": "config.json",
                            "url": "https://storage.test/cfg",
                        },
                        {
                            "name": "weights.safetensors",
                            "url": "https://storage.test/w",
                        },
                    ]
                },
            )
        raise AssertionError(f"Unexpected URL: {request.url}")

    client = _make_client(handler)

    result = resolve("oumi://models/m1", client=client, cache_dir=tmp_path)

    assert isinstance(result, Path)
    assert result == tmp_path / "models" / "proj-1" / "m1" / "v2"
    assert (result / "config.json").read_bytes() == b"/cfg"


def test_resolve_model_pinned_version_uses_version_endpoint(tmp_path):
    download_paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "storage.test":
            return httpx.Response(200, content=b"data")
        download_paths.append(request.url.path)
        if request.url.path.endswith(
            "/models/m1/versions/v7:download"
        ):
            return httpx.Response(
                200,
                json={
                    "files": [
                        {
                            "name": "x.bin",
                            "url": "https://storage.test/x",
                        }
                    ]
                },
            )
        raise AssertionError(f"Unexpected URL: {request.url}")

    client = _make_client(handler)

    result = resolve(
        "oumi://models/m1@v7",
        client=client,
        cache_dir=tmp_path,
    )

    assert isinstance(result, Path)
    assert any(
        p.endswith("/models/m1/versions/v7:download") for p in download_paths
    )


# ----------------------------------------------------- judges/evaluators ---


def test_resolve_evaluator_returns_payload():
    payload = {
        "id": 1,
        "displayName": "quality",
        "evaluatorType": "judge",
        "version": 1,
    }

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/evaluators/q1"):
            return httpx.Response(200, json=payload)
        raise AssertionError(f"Unexpected URL: {request.url}")

    client = _make_client(handler)

    out = resolve("oumi://evaluators/q1", client=client)

    assert out == payload


def test_resolve_judges_rejects_non_judge_evaluator():
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": 1,
                "displayName": "cls",
                "evaluatorType": "classification",
            },
        )

    client = _make_client(handler)

    with pytest.raises(PlatformError, match="not 'judge'"):
        resolve("oumi://judges/cls", client=client)


def test_resolve_judges_accepts_judge_type():
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": 1,
                "displayName": "qual",
                "evaluatorType": "judge",
            },
        )

    client = _make_client(handler)

    out = resolve_evaluator(parse_uri("oumi://judges/qual"), client=client)

    assert out["evaluatorType"] == "judge"


# ------------------------------------------------------------- recipes ---


def test_resolve_recipe_returns_payload():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/recipes/sft"):
            return httpx.Response(
                200,
                json={"id": 1, "displayName": "sft", "version": 1},
            )
        raise AssertionError(f"Unexpected URL: {request.url}")

    client = _make_client(handler)

    out = resolve_recipe(parse_uri("oumi://recipes/sft"), client=client)

    assert out["displayName"] == "sft"


# ---------------------------------------------------------- top-level ---


def test_resolve_rejects_wrong_kind_helpers():
    parsed = parse_uri("oumi://models/m1")
    with pytest.raises(ValueError):
        resolve_dataset(parsed, client=None)  # wrong dispatcher


def test_resolve_model_finds_version_via_metadata_when_unpinned(tmp_path):
    """When the model lookup payload lacks a versionName, fall back to version."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "storage.test":
            return httpx.Response(200, content=b"data")
        if request.url.path.endswith("/models/m1"):
            return httpx.Response(200, json={"version": 5})  # no versionName
        if request.url.path.endswith("/models/m1:download"):
            return httpx.Response(
                200,
                json={
                    "files": [
                        {
                            "name": "x.bin",
                            "url": "https://storage.test/x",
                        }
                    ]
                },
            )
        raise AssertionError(f"Unexpected URL: {request.url}")

    client = _make_client(handler)

    result = resolve("oumi://models/m1", client=client, cache_dir=tmp_path)

    assert isinstance(result, Path)
    assert result.name == "5"  # stringified version
