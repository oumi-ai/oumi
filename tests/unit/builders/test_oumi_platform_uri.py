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

"""Tests that ``oumi://`` URIs are intercepted by the data and model builders.

These tests stub out :mod:`oumi.platform.resolver` so they neither hit the
network nor depend on a real platform; they only verify that the builder
calls the resolver with the right URI and substitutes the resolved local
path back into the rest of the pipeline.
"""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from oumi.builders.data import _maybe_load_oumi_platform_dataset
from oumi.builders.models import _maybe_resolve_oumi_model_uri
from oumi.core.configs import DatasetParams, ModelParams


def test_dataset_uri_resolves_to_local_jsonl(tmp_path):
    """An ``oumi://datasets/...`` URI is fetched and loaded as raw JSONL."""
    jsonl = tmp_path / "data.jsonl"
    jsonl.write_text(
        json.dumps({"text": "hello"}) + "\n" + json.dumps({"text": "world"}) + "\n"
    )

    with patch("oumi.platform.resolver.resolve_dataset", return_value=jsonl) as r:
        params = DatasetParams(dataset_name="oumi://datasets/abc@v1")
        result = _maybe_load_oumi_platform_dataset(params, stream=False)

    assert result is not None
    assert r.call_count == 1
    rows = list(result)
    assert rows == [{"text": "hello"}, {"text": "world"}]


def test_dataset_non_oumi_uri_returns_none():
    """Non-platform names fall through to normal loading."""
    params = DatasetParams(dataset_name="HuggingFaceH4/ultrachat_200k")

    # Resolver must not be invoked.
    with patch("oumi.platform.resolver.resolve_dataset") as resolve:
        result = _maybe_load_oumi_platform_dataset(params, stream=False)
    resolve.assert_not_called()
    assert result is None


def test_dataset_wrong_kind_uri_raises():
    """Misusing a non-dataset URI in dataset_name is a programming error."""
    params = DatasetParams(dataset_name="oumi://models/m1")

    with pytest.raises(ValueError, match="oumi://datasets"):
        _maybe_load_oumi_platform_dataset(params, stream=False)


def test_model_uri_rewrites_model_name(tmp_path: Path):
    """An ``oumi://models/...`` URI is downloaded and substituted in place."""
    local_dir = tmp_path / "downloaded-model"
    local_dir.mkdir()
    (local_dir / "config.json").write_text("{}")

    params = ModelParams(model_name="oumi://models/m1@v3")
    with patch(
        "oumi.platform.resolver.resolve_model", return_value=local_dir
    ) as r:
        _maybe_resolve_oumi_model_uri(params)

    assert r.call_count == 1
    assert params.model_name == str(local_dir)


def test_model_non_oumi_uri_is_passthrough():
    """A normal model name is left unchanged and the resolver is not invoked."""
    params = ModelParams(model_name="meta-llama/Meta-Llama-3-8B")

    with patch("oumi.platform.resolver.resolve_model") as resolve:
        _maybe_resolve_oumi_model_uri(params)
    resolve.assert_not_called()
    assert params.model_name == "meta-llama/Meta-Llama-3-8B"


def test_model_wrong_kind_uri_raises():
    params = ModelParams(model_name="oumi://datasets/abc")

    with pytest.raises(ValueError, match="oumi://models"):
        _maybe_resolve_oumi_model_uri(params)
