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

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from oumi.inference.vllm_inference_engine import (
    _cuda_toolkit_below_12_6,
    _model_uses_gdn,
    _parse_nvcc_release_version,
    _should_force_triton_gdn_backend,
)

_MODULE = "oumi.inference.vllm_inference_engine"


@pytest.fixture(autouse=True)
def _clear_caches():
    _cuda_toolkit_below_12_6.cache_clear()
    _model_uses_gdn.cache_clear()
    yield
    _cuda_toolkit_below_12_6.cache_clear()
    _model_uses_gdn.cache_clear()


@pytest.mark.parametrize(
    "output,expected",
    [
        ("Cuda compilation tools, release 12.4, V12.4.131", (12, 4)),
        ("release 12.6, V12.6.20", (12, 6)),
        ("release 13.0", (13, 0)),
        ("no version here", None),
        ("", None),
    ],
)
def test_parse_nvcc_release_version(output, expected):
    assert _parse_nvcc_release_version(output) == expected


@pytest.mark.parametrize(
    "release,expected",
    [
        ("release 12.4, V12.4.131", True),  # < 12.6
        ("release 12.5, V12.5.0", True),
        ("release 12.6, V12.6.20", False),  # >= 12.6
        ("release 13.0, V13.0.0", False),
        ("unparseable", False),  # unknown → don't change behavior
    ],
)
def test_cuda_toolkit_below_12_6_by_version(release, expected):
    with (
        patch(f"{_MODULE}.shutil.which", return_value="/usr/local/cuda/bin/nvcc"),
        patch(f"{_MODULE}.subprocess.run") as mock_run,
    ):
        mock_run.return_value.stdout = release
        assert _cuda_toolkit_below_12_6() is expected


def test_cuda_toolkit_below_12_6_no_nvcc():
    # No nvcc on PATH and none under CUDA_HOME → treat as too old.
    with (
        patch(f"{_MODULE}.shutil.which", return_value=None),
        patch.dict(f"{_MODULE}.os.environ", {}, clear=True),
        patch(f"{_MODULE}.Path.is_file", return_value=False),
    ):
        assert _cuda_toolkit_below_12_6() is True


@pytest.mark.parametrize(
    "model_type,expected",
    [
        ("qwen3_5", True),
        ("qwen3_next", True),
        ("qwen3", False),  # non-GDN
        ("llama", False),
    ],
)
def test_model_uses_gdn(model_type, expected):
    with patch("transformers.AutoConfig.from_pretrained") as mock_cfg:
        mock_cfg.return_value = SimpleNamespace(model_type=model_type)
        assert _model_uses_gdn("some/model", False) is expected


def test_model_uses_gdn_unreadable_config():
    # Config can't be loaded (e.g. bad path) → False, never raises.
    with patch("transformers.AutoConfig.from_pretrained", side_effect=OSError("boom")):
        assert _model_uses_gdn("missing/model", False) is False


@pytest.mark.parametrize(
    "accepts,old_toolkit,is_gdn,expected",
    [
        (True, True, True, True),  # all conditions met → force triton
        (True, True, False, False),  # non-GDN model → no-op
        (True, False, True, False),  # modern toolkit → keep flashinfer
        (False, True, True, False),  # vLLM can't take additional_config
    ],
)
def test_should_force_triton_gdn_backend(accepts, old_toolkit, is_gdn, expected):
    with (
        patch(f"{_MODULE}._vllm_accepts_additional_config", return_value=accepts),
        patch(f"{_MODULE}._cuda_toolkit_below_12_6", return_value=old_toolkit),
        patch(f"{_MODULE}._model_uses_gdn", return_value=is_gdn),
    ):
        assert _should_force_triton_gdn_backend("some/model", False) is expected
