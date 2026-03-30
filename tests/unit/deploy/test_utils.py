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

"""Unit tests for oumi.deploy.utils."""

import logging
from unittest.mock import MagicMock, patch

import pytest

from oumi.deploy.utils import (
    check_hf_model_accessibility,
    is_huggingface_repo_id,
    is_huggingface_url,
    resolve_hf_token,
    warn_if_private_model_missing_token,
)


class TestIsHuggingfaceRepoId:
    def test_standard_repo_ids(self):
        assert is_huggingface_repo_id("Qwen/Qwen3-4B") is True
        assert is_huggingface_repo_id("meta-llama/Llama-3-8B") is True
        assert is_huggingface_repo_id("deepseek-ai/DeepSeek-R1") is True

    def test_repo_id_with_dots_and_underscores(self):
        assert is_huggingface_repo_id("user/model_v1.0") is True

    def test_rejects_single_segment(self):
        assert is_huggingface_repo_id("Qwen3-4B") is False

    def test_rejects_three_segments(self):
        assert is_huggingface_repo_id("org/sub/model") is False

    def test_rejects_urls(self):
        assert is_huggingface_repo_id("https://huggingface.co/Qwen/Qwen3") is False

    def test_rejects_empty(self):
        assert is_huggingface_repo_id("") is False

    def test_rejects_spaces(self):
        assert is_huggingface_repo_id("Qwen/ Qwen3") is False


class TestIsHuggingfaceUrl:
    def test_https_url(self):
        assert is_huggingface_url("https://huggingface.co/Qwen/Qwen3-4B") is True

    def test_http_url(self):
        assert is_huggingface_url("http://huggingface.co/Qwen/Qwen3-4B") is True

    def test_rejects_other_urls(self):
        assert is_huggingface_url("https://github.com/org/repo") is False

    def test_rejects_repo_id(self):
        assert is_huggingface_url("Qwen/Qwen3-4B") is False


class TestResolveHfToken:
    def test_explicit_key_takes_priority(self):
        with patch.dict("os.environ", {"HF_TOKEN": "env-token"}):
            assert resolve_hf_token("explicit-token") == "explicit-token"

    def test_falls_back_to_env_var(self):
        with patch.dict("os.environ", {"HF_TOKEN": "env-token"}):
            assert resolve_hf_token() == "env-token"

    def test_returns_empty_when_nothing_set(self):
        with patch.dict("os.environ", {}, clear=True):
            assert resolve_hf_token() == ""

    def test_empty_string_key_falls_through(self):
        with patch.dict("os.environ", {"HF_TOKEN": "env-token"}):
            assert resolve_hf_token("") == "env-token"

    def test_none_key_falls_through(self):
        with patch.dict("os.environ", {"HF_TOKEN": "env-token"}):
            assert resolve_hf_token(None) == "env-token"


class TestCheckHfModelAccessibility:
    def test_non_repo_id_returns_true(self):
        assert check_hf_model_accessibility("not-a-repo-id") is True

    def test_returns_true_when_hub_not_installed(self):
        with patch.dict("sys.modules", {"huggingface_hub": None}):
            with patch(
                "oumi.deploy.utils.check_hf_model_accessibility",
                wraps=check_hf_model_accessibility,
            ):
                assert check_hf_model_accessibility("single-word") is True

    def test_public_model_returns_true(self):
        with patch("oumi.deploy.utils.check_hf_model_accessibility") as mock_check:
            mock_check.return_value = True
            assert mock_check("Qwen/Qwen3-4B") is True

    def test_gated_model_returns_false(self):
        try:
            from huggingface_hub.utils import GatedRepoError

            with patch(
                "huggingface_hub.model_info",
                side_effect=GatedRepoError("gated", response=MagicMock()),
            ):
                assert check_hf_model_accessibility("meta-llama/Llama-3-8B") is False
        except ImportError:
            pytest.skip("huggingface_hub not installed")

    def test_not_found_model_returns_false(self):
        try:
            from huggingface_hub.utils import RepositoryNotFoundError

            with patch(
                "huggingface_hub.model_info",
                side_effect=RepositoryNotFoundError("not found", response=MagicMock()),
            ):
                assert check_hf_model_accessibility("org/nonexistent") is False
        except ImportError:
            pytest.skip("huggingface_hub not installed")

    def test_unexpected_error_returns_true(self):
        try:
            with patch(
                "huggingface_hub.model_info",
                side_effect=RuntimeError("network error"),
            ):
                assert check_hf_model_accessibility("org/model") is True
        except ImportError:
            pytest.skip("huggingface_hub not installed")


class TestWarnIfPrivateModelMissingToken:
    def test_no_warning_when_token_is_set(self, caplog):
        with caplog.at_level(logging.WARNING, logger="oumi.deploy.utils"):
            warn_if_private_model_missing_token("Qwen/Qwen3-4B", "my-token")
        assert not any("gated or private" in m for m in caplog.messages)

    def test_warns_when_model_is_private_and_no_token(self, caplog):
        with (
            patch("oumi.deploy.utils.check_hf_model_accessibility", return_value=False),
            caplog.at_level(logging.WARNING, logger="oumi.deploy.utils"),
        ):
            warn_if_private_model_missing_token("meta-llama/Llama-3-8B", "")

        assert any("gated or private" in m for m in caplog.messages)

    def test_no_warning_when_model_is_public(self, caplog):
        with (
            patch("oumi.deploy.utils.check_hf_model_accessibility", return_value=True),
            caplog.at_level(logging.WARNING, logger="oumi.deploy.utils"),
        ):
            warn_if_private_model_missing_token("Qwen/Qwen3-4B", "")

        assert not any("gated or private" in m for m in caplog.messages)
