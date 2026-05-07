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

import httpx
import pytest

from oumi.deploy.errors import (
    DeployApiError,
    DeployInvalidRequestError,
    DeployNotFoundError,
    DeployRateLimitError,
    DeployTransientError,
)
from oumi.deploy.utils import (
    check_hf_model_accessibility,
    is_huggingface_repo_id,
    is_huggingface_url,
    raise_api_error,
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


def _make_response(
    status_code: int,
    json_body: dict | list | str | None = None,
    text: str = "",
    method: str = "POST",
    url: str = "https://api.fireworks.ai/v1/accounts/admin-xyz/models/foo",
    json_raises: bool = False,
) -> MagicMock:
    """Build a MagicMock httpx.Response for ``raise_api_error`` tests.

    Set *json_raises* to simulate a non-JSON body where ``response.json()``
    raises; in that case *text* is used as the fallback detail source.
    """
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.text = text
    if json_raises:
        resp.json.side_effect = ValueError("not json")
    else:
        resp.json.return_value = json_body
    req = MagicMock()
    req.method = method
    req.url = url
    req.content = b'{"key": "value"}'
    resp.request = req
    return resp


class TestRaiseApiErrorStatusCodeMapping:
    """raise_api_error picks the typed subclass from the HTTP status."""

    def test_http_500_empty_body(self):
        """HTTP 500 with empty body → DeployTransientError with sanitized str."""
        resp = _make_response(
            500,
            json_raises=True,
            text="",
            url="https://api.fireworks.ai/x:validateUpload",
        )
        with pytest.raises(DeployTransientError) as exc_info:
            raise_api_error(resp, "validate upload for model 'foo'")

        exc = exc_info.value
        assert exc.detail == "(no details)"
        assert exc.status_code == 500
        assert exc.context == "validate upload for model 'foo'"
        rendered = str(exc)
        assert rendered == (
            "The deployment service is temporarily unavailable. Please retry shortly."
        )
        # No URL, method, or empty-colon artifact
        assert "fireworks.ai" not in rendered
        assert "POST" not in rendered

    def test_http_400_without_classifier_is_base_invalid_request(self):
        """Without a classify_4xx hook, 400 is the base DeployInvalidRequestError."""
        resp = _make_response(400, {"message": "some specific provider detail"})
        with pytest.raises(DeployInvalidRequestError) as exc_info:
            raise_api_error(resp, "op")

        assert type(exc_info.value) is DeployInvalidRequestError
        assert exc_info.value.status_code == 400

    def test_classify_4xx_hook_refines_400(self):
        """A classify_4xx hook selects a provider-specific subclass on 400."""

        class CustomSubclass(DeployInvalidRequestError):
            pass

        def classifier(detail: str) -> type[DeployInvalidRequestError]:
            if "magic" in detail:
                return CustomSubclass
            return DeployInvalidRequestError

        resp = _make_response(400, {"message": "magic word present"})
        with pytest.raises(CustomSubclass):
            raise_api_error(resp, "op", classify_4xx=classifier)

    def test_classify_4xx_hook_fallthrough_when_no_match(self):
        """Classifier returning the base class → exception is the base class."""

        def classifier(_: str) -> type[DeployInvalidRequestError]:
            return DeployInvalidRequestError

        resp = _make_response(400, {"message": "no match"})
        with pytest.raises(DeployInvalidRequestError) as exc_info:
            raise_api_error(resp, "op", classify_4xx=classifier)
        assert type(exc_info.value) is DeployInvalidRequestError

    def test_classify_4xx_hook_not_consulted_for_non_4xx(self):
        """Classifier is only consulted for 400/422 — e.g. 500 ignores it."""
        calls = []

        def classifier(detail: str) -> type[DeployInvalidRequestError]:
            calls.append(detail)
            return DeployInvalidRequestError

        resp = _make_response(500, {"message": "oops"})
        with pytest.raises(DeployTransientError):
            raise_api_error(resp, "op", classify_4xx=classifier)
        assert calls == []

    def test_http_404_not_found(self):
        resp = _make_response(404, {"message": "model not found"})
        with pytest.raises(DeployNotFoundError) as exc_info:
            raise_api_error(resp, "get model 'foo'")

        exc = exc_info.value
        assert exc.status_code == 404
        assert exc.detail == "model not found"
        assert "fireworks.ai" not in str(exc)

    def test_http_429_rate_limit(self):
        resp = _make_response(
            429, {"error": {"code": "RATE_LIMITED", "message": "slow down"}}
        )
        with pytest.raises(DeployRateLimitError) as exc_info:
            raise_api_error(resp, "create endpoint")

        exc = exc_info.value
        assert exc.status_code == 429
        assert exc.detail == "slow down"

    def test_http_422_is_invalid_request(self):
        resp = _make_response(422, {"detail": "bad field"})
        with pytest.raises(DeployInvalidRequestError) as exc_info:
            raise_api_error(resp, "update endpoint")
        assert exc_info.value.status_code == 422

    @pytest.mark.parametrize("status", [401, 403, 409, 410, 418])
    def test_unclassified_4xx_uses_base_class(self, status):
        """401/403/409/etc. should fall through to the base DeployApiError.

        401/403 in particular are backend-config issues, not user-input
        errors, so they must NOT be routed through DeployInvalidRequestError.
        """
        resp = _make_response(status, {"message": "nope"})
        with pytest.raises(DeployApiError) as exc_info:
            raise_api_error(resp, "op")

        exc = exc_info.value
        # Base class — not any of the modeled subclasses
        assert type(exc) is DeployApiError
        assert not isinstance(exc, DeployInvalidRequestError)
        assert not isinstance(exc, DeployNotFoundError)
        assert not isinstance(exc, DeployRateLimitError)
        assert not isinstance(exc, DeployTransientError)
        assert exc.status_code == status

    def test_non_error_status_raises_value_error(self):
        """status < 400 means raise_api_error was called on a 2xx/3xx — caller bug."""
        resp = _make_response(200, {"message": "ok"})
        with pytest.raises(ValueError, match="non-error status 200"):
            raise_api_error(resp, "op")


class TestRaiseApiErrorBodyParsing:
    """Body-shape fallbacks (preserve existing behavior)."""

    def test_nested_error_message(self):
        resp = _make_response(400, {"error": {"message": "x", "code": "INVALID"}})
        with pytest.raises(DeployInvalidRequestError) as exc_info:
            raise_api_error(resp, "op")
        assert exc_info.value.detail == "x"

    def test_top_level_message(self):
        resp = _make_response(400, {"message": "x"})
        with pytest.raises(DeployInvalidRequestError) as exc_info:
            raise_api_error(resp, "op")
        assert exc_info.value.detail == "x"

    def test_detail_field(self):
        resp = _make_response(400, {"detail": "x"})
        with pytest.raises(DeployInvalidRequestError) as exc_info:
            raise_api_error(resp, "op")
        assert exc_info.value.detail == "x"

    def test_non_json_body_falls_back_to_text(self):
        resp = _make_response(500, json_raises=True, text="raw text from upstream")
        with pytest.raises(DeployTransientError) as exc_info:
            raise_api_error(resp, "op")
        assert exc_info.value.detail == "raw text from upstream"

    def test_json_raises_and_no_text(self):
        resp = _make_response(500, json_raises=True, text="")
        with pytest.raises(DeployTransientError) as exc_info:
            raise_api_error(resp, "op")
        assert exc_info.value.detail == "(no details)"

    def test_dict_without_known_fields_stringifies(self):
        body = {"unexpected": "shape"}
        resp = _make_response(400, body)
        with pytest.raises(DeployInvalidRequestError) as exc_info:
            raise_api_error(resp, "op")
        assert exc_info.value.detail == str(body)


class TestRaiseApiErrorMessageDoesNotLeakRequestDetails:
    """str(exc) never includes method, URL, status code, or request body."""

    def test_method_and_url_never_in_message(self):
        resp = _make_response(
            400,
            {"message": "bad request"},
            method="POST",
            url="https://api.fireworks.ai/v1/accounts/admin-SECRET/deployments?deploymentId=x",
        )
        with pytest.raises(DeployInvalidRequestError) as exc_info:
            raise_api_error(resp, "create endpoint")

        rendered = str(exc_info.value)
        assert "POST" not in rendered
        assert "fireworks.ai" not in rendered
        assert "admin-SECRET" not in rendered
        assert "HTTP 400" not in rendered
        assert "deploymentId" not in rendered

    def test_structured_attributes_populated(self):
        resp = _make_response(
            400,
            {"message": "bad"},
            method="DELETE",
            url="https://api.fireworks.ai/v1/x",
        )
        with pytest.raises(DeployInvalidRequestError) as exc_info:
            raise_api_error(resp, "ctx")

        exc = exc_info.value
        assert exc.status_code == 400
        assert exc.detail == "bad"
        assert exc.method == "DELETE"
        assert exc.url == "https://api.fireworks.ai/v1/x"
        assert exc.context == "ctx"


class TestDeployApiErrorHierarchy:
    """Status-code subclasses inherit from DeployApiError."""

    def test_subclasses_inherit_from_base(self):
        for cls in (
            DeployInvalidRequestError,
            DeployNotFoundError,
            DeployRateLimitError,
            DeployTransientError,
        ):
            assert issubclass(cls, DeployApiError)
