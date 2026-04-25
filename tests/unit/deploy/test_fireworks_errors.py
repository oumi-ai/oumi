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

"""Unit tests for oumi.deploy.fireworks_errors.

Split into two layers:

1. Pure-function tests of ``classify_fireworks_invalid_request`` — detail
   string → subclass. Adding a new Fireworks-specific error type is a
   one-line test here.

2. One wiring integration test per sub-type, exercising the classifier
   through ``raise_api_error(..., classify_4xx=...)`` to verify the
   end-to-end FireworksClient path produces the right subclass on a real
   response shape.
"""

from unittest.mock import MagicMock

import httpx
import pytest

from oumi.deploy.errors import DeployInvalidRequestError
from oumi.deploy.fireworks_errors import (
    FireworksAdapterMismatchError,
    FireworksUnsupportedHardwareError,
    classify_fireworks_invalid_request,
)
from oumi.deploy.utils import raise_api_error


class TestClassifyFireworksInvalidRequest:
    """Pure-function tests — detail string → matching subclass."""

    def test_unsupported_hardware_signature(self):
        detail = (
            "invalid deployment: model type qwen3 is not supported on NVIDIA_A100_80GB"
        )
        assert (
            classify_fireworks_invalid_request(detail)
            is FireworksUnsupportedHardwareError
        )

    def test_adapter_mismatch_signature(self):
        detail = (
            "LoRA validation failed: LoRA keys reference non-existent base "
            "model parameters: model.vision_tower..."
        )
        assert (
            classify_fireworks_invalid_request(detail) is FireworksAdapterMismatchError
        )

    def test_invalid_deployment_without_unsupported_on_falls_through(self):
        """Both markers required — 'invalid deployment' alone is not enough."""
        detail = "invalid deployment: some other reason"
        assert classify_fireworks_invalid_request(detail) is DeployInvalidRequestError

    def test_lora_in_middle_is_not_a_match(self):
        """The signature requires `startswith`, not substring — guards against
        incidental mentions of `LoRA validation failed` inside other messages.
        """
        detail = "prefix text LoRA validation failed trailing"
        assert classify_fireworks_invalid_request(detail) is DeployInvalidRequestError

    def test_unrelated_400_detail_falls_through(self):
        assert (
            classify_fireworks_invalid_request("something else entirely")
            is DeployInvalidRequestError
        )

    def test_empty_detail_falls_through(self):
        assert classify_fireworks_invalid_request("") is DeployInvalidRequestError


def _make_response(status_code: int, json_body: dict) -> MagicMock:
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.text = ""
    resp.json.return_value = json_body
    req = MagicMock()
    req.method = "POST"
    req.url = "https://api.fireworks.ai/v1/accounts/admin-xyz/models/foo"
    req.content = b""
    resp.request = req
    return resp


class TestFireworksClassifierWiring:
    """Integration tests — classifier flowing through raise_api_error.

    These prove the plumbing holds end-to-end on verbatim production
    failure-mode response shapes.
    """

    def test_unsupported_hardware_wires_through(self):
        detail = (
            "invalid deployment: model type qwen3 is not supported on NVIDIA_A100_80GB"
        )
        resp = _make_response(400, {"message": detail})
        with pytest.raises(FireworksUnsupportedHardwareError) as exc_info:
            raise_api_error(
                resp,
                "create endpoint for model 'foo'",
                classify_4xx=classify_fireworks_invalid_request,
            )
        exc = exc_info.value
        assert exc.status_code == 400
        assert exc.detail == detail
        rendered = str(exc)
        # Provider-specific detail preserved; method/URL must NOT leak
        assert "NVIDIA_A100_80GB" in rendered
        assert "POST" not in rendered
        assert "fireworks.ai" not in rendered

    def test_adapter_mismatch_wires_through(self):
        detail = (
            "LoRA validation failed: LoRA keys reference non-existent base "
            "model parameters: model.vision_tower.vision_model.encoder."
            "layers.6.self_attn.k_proj.weight (from LoRA key: "
            "base_model.model.model.vision_tower...)"
        )
        resp = _make_response(400, {"message": detail})
        with pytest.raises(FireworksAdapterMismatchError) as exc_info:
            raise_api_error(
                resp,
                "validate upload for model 'foo'",
                classify_4xx=classify_fireworks_invalid_request,
            )
        exc = exc_info.value
        assert exc.detail == detail
        rendered = str(exc)
        assert "POST" not in rendered
        assert "fireworks.ai" not in rendered
