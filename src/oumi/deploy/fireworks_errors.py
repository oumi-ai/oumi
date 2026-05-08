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

"""Fireworks-specific typed errors and 4xx classifier.

Two patterns:

- 400/422: subclass :class:`DeployInvalidRequestError`, routed by
  :func:`classify_fireworks_invalid_request` (used as the ``classify_4xx``
  hook in :func:`oumi.deploy.utils.check_response`).
- Other 4xx (e.g. 409): subclass :class:`DeployApiError` and raise
  inline from the client where the status is detected.
"""

from oumi.deploy.errors import DeployApiError, DeployInvalidRequestError


class FireworksUnsupportedHardwareError(DeployInvalidRequestError):
    """HTTP 400 — Fireworks rejected an unsupported model/hardware pair.

    Detected on responses whose detail begins with ``"invalid deployment"``
    and contains either:

    - ``"is not supported on"`` (e.g. ``"model type qwen3 is not supported
      on NVIDIA_A100_80GB"``), or
    - ``"requires one of"`` (e.g. ``"model type gpt_oss requires one of:
      NVIDIA_H100_80GB, NVIDIA_B200_180GB, ..."``).

    Both phrasings mean the same thing semantically — the requested
    accelerator is incompatible with the model — and consumers fall back
    to the next accelerator on either.
    """


class FireworksAdapterMismatchError(DeployInvalidRequestError):
    """HTTP 400 — Fireworks rejected an adapter mismatched with its base model.

    Detected on responses whose detail begins with ``"LoRA validation failed"``.
    """


class FireworksConflictError(DeployApiError):
    """HTTP 409 — model resource already exists on Fireworks.

    Concurrent callers that hit ``POST /v1/accounts/{id}/models`` for the
    same model ID race; the loser sees 409. Catch this to wait for the
    winner (e.g. poll :meth:`FireworksDeploymentClient.get_model`) instead
    of retrying.
    """


def classify_fireworks_invalid_request(
    detail: str,
) -> type[DeployInvalidRequestError]:
    """Returns the most specific subclass for a Fireworks 4xx detail string.

    Returns :class:`DeployInvalidRequestError` itself when no signature
    matches — callers treat that as "fall through to the base class." Never
    returns ``None`` and never raises.

    Args:
        detail: The provider-supplied ``detail`` string (already extracted
            from the response body by :func:`raise_api_error`).

    Returns:
        The matching :class:`DeployInvalidRequestError` subclass, or the
        base class when no signature applies.
    """
    if detail.startswith("invalid deployment") and (
        "is not supported on" in detail or "requires one of" in detail
    ):
        return FireworksUnsupportedHardwareError
    if detail.startswith("LoRA validation failed"):
        return FireworksAdapterMismatchError
    return DeployInvalidRequestError
