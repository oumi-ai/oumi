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

These subclass :class:`oumi.deploy.errors.DeployInvalidRequestError` so
consumers can still catch the generic base, but they expose Fireworks-only
failure modes that are detected via Fireworks-specific detail strings.

Add new Fireworks error types here as they're discovered. Each new type
needs:

1. A subclass of :class:`DeployInvalidRequestError` below.
2. A signature check in :func:`classify_fireworks_invalid_request`.
3. One unit test for the classifier (detail string → class) in
   ``tests/unit/deploy/test_fireworks_errors.py``.

The classifier is wired into :class:`FireworksDeploymentClient` via the
``classify_4xx`` hook on :func:`oumi.deploy.utils.check_response`, so no
shared-module changes are required when extending.
"""

from oumi.deploy.errors import DeployInvalidRequestError


class FireworksUnsupportedHardwareError(DeployInvalidRequestError):
    """HTTP 400 — Fireworks rejected an unsupported model/hardware pair.

    Detected on responses whose detail begins with ``"invalid deployment"``
    and contains ``"is not supported on"``.
    """


class FireworksAdapterMismatchError(DeployInvalidRequestError):
    """HTTP 400 — Fireworks rejected an adapter mismatched with its base model.

    Detected on responses whose detail begins with ``"LoRA validation failed"``.
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
    if detail.startswith("invalid deployment") and "is not supported on" in detail:
        return FireworksUnsupportedHardwareError
    if detail.startswith("LoRA validation failed"):
        return FireworksAdapterMismatchError
    return DeployInvalidRequestError
