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

"""Shared utilities for the deploy module."""

import logging
import os
import re
from collections.abc import Callable

import httpx

from oumi.deploy.errors import (
    DeployApiError,
    DeployInvalidRequestError,
    DeployNotFoundError,
    DeployRateLimitError,
    DeployTransientError,
)

logger = logging.getLogger(__name__)

# Callable that refines a 4xx (400/422) detail string to a specific subclass.
# Convention: return DeployInvalidRequestError itself when no provider-specific
# signature matches. Never return None; never raise.
InvalidRequestClassifier = Callable[[str], type[DeployInvalidRequestError]]

_HF_REPO_ID_RE = re.compile(r"^[A-Za-z0-9_.\-]+/[A-Za-z0-9_.\-]+$")


def is_huggingface_repo_id(source: str) -> bool:
    """Returns ``True`` if *source* looks like a HuggingFace ``owner/repo`` ID."""
    return bool(_HF_REPO_ID_RE.match(source))


def is_huggingface_url(source: str) -> bool:
    """Returns ``True`` if *source* is a ``huggingface.co`` URL."""
    return source.startswith(("https://huggingface.co/", "http://huggingface.co/"))


def check_hf_model_accessibility(model_id: str) -> bool:
    """Check whether a HuggingFace model is publicly accessible.

    Makes an unauthenticated call to the HuggingFace Hub API. Returns
    ``True`` if the model can be accessed without a token, ``False`` if it
    is gated, private, or not found.

    This is a best-effort check — network errors are caught and treated as
    "unknown" (returns ``True`` to avoid blocking the user).

    Args:
        model_id: HuggingFace repo ID (e.g., ``Qwen/Qwen2.5-72B-Instruct``).

    Returns:
        ``True`` if the model is publicly accessible, ``False`` otherwise.
    """
    if not is_huggingface_repo_id(model_id):
        return True

    try:
        from huggingface_hub import model_info
        from huggingface_hub.utils import (
            GatedRepoError,
            RepositoryNotFoundError,
        )
    except ImportError:
        logger.debug("huggingface_hub not installed; skipping accessibility check.")
        return True

    try:
        model_info(model_id, token=False)
        return True
    except GatedRepoError:
        return False
    except RepositoryNotFoundError:
        return False
    except Exception:
        logger.debug(
            "Unexpected error checking HF model '%s'; assuming public.",
            model_id,
            exc_info=True,
        )
        return True


def resolve_hf_token(model_access_key: str | None = None) -> str:
    """Resolve the HuggingFace token to use for Parasail API calls.

    Priority:
    1. Explicit *model_access_key* (caller-provided).
    2. ``HF_TOKEN`` environment variable.
    3. Empty string (public models need no token).

    Args:
        model_access_key: An explicit HuggingFace token. Takes highest
            priority when non-empty.

    Returns:
        The resolved token string (may be empty).
    """
    if model_access_key:
        return model_access_key
    return os.environ.get("HF_TOKEN", "")


def warn_if_private_model_missing_token(
    model_id: str,
    resolved_token: str = "",
) -> None:
    """Emit a warning if a HuggingFace model appears private and no token is set.

    Args:
        model_id: HuggingFace repo ID or URL.
        resolved_token: Pre-resolved token (from :func:`resolve_hf_token`).
            If non-empty the check is skipped.
    """
    if resolved_token:
        return

    if not check_hf_model_accessibility(model_id):
        logger.warning(
            "Model '%s' appears to be gated or private on HuggingFace, "
            "but no HuggingFace token was found. Set the HF_TOKEN environment "
            "variable or pass --model-access-key to avoid authentication errors "
            "from Parasail.",
            model_id,
        )


# ---------------------------------------------------------------------------
# HTTP response error helpers
# ---------------------------------------------------------------------------


def raise_api_error(
    response: httpx.Response,
    context: str,
    *,
    classify_4xx: InvalidRequestClassifier | None = None,
) -> None:
    """Extract the human-readable message from a provider error response and raise.

    Provider APIs (Fireworks, Parasail, etc.) return JSON bodies on errors in
    various shapes.  This function tries common fields and falls back to the
    raw response text::

        {"error": {"message": "...", "code": "INVALID_ARGUMENT", ...}}
        {"message": "...", "code": 400}
        {"detail": "..."}

    The HTTP status code selects a typed subclass of
    :class:`~oumi.deploy.errors.DeployApiError`. The raised exception's
    string form contains only ``context`` and ``detail`` — the request
    method, URL, and body are attached as structured attributes and logged
    at DEBUG level so internal attribution never leaks into user-facing
    error messages.

    Provider-specific 4xx sub-types (for example Fireworks'
    ``FireworksUnsupportedHardwareError``) live in the provider's module.
    Callers pass an ``classify_4xx`` function to refine 400/422 responses
    by detail-string signature. By convention the classifier returns
    :class:`DeployInvalidRequestError` itself when no signature matches.

    Args:
        response: The failed HTTP response.
        context: Short description of the operation that failed (used in the
            raised message, e.g. ``"create deployment 'my-ep'"``).
        classify_4xx: Optional provider-specific classifier for 400/422
            responses. Called with the extracted detail string; must return
            a :class:`DeployInvalidRequestError` subclass (or the base class
            itself when no provider-specific signature matches). Only
            consulted for 400/422 — other statuses use the generic mapping.

    Raises:
        DeployNotFoundError: Response status is 404.
        DeployRateLimitError: Response status is 429.
        DeployInvalidRequestError: Response status is 400 or 422. May be
            a more specific subclass when ``classify_4xx`` is provided and
            recognizes the detail.
        DeployTransientError: Response status is 5xx.
        DeployApiError: Any other 4xx status (401/403/409/etc.) — handled
            via the base class since the appropriate disposition depends on
            the specific code and no modeled subclass applies.
        ValueError: ``raise_api_error`` was called on a non-error response
            (status ``< 400``). Indicates a caller bug — :func:`check_response`
            is supposed to gate this path on ``not response.is_success``.
    """
    detail: str
    try:
        body = response.json()
        if isinstance(body, dict):
            if "error" in body:
                err = body["error"]
                detail = (
                    err.get("message", str(err)) if isinstance(err, dict) else str(err)
                )
            elif "message" in body:
                detail = str(body["message"])
            elif "detail" in body:
                detail = str(body["detail"])
            else:
                detail = str(body)
        else:
            detail = str(body)
    except Exception:
        detail = response.text or "(no details)"

    req = response.request
    try:
        req_body = req.content.decode("utf-8", errors="replace") or "(empty)"
    except Exception:
        req_body = "(unreadable)"
    logger.debug(
        "API error for %s %s (HTTP %d): %s",
        req.method,
        req.url,
        response.status_code,
        detail,
    )
    logger.debug("API error request body for %s %s: %s", req.method, req.url, req_body)

    status = response.status_code
    exc_cls: type[DeployApiError]
    if status == 404:
        exc_cls = DeployNotFoundError
    elif status == 429:
        exc_cls = DeployRateLimitError
    elif status in (400, 422):
        exc_cls = classify_4xx(detail) if classify_4xx else DeployInvalidRequestError
    elif status >= 500:
        exc_cls = DeployTransientError
    elif status >= 400:
        # Unclassified 4xx (401/403/409/etc.). Use the base class rather
        # than DeployInvalidRequestError — 401/403 especially are backend
        # config issues, not user-input errors, and shouldn't be routed to
        # InvalidArgumentError-style downstream handling.
        exc_cls = DeployApiError
    else:
        raise ValueError(
            f"raise_api_error called with non-error status {status}; "
            "check_response should gate on response.is_success."
        )

    raise exc_cls(
        detail=detail,
        status_code=status,
        method=req.method,
        url=str(req.url),
        context=context,
    )


def check_response(
    response: httpx.Response,
    context: str,
    *,
    classify_4xx: InvalidRequestClassifier | None = None,
) -> None:
    """Raises a typed :class:`DeployApiError` if the response is not successful.

    Args:
        response: The HTTP response to check.
        context: Short description of the operation (passed to :func:`raise_api_error`).
        classify_4xx: Optional provider-specific 4xx classifier, passed
            through to :func:`raise_api_error`. See that function for the
            convention (return :class:`DeployInvalidRequestError` when no
            signature matches).
    """
    if not response.is_success:
        raise_api_error(response, context, classify_4xx=classify_4xx)
