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

logger = logging.getLogger(__name__)

_HF_REPO_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.\-]+$")


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
            f"Unexpected error checking HF model '{model_id}'; assuming public.",
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
            f"Model '{model_id}' appears to be gated or private on HuggingFace, "
            "but no HuggingFace token was found. Set the HF_TOKEN environment "
            "variable or pass --model-access-key to avoid authentication errors "
            "from Parasail."
        )
