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

"""Environment helpers for the Oumi MCP server.

Utilities for detecting missing cloud env vars and stripping
Oumi-specific env overrides that can silently change launcher behaviour.
"""

import logging
import os

logger = logging.getLogger(__name__)

_CLOUD_ENV_VAR_HINTS: dict[str, str] = {
    "WANDB_API_KEY": "Weights & Biases logging",
    "WANDB_PROJECT": "Weights & Biases project name",
    "HF_TOKEN": "HuggingFace token (alternative to ~/.cache/huggingface/token)",
    "COMET_API_KEY": "Comet ML logging",
}


def _build_missing_env_warning(envs: dict[str, str] | None) -> str:
    """Return a warning string listing local env vars that won't reach the remote VM."""
    missing = []
    for var, description in _CLOUD_ENV_VAR_HINTS.items():
        if os.environ.get(var) and (not envs or var not in envs):
            missing.append(f"  - {var} ({description})")
    if not missing:
        return ""
    return (
        "\n\nWARNING: These env vars exist locally but won't be set on the remote VM:\n"
        + "\n".join(missing)
        + '\n  Pass them via the `envs` parameter: envs={"WANDB_API_KEY": "..."}'
    )


_OUMI_ENV_OVERRIDES = ("OUMI_USE_SPOT_VM", "OUMI_FORCE_EDITABLE_INSTALL")


def _strip_oumi_env_overrides() -> None:
    """Remove oumi env vars that silently override launcher config values.

    These are CLI convenience toggles (e.g. "always use spot") that make
    sense for interactive ``oumi launch up`` but break programmatic callers
    like this MCP server — the tool's explicit parameters should be the
    sole source of truth.
    """
    for var in _OUMI_ENV_OVERRIDES:
        val = os.environ.pop(var, None)
        if val:
            logger.info("Stripped inherited env var %s=%r from MCP process", var, val)
