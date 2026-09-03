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

"""Shared helpers for the HealthBench consolidation scripts.

The OpenAI key on this machine lives in the repo-root `.env` as
`INTERNAL_OPENAI_API_KEY`, not as the `OPENAI_API_KEY` the SDK looks for by
default, and `.env` is not sourced automatically. `load_api_key` bridges that
gap so the scripts work whether or not the caller exported the variable.
"""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_KEY_VARNAME = "INTERNAL_OPENAI_API_KEY"


def repo_root() -> Path:
    """Returns the repository root (four levels above this file)."""
    return Path(__file__).resolve().parents[4]


def load_dotenv_value(varname: str) -> str | None:
    """Reads one variable from the repo-root .env without importing dotenv."""
    dotenv = repo_root() / ".env"
    if not dotenv.exists():
        return None
    for line in dotenv.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        if name.strip() == varname:
            return value.strip().strip('"').strip("'")
    return None


def load_api_key(varname: str = DEFAULT_KEY_VARNAME) -> str:
    """Returns the OpenAI API key from the environment or the repo-root .env."""
    key = os.environ.get(varname) or load_dotenv_value(varname)
    if not key:
        raise RuntimeError(
            f"No API key found. Export {varname} or add it to {repo_root() / '.env'}."
        )
    return key
