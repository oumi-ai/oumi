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

"""Prompt loading utilities for the onboard wizard.

This module provides functions to load and render Jinja templates for LLM prompts.
"""

from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, Template

# Directory containing prompt templates
_PROMPTS_DIR = Path(__file__).parent

# Jinja environment with file system loader
_env = Environment(
    loader=FileSystemLoader(str(_PROMPTS_DIR)),
    autoescape=False,
    trim_blocks=True,
    lstrip_blocks=True,
)


def load_prompt(name: str, **kwargs: Any) -> str:
    """Load and render a prompt template.

    Args:
        name: Name of the prompt template (without .jinja extension).
        **kwargs: Variables to pass to the template.

    Returns:
        Rendered prompt string.

    Raises:
        FileNotFoundError: If the template file does not exist.
    """
    template = _env.get_template(f"{name}.jinja")
    return template.render(**kwargs)


def get_template(name: str) -> Template:
    """Get a Jinja template object for more advanced rendering.

    Args:
        name: Name of the prompt template (without .jinja extension).

    Returns:
        Jinja Template object.

    Raises:
        FileNotFoundError: If the template file does not exist.
    """
    return _env.get_template(f"{name}.jinja")


__all__ = ["load_prompt", "get_template"]
