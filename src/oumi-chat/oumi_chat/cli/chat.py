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

"""Chat CLI command for Oumi Chat package."""

from typing import Annotated, Optional

import typer

import oumi.cli.cli_utils as cli_utils


def chat(
    ctx: typer.Context,
    config: Annotated[
        str,
        typer.Option(
            *cli_utils.CONFIG_FLAGS,
            help="Path to the configuration file for inference.",
        ),
    ],
    image: Annotated[
        Optional[str],
        typer.Option(
            "--image",
            help=(
                "File path or URL of an input image to be used with image+text VLLMs."
            ),
        ),
    ] = None,
    system_prompt: Annotated[
        Optional[str],
        typer.Option(
            "--system-prompt",
            help="System prompt for task-specific instructions.",
        ),
    ] = None,
    level: cli_utils.LOG_LEVEL_TYPE = None,
):
    """Start an interactive chat session with a model.

    Args:
        ctx: The Typer context object.
        config: Path to the configuration file for inference.
        image: Path to the input image for `image+text` VLLMs.
        system_prompt: System prompt for task-specific instructions.
        level: The logging level for the specified command.
    """
    # Import here to avoid circular dependencies
    from oumi.cli.infer import infer

    # Call infer with interactive=True
    return infer(
        ctx=ctx,
        config=config,
        interactive=True,  # Always interactive for chat
        server_mode=False,
        host="0.0.0.0",
        port=8000,
        image=image,
        system_prompt=system_prompt,
        level=level,
    )
