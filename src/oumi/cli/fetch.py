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

from pathlib import Path
from typing import Annotated, Optional

import typer

from oumi.cli.cli_utils import _OUMI_PREFIX, resolve_and_fetch_config
from oumi.utils.logging import logger

OUMI_GITHUB_RAW = "https://raw.githubusercontent.com/oumi-ai/oumi/main"


def fetch(
    config_path: Annotated[
        str,
        typer.Argument(
            help="Path to config "
            "(e.g. oumi://configs/recipes/smollm/inference/135m_infer.yaml)"
        ),
    ],
    output_dir: Annotated[
        Optional[Path],
        typer.Option(
            "--output-dir",
            "-o",
            help=(
                "Directory to save configs "
                "(defaults to OUMI_DIR env var or ~/.oumi/fetch)"
            ),
        ),
    ] = None,
    force: Annotated[
        bool, typer.Option("--force", "-f", help="Overwrite existing config if present")
    ] = False,
) -> None:
    """Fetch configuration files from GitHub repository."""
    if not config_path.lower().startswith(_OUMI_PREFIX):
        logger.info(f"Prepending {_OUMI_PREFIX} to config path")
        config_path = _OUMI_PREFIX + config_path
    _ = resolve_and_fetch_config(config_path, output_dir, force)
