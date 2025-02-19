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

import requests
import typer
import yaml
from requests.exceptions import RequestException

from oumi.cli.cli_utils import resolve_oumi_prefix
from oumi.utils.logging import logger

OUMI_GITHUB_RAW = "https://raw.githubusercontent.com/oumi-ai/oumi/main"
OUMI_DIR = "~/.oumi/configs"


def fetch(
    config_path: Annotated[
        str,
        typer.Argument(
            help="Path to config (e.g. oumi://smollm/inference/135m_infer.yaml)"
        ),
    ],
    output_dir: Annotated[
        Optional[Path],
        typer.Option(
            "--output-dir",
            "-o",
            help=(
                "Directory to save configs "
                "(defaults to OUMI_DIR env var or ~/.oumi/configs)"
            ),
        ),
    ] = None,
    force: Annotated[
        bool, typer.Option("--force", "-f", help="Overwrite existing config if present")
    ] = False,
) -> None:
    """Fetch configuration files from GitHub repository."""
    # Remove oumi:// prefix if present
    config_path, config_dir = resolve_oumi_prefix(config_path, output_dir)

    try:
        # Check destination first
        local_path = (config_dir or Path(OUMI_DIR).expanduser()) / config_path
        if local_path.exists() and not force:
            msg = f"Config already exists at {local_path}. Use --force to overwrite"
            logger.error(msg)
            typer.echo(msg, err=True)
            raise typer.Exit(code=1)

        # Fetch from GitHub
        github_url = f"{OUMI_GITHUB_RAW}/{config_path.lstrip('/')}"
        response = requests.get(github_url)
        response.raise_for_status()
        config_content = response.text

        # Validate YAML
        yaml.safe_load(config_content)

        # Save to destination
        if local_path.exists():
            logger.warning(f"Overwriting existing config at {local_path}")
        local_path.parent.mkdir(parents=True, exist_ok=True)

        with open(local_path, "w") as f:
            f.write(config_content)
        logger.info(f"Successfully downloaded config to {local_path}")

    except RequestException as e:
        logger.error(f"Failed to download config from GitHub: {e}")
        raise typer.Exit(1)
    except yaml.YAMLError:
        logger.error("Invalid YAML configuration")
        raise typer.Exit(1)
