from pathlib import Path
from typing import Annotated, Optional

import requests
import typer
import yaml

from oumi.cli.cli_utils import resolve_oumi_prefix
from oumi.utils.logging import logger

OUMI_GITHUB_RAW = "https://raw.githubusercontent.com/oumi-ai/oumi/main/configs/recipes"
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
) -> None:
    """Fetch configuration files from GitHub repository."""
    # Remove oumi:// prefix if present
    if config_path.startswith("oumi://"):
        config_path, config_dir = resolve_oumi_prefix(config_path, output_dir)

    else:
        # raise error
        logger.error("Invalid config path")
        raise typer.Exit(1)

    try:
        # Fetch from GitHub
        github_url = f"{OUMI_GITHUB_RAW}/{config_path}"
        response = requests.get(github_url)
        response.raise_for_status()
        config_content = response.text

        # Validate YAML
        yaml.safe_load(config_content)

        # Save to destination
        local_path = (config_dir or Path(OUMI_DIR).expanduser()) / config_path
        local_path.parent.mkdir(parents=True, exist_ok=True)

        with open(local_path, "w") as f:
            f.write(config_content)

        logger.info(f"Successfully downloaded config to {local_path}")

    except requests.RequestException as e:
        logger.error(f"Failed to download config from GitHub: {e}")
        raise typer.Exit(1)
    except yaml.YAMLError:
        logger.error("Invalid YAML configuration")
        raise typer.Exit(1)
