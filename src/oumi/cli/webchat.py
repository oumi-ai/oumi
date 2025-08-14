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

"""WebChat CLI command for launching the web interface."""

import asyncio
import os
import threading
import time
from typing import Annotated, Optional

import typer

import oumi.cli.cli_utils as cli_utils
from oumi.cli.alias import AliasType, try_get_config_name_for_alias
from oumi.utils.logging import logger


def webchat(
    ctx: typer.Context,
    config: Annotated[
        str,
        typer.Option(
            *cli_utils.CONFIG_FLAGS,
            help="Path to the configuration file for inference.",
        ),
    ],
    host: Annotated[
        str,
        typer.Option("--host", help="Host address for both backend and frontend."),
    ] = "0.0.0.0",
    backend_port: Annotated[
        int,
        typer.Option("--backend-port", help="Port for backend server."),
    ] = 8000,
    frontend_port: Annotated[
        int,
        typer.Option("--frontend-port", help="Port for frontend interface."),
    ] = 7860,
    share: Annotated[
        bool,
        typer.Option("--share", help="Create a public Gradio link."),
    ] = False,
    system_prompt: Annotated[
        Optional[str],
        typer.Option(
            "--system-prompt",
            help="System prompt for task-specific instructions.",
        ),
    ] = None,
    level: cli_utils.LOG_LEVEL_TYPE = None,
):
    """Launch Oumi WebChat - a web-based interface for interactive chat.
    
    This command starts both the backend API server and the frontend web interface,
    providing full access to all Oumi chat features through a browser.
    
    Args:
        ctx: The Typer context object.
        config: Path to the configuration file for inference.
        host: Host address for both servers.
        backend_port: Port for the backend API server.
        frontend_port: Port for the frontend web interface.
        share: Whether to create a public Gradio link.
        system_prompt: System prompt for task-specific instructions.
        level: The logging level for the specified command.
    """
    extra_args = cli_utils.parse_extra_cli_args(ctx)

    config = str(
        cli_utils.resolve_and_fetch_config(
            try_get_config_name_for_alias(config, AliasType.INFER),
        )
    )

    # Delayed imports to avoid loading heavy dependencies unless needed
    from oumi.core.configs import InferenceConfig
    from oumi.webchat.interface import launch_webchat
    from oumi.webchat.server import run_webchat_server
    # End imports

    parsed_config: InferenceConfig = InferenceConfig.from_yaml_and_arg_list(
        config, extra_args, logger=logger
    )
    parsed_config.finalize_and_validate()

    # Print configuration for verification
    parsed_config.print_config(logger)

    # https://stackoverflow.com/questions/62691279/how-to-disable-tokenizers-parallelism-true-false-warning
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Start backend server in a separate thread
    def run_backend():
        """Run the backend server."""
        try:
            run_webchat_server(
                config=parsed_config,
                host=host,
                port=backend_port,
                system_prompt=system_prompt,
            )
        except Exception as e:
            logger.error(f"Backend server error: {e}")
            raise

    backend_thread = threading.Thread(target=run_backend, daemon=True)
    backend_thread.start()

    # Wait a moment for backend to start
    time.sleep(2)

    # Test backend connection
    backend_url = f"http://{host if host != '0.0.0.0' else 'localhost'}:{backend_port}"
    logger.info(f"Testing backend connection at {backend_url}")
    
    try:
        import requests
        response = requests.get(f"{backend_url}/health", timeout=5)
        if response.status_code == 200:
            logger.info("✅ Backend server is running")
        else:
            logger.warning(f"⚠️  Backend server returned status {response.status_code}")
    except Exception as e:
        logger.warning(f"⚠️  Could not connect to backend: {e}")
        logger.info("Proceeding anyway - backend may need more time to start")

    # Launch frontend interface
    try:
        launch_webchat(
            config=parsed_config,
            server_url=backend_url,
            share=share,
            server_name=host,
            server_port=frontend_port,
        )
    except KeyboardInterrupt:
        logger.info("🛑 WebChat stopped by user")
    except Exception as e:
        logger.error(f"Frontend error: {e}")
        raise


def webchat_server(
    ctx: typer.Context,
    config: Annotated[
        str,
        typer.Option(
            *cli_utils.CONFIG_FLAGS,
            help="Path to the configuration file for inference.",
        ),
    ],
    host: Annotated[
        str,
        typer.Option("--host", help="Host address for server."),
    ] = "0.0.0.0",
    port: Annotated[
        int,
        typer.Option("--port", help="Port for server."),
    ] = 8000,
    system_prompt: Annotated[
        Optional[str],
        typer.Option(
            "--system-prompt",
            help="System prompt for task-specific instructions.",
        ),
    ] = None,
    level: cli_utils.LOG_LEVEL_TYPE = None,
):
    """Launch only the WebChat backend server (no frontend).
    
    This is useful for development or when you want to run the frontend separately.
    
    Args:
        ctx: The Typer context object.
        config: Path to the configuration file for inference.
        host: Host address for server.
        port: Port for server.
        system_prompt: System prompt for task-specific instructions.
        level: The logging level for the specified command.
    """
    extra_args = cli_utils.parse_extra_cli_args(ctx)

    config = str(
        cli_utils.resolve_and_fetch_config(
            try_get_config_name_for_alias(config, AliasType.INFER),
        )
    )

    # Delayed imports
    from oumi.core.configs import InferenceConfig
    from oumi.webchat.server import run_webchat_server
    # End imports

    parsed_config: InferenceConfig = InferenceConfig.from_yaml_and_arg_list(
        config, extra_args, logger=logger
    )
    parsed_config.finalize_and_validate()

    # Print configuration for verification
    parsed_config.print_config(logger)

    # https://stackoverflow.com/questions/62691279/how-to-disable-tokenizers-parallelism-true-false-warning
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Run backend server
    run_webchat_server(
        config=parsed_config,
        host=host,
        port=port,
        system_prompt=system_prompt,
    )