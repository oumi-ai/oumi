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

import os
import socket
import subprocess
import threading
import time
from typing import Annotated, Optional

import typer

import oumi.cli.cli_utils as cli_utils
from oumi.cli.alias import AliasType, try_get_config_name_for_alias
from oumi.utils.logging import logger


def check_port_availability(port: int) -> tuple[bool, str]:
    """Check if a port is available for use.

    Args:
        port: Port number to check.

    Returns:
        Tuple of (is_available, error_message_if_not_available).
    """
    try:
        # Check if port is already in use
        result = subprocess.run(
            ["lsof", "-i", f":{port}"], capture_output=True, text=True, timeout=5
        )

        if result.returncode == 0 and result.stdout.strip():
            # Port is in use - extract process info
            lines = result.stdout.strip().split("\n")
            if len(lines) > 1:  # Skip header line
                process_line = lines[1].split()
                if len(process_line) >= 2:
                    command = process_line[0]
                    pid = process_line[1]
                    return (
                        False,
                        f"Port {port} is already in use by {command} (PID {pid})",
                    )
            return False, f"Port {port} is already in use"

        # Try to bind to the port to confirm availability
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(("", port))
            return True, ""
        except OSError as e:
            return False, f"Cannot bind to port {port}: {e}"
        finally:
            sock.close()

    except subprocess.TimeoutExpired:
        logger.warning("Port check timed out, assuming port is available")
        return True, ""
    except Exception as e:
        logger.warning(f"Port check failed: {e}, assuming port is available")
        return True, ""


def find_available_port(start_port: int = 9000, max_attempts: int = 100) -> int:
    """Find an available port starting from start_port.

    Args:
        start_port: Port to start checking from.
        max_attempts: Maximum number of ports to try.

    Returns:
        An available port number.

    Raises:
        RuntimeError: If no available port is found.
    """
    for port in range(start_port, start_port + max_attempts):
        is_available, _ = check_port_availability(port)
        if is_available:
            return port

    raise RuntimeError(
        f"Could not find an available port in range {start_port}-{start_port + max_attempts - 1}. "
        f"Please specify a custom port with --backend-port or --port option."
    )


def wait_for_backend_health(
    backend_url: str, timeout: int, check_interval: float = 2.0
) -> bool:
    """Wait for backend to become healthy with async polling.

    Args:
        backend_url: Base URL of the backend server.
        timeout: Maximum time to wait in seconds.
        check_interval: Time between health checks in seconds.

    Returns:
        True if backend becomes healthy, False if timeout.

    Raises:
        Exception: If backend startup fails definitively.
    """
    import time

    import requests

    start_time = time.time()
    # Try multiple endpoints to determine if backend is ready
    test_endpoints = [
        f"{backend_url}/health",
        f"{backend_url}/v1/chat/completions",  # Fallback to main API endpoint
    ]
    attempt = 0

    logger.info(f"⏳ Waiting for backend readiness at {backend_url}")
    logger.info(f"🕐 Timeout: {timeout}s, Check interval: {check_interval}s")

    while time.time() - start_time < timeout:
        attempt += 1
        elapsed = time.time() - start_time

        for endpoint in test_endpoints:
            try:
                if endpoint.endswith("/health"):
                    # Try health endpoint
                    response = requests.get(endpoint, timeout=5.0)
                    if response.status_code == 200:
                        health_data = response.json()
                        logger.info(f"✅ Backend is healthy! (attempt {attempt})")
                        logger.info(
                            f"📊 Health status: {health_data.get('status', 'unknown')}"
                        )
                        return True
                    elif response.status_code == 500:
                        logger.debug("⚠️  Health endpoint error, trying fallback...")
                        continue
                else:
                    # Try chat completions endpoint with a test payload
                    response = requests.post(
                        endpoint,
                        json={
                            "messages": [{"role": "user", "content": "test"}],
                            "max_tokens": 1,
                            "stream": False,
                        },
                        timeout=10.0,
                    )
                    # Accept any response that's not a connection error
                    if response.status_code in [200, 400, 422]:  # API is responding
                        logger.info(
                            f"✅ Backend API is responding! (attempt {attempt}, {elapsed:.1f}s)"
                        )
                        return True
                    else:
                        logger.debug(
                            f"❌ API check failed: HTTP {response.status_code}"
                        )

            except requests.exceptions.ConnectionError:
                # Backend not ready yet - this is expected during startup
                logger.info(f"⏳ Backend starting... ({elapsed:.1f}s/{timeout}s)")
                break  # Try again after sleep
            except requests.exceptions.Timeout:
                logger.debug(f"⚠️  Request timeout (attempt {attempt})")
                continue  # Try next endpoint
            except Exception as e:
                logger.debug(f"⚠️  Request error (attempt {attempt}): {e}")
                continue  # Try next endpoint

        # Wait before next check
        time.sleep(check_interval)

    # Timeout reached
    elapsed = time.time() - start_time
    logger.error(
        f"💥 Backend failed to start within {timeout}s (elapsed: {elapsed:.1f}s)"
    )
    return False


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
    ] = 9000,
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
    backend_timeout: Annotated[
        int,
        typer.Option(
            "--backend-timeout", help="Timeout in seconds to wait for backend startup."
        ),
    ] = 120,
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
        backend_timeout: Timeout in seconds to wait for backend startup.
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

    # Check backend port availability and find alternative if needed
    is_backend_available, backend_error_msg = check_port_availability(backend_port)
    if not is_backend_available:
        logger.warning(f"❌ {backend_error_msg}")
        logger.info("🔍 Searching for an available port...")
        try:
            backend_port = find_available_port(start_port=backend_port)
            logger.info(f"✅ Found available backend port: {backend_port}")
        except RuntimeError as e:
            logger.error(f"💥 Port selection failed: {e}")
            raise typer.Exit(1)
    else:
        logger.info(f"✅ Backend port {backend_port} is available")

    # Check frontend port availability and find alternative if needed
    is_frontend_available, frontend_error_msg = check_port_availability(frontend_port)
    if not is_frontend_available:
        logger.warning(f"❌ {frontend_error_msg}")
        logger.info("🔍 Searching for an available frontend port...")
        try:
            frontend_port = find_available_port(start_port=frontend_port)
            logger.info(f"✅ Found available frontend port: {frontend_port}")
        except RuntimeError as e:
            logger.error(f"💥 Frontend port selection failed: {e}")
            raise typer.Exit(1)
    else:
        logger.info(f"✅ Frontend port {frontend_port} is available")

    # Start backend server in a separate thread
    backend_error = []  # Shared list to capture errors

    def run_backend():
        """Run the backend server."""
        try:
            logger.info(f"🔧 Starting backend server on {host}:{backend_port}")
            run_webchat_server(
                config=parsed_config,
                host=host,
                port=backend_port,
                system_prompt=system_prompt,
            )
        except Exception as e:
            logger.error(f"Backend server error: {e}")
            backend_error.append(str(e))
            raise

    backend_thread = threading.Thread(target=run_backend, daemon=True)
    backend_thread.start()

    # Wait for backend to become healthy with proper polling
    backend_url = f"http://{host if host != '0.0.0.0' else 'localhost'}:{backend_port}"

    # Check for immediate backend startup errors
    time.sleep(1)  # Brief wait to catch immediate errors
    if backend_error:
        raise RuntimeError(f"Backend failed to start: {backend_error[0]}")

    # Poll for backend health with configurable timeout
    backend_ready = wait_for_backend_health(backend_url, backend_timeout)

    if not backend_ready:
        if backend_error:
            raise RuntimeError(f"Backend startup failed: {backend_error[0]}")
        else:
            raise RuntimeError(
                f"Backend did not become healthy within {backend_timeout}s. "
                "Try increasing --backend-timeout or check server logs for errors."
            )

    # Launch React frontend (backend is confirmed healthy)
    logger.info(f"🌐 Launching React WebChat frontend at http://{host}:{frontend_port}")
    logger.info(f"🔗 Connected to backend: {backend_url}")
    
    # Get the path to the frontend directory
    
    # Find the oumi root directory
    current_file = os.path.abspath(__file__)
    oumi_src_dir = os.path.dirname(os.path.dirname(current_file))  # src/oumi/
    oumi_root_dir = os.path.dirname(os.path.dirname(oumi_src_dir))  # oumi/
    frontend_dir = os.path.join(oumi_root_dir, "frontend")
    
    if not os.path.exists(frontend_dir):
        logger.error(f"❌ Frontend directory not found: {frontend_dir}")
        logger.info("💡 Make sure the Next.js frontend is set up in frontend/")
        raise typer.Exit(1)
    
    # Launch Next.js development server
    try:
        
        env = os.environ.copy()
        env["NEXT_PUBLIC_BACKEND_URL"] = backend_url
        
        logger.info(f"🚀 Starting Next.js development server in: {frontend_dir}")
        
        # Use the npm run dev:full command which starts both backend and frontend
        if share:
            logger.warning("⚠️  --share option not supported with React frontend yet")
            
        # Start Next.js frontend only (backend is already running)
        result = subprocess.run(
            ["npm", "run", "dev", "--", "-p", str(frontend_port)],
            cwd=frontend_dir,
            env=env,
            check=True
        )
        
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed to start React frontend: {e}")
        logger.info("💡 Make sure to run 'npm install' in the frontend directory first")
        raise typer.Exit(1)
    except KeyboardInterrupt:
        logger.info("🛑 WebChat stopped by user")
    except FileNotFoundError:
        logger.error("❌ npm not found. Please install Node.js and npm")
        logger.info("💡 Visit: https://nodejs.org/ to install Node.js")
        raise typer.Exit(1)
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
    ] = 9000,
    system_prompt: Annotated[
        Optional[str],
        typer.Option(
            "--system-prompt",
            help="System prompt for task-specific instructions.",
        ),
    ] = None,
    wait_healthy: Annotated[
        bool,
        typer.Option(
            "--wait-healthy", help="Wait for server to become healthy before returning."
        ),
    ] = False,
    health_timeout: Annotated[
        int,
        typer.Option(
            "--health-timeout",
            help="Timeout for health check when --wait-healthy is used.",
        ),
    ] = 60,
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
        wait_healthy: Wait for server to become healthy before returning.
        health_timeout: Timeout for health check when --wait-healthy is used.
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

    # Check port availability and find alternative if needed
    is_port_available, port_error_msg = check_port_availability(port)
    if not is_port_available:
        logger.warning(f"❌ {port_error_msg}")
        logger.info("🔍 Searching for an available port...")
        try:
            port = find_available_port(start_port=port)
            logger.info(f"✅ Found available port: {port}")
        except RuntimeError as e:
            logger.error(f"💥 Port selection failed: {e}")
            raise typer.Exit(1)
    else:
        logger.info(f"✅ Port {port} is available")

    # Run backend server
    if wait_healthy:
        # Start server in background thread and wait for health
        server_error = []

        def run_server():
            try:
                run_webchat_server(
                    config=parsed_config,
                    host=host,
                    port=port,
                    system_prompt=system_prompt,
                )
            except Exception as e:
                logger.error(f"Server startup error: {e}")
                server_error.append(str(e))
                raise

        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()

        # Wait for health
        backend_url = f"http://{host if host != '0.0.0.0' else 'localhost'}:{port}"

        # Check for immediate errors
        time.sleep(1)
        if server_error:
            raise RuntimeError(f"Server failed to start: {server_error[0]}")

        # Poll for health
        server_ready = wait_for_backend_health(backend_url, health_timeout)

        if not server_ready:
            if server_error:
                raise RuntimeError(f"Server startup failed: {server_error[0]}")
            else:
                raise RuntimeError(
                    f"Server did not become healthy within {health_timeout}s. "
                    "Try increasing --health-timeout or check server logs."
                )

        logger.info("🎉 WebChat server is healthy and ready!")

        # Keep server running
        try:
            while server_thread.is_alive():
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("🛑 Server stopped by user")
    else:
        # Run server directly (original behavior)
        run_webchat_server(
            config=parsed_config,
            host=host,
            port=port,
            system_prompt=system_prompt,
        )


# Create webchat subcommand app
webchat_app = typer.Typer(
    help="WebChat interface commands.",
    pretty_exceptions_enable=False,
    no_args_is_help=True
)


# Rename webchat_server to webchat_serve for consistency
def webchat_serve(
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
        typer.Option("--host", help="Host address for backend server."),
    ] = "0.0.0.0",
    port: Annotated[
        int,
        typer.Option("--port", help="Port for backend server."),
    ] = 9000,
    system_prompt: Annotated[
        Optional[str],
        typer.Option(
            "--system-prompt",
            help="System prompt for task-specific instructions.",
        ),
    ] = None,
    level: cli_utils.LOG_LEVEL_TYPE = None,
):
    """Launch Oumi WebChat backend server only.

    This command starts only the backend API server, providing a REST API
    for chat completions and other Oumi features. Use this when you have
    a separate frontend (like Electron) that needs to connect to the backend.

    Args:
        ctx: The Typer context object.
        config: Path to the configuration file for inference.
        host: Host address for the backend server.
        port: Port for the backend API server.
        system_prompt: System prompt for task-specific instructions.
        level: The logging level for the specified command.
    """
    # Call the existing webchat_server function with the same parameters
    webchat_server(
        ctx=ctx,
        config=config,
        host=host,
        port=port,
        system_prompt=system_prompt,
        wait_healthy=False,  # Don't wait for health by default
        health_timeout=60,   # Default health timeout
        level=level,
    )


# Create webchat_launch as an alias to the current webchat function
def webchat_launch(
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
    ] = 9000,
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
    backend_timeout: Annotated[
        int,
        typer.Option(
            "--backend-timeout", help="Timeout in seconds to wait for backend startup."
        ),
    ] = 120,
    level: cli_utils.LOG_LEVEL_TYPE = None,
):
    """Launch Oumi WebChat full-stack - backend + frontend interface.

    This command starts both the backend API server and the frontend web interface,
    providing full access to all Oumi chat features through a browser. This is the
    traditional full-stack deployment mode.

    Args:
        ctx: The Typer context object.
        config: Path to the configuration file for inference.
        host: Host address for both servers.
        backend_port: Port for the backend API server.
        frontend_port: Port for the frontend web interface.
        share: Whether to create a public Gradio link.
        system_prompt: System prompt for task-specific instructions.
        backend_timeout: Timeout in seconds to wait for backend startup.
        level: The logging level for the specified command.
    """
    # Call the original full-stack implementation
    return _original_webchat_fullstack(
        ctx=ctx,
        config=config,
        host=host,
        backend_port=backend_port,
        frontend_port=frontend_port,
        share=share,
        system_prompt=system_prompt,
        backend_timeout=backend_timeout,
        level=level,
    )


# Register subcommands with the webchat app
webchat_app.command("serve", help="Launch backend server only")(webchat_serve)
webchat_app.command("launch", help="Launch full-stack (backend + frontend)")(webchat_launch)


# Store the original webchat function before redefining it
def _original_webchat_fullstack(
    ctx: typer.Context,
    config: str,
    host: str = "0.0.0.0", 
    backend_port: int = 9000,
    frontend_port: int = 7860,
    share: bool = False,
    system_prompt: Optional[str] = None,
    backend_timeout: int = 120,
    level = None,
):
    """The original full-stack webchat implementation."""
    extra_args = cli_utils.parse_extra_cli_args(ctx)

    config = str(
        cli_utils.resolve_and_fetch_config(
            try_get_config_name_for_alias(config, AliasType.INFER),
        )
    )

    # Delayed imports to avoid loading heavy dependencies unless needed
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

    # Check backend port availability and find alternative if needed
    is_backend_available, backend_error_msg = check_port_availability(backend_port)
    if not is_backend_available:
        logger.warning(f"❌ {backend_error_msg}")
        logger.info("🔍 Searching for an available port...")
        try:
            backend_port = find_available_port(start_port=backend_port)
            logger.info(f"✅ Found available backend port: {backend_port}")
        except RuntimeError as e:
            logger.error(f"💥 Port selection failed: {e}")
            raise typer.Exit(1)
    else:
        logger.info(f"✅ Backend port {backend_port} is available")

    # Check frontend port availability and find alternative if needed
    is_frontend_available, frontend_error_msg = check_port_availability(frontend_port)
    if not is_frontend_available:
        logger.warning(f"❌ {frontend_error_msg}")
        logger.info("🔍 Searching for an available frontend port...")
        try:
            frontend_port = find_available_port(start_port=frontend_port)
            logger.info(f"✅ Found available frontend port: {frontend_port}")
        except RuntimeError as e:
            logger.error(f"💥 Frontend port selection failed: {e}")
            raise typer.Exit(1)
    else:
        logger.info(f"✅ Frontend port {frontend_port} is available")

    # Start backend server in a separate thread
    backend_error = []  # Shared list to capture errors

    def run_backend():
        """Run the backend server."""
        try:
            logger.info(f"🔧 Starting backend server on {host}:{backend_port}")
            run_webchat_server(
                config=parsed_config,
                host=host,
                port=backend_port,
                system_prompt=system_prompt,
            )
        except Exception as e:
            logger.error(f"Backend server error: {e}")
            backend_error.append(str(e))
            raise

    backend_thread = threading.Thread(target=run_backend, daemon=True)
    backend_thread.start()

    # Wait for backend to become healthy with proper polling
    backend_url = f"http://{host if host != '0.0.0.0' else 'localhost'}:{backend_port}"

    # Check for immediate backend startup errors
    time.sleep(1)  # Brief wait to catch immediate errors
    if backend_error:
        raise RuntimeError(f"Backend failed to start: {backend_error[0]}")

    # Poll for backend health with configurable timeout
    backend_ready = wait_for_backend_health(backend_url, backend_timeout)

    if not backend_ready:
        if backend_error:
            raise RuntimeError(f"Backend startup failed: {backend_error[0]}")
        else:
            raise RuntimeError(
                f"Backend did not become healthy within {backend_timeout}s. "
                "Try increasing --backend-timeout or check server logs for errors."
            )

    # Launch React frontend (backend is confirmed healthy)
    logger.info(f"🌐 Launching React WebChat frontend at http://{host}:{frontend_port}")
    logger.info(f"🔗 Connected to backend: {backend_url}")
    
    # Get the path to the frontend directory
    
    # Find the oumi root directory
    current_file = os.path.abspath(__file__)
    oumi_src_dir = os.path.dirname(os.path.dirname(current_file))  # src/oumi/
    oumi_root_dir = os.path.dirname(os.path.dirname(oumi_src_dir))  # oumi/
    frontend_dir = os.path.join(oumi_root_dir, "frontend")
    
    if not os.path.exists(frontend_dir):
        logger.error(f"❌ Frontend directory not found: {frontend_dir}")
        logger.info("💡 Make sure the Next.js frontend is set up in frontend/")
        raise typer.Exit(1)
    
    # Launch Next.js development server
    try:
        
        env = os.environ.copy()
        env["NEXT_PUBLIC_BACKEND_URL"] = backend_url
        
        logger.info(f"🚀 Starting Next.js development server in: {frontend_dir}")
        
        # Use the npm run dev:full command which starts both backend and frontend
        if share:
            logger.warning("⚠️  --share option not supported with React frontend yet")
            
        # Start Next.js frontend only (backend is already running)
        result = subprocess.run(
            ["npm", "run", "dev", "--", "-p", str(frontend_port)],
            cwd=frontend_dir,
            env=env,
            check=True
        )
        
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed to start React frontend: {e}")
        logger.info("💡 Make sure to run 'npm install' in the frontend directory first")
        raise typer.Exit(1)
    except KeyboardInterrupt:
        logger.info("🛑 WebChat stopped by user")
    except FileNotFoundError:
        logger.error("❌ npm not found. Please install Node.js and npm")
        logger.info("💡 Visit: https://nodejs.org/ to install Node.js")
        raise typer.Exit(1)
    except Exception as e:
        logger.error(f"Frontend error: {e}")
        raise typer.Exit(1)


def webchat(ctx: typer.Context):
    """Launch WebChat interface for interactive chat with models.
    
    Use 'oumi webchat serve' for backend-only mode (recommended for Electron apps).
    Use 'oumi webchat launch' for full-stack mode (backend + React frontend).
    
    This command now requires a subcommand for clarity. Use:
    - 'serve' for backend-only mode
    - 'launch' for full-stack mode (traditional behavior)
    """
    # Show help since subcommand is required
    print("Error: Missing subcommand.")
    print("")
    print("Available subcommands:")
    print("  serve   - Launch backend server only")
    print("  launch  - Launch full-stack (backend + frontend)")
    print("")
    print("Examples:")
    print("  oumi webchat serve -c config.yaml    # Backend only")
    print("  oumi webchat launch -c config.yaml   # Full stack")
    print("")
    print("Use 'oumi webchat [SUBCOMMAND] --help' for more information.")
    raise typer.Exit(1)
