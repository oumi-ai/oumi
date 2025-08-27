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

"""DEPRECATED: Legacy interface module for Gradio WebChat.

This module is deprecated and kept for backwards compatibility only.
The new WebChat interface is a React + Next.js application located in frontend/
"""

import os
import subprocess
import sys
from pathlib import Path

from oumi.core.configs import InferenceConfig
from oumi.utils.logging import logger


def create_webchat_interface(
    config: InferenceConfig, server_url: str = "http://localhost:9000"
):
    """DEPRECATED: Create webchat interface.
    
    This function is deprecated. The new WebChat interface is a React + Next.js
    application that should be launched using the CLI command or npm scripts.
    
    Args:
        config: Inference configuration.
        server_url: URL of the WebChat server.
        
    Raises:
        DeprecationWarning: This function is deprecated.
    """
    import warnings
    warnings.warn(
        "create_webchat_interface is deprecated. Use the React frontend in frontend/ "
        "or the 'oumi webchat' CLI command instead.",
        DeprecationWarning,
        stacklevel=2
    )
    logger.warning("⚠️  create_webchat_interface is deprecated")
    logger.info("💡 Use the React frontend in frontend/ or 'oumi webchat' CLI command")


def launch_webchat(
    config: InferenceConfig,
    server_url: str = "http://localhost:9000",
    share: bool = False,
    server_name: str = "0.0.0.0",
    server_port: int = 3000,
):
    """DEPRECATED: Launch the WebChat interface.
    
    This function now launches the new React + Next.js frontend instead of Gradio.
    
    Args:
        config: Inference configuration.
        server_url: URL of the WebChat server.
        share: Whether to create a public link (not supported in React version).
        server_name: Server hostname.
        server_port: Server port.
        
    Raises:
        RuntimeError: If the React frontend cannot be started.
    """
    import warnings
    warnings.warn(
        "launch_webchat is deprecated. Use the 'oumi webchat' CLI command instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    logger.warning("⚠️  launch_webchat is deprecated")
    logger.info("💡 Starting React frontend as fallback...")
    
    if share:
        logger.warning("⚠️  --share option not supported with React frontend")
    
    # Find the frontend directory
    current_file = Path(__file__).resolve()
    oumi_root = current_file.parents[4]  # Go up: webchat -> oumi -> src -> oumi -> root
    frontend_dir = oumi_root / "frontend"
    
    if not frontend_dir.exists():
        raise RuntimeError(
            f"React frontend not found at {frontend_dir}. "
            "Please set up the Next.js frontend first."
        )
    
    # Set environment variables for the frontend
    env = os.environ.copy()
    env["NEXT_PUBLIC_BACKEND_URL"] = server_url
    
    logger.info(f"🚀 Launching React WebChat at http://{server_name}:{server_port}")
    logger.info(f"📁 Frontend directory: {frontend_dir}")
    
    try:
        # Start Next.js development server
        subprocess.run(
            ["npm", "run", "dev", "--", "--port", str(server_port), "--hostname", server_name],
            cwd=frontend_dir,
            env=env,
            check=True
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to start React frontend: {e}")
    except FileNotFoundError:
        raise RuntimeError(
            "npm not found. Please install Node.js and npm. "
            "Visit https://nodejs.org/ for installation instructions."
        )