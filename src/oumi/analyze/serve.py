"""HTTP server for the React-based analyze web UI.

This module provides a simple HTTP server that serves:
1. Static React app files from the dist/ folder
2. JSON data from the analyze storage (~/.oumi/analyze/)
"""

import http.server
import json
import logging
import os
import shutil
import socketserver
import tempfile
import threading
import webbrowser
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class AnalyzeUIHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP handler that serves React app and JSON data."""

    def __init__(self, *args, data_dir: Path, **kwargs):
        self.data_dir = data_dir
        super().__init__(*args, **kwargs)

    def do_GET(self):
        """Handle GET requests."""
        # Route /data/* requests to the data directory
        if self.path.startswith("/data/"):
            self.serve_data()
        else:
            # Serve static files
            super().do_GET()

    def serve_data(self):
        """Serve JSON data files from the storage directory."""
        # Extract the path after /data/
        data_path = self.path[6:]  # Remove "/data/" prefix

        # Security: prevent directory traversal
        if ".." in data_path:
            self.send_error(403, "Forbidden")
            return

        file_path = self.data_dir / data_path

        if not file_path.exists():
            self.send_error(404, "File not found")
            return

        if not file_path.is_file():
            self.send_error(403, "Not a file")
            return

        # Serve the file
        try:
            with open(file_path, "rb") as f:
                content = f.read()

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", len(content))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(content)
        except Exception as e:
            logger.error(f"Error serving {file_path}: {e}")
            self.send_error(500, str(e))

    def log_message(self, format: str, *args: Any) -> None:
        """Suppress default logging unless debug is enabled."""
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"{self.address_string()} - {format % args}")


def create_handler(data_dir: Path):
    """Create a handler class with the data directory bound."""

    class BoundHandler(AnalyzeUIHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, data_dir=data_dir, **kwargs)

    return BoundHandler


def prepare_serving_directory(storage_dir: Path) -> Path:
    """Prepare the serving directory with React app and JSON data.

    Args:
        storage_dir: Path to the analyze storage directory (~/.oumi/analyze/)

    Returns:
        Path to the temporary serving directory
    """
    # Create temp directory
    serve_dir = Path(tempfile.mkdtemp(prefix="oumi_analyze_ui_"))

    # Copy React dist files
    web_dist = Path(__file__).parent / "web" / "dist"
    if web_dist.exists():
        # Copy all dist files to serve_dir
        for item in web_dist.iterdir():
            dest = serve_dir / item.name
            if item.is_dir():
                shutil.copytree(item, dest)
            else:
                shutil.copy2(item, dest)
    else:
        # Create a placeholder index.html if dist doesn't exist
        logger.warning(
            f"React dist not found at {web_dist}. "
            "Run 'npm run build' in src/oumi/analyze/web/ to build the UI."
        )
        (serve_dir / "index.html").write_text(
            """<!DOCTYPE html>
<html>
<head><title>Oumi Analyze</title></head>
<body>
<h1>React UI Not Built</h1>
<p>The React UI has not been built yet.</p>
<p>To build it, run:</p>
<pre>
cd src/oumi/analyze/web
npm install
npm run build
</pre>
<p>Then restart <code>oumi analyze view</code>.</p>
</body>
</html>"""
        )

    # Create data directory with symlinks to storage
    data_dir = serve_dir / "data"
    data_dir.mkdir(exist_ok=True)

    # Copy/link index.json
    index_path = storage_dir / "index.json"
    if index_path.exists():
        shutil.copy2(index_path, data_dir / "index.json")
    else:
        # Create empty index
        (data_dir / "index.json").write_text('{"evals": []}')

    # Copy/link evals directory
    evals_dir = storage_dir / "evals"
    if evals_dir.exists():
        dest_evals = data_dir / "evals"
        dest_evals.mkdir(exist_ok=True)
        for eval_file in evals_dir.glob("*.json"):
            shutil.copy2(eval_file, dest_evals / eval_file.name)

    return serve_dir


def serve_ui(
    port: int = 8765,
    host: str = "localhost",
    storage_dir: Path | None = None,
    open_browser: bool = True,
) -> None:
    """Start the HTTP server for the React UI.

    Args:
        port: Port to run the server on.
        host: Host address to bind to.
        storage_dir: Path to analyze storage. Defaults to ~/.oumi/analyze/
        open_browser: Whether to open the browser automatically.
    """
    if storage_dir is None:
        storage_dir = Path.home() / ".oumi" / "analyze"

    # Prepare serving directory
    serve_dir = prepare_serving_directory(storage_dir)
    data_dir = serve_dir / "data"

    # Create handler with data dir
    handler_class = create_handler(data_dir)

    # Change to serve directory
    original_dir = os.getcwd()
    os.chdir(serve_dir)

    try:
        # Allow address reuse
        socketserver.TCPServer.allow_reuse_address = True

        with socketserver.TCPServer((host, port), handler_class) as httpd:
            url = f"http://{host}:{port}"
            print(f"\n✨ Oumi Analyze UI running at: {url}")
            print("Press Ctrl+C to stop.\n")

            # Open browser
            if open_browser:

                def open_browser_delayed():
                    webbrowser.open(url)

                timer = threading.Timer(0.5, open_browser_delayed)
                timer.start()

            # Serve forever
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n👋 Server stopped.")
    finally:
        os.chdir(original_dir)
        # Clean up temp directory
        try:
            shutil.rmtree(serve_dir)
        except Exception as e:
            logger.warning(f"Failed to clean up temp directory: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Serve Oumi Analyze UI")
    parser.add_argument("--port", type=int, default=8765, help="Port to run on")
    parser.add_argument("--host", default="localhost", help="Host to bind to")
    parser.add_argument(
        "--no-browser", action="store_true", help="Don't open browser automatically"
    )
    args = parser.parse_args()

    serve_ui(port=args.port, host=args.host, open_browser=not args.no_browser)
