"""HTTP server for the React-based analyze web UI.

This module provides a simple HTTP server that serves:
1. Static React app files from the dist/ folder
2. JSON data from the analyze storage (~/.oumi/analyze/)
3. API endpoints for running analysis and tracking progress
"""

import http.server
import json
import logging
import os
import shutil
import socketserver
import subprocess
import tempfile
import threading
import time
import uuid
import webbrowser
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# In-memory job storage
@dataclass
class AnalysisJob:
    """Tracks a running analysis job."""
    id: str
    status: str  # 'pending', 'running', 'completed', 'failed'
    config_path: str
    output_path: str
    progress: int = 0
    total: int = 100
    message: str = ""
    error: str | None = None
    eval_id: str | None = None
    log_lines: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

# Global job registry
_jobs: dict[str, AnalysisJob] = {}


class AnalyzeUIHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP handler that serves React app and JSON data."""

    def __init__(self, *args, data_dir: Path, storage_dir: Path, **kwargs):
        self.data_dir = data_dir
        self.storage_dir = storage_dir
        super().__init__(*args, **kwargs)

    def do_OPTIONS(self):
        """Handle CORS preflight requests."""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        """Handle GET requests."""
        # Route /data/* requests to the data directory
        if self.path.startswith("/data/"):
            self.serve_data()
        elif self.path.startswith("/api/jobs/"):
            self.get_job_status()
        elif self.path == "/api/jobs":
            self.list_jobs()
        else:
            # Serve static files
            super().do_GET()

    def do_POST(self):
        """Handle POST requests."""
        if self.path == "/api/run":
            self.start_analysis()
        elif self.path == "/api/rename":
            self.rename_eval()
        elif self.path == "/api/delete":
            self.delete_eval()
        else:
            self.send_error(404, "Not found")

    def delete_eval(self):
        """Delete an eval."""
        try:
            from oumi.analyze.storage import AnalyzeStorage

            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            data = json.loads(body)

            eval_id = data.get("eval_id", "")

            if not eval_id:
                self._send_json({"error": "eval_id required"}, 400)
                return

            storage = AnalyzeStorage()
            success = storage.delete_eval(eval_id)

            if success:
                # Also update the served copy
                self._refresh_storage_index(storage.base_dir)
                # Remove the eval file from served directory
                eval_file = self.data_dir / "evals" / f"{eval_id}.json"
                if eval_file.exists():
                    eval_file.unlink()
                self._send_json({"success": True, "eval_id": eval_id})
            else:
                self._send_json({"error": "Eval not found"}, 404)

        except Exception as e:
            logger.error(f"Error deleting eval: {e}")
            self._send_json({"error": str(e)}, 500)

    def rename_eval(self):
        """Rename an eval."""
        try:
            from oumi.analyze.storage import AnalyzeStorage

            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            data = json.loads(body)

            eval_id = data.get("eval_id", "")
            new_name = data.get("name", "")

            if not eval_id or not new_name:
                self._send_json({"error": "eval_id and name required"}, 400)
                return

            storage = AnalyzeStorage()
            success = storage.rename_eval(eval_id, new_name)

            if success:
                # Also update the served copy
                self._refresh_storage_index(storage.base_dir)
                self._send_json({"success": True, "eval_id": eval_id, "name": new_name})
            else:
                self._send_json({"error": "Eval not found"}, 404)

        except Exception as e:
            logger.error(f"Error renaming eval: {e}")
            self._send_json({"error": str(e)}, 500)

    def _send_json(self, data: dict, status: int = 200):
        """Send a JSON response."""
        content = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(content))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(content)

    def start_analysis(self):
        """Start a new analysis job."""
        try:
            # Read request body
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            data = json.loads(body)

            yaml_config = data.get("config", "")
            if not yaml_config:
                self._send_json({"error": "No config provided"}, 400)
                return

            # Create job
            job_id = str(uuid.uuid4())[:8]
            
            # Save config to temp file
            config_dir = Path(tempfile.gettempdir()) / "oumi_analyze_jobs"
            config_dir.mkdir(exist_ok=True)
            config_path = config_dir / f"{job_id}.yaml"
            config_path.write_text(yaml_config)

            # Parse output_path from config
            output_path = "./analysis_output"
            for line in yaml_config.split("\n"):
                if line.startswith("output_path:"):
                    output_path = line.split(":", 1)[1].strip()
                    break

            job = AnalysisJob(
                id=job_id,
                status="pending",
                config_path=str(config_path),
                output_path=output_path,
                message="Starting analysis...",
            )
            _jobs[job_id] = job

            # Start analysis in background thread
            thread = threading.Thread(
                target=self._run_analysis_job,
                args=(job, self.storage_dir),
                daemon=True,
            )
            thread.start()

            self._send_json({
                "job_id": job_id,
                "status": "pending",
                "message": "Analysis job created",
            })

        except Exception as e:
            logger.error(f"Error starting analysis: {e}")
            self._send_json({"error": str(e)}, 500)

    def _run_analysis_job(self, job: AnalysisJob, storage_dir: Path):
        """Run the analysis job in a background thread."""
        try:
            job.status = "running"
            job.message = "Running analysis..."

            # Run the oumi analyze command
            cmd = [
                "oumi", "analyze",
                "--config", job.config_path,
                "--typed",
            ]
            
            logger.info(f"Running: {' '.join(cmd)}")
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            # Read output line by line
            for line in iter(process.stdout.readline, ""):
                line = line.rstrip()
                job.log_lines.append(line)
                
                # Parse progress from output
                if "Analyzing" in line and "/" in line:
                    # Try to parse "Analyzing 5/100..." style output
                    try:
                        parts = line.split()
                        for part in parts:
                            if "/" in part:
                                nums = part.split("/")
                                if len(nums) == 2 and nums[0].isdigit() and nums[1].isdigit():
                                    job.progress = int(nums[0])
                                    job.total = int(nums[1])
                                    break
                    except (ValueError, IndexError):
                        pass
                
                # Update message with latest line
                if line:
                    job.message = line[-100:]  # Truncate long lines

            process.wait()

            if process.returncode == 0:
                job.status = "completed"
                job.progress = job.total
                job.message = "Analysis completed successfully!"
                
                # Refresh the index.json to include new eval
                self._refresh_storage_index(storage_dir)
                
                # Try to find the eval ID from output path
                output_dir = Path(job.output_path)
                if output_dir.exists():
                    # Look for the eval in storage
                    index_path = storage_dir / "index.json"
                    if index_path.exists():
                        index_data = json.loads(index_path.read_text())
                        evals = index_data.get("evals", [])
                        if evals:
                            # Get the most recent eval
                            job.eval_id = evals[0].get("id")
            else:
                job.status = "failed"
                job.error = f"Analysis failed with exit code {process.returncode}"
                job.message = job.error

        except Exception as e:
            logger.error(f"Error running analysis job {job.id}: {e}")
            job.status = "failed"
            job.error = str(e)
            job.message = f"Error: {e}"

    def _refresh_storage_index(self, storage_dir: Path):
        """Refresh the serving directory with latest files from storage."""
        try:
            # Copy updated index.json
            index_src = storage_dir / "index.json"
            index_dst = self.data_dir / "index.json"
            if index_src.exists():
                shutil.copy2(index_src, index_dst)
            
            # Copy any new eval files
            evals_src = storage_dir / "evals"
            evals_dst = self.data_dir / "evals"
            if evals_src.exists():
                evals_dst.mkdir(exist_ok=True)
                for eval_file in evals_src.glob("*.json"):
                    shutil.copy2(eval_file, evals_dst / eval_file.name)
                    
            logger.info("Refreshed storage index")
        except Exception as e:
            logger.error(f"Error refreshing storage index: {e}")

    def get_job_status(self):
        """Get the status of a specific job."""
        # Extract job ID from path: /api/jobs/{job_id}
        job_id = self.path.split("/")[-1]
        
        job = _jobs.get(job_id)
        if not job:
            self._send_json({"error": "Job not found"}, 404)
            return

        self._send_json({
            "id": job.id,
            "status": job.status,
            "progress": job.progress,
            "total": job.total,
            "message": job.message,
            "error": job.error,
            "eval_id": job.eval_id,
            "log_lines": job.log_lines[-50:],  # Last 50 lines
        })

    def list_jobs(self):
        """List all jobs."""
        jobs_list = [
            {
                "id": job.id,
                "status": job.status,
                "progress": job.progress,
                "total": job.total,
                "message": job.message,
            }
            for job in _jobs.values()
        ]
        self._send_json({"jobs": jobs_list})

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


def create_handler(data_dir: Path, storage_dir: Path):
    """Create a handler class with the data and storage directories bound."""

    class BoundHandler(AnalyzeUIHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, data_dir=data_dir, storage_dir=storage_dir, **kwargs)

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

    # Create handler with data dir and storage dir
    handler_class = create_handler(data_dir, storage_dir)

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
