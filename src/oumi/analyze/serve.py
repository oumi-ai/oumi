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
        elif self.path == "/api/run-tests-only":
            self.run_tests_only()
        elif self.path == "/api/rename":
            self.rename_eval()
        elif self.path == "/api/delete":
            self.delete_eval()
        elif self.path == "/api/upload-dataset":
            self.upload_dataset()
        elif self.path == "/api/suggest":
            self.generate_suggestions()
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

    def upload_dataset(self):
        """Upload a dataset file and return the path."""
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            data = json.loads(body)

            filename = data.get("filename", "uploaded_dataset.jsonl")
            content = data.get("content", "")

            if not content:
                self._send_json({"error": "No file content provided"}, 400)
                return

            # Save to temp directory
            upload_dir = Path(tempfile.gettempdir()) / "oumi_analyze_uploads"
            upload_dir.mkdir(exist_ok=True)

            # Use a unique filename to avoid conflicts
            import uuid

            unique_filename = f"{uuid.uuid4().hex[:8]}_{filename}"
            file_path = upload_dir / unique_filename

            with open(file_path, "w") as f:
                f.write(content)

            logger.info(f"Uploaded dataset to: {file_path}")
            self._send_json({"success": True, "path": str(file_path)})

        except Exception as e:
            logger.error(f"Error uploading dataset: {e}")
            self._send_json({"error": str(e)}, 500)

    def generate_suggestions(self):
        """Generate AI-powered suggestions for analyzers, metrics, and tests."""
        try:
            from oumi.analyze.suggest import (
                generate_suggestions,
                suggestion_response_to_dict,
            )

            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            data = json.loads(body)

            dataset_path = data.get("dataset_path")
            dataset_name = data.get("dataset_name")
            split = data.get("split", "train")
            subset = data.get("subset")
            sample_count = data.get("sample_count", 5)

            if not dataset_path and not dataset_name:
                self._send_json(
                    {"error": "Either dataset_path or dataset_name is required"}, 400
                )
                return

            logger.info(
                f"Generating suggestions for dataset: "
                f"{dataset_path or dataset_name} (samples={sample_count})"
            )

            # Generate suggestions
            response = generate_suggestions(
                dataset_path=dataset_path,
                dataset_name=dataset_name,
                split=split,
                subset=subset,
                sample_count=sample_count,
            )

            # Convert to dict for JSON response
            result = suggestion_response_to_dict(response)

            if response.error:
                self._send_json(result, 500)
            else:
                self._send_json(result)

        except Exception as e:
            logger.error(f"Error generating suggestions: {e}")
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

    def run_tests_only(self):
        """Run tests only, reusing cached analyzer results from a parent eval."""
        try:
            import yaml
            from oumi.analyze.storage import AnalyzeStorage
            from oumi.analyze.testing.engine import TestEngine, TestConfig, TestType
            from oumi.analyze.testing.results import TestSeverity
            from oumi.analyze.config import TestConfigYAML
            
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            data = json.loads(body)

            parent_eval_id = data.get("parent_eval_id")
            yaml_config = data.get("config", "")
            
            if not parent_eval_id:
                self._send_json({"error": "No parent_eval_id provided"}, 400)
                return
            if not yaml_config:
                self._send_json({"error": "No config provided"}, 400)
                return

            # Load parent eval data
            storage = AnalyzeStorage(self.storage_dir)
            parent_eval = storage.load_eval(parent_eval_id)
            if not parent_eval:
                self._send_json({"error": f"Parent eval {parent_eval_id} not found"}, 404)
                return

            # Parse new config to get tests
            new_config = yaml.safe_load(yaml_config)
            new_tests = new_config.get("tests", [])
            
            # Convert test configs to TestConfig objects
            test_configs = []
            for test_data in new_tests:
                test_config = TestConfig(
                    id=test_data.get("id", ""),
                    type=TestType(test_data.get("type", "threshold")),
                    metric=test_data.get("metric", ""),
                    severity=TestSeverity(test_data.get("severity", "medium")),
                    title=test_data.get("title", ""),
                    description=test_data.get("description", ""),
                    operator=test_data.get("operator"),
                    value=test_data.get("value"),
                    condition=test_data.get("condition"),
                    max_percentage=test_data.get("max_percentage"),
                    min_percentage=test_data.get("min_percentage"),
                    min_value=test_data.get("min_value"),
                    max_value=test_data.get("max_value"),
                )
                test_configs.append(test_config)

            # Convert parent analysis results to format expected by test engine
            # The test engine expects dict[str, list[BaseModel]], but we have dict[str, list[dict]]
            # We need to wrap the dicts in a simple object
            from pydantic import BaseModel
            
            class DictWrapper(BaseModel):
                """Wrapper to make dict accessible as BaseModel."""
                class Config:
                    extra = "allow"
            
            analysis_results = {}
            for analyzer_name, results in parent_eval.analysis_results.items():
                analysis_results[analyzer_name] = [DictWrapper(**r) for r in results]

            # Run tests
            engine = TestEngine(test_configs)
            test_summary = engine.run(analysis_results)

            # Create new eval with same analysis results but new tests
            new_eval_name = new_config.get("eval_name", f"{parent_eval.metadata.name}_v2")
            
            # Save the new eval
            new_eval_id = storage.save_eval(
                name=new_eval_name,
                config=new_config,
                analysis_results=parent_eval.analysis_results,
                test_results=test_summary.to_dict(),
                conversations=parent_eval.conversations,
            )

            # Refresh index
            self._refresh_storage_index(self.storage_dir)

            self._send_json({
                "status": "completed",
                "eval_id": new_eval_id,
                "message": "Tests re-run successfully with cached analyzer results",
                "tests_passed": test_summary.passed_tests,
                "tests_total": test_summary.total_tests,
            })

        except Exception as e:
            logger.error(f"Error running tests only: {e}")
            import traceback
            traceback.print_exc()
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
