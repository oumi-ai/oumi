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

"""Modal client for running jobs on Modal.com infrastructure."""

import json
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from subprocess import PIPE, Popen
from threading import Lock, Thread
from typing import Optional

from oumi.core.configs import JobConfig
from oumi.core.launcher import JobState, JobStatus
from oumi.utils.logging import logger


@dataclass
class _ModalJobInfo:
    """Internal tracking info for a Modal job."""

    status: JobStatus
    config: JobConfig
    app_id: Optional[str] = None
    modal_function_call_id: Optional[str] = None
    log_file: Optional[str] = None
    generated_app_path: Optional[str] = None


class ModalClient:
    """A client for running jobs on Modal.com infrastructure.

    This client uses dynamic code generation (Option A) to create Modal apps
    from Oumi JobConfig specifications. Jobs are submitted via the Modal CLI
    and tracked locally.
    """

    # Directory for storing job metadata and generated apps
    _OUMI_MODAL_DIR = ".oumi_modal"
    # Supported GPU types on Modal
    _SUPPORTED_GPUS = {
        "T4",
        "L4",
        "A10",
        "A100",
        "A100-40GB",
        "A100-80GB",
        "L40S",
        "H100",
        "H200",
        "B200",
    }
    # Default timeout for Modal functions (24 hours)
    _DEFAULT_TIMEOUT = 86400

    def __init__(self, working_dir: Optional[str] = None):
        """Initializes a new instance of the ModalClient.

        Args:
            working_dir: Directory to store job metadata and generated apps.
                        Defaults to current directory.
        """
        self._mutex = Lock()
        self._working_dir = Path(working_dir) if working_dir else Path.cwd()
        self._modal_dir = self._working_dir / self._OUMI_MODAL_DIR
        self._modal_dir.mkdir(parents=True, exist_ok=True)

        # In-memory job tracking
        self._jobs: dict[str, _ModalJobInfo] = {}
        self._next_job_id = 0

        # Load any existing job metadata
        self._load_job_metadata()

        # Background worker for polling job status
        self._running = True
        self._worker = Thread(target=self._status_poll_loop, daemon=True)
        self._worker.start()

    def _load_job_metadata(self) -> None:
        """Load job metadata from disk."""
        metadata_file = self._modal_dir / "jobs.json"
        if metadata_file.exists():
            try:
                with open(metadata_file) as f:
                    data = json.load(f)
                    self._next_job_id = data.get("next_job_id", 0)
                    # Note: We don't restore full job objects, just the ID counter
            except (OSError, json.JSONDecodeError) as e:
                logger.warning(f"Failed to load job metadata: {e}")

    def _save_job_metadata(self) -> None:
        """Save job metadata to disk."""
        metadata_file = self._modal_dir / "jobs.json"
        try:
            with open(metadata_file, "w") as f:
                json.dump({"next_job_id": self._next_job_id}, f)
        except OSError as e:
            logger.warning(f"Failed to save job metadata: {e}")

    def _generate_job_id(self) -> str:
        """Generate a unique job ID."""
        job_id = f"modal-{self._next_job_id}-{uuid.uuid4().hex[:8]}"
        self._next_job_id += 1
        self._save_job_metadata()
        return job_id

    def _convert_accelerator_to_modal_gpu(
        self, accelerator: Optional[str]
    ) -> Optional[str]:
        """Convert Oumi accelerator spec to Modal GPU spec.

        Args:
            accelerator: Oumi accelerator string, e.g., "A100", "A100:4", "H100:8"

        Returns:
            Modal GPU string, e.g., "A100", "A100:4", or None if no GPU
        """
        if not accelerator:
            return None

        # Parse accelerator string (e.g., "A100:4" -> ("A100", "4"))
        match = re.match(r"([A-Za-z0-9-]+)(?::(\d+))?", accelerator)
        if not match:
            logger.warning(f"Could not parse accelerator spec: {accelerator}")
            return None

        gpu_type = match.group(1).upper()
        gpu_count = match.group(2)

        # Normalize GPU type names
        gpu_type_map = {
            "V100": None,  # Not supported on Modal
            "A100": "A100",
            "A100-40GB": "A100-40GB",
            "A100-80GB": "A100-80GB",
            "H100": "H100",
            "H200": "H200",
            "L4": "L4",
            "L40S": "L40S",
            "T4": "T4",
            "A10G": "A10",
            "A10": "A10",
            "B200": "B200",
        }

        modal_gpu = gpu_type_map.get(gpu_type)
        if modal_gpu is None:
            logger.warning(
                f"GPU type {gpu_type} is not supported on Modal. "
                f"Supported types: {self._SUPPORTED_GPUS}"
            )
            return None

        if gpu_count:
            return f"{modal_gpu}:{gpu_count}"
        return modal_gpu

    def _convert_memory_to_modal(self, memory: Optional[str]) -> Optional[int]:
        """Convert Oumi memory spec to Modal memory (in MB).

        Args:
            memory: Oumi memory string, e.g., "256GB", "128+"

        Returns:
            Memory in MB for Modal, or None
        """
        if not memory:
            return None

        # Remove any trailing + or modifiers
        memory = memory.rstrip("+")

        # Parse memory value
        match = re.match(r"(\d+(?:\.\d+)?)\s*([KMGT]?B?)?", memory, re.IGNORECASE)
        if not match:
            return None

        value = float(match.group(1))
        unit = (match.group(2) or "GB").upper()

        # Convert to MB
        multipliers = {"B": 1 / (1024 * 1024), "KB": 1 / 1024, "MB": 1, "GB": 1024}
        # Handle cases like "256G" without B
        if unit.endswith("B"):
            multiplier = multipliers.get(unit, 1024)
        else:
            multiplier = multipliers.get(unit + "B", 1024)

        return int(value * multiplier)

    def _generate_modal_app_code(self, job: JobConfig, job_id: str) -> str:
        """Generate Modal app code from JobConfig.

        Args:
            job: The job configuration
            job_id: The unique job ID

        Returns:
            Python code string for the Modal app
        """
        # Convert resources
        gpu_spec = self._convert_accelerator_to_modal_gpu(job.resources.accelerators)
        memory_mb = self._convert_memory_to_modal(job.resources.memory)

        # Build function decorator arguments
        func_args = [f"timeout={self._DEFAULT_TIMEOUT}"]

        if gpu_spec:
            func_args.append(f'gpu="{gpu_spec}"')

        if memory_mb:
            func_args.append(f"memory={memory_mb}")

        if job.resources.cpus:
            try:
                cpu_count = float(job.resources.cpus.rstrip("+"))
                func_args.append(f"cpu={cpu_count}")
            except ValueError:
                pass

        # Build image setup commands
        setup_commands = []
        if job.setup:
            # Split setup into individual commands for run_commands
            for line in job.setup.strip().split("\n"):
                line = line.strip()
                if line and not line.startswith("#"):
                    # Escape quotes in the command
                    escaped_line = line.replace("\\", "\\\\").replace('"', '\\"')
                    setup_commands.append(escaped_line)

        # Generate image definition
        if setup_commands:
            setup_cmd_str = '",\n        "'.join(setup_commands)
            image_def = f"""image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "curl", "wget")
    .run_commands(
        "{setup_cmd_str}"
    )
)"""
        else:
            image_def = (
                'image = modal.Image.debian_slim(python_version="3.11")'
                '.apt_install("git", "curl", "wget")'
            )

        # Add image to function args
        func_args.append("image=image")

        # Handle environment variables
        env_dict_str = ""
        if job.envs:
            env_items = ", ".join(f'"{k}": "{v}"' for k, v in job.envs.items())
            env_dict_str = f"\n    env = {{{env_items}}}"
            func_args.append("secrets=[modal.Secret.from_dict(env)]")

        # Handle volumes/storage mounts
        volumes_setup = ""
        volumes_arg = ""
        if job.storage_mounts:
            volume_defs = []
            volume_mounts = []
            for mount_path, storage in job.storage_mounts.items():
                vol_name = f"oumi-{storage.source.replace('/', '-').replace(':', '-')}"
                vol_var = vol_name.replace("-", "_")
                volume_defs.append(
                    f'{vol_var} = modal.Volume.from_name("{vol_name}", '
                    f"create_if_missing=True)"
                )
                volume_mounts.append(f'"{mount_path}": {vol_var}')

            if volume_defs:
                volumes_setup = "\n".join(volume_defs)
                volumes_arg = f"volumes={{{', '.join(volume_mounts)}}}"
                func_args.append(volumes_arg)

        # Escape the run script for embedding in Python code
        run_script = job.run.replace("\\", "\\\\").replace('"""', '\\"\\"\\"')

        # Build the complete app code
        func_args_str = ",\n    ".join(func_args)

        app_code = f'''#!/usr/bin/env python3
"""Auto-generated Modal app for Oumi job: {job_id}"""

import subprocess
import sys
import os

import modal

app = modal.App("oumi-{job_id}")

{image_def}

{volumes_setup}
{env_dict_str}

@app.function(
    {func_args_str}
)
def run_oumi_job():
    """Execute the Oumi job."""
    print("=" * 60)
    print("Starting Oumi job: {job_id}")
    print("=" * 60)

    # Set working directory if specified
    working_dir = "{job.working_dir or "/root"}"
    if working_dir and os.path.exists(working_dir):
        os.chdir(working_dir)

    # Execute the run script
    run_script = """
{run_script}
"""

    result = subprocess.run(
        run_script,
        shell=True,
        executable="/bin/bash",
        capture_output=False,
    )

    print("=" * 60)
    print(f"Job completed with exit code: {{result.returncode}}")
    print("=" * 60)

    if result.returncode != 0:
        raise RuntimeError(f"Job failed with exit code {{result.returncode}}")

    return {{"status": "completed", "exit_code": result.returncode}}


@app.local_entrypoint()
def main():
    """Local entrypoint for running the job."""
    result = run_oumi_job.remote()
    print(f"Job result: {{result}}")
    return result
'''
        return app_code

    def _run_modal_command(
        self, args: list[str], cwd: Optional[str] = None
    ) -> tuple[int, str, str]:
        """Run a Modal CLI command.

        Args:
            args: Command arguments (without 'modal' prefix)
            cwd: Working directory for the command

        Returns:
            Tuple of (return_code, stdout, stderr)
        """
        cmd = ["modal"] + args
        logger.debug(f"Running Modal command: {' '.join(cmd)}")

        try:
            process = Popen(
                cmd,
                stdout=PIPE,
                stderr=PIPE,
                cwd=cwd,
                text=True,
            )
            stdout, stderr = process.communicate(timeout=600)
            return process.returncode, stdout, stderr
        except Exception as e:
            return 1, "", str(e)

    def submit_job(self, job: JobConfig) -> JobStatus:
        """Submit a job to Modal.

        Args:
            job: The job configuration to submit

        Returns:
            JobStatus with the initial pending status
        """
        with self._mutex:
            job_id = self._generate_job_id()
            job_name = job.name or job_id

            # Generate the Modal app code
            app_code = self._generate_modal_app_code(job, job_id)

            # Write the generated app to a file
            app_dir = self._modal_dir / "apps" / job_id
            app_dir.mkdir(parents=True, exist_ok=True)
            app_path = app_dir / "app.py"

            with open(app_path, "w") as f:
                f.write(app_code)

            logger.info(f"Generated Modal app at: {app_path}")

            # Create log file
            log_file = app_dir / "output.log"

            # Create initial job status
            status = JobStatus(
                name=job_name,
                id=job_id,
                status="SUBMITTING",
                cluster="modal",
                metadata=f"Generated app: {app_path}",
                done=False,
                state=JobState.PENDING,
            )

            # Store job info
            self._jobs[job_id] = _ModalJobInfo(
                status=status,
                config=job,
                log_file=str(log_file),
                generated_app_path=str(app_path),
            )

        # Submit the job in a separate thread to not block
        submit_thread = Thread(
            target=self._submit_job_async, args=(job_id, str(app_path), str(log_file))
        )
        submit_thread.start()

        return status

    def _submit_job_async(self, job_id: str, app_path: str, log_file: str) -> None:
        """Submit a job asynchronously.

        Args:
            job_id: The job ID
            app_path: Path to the generated Modal app
            log_file: Path to the log file
        """
        with self._mutex:
            if job_id not in self._jobs:
                return
            self._jobs[job_id].status.status = "STARTING"
            self._jobs[job_id].status.state = JobState.PENDING

        try:
            # Run the Modal app with --detach to run in background
            with open(log_file, "w") as log_f:
                process = Popen(
                    ["modal", "run", "--detach", app_path],
                    stdout=log_f,
                    stderr=log_f,
                    text=True,
                )
                # Wait for the detach command to complete
                return_code = process.wait(timeout=300)

            if return_code == 0:
                with self._mutex:
                    if job_id in self._jobs:
                        self._jobs[job_id].status.status = "RUNNING"
                        self._jobs[job_id].status.state = JobState.RUNNING
                        self._jobs[
                            job_id
                        ].status.metadata = (
                            f"Job submitted successfully. Logs: {log_file}"
                        )
            else:
                with open(log_file) as f:
                    error_output = f.read()[-1000:]  # Last 1000 chars
                with self._mutex:
                    if job_id in self._jobs:
                        self._jobs[job_id].status.status = "FAILED"
                        self._jobs[job_id].status.state = JobState.FAILED
                        self._jobs[job_id].status.done = True
                        self._jobs[
                            job_id
                        ].status.metadata = f"Job submission failed: {error_output}"

        except Exception as e:
            with self._mutex:
                if job_id in self._jobs:
                    self._jobs[job_id].status.status = "FAILED"
                    self._jobs[job_id].status.state = JobState.FAILED
                    self._jobs[job_id].status.done = True
                    self._jobs[job_id].status.metadata = f"Submission error: {str(e)}"

    def _status_poll_loop(self) -> None:
        """Background loop to poll job status from Modal."""
        while self._running:
            try:
                self._poll_running_jobs()
            except Exception as e:
                logger.debug(f"Error polling job status: {e}")
            time.sleep(10)  # Poll every 10 seconds

    def _poll_running_jobs(self) -> None:
        """Poll status of running jobs."""
        with self._mutex:
            running_jobs = [
                (job_id, info)
                for job_id, info in self._jobs.items()
                if info.status.state == JobState.RUNNING
            ]

        for job_id, info in running_jobs:
            try:
                # Check if the job has completed by looking at Modal app list
                # This is a simplified check - in production, we'd use Modal's API
                return_code, stdout, stderr = self._run_modal_command(
                    ["app", "list", "--json"]
                )

                if return_code == 0 and stdout:
                    try:
                        apps = json.loads(stdout)
                        app_name = f"oumi-{job_id}"
                        app_info = next(
                            (a for a in apps if a.get("name") == app_name), None
                        )

                        if app_info:
                            state = app_info.get("state", "").lower()
                            if state == "deployed":
                                # Job is still running or completed
                                pass
                            elif state in ("stopped", "deleted"):
                                with self._mutex:
                                    if job_id in self._jobs:
                                        self._jobs[job_id].status.status = "COMPLETED"
                                        self._jobs[
                                            job_id
                                        ].status.state = JobState.SUCCEEDED
                                        self._jobs[job_id].status.done = True
                    except json.JSONDecodeError:
                        pass

            except Exception as e:
                logger.debug(f"Error polling job {job_id}: {e}")

    def get_job(self, job_id: str) -> Optional[JobStatus]:
        """Get the status of a specific job.

        Args:
            job_id: The job ID to query

        Returns:
            JobStatus if found, None otherwise
        """
        with self._mutex:
            if job_id in self._jobs:
                return self._jobs[job_id].status
        return None

    def list_jobs(self) -> list[JobStatus]:
        """List all tracked jobs.

        Returns:
            List of JobStatus objects
        """
        with self._mutex:
            return [info.status for info in self._jobs.values()]

    def cancel(self, job_id: str) -> Optional[JobStatus]:
        """Cancel a running job.

        Args:
            job_id: The job ID to cancel

        Returns:
            Updated JobStatus if found, None otherwise
        """
        with self._mutex:
            if job_id not in self._jobs:
                return None

            job_info = self._jobs[job_id]

            # If job is already done, just return status
            if job_info.status.done:
                return job_info.status

        # Try to stop the Modal app
        app_name = f"oumi-{job_id}"
        return_code, stdout, stderr = self._run_modal_command(["app", "stop", app_name])

        with self._mutex:
            if job_id in self._jobs:
                if return_code == 0:
                    self._jobs[job_id].status.status = "CANCELLED"
                    self._jobs[job_id].status.state = JobState.CANCELLED
                    self._jobs[job_id].status.done = True
                    self._jobs[job_id].status.metadata = "Job cancelled by user"
                return self._jobs[job_id].status

        return None

    def get_logs(self, job_id: str) -> Optional[str]:
        """Get logs for a job.

        Args:
            job_id: The job ID to get logs for

        Returns:
            Log content as string, or None if not found
        """
        with self._mutex:
            if job_id not in self._jobs:
                return None

            log_file = self._jobs[job_id].log_file

        if log_file and Path(log_file).exists():
            try:
                with open(log_file) as f:
                    return f.read()
            except OSError:
                pass

        # Try to get logs from Modal
        app_name = f"oumi-{job_id}"
        return_code, stdout, stderr = self._run_modal_command(["app", "logs", app_name])

        if return_code == 0:
            return stdout

        return None

    def shutdown(self) -> None:
        """Shutdown the client and stop background threads."""
        self._running = False
