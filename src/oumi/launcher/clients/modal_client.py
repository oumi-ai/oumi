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

"""Client wrapping the Modal SDK for the Oumi launcher.

Modal (modal.com) is a serverless GPU platform. There is no long-lived
``cluster`` concept — every job is a ``FunctionCall``. We model each
``FunctionCall`` as a single-job cluster: the cluster name is the
``FunctionCall.object_id`` and ``down()`` cancels the call if still pending.

Image, GPU, and secrets are derived from the ``JobConfig`` at launch time.
``setup`` is baked into the image (content-addressed cache) and ``run`` is
executed by a remote subprocess inside the function.
"""

from __future__ import annotations

import io
import re
import shlex
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, cast

from oumi.core.configs import JobConfig
from oumi.core.launcher import ClusterNotFoundError, JobState, JobStatus
from oumi.utils.logging import logger

if TYPE_CHECKING:
    import modal


_DEFAULT_TIMEOUT_S = 24 * 60 * 60  # 24h


def _import_modal() -> Any:
    """Imports the modal SDK lazily to avoid hard-importing it at module load."""
    import modal  # noqa: PLC0415

    return modal


class ModalLogStream(io.TextIOBase):
    """Wraps a Modal log iterator into a ``readline()``-capable stream."""

    def __init__(self, iterator: Iterator[str]):
        """Initializes a new instance of the ModalLogStream class."""
        self._iterator = iterator

    def readline(self) -> str:  # noqa: D401
        """Reads the next chunk from the wrapped iterator."""
        for chunk in self._iterator:
            if chunk is None:
                return ""
            return chunk
        return ""


def _build_image(modal_lib: Any, job: JobConfig) -> Any:
    """Builds a ``modal.Image`` from the JobConfig.

    ``resources.image_id`` (``docker:<ref>``) → ``Image.from_registry(<ref>)``.
    Otherwise we start from a generic Python base image with ``apt`` tooling
    baked in for the common setup paths.

    Note: ``job.setup`` is intentionally NOT baked into the image. Modal's
    image build runs without secrets attached, so any setup step that
    consumes ``$HF_TOKEN`` (e.g. ``hf download``) or other env-derived
    credentials would fail at build time. We instead concatenate ``setup``
    and ``run`` and execute them together inside the function body, where
    secrets are present and the SkyPilot-compatible script can run with
    ``set -e`` semantics. Modal's apt/pip caches still amortize across
    jobs that reuse the same base image.
    """
    if job.resources.image_id and str(job.resources.image_id).startswith("docker:"):
        return modal_lib.Image.from_registry(
            job.resources.image_id.removeprefix("docker:")
        )
    return (
        modal_lib.Image.debian_slim()
        .apt_install("zip", "curl", "git")
        .pip_install("uv", "awscli")
    )


_SUDO_RE = re.compile(r"\bsudo\s+")


def _strip_sudo(script: str) -> str:
    """Strips ``sudo`` invocations from a shell script.

    Modal containers run as root and don't ship ``sudo``. SkyPilot setup
    scripts authored for cloud VMs (Lambda, Nebius, GCP) frequently call
    ``sudo apt-get …`` directly or chained after ``&&``/``;``/``||``.
    Stripping the token in-place lets the same script run unchanged
    inside a Modal function.
    """
    return _SUDO_RE.sub("", script)


def _build_secret(modal_lib: Any, envs: dict[str, str]) -> Any | None:
    """Bundles the JobConfig env dict into a single ``modal.Secret``."""
    if not envs:
        return None
    return modal_lib.Secret.from_dict({k: str(v) for k, v in envs.items()})


_LAUNCHER_APP_NAME = "oumi-launcher"


def _sandbox_state(sandbox: Any) -> JobState:
    """Maps a ``modal.Sandbox`` poll result to a :class:`JobState`."""
    rc = sandbox.poll()
    if rc is None:
        # ``poll()`` returns None while the sandbox is still running. Modal
        # transitions through internal pending → running states which we
        # collapse into RUNNING for caller-facing reporting.
        return JobState.RUNNING
    if rc == 0:
        return JobState.SUCCEEDED
    return JobState.FAILED


class ModalClient:
    """A wrapped client for communicating with Modal."""

    def __init__(self) -> None:
        """Initializes a new instance of the ModalClient class."""
        self._modal = _import_modal()

    # ----- launch / lifecycle -----

    def launch(
        self, job: JobConfig, cluster_name: str | None = None, **kwargs: Any
    ) -> JobStatus:
        """Creates a detached ``modal.Sandbox`` for the provided job.

        ``cluster_name`` is ignored — Modal generates a stable opaque
        ``Sandbox.object_id`` returned as the cluster name on the
        resulting JobStatus.

        We use ``Sandbox`` (not ``Function.spawn``) because sandboxes
        persist beyond the Python process that creates them, which is
        the lifecycle our launcher pattern needs.
        """
        modal_lib = self._modal
        image = _build_image(modal_lib, job)
        secret = _build_secret(modal_lib, job.envs)

        gpu = job.resources.accelerators
        timeout = int(kwargs.get("timeout", _DEFAULT_TIMEOUT_S))

        # ``setup`` runs inside the sandbox (not at image-build time) so
        # secrets injected via ``modal.Secret`` are visible.
        cleaned_setup = _strip_sudo(job.setup) if job.setup else ""
        full_script = (
            f"set -e\n{cleaned_setup}\n{job.run}" if cleaned_setup else job.run
        )

        # ``App.lookup`` returns a persistent app reference; sandboxes
        # don't require an active ``with app.run()`` context.
        app = modal_lib.App.lookup(_LAUNCHER_APP_NAME, create_if_missing=True)

        sandbox = modal_lib.Sandbox.create(
            "/bin/bash",
            "-lc",
            full_script,
            app=app,
            image=image,
            gpu=gpu,
            secrets=[secret] if secret else [],
            timeout=timeout,
        )
        sandbox_id = sandbox.object_id

        logger.info(
            f"Launched Modal sandbox={sandbox_id} gpu={gpu} timeout={timeout}s"
        )
        return JobStatus(
            name=job.name or sandbox_id,
            id=sandbox_id,
            cluster=sandbox_id,
            status=str(JobState.PENDING.value),
            metadata=shlex.quote(_LAUNCHER_APP_NAME),
            done=False,
            state=JobState.PENDING,
            cost_per_hour=self.estimate_cost_per_hour(gpu),
        )

    def get_call(self, call_id: str) -> Any:
        """Resolves a ``Sandbox`` by its opaque ID, raising if missing."""
        modal_lib = self._modal
        try:
            return modal_lib.Sandbox.from_id(call_id)
        except Exception as e:  # noqa: BLE001
            raise ClusterNotFoundError(
                f"Modal sandbox '{call_id}' not found"
            ) from e

    def get_status(self, call_id: str) -> JobStatus:
        """Returns the current :class:`JobStatus` for ``call_id``."""
        sandbox = self.get_call(call_id)
        state = _sandbox_state(sandbox)
        return JobStatus(
            name=call_id,
            id=call_id,
            cluster=call_id,
            status=state.value,
            metadata="",
            done=state in (JobState.SUCCEEDED, JobState.FAILED, JobState.CANCELLED),
            state=state,
        )

    def cancel(self, call_id: str) -> None:
        """Terminates the sandbox if it is still running."""
        sandbox = self.get_call(call_id)
        try:
            sandbox.terminate()
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Modal terminate({call_id}) failed: {e!r}")

    def get_logs_stream(self, call_id: str) -> ModalLogStream:
        """Returns a streaming readline()-style log stream for ``call_id``."""
        sandbox = self.get_call(call_id)
        # ``Sandbox.stdout`` is an async iterator of log chunks. Materialize
        # to a list synchronously for the worker's blocking log-tail path.
        chunks: list[str] = []
        for stream_attr in ("stdout", "stderr"):
            stream = getattr(sandbox, stream_attr, None)
            if stream is None:
                continue
            try:
                for line in stream:
                    chunks.append(str(line))
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Modal {stream_attr} read failed: {e!r}")
        return ModalLogStream(cast("Iterator[str]", iter(chunks)))

    # ----- pricing -----

    # GPU $/hr list pricing as of 2026-05. Used for billing parity with
    # SkyPilot's ``handle.get_hourly_price()``. Kept conservative and easy to
    # bump; not authoritative.
    _GPU_HOURLY_USD: dict[str, float] = {
        "T4": 0.59,
        "L4": 0.80,
        "A10G": 1.10,
        "A100": 2.10,
        "A100-40GB": 2.10,
        "A100-80GB": 2.50,
        "L40S": 1.95,
        "H100": 3.95,
        "H200": 4.55,
        "B200": 6.25,
    }

    @classmethod
    def estimate_cost_per_hour(cls, gpu: str | None) -> float | None:
        """Estimates total $/hr for a Modal GPU spec like ``H100:8``.

        Returns ``None`` if the GPU type is unknown.
        """
        if not gpu:
            return None
        spec = str(gpu).split(":")
        gpu_type = spec[0].strip()
        count = int(spec[1]) if len(spec) > 1 else 1
        unit = cls._GPU_HOURLY_USD.get(gpu_type)
        if unit is None:
            return None
        return round(unit * count, 4)
