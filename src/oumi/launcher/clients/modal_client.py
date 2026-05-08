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
import shlex
import time
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, cast

from oumi.core.configs import JobConfig
from oumi.core.launcher import ClusterNotFoundError, JobState, JobStatus
from oumi.utils.logging import logger

if TYPE_CHECKING:
    import modal


_DEFAULT_TIMEOUT_S = 24 * 60 * 60  # 24h
_DEFAULT_BASE_IMAGE = "python:3.11-slim"


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
    Otherwise we start from a slim Python base image. ``setup`` is baked in
    via ``run_commands`` so Modal's content-addressed cache reuses it across
    jobs with identical setup scripts.
    """
    base = (
        modal_lib.Image.from_registry(job.resources.image_id.removeprefix("docker:"))
        if job.resources.image_id
        and str(job.resources.image_id).startswith("docker:")
        else modal_lib.Image.debian_slim().pip_install("uv")
    )
    if not job.resources.image_id:
        # Fall back to a generic registry image when none is specified.
        base = modal_lib.Image.from_registry(_DEFAULT_BASE_IMAGE)
    if job.setup:
        base = base.run_commands([job.setup])
    return base


def _build_secret(modal_lib: Any, envs: dict[str, str]) -> Any | None:
    """Bundles the JobConfig env dict into a single ``modal.Secret``."""
    if not envs:
        return None
    return modal_lib.Secret.from_dict({k: str(v) for k, v in envs.items()})


def _function_call_state(call: Any) -> JobState:
    """Maps a ``modal.FunctionCall`` status to a :class:`JobState`."""
    # Modal's public API does not expose status directly. Probe with get(timeout=0).
    # Raises TimeoutError while pending/running, returns on success, raises
    # FunctionCallTerminationError or the user exception on failure.
    modal_lib = _import_modal()
    try:
        call.get(timeout=0)
        return JobState.SUCCEEDED
    except modal_lib.exception.OutputExpiredError:
        # Result was discarded — treat as terminal failure.
        return JobState.FAILED
    except getattr(modal_lib.exception, "FunctionTimeoutError", Exception):
        return JobState.RUNNING
    except TimeoutError:
        return JobState.RUNNING
    except Exception:
        # Any other surfaced exception means the user code raised → failed.
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
        """Spawns a Modal FunctionCall for the provided job.

        ``cluster_name`` is ignored — Modal generates a stable opaque ID
        which is returned as the cluster name on the resulting JobStatus.
        """
        modal_lib = self._modal
        app_name = (
            cluster_name or job.name or f"oumi-{int(time.time() * 1000)}"
        ).replace("_", "-")
        app = modal_lib.App(app_name)
        image = _build_image(modal_lib, job)
        secret = _build_secret(modal_lib, job.envs)

        gpu = job.resources.accelerators
        timeout = int(kwargs.get("timeout", _DEFAULT_TIMEOUT_S))

        @app.function(
            image=image,
            gpu=gpu,
            secrets=[secret] if secret else [],
            timeout=timeout,
        )
        def _run(run_script: str) -> int:
            import subprocess  # noqa: PLC0415

            return subprocess.run(
                ["/bin/bash", "-lc", run_script], check=True
            ).returncode

        with app.run():
            call = _run.spawn(job.run)
            call_id = call.object_id

        logger.info(
            f"Launched Modal app={app_name} call_id={call_id} gpu={gpu} "
            f"timeout={timeout}s"
        )
        return JobStatus(
            name=job.name or app_name,
            id=call_id,
            cluster=call_id,
            status=str(JobState.PENDING.value),
            metadata=shlex.quote(app_name),
            done=False,
            state=JobState.PENDING,
            cost_per_hour=self.estimate_cost_per_hour(gpu),
        )

    def get_call(self, call_id: str) -> Any:
        """Resolves a ``FunctionCall`` by its opaque ID, raising if missing."""
        modal_lib = self._modal
        try:
            return modal_lib.FunctionCall.from_id(call_id)
        except Exception as e:  # noqa: BLE001
            raise ClusterNotFoundError(
                f"Modal FunctionCall '{call_id}' not found"
            ) from e

    def get_status(self, call_id: str) -> JobStatus:
        """Returns the current :class:`JobStatus` for ``call_id``."""
        call = self.get_call(call_id)
        state = _function_call_state(call)
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
        """Cancels the FunctionCall if it is still pending or running."""
        call = self.get_call(call_id)
        try:
            call.cancel()
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Modal cancel({call_id}) failed: {e!r}")

    def get_logs_stream(self, call_id: str) -> ModalLogStream:
        """Returns a streaming readline()-style log stream for ``call_id``."""
        call = self.get_call(call_id)
        # ``FunctionCall.logs`` is the supported log iterator on recent
        # modal versions; fall back to a no-op iterator otherwise.
        logs_fn: Any = getattr(call, "logs", None)
        raw: Any = logs_fn() if callable(logs_fn) else []
        return ModalLogStream(cast("Iterator[str]", iter(raw)))

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
