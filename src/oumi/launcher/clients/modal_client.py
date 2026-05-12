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
``cluster`` concept — every job is a ``modal.Sandbox`` that persists
beyond the calling Python process. ``ModalClient`` translates a
``JobConfig`` into a sandbox launch and exposes status/cancel/log
primitives via the sandbox's opaque ``object_id``.

Image, GPU, secrets, and a workspace-scoped HuggingFace cache volume
are derived from the ``JobConfig`` at launch time. ``setup`` and
``run`` are concatenated into a single shell script and executed
together inside the sandbox so secrets injected via ``modal.Secret``
are visible (image-build time has no secrets attached). Sandboxes are
tagged with the caller's logical cluster name so ``ModalCluster.down()``
can find and terminate them across worker restarts.
"""

from __future__ import annotations

import io
import re
import shlex
from collections.abc import Iterator
from typing import Any, cast

from oumi.core.configs import JobConfig
from oumi.core.launcher import ClusterNotFoundError, JobState, JobStatus
from oumi.utils.logging import logger

# ``modal`` is an optional extra (``oumi[modal]``); we lazy-import at
# runtime via :func:`_import_modal` so importing this module doesn't
# require the SDK. Public types are kept as ``Any`` rather than
# referencing ``modal.*`` so pyright does not require ``modal`` on the
# typecheck path.

_DEFAULT_TIMEOUT_S = 24 * 60 * 60  # 24h


def _import_modal() -> Any:
    """Imports the modal SDK lazily to avoid hard-importing it at module load.

    ``modal`` is shipped as an optional extra (``oumi[modal]``) so the
    SDK is absent from default install / typecheck environments. The
    ``pyright: ignore`` keeps CI green when ``modal`` isn't present;
    runtime callers that hit this path must have installed the extra.
    """
    import modal  # noqa: PLC0415 # pyright: ignore[reportMissingImports]

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
    # ``uv_pip_install`` is Modal's recommended replacement for
    # ``pip_install`` — uv is faster and Modal handles its bootstrap
    # internally so we don't need to install uv as a separate step.
    return (
        modal_lib.Image.debian_slim()
        .apt_install("zip", "curl", "git")
        .uv_pip_install("awscli")
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

#: Tag key applied to every sandbox at launch time. Used by
#: ``ModalCluster.down()`` to find sandboxes across worker restarts via
#: ``Sandbox.list(tags=...)``, so cleanup doesn't depend on in-process state.
_CLUSTER_TAG = "oumi_cluster"

#: Name of the Modal Volume mounted at ``/root/.cache/huggingface``. Persists
#: HuggingFace model/tokenizer downloads across sandboxes so repeated
#: training of the same model skips the multi-GB ``hf download`` step.
_HF_CACHE_VOLUME_NAME = "oumi-hf-cache"

#: Container path where the HuggingFace cache volume is mounted. Matches
#: the default ``HF_HOME``/``HUGGINGFACE_HUB_CACHE`` location for root,
#: so no setup-script changes are needed to take advantage of the cache.
_HF_CACHE_MOUNT_PATH = "/root/.cache/huggingface"


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
    """A wrapped client for communicating with Modal.

    Tracks the cluster_name → sandbox_ids mapping in-process so
    ``ModalCluster.down()`` can find the sandboxes spawned under a
    given logical cluster name. Across worker restarts the mapping is
    lost, so cleanup falls back to per-sandbox cancel via job_id which
    the worker also persists in the operation record.
    """

    def __init__(self) -> None:
        """Initializes a new instance of the ModalClient class."""
        self._modal = _import_modal()
        self._cluster_to_sandboxes: dict[str, list[str]] = {}

    # ----- launch / lifecycle -----

    def launch(
        self, job: JobConfig, cluster_name: str | None = None, **kwargs: Any
    ) -> JobStatus:
        """Creates a detached ``modal.Sandbox`` for the provided job.

        Modal has no native cluster concept. The ``cluster_name`` argument
        becomes a logical label returned on ``JobStatus.cluster`` so the
        caller can group multiple sandboxes (e.g. retries) under one name.
        ``JobStatus.id`` is the opaque ``Sandbox.object_id`` and is the
        canonical handle for status / cancel / log lookups.

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

        # Workspace-scoped HF cache. Mounting at the default cache path
        # means ``hf download`` populates the volume on first run and
        # short-circuits on subsequent runs of the same model.
        hf_cache_volume = modal_lib.Volume.from_name(
            _HF_CACHE_VOLUME_NAME, create_if_missing=True
        )

        sandbox = modal_lib.Sandbox.create(
            "/bin/bash",
            "-lc",
            full_script,
            app=app,
            image=image,
            gpu=gpu,
            secrets=[secret] if secret else [],
            timeout=timeout,
            volumes={_HF_CACHE_MOUNT_PATH: hf_cache_volume},
        )
        sandbox_id = sandbox.object_id
        effective_cluster = cluster_name or sandbox_id
        self._cluster_to_sandboxes.setdefault(effective_cluster, []).append(sandbox_id)
        # Tag the sandbox so ``find_sandboxes_for_cluster`` can locate it
        # across worker restarts. Best-effort — if tagging fails the
        # in-process tracker still works for the same-process case.
        try:
            sandbox.set_tags({_CLUSTER_TAG: effective_cluster})
        except Exception as e:  # noqa: BLE001
            logger.warning(
                f"Failed to tag Modal sandbox {sandbox_id} with cluster="
                f"{effective_cluster}: {e!r}"
            )

        logger.info(
            f"Launched Modal sandbox={sandbox_id} cluster={effective_cluster} "
            f"gpu={gpu} timeout={timeout}s"
        )
        return JobStatus(
            name=job.name or sandbox_id,
            id=sandbox_id,
            cluster=effective_cluster,
            status=str(JobState.PENDING.value),
            metadata=shlex.quote(_LAUNCHER_APP_NAME),
            done=False,
            state=JobState.PENDING,
            # ``cost_per_hour`` is intentionally left unset (None). Modal
            # doesn't expose pricing via its Python SDK, so any $/hr
            # number would have to come from a hand-maintained table.
            # That table belongs in the caller (e.g. an enterprise
            # billing layer) rather than in the OSS launcher.
        )

    def sandboxes_for_cluster(self, cluster_name: str) -> list[str]:
        """Returns the sandbox IDs spawned under ``cluster_name`` in this process."""
        return list(self._cluster_to_sandboxes.get(cluster_name, []))

    def find_sandboxes_for_cluster(self, cluster_name: str) -> list[str]:
        """Returns the sandbox IDs tagged with ``cluster_name`` on Modal.

        Stateless lookup via ``Sandbox.list(tags=...)`` — works across
        worker restarts (unlike :meth:`sandboxes_for_cluster`, which
        only sees launches from the current process). Falls back to
        the in-process tracker if the Modal API call fails or returns
        nothing.
        """
        ids: list[str] = []
        try:
            for sandbox in self._modal.Sandbox.list(tags={_CLUSTER_TAG: cluster_name}):
                ids.append(sandbox.object_id)
        except Exception as e:  # noqa: BLE001
            logger.warning(
                f"Modal Sandbox.list(tags={{{_CLUSTER_TAG}={cluster_name}}}) "
                f"failed: {e!r}; falling back to in-process tracker"
            )
        if ids:
            return ids
        return self.sandboxes_for_cluster(cluster_name)

    def get_call(self, call_id: str) -> Any:
        """Resolves a ``Sandbox`` by its opaque ID, raising if missing."""
        modal_lib = self._modal
        try:
            return modal_lib.Sandbox.from_id(call_id)
        except Exception as e:  # noqa: BLE001
            raise ClusterNotFoundError(f"Modal sandbox '{call_id}' not found") from e

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

