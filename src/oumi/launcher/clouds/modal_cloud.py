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

"""Modal cloud backend for running jobs on Modal.com infrastructure.

Modal is a serverless platform for running code in the cloud. Unlike traditional
cloud providers, Modal doesn't have the concept of persistent clusters - instead,
it runs functions on-demand with automatic scaling.

This module provides a cloud backend that integrates Modal with Oumi's launcher
framework, using dynamic code generation to convert JobConfig specifications
into Modal apps.

Features:
- Automatic GPU allocation (H100, A100, L4, T4, etc.)
- Support for up to 8 GPUs per container
- Automatic container image building from setup scripts
- Integration with Modal Volumes for persistent storage

Limitations:
- Multi-node training is in early preview on Modal
- No persistent clusters (serverless model)
- Region/zone selection is handled automatically by Modal

Usage:
    Set `resources.cloud: modal` in your job configuration:

    ```yaml
    name: my-training-job
    resources:
      cloud: modal
      accelerators: "H100:4"
    setup: |
      pip install torch transformers
    run: |
      python train.py
    ```

Requirements:
    - Modal CLI installed and authenticated (`pip install modal && modal token new`)
    - MODAL_TOKEN_ID and MODAL_TOKEN_SECRET environment variables (or ~/.modal.toml)
"""

from typing import Optional

from oumi.core.configs import JobConfig
from oumi.core.launcher import BaseCloud, BaseCluster, JobStatus
from oumi.core.registry import register_cloud_builder
from oumi.launcher.clients.modal_client import ModalClient
from oumi.launcher.clusters.modal_cluster import ModalCluster
from oumi.utils.logging import logger


class ModalCloud(BaseCloud):
    """A resource pool for managing jobs on Modal.com.

    Modal is a serverless platform, so unlike traditional cloud backends:
    - There's no persistent cluster infrastructure to manage
    - Jobs run on-demand with automatic scaling
    - "Clusters" in this context are logical groupings of jobs

    This class provides a unified interface for running Oumi jobs on Modal
    while maintaining compatibility with the launcher framework.
    """

    # The default cluster name. Used when no cluster name is provided.
    _DEFAULT_CLUSTER = "modal"

    def __init__(self):
        """Initializes a new instance of the ModalCloud class."""
        # A mapping from cluster names to ModalCluster instances.
        # Each "cluster" is a logical grouping that shares a ModalClient.
        self._clusters: dict[str, ModalCluster] = {}
        self._clients: dict[str, ModalClient] = {}

    def _get_or_create_cluster(self, name: str) -> ModalCluster:
        """Gets the cluster with the specified name, or creates one if needed.

        Args:
            name: The name of the cluster

        Returns:
            ModalCluster: The cluster instance
        """
        if name not in self._clusters:
            # Create a new client for this cluster
            client = ModalClient()
            self._clients[name] = client
            self._clusters[name] = ModalCluster(name, client)
            logger.info(f"Created Modal cluster: {name}")
        return self._clusters[name]

    def up_cluster(self, job: JobConfig, name: Optional[str], **kwargs) -> JobStatus:
        """Creates a cluster and starts the provided Job.

        For Modal (serverless), "creating a cluster" simply means organizing
        jobs under a logical name. The actual infrastructure is provisioned
        on-demand when the job runs.

        Args:
            job: The job configuration to run
            name: Optional cluster name for organizing jobs
            **kwargs: Additional arguments (unused)

        Returns:
            JobStatus: The initial status of the submitted job

        Raises:
            RuntimeError: If job submission fails
        """
        cluster_name = name or self._DEFAULT_CLUSTER
        cluster = self._get_or_create_cluster(cluster_name)

        logger.info(f"Submitting job to Modal cluster '{cluster_name}'")
        job_status = cluster.run_job(job)

        if not job_status:
            raise RuntimeError("Failed to submit job to Modal.")

        return job_status

    def get_cluster(self, name: str) -> Optional[BaseCluster]:
        """Gets the cluster with the specified name, or None if not found.

        Args:
            name: The name of the cluster to retrieve

        Returns:
            The ModalCluster if found, None otherwise
        """
        return self._clusters.get(name)

    def list_clusters(self) -> list[BaseCluster]:
        """Lists all active clusters on this cloud.

        Returns:
            List of ModalCluster instances
        """
        return list(self._clusters.values())


@register_cloud_builder("modal")
def modal_cloud_builder() -> ModalCloud:
    """Builds a ModalCloud instance.

    This function is registered with Oumi's cloud registry and is called
    when a job specifies `resources.cloud: modal`.

    Returns:
        A new ModalCloud instance
    """
    return ModalCloud()
