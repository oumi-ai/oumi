from typing import Optional, Union

from oumi.core.configs import JobConfig
from oumi.core.launcher import BaseCloud, BaseCluster, JobStatus
from oumi.core.registry import REGISTRY, RegistryType


class Launcher:
    """A class for managing the lifecycle of jobs on different clouds."""

    def __init__(self):
        """Initializes a new instance of the Launcher class."""
        self._clouds = dict()
        self._initialize_new_clouds()

    def _initialize_new_clouds(self) -> None:
        """Initializes new clouds. Existing clouds are not re-initialized."""
        for name, builder in REGISTRY.get_all(RegistryType.CLOUD).items():
            if name not in self._clouds:
                self._clouds[name] = builder()

    def _get_cloud_by_name(self, cloud: str) -> BaseCloud:
        """Gets the cloud instance for the specified cloud name."""
        if cloud not in self._clouds:
            cloud_builder = REGISTRY.get(cloud, RegistryType.CLOUD)
            if not cloud_builder:
                raise ValueError(f"Cloud {cloud} not found in the registry.")
            self._clouds[cloud] = cloud_builder()
        return self._clouds[cloud]

    def get_cloud(self, job_or_cloud: Union[JobConfig, str]) -> BaseCloud:
        """Gets the cloud instance for the specified job."""
        if isinstance(job_or_cloud, str):
            return self._get_cloud_by_name(job_or_cloud)
        return self._get_cloud_by_name(job_or_cloud.resources.cloud)

    def up(
        self, job: JobConfig, cluster_name: Optional[str], **kwargs
    ) -> tuple[BaseCluster, JobStatus]:
        """Creates a new cluster and starts the specified job on it."""
        cloud = self.get_cloud(job)
        job_status = cloud.up_cluster(job, cluster_name, **kwargs)
        cluster = cloud.get_cluster(job_status.cluster)
        if not cluster:
            raise RuntimeError(f"Cluster {job_status.cluster} not found.")
        return (cluster, job_status)

    def run(self, job: JobConfig, cluster_name: str) -> JobStatus:
        """Runs the specified job on the specified cluster.

        Args:
            job: The job configuration.
            cluster_name: The name of the cluster to run the job on.

        Returns:
            Optional[JobStatus]: The status of the job.
        """
        cloud = self.get_cloud(job)
        cluster = cloud.get_cluster(cluster_name)
        if not cluster:
            raise ValueError(f"Cluster {cluster_name} not found.")
        return cluster.run_job(job)

    def stop(self, job_id: str, cloud_name: str, cluster_name: str) -> JobStatus:
        """Stops the specified job."""
        cloud = self._get_cloud_by_name(cloud_name)
        cluster = cloud.get_cluster(cluster_name)
        if not cluster:
            raise ValueError(f"Cluster {cluster_name} not found.")
        return cluster.stop_job(job_id)

    def down(self, cloud_name: str, cluster_name: str) -> None:
        """Turns down the specified cluster."""
        cloud = self._get_cloud_by_name(cloud_name)
        cluster = cloud.get_cluster(cluster_name)
        if not cluster:
            raise ValueError(f"Cluster {cluster_name} not found.")
        cluster.down()

    def status(self) -> list[JobStatus]:
        """Gets the status of all jobs across all clusters."""
        # Pick up any newly registered cloud builders.
        self._initialize_new_clouds()
        statuses = []
        for _, cloud in self._clouds.items():
            for cluster in cloud.list_clusters():
                statuses.extend(cluster.get_jobs())
        return statuses

    def which_clouds(self) -> list[str]:
        """Gets the names of all clouds in the registry."""
        return [name for name, _ in REGISTRY.get_all(RegistryType.CLOUD).items()]


LAUNCHER = Launcher()
# Explicitly expose the public methods of the default Launcher instance.
#: A convenience method for Launcher.down.
down = LAUNCHER.down
#: A convenience method for Launcher.get_cloud.
get_cloud = LAUNCHER.get_cloud
#: A convenience method for Launcher.run.
run = LAUNCHER.run
#: A convenience method for Launcher.status.
status = LAUNCHER.status
#: A convenience method for Launcher.stop.
stop = LAUNCHER.stop
#: A convenience method for Launcher.up.
up = LAUNCHER.up
#: A convenience method for Launcher.which_clouds.
which_clouds = LAUNCHER.which_clouds
