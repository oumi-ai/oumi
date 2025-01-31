# Custom Clusters
Similar to custom dataset and model classes, you can register a class for your own cluster so you can launch Oumi jobs on custom clusters that are not supported out of the box.

This guide is specifically geared towards individuals who have access to a compute cluster that's not hosted on a common cloud provider (e.g. University/personal compute clusters).

## The Oumi Launcher Hierarchy

### Preface
Before diving into this tutorial, lets discuss the hierarchy of the Oumi Launcher. At this point, it's worth reading through our tutorial on {doc}`/user_guides/launch/deploy` to better understand the end-to-end flow of the launcher. Already read it? Great!

### Overview
At a high level, the Oumi Launcher is composed of 3 tiers of objects: `Clouds`, `Clusters`, and `Clients`. The Launcher holds an instance of each unique `Cloud`. These `Clouds`, in turn, are responsible for creating compute `Clusters`. And `Clusters` coordinate running jobs. All communication with remote APIs happens via the `Client`.

#### Clouds
A Cloud class must implement the {class}`~oumi.core.launcher.BaseCloud` abstract class. The Launcher will only create one instance of each Cloud, so it's important that a single Cloud object is capable of turning up and down multiple clusters.

You can find several implementations of Clouds [here](https://github.com/oumi-ai/oumi/tree/main/src/oumi/launcher/clouds).

#### Clusters
A Cluster class must implement the {class}`~oumi.core.launcher.BaseCluster` abstract class. A cluster represents a single instance of hardware. For custom clusters (such as having access to a super computer that can only run one job at a time), it may be the case that you only need one cluster to represent your hardware setup.

You can find several implementations of Clusters [here](https://github.com/oumi-ai/oumi/tree/main/src/oumi/launcher/clusters).

#### Clients
Clients are a completely optional but highly encouraged class. Clients should encapsulate all logic that calls remote APIs related to your cloud. While this logic could be encapsulated with your Cluster and Cloud classes, having a dedicated class for this purpose greatly simplifies your Cloud and Cluster logic.

You can find several implementations of Clients [here](https://github.com/oumi-ai/oumi/tree/main/src/oumi/launcher/clients).

## Creating a CustomClient Class
Let's get started by creating a client for our new cloud, `CustomCloud`. Let's create a simple client that randomly sets the state of the job on submission. It also supports canceling jobs, and turning down clusters:

``` {code-block} python
import random
from enum import Enum
from typing import Optional

from oumi.core.configs import JobConfig
from oumi.core.launcher import JobStatus


class _JobState(Enum):
    """An enumeration of the possible states of a job."""

    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELED = "CANCELED"


class CustomClient:
    """A client for running jobs locally in a subprocess."""

    def __init__(self):
        """Initializes a new instance of the CustomClient class."""
        self._jobs = []

    def submit_job(self, job: JobConfig) -> JobStatus:
        """Pretends to run the specified job on this cluster."""
        job_id = str(len(self._jobs))
        name = job.name if job.name else job_id
        # Pick a random status
        status = random.choice([state for state in _JobState])
        job_status = JobStatus(
            name=name,
            id=job_id,
            status=status.value,
            cluster="",
            metadata="",
            done=False,
        )
        self._jobs.append(job_status)
        return job_status

    def list_jobs(self) -> list[JobStatus]:
        """Returns a list of job statuses."""
        return self._jobs

    def get_job(self, job_id: str) -> Optional[JobStatus]:
        """Gets the specified job's status.

        Args:
            job_id: The ID of the job to get.

        Returns:
            The job status if found, None otherwise.
        """
        job_list = self.list_jobs()
        for job in job_list:
            if job.id == job_id:
                return job
        return None

    def cancel(self, job_id) -> Optional[JobStatus]:
        """Cancels the specified job.

        Args:
            job_id: The ID of the job to cancel.

        Returns:
            The job status if found, None otherwise.
        """
        int_id = int(job_id)
        if int_id > len(self._jobs):
            return None
        job_status = self._jobs[int_id]
        job_status.status = _JobState.CANCELED.value
        return job_status

    def turndown_cluster(self, cluster_name: str):
        """Turns down the cluster."""
        print(f"Turning down cluster {cluster_name}...")
        pass
```

## Creating a CustomCluster Class
Now that we have a client that talks to our API, we can use the Client to build a Cluster!

``` {code-block} python
from typing import Any, Optional

from oumi.core.launcher import BaseCluster


class CustomCluster(BaseCluster):
    """A custom cluster implementation."""

    def __init__(self, name: str, client: CustomClient) -> None:
        """Initializes a new instance of the CustomCluster class."""
        self._name = name
        self._client = client

    def __eq__(self, other: Any) -> bool:
        """Checks if two LocalClusters are equal."""
        if not isinstance(other, CustomCluster):
            return False
        return self.name() == other.name()

    def name(self) -> str:
        """Gets the name of the cluster."""
        return self._name

    def get_job(self, job_id: str) -> Optional[JobStatus]:
        """Gets the jobs on this cluster if it exists, else returns None."""
        for job in self.get_jobs():
            if job.id == job_id:
                return job
        return None

    def get_jobs(self) -> list[JobStatus]:
        """Lists the jobs on this cluster."""
        jobs = self._client.list_jobs()
        for job in jobs:
            job.cluster = self._name
        return jobs

    def cancel_job(self, job_id: str) -> JobStatus:
        """Cancels the specified job on this cluster."""
        self._client.cancel(job_id)
        job = self.get_job(job_id)
        if job is None:
            raise RuntimeError(f"Job {job_id} not found.")
        return job

    def run_job(self, job: JobConfig) -> JobStatus:
        """Runs the specified job on this cluster.

        Args:
            job: The job to run.

        Returns:
            The job status.
        """
        job_status = self._client.submit_job(job)
        job_status.cluster = self._name
        return job_status

    def down(self) -> None:
        """Cancel all jobs and turn down the cluster."""
        for job in self.get_jobs():
            self.cancel_job(job.id)
        self._client.turndown_cluster(self._name)

    def stop(self) -> None:
        """Cancel all jobs and turn down the cluster."""
        self.down()
```

## Creating a CustomCloud Class
Let's create a CustomCloud to manage our clusters:

``` {code-block} python
from oumi.core.launcher import BaseCloud


class CustomCloud(BaseCloud):
    """A resource pool for managing Local jobs."""

    # The default cluster name. Used when no cluster name is provided.
    _DEFAULT_CLUSTER = "custom"

    def __init__(self):
        """Initializes a new instance of the LocalCloud class."""
        # A mapping from cluster names to Local Cluster instances.
        self._clusters = {}

    def _get_or_create_cluster(self, name: str) -> CustomCluster:
        """Gets the cluster with the specified name, or creates one if it doesn't exist.

        Args:
            name: The name of the cluster.

        Returns:
            LocalCluster: The cluster instance.
        """
        if name not in self._clusters:
            self._clusters[name] = CustomCluster(name, CustomClient())
        return self._clusters[name]

    def up_cluster(self, job: JobConfig, name: Optional[str]) -> JobStatus:
        """Creates a cluster and starts the provided Job."""
        # The default cluster.
        cluster_name = name or self._DEFAULT_CLUSTER
        cluster = self._get_or_create_cluster(cluster_name)
        job_status = cluster.run_job(job)
        if not job_status:
            raise RuntimeError("Failed to start job.")
        return job_status

    def get_cluster(self, name) -> Optional[BaseCluster]:
        """Gets the cluster with the specified name, or None if not found."""
        clusters = self.list_clusters()
        for cluster in clusters:
            if cluster.name() == name:
                return cluster
        return None

    def list_clusters(self) -> list[BaseCluster]:
        """Lists the active clusters on this cloud."""
        return list(self._clusters.values())
```

Now all that's left to do is register your CustomCloud!

## Registering Your CustomCloud
By implementing the BaseCloud class, you are now ready to register your cloud with Oumi. First, let's take a look at the clouds that are already registered:

``` {code-block} python
import oumi.launcher as launcher

print(launcher.which_clouds())
```

You can register your cloud by implementing a builder method. This method must take no arguments and must return a new instance of your CustomCloud:

``` {code-block} python
from oumi.core.registry import register_cloud_builder


@register_cloud_builder("custom")
def Local_cloud_builder() -> CustomCloud:
    """Builds a LocalCloud instance."""
    return CustomCloud()
```

Let's take another look at our registered clouds now:

``` {code-block} python
print(launcher.which_clouds())
```

Great, our CustomCloud is there!

### Using Your CustomCloud via the CLI

**‼️ Important ‼️** A few extra steps are needed to use your cloud from the CLI.

First, you need to create a `requirements.txt` file.

This is a simple text file where each line contains the absolute filepath to any python
that interface with the Oumi registry. In this guide, that means any files we wrote that
contain a `@register_cloud_builder` decorator.

Let's say you created your CustomCloud class in a file saved at
`/path/to/custom_cloud.py`, and your `requirements` file at
`/another/path/requirements.txt`.
Your `requirements.txt` file should look like:
```
/path/to/custom_cloud.py
```

Now that you've created your `requirements.txt` file, you simply need to set the
`OUMI_EXTRA_DEPS_FILE` environment variable with the path of your `requirements.txt`
file and the Oumi CLI will automatically pick up your changes!

``` {code-block} shell
export OUMI_EXTRA_DEPS_FILE=/another/path/requirements.txt
```

You can verify that your cloud is now installed by running:

``` {code-block} shell
oumi launch which
```

## Running a Job on Your Cloud

Let's take our new Cloud for a spin:



::::{tab-set}
:::{tab-item} CLI

We can kick off a job:
``` {code-block} shell
oumi launch up --cluster first_cluster -c configs/recipes/smollm/sft/135m/quickstart_gcp_job.yaml  --resources.cloud custom

oumi launch status
```

And now let's turn down our cluster:

``` {code-block} shell
oumi launch down --cluster first_cluster
```
:::

:::{tab-item} Python
``` {code-block} python
job = launcher.JobConfig(name="test")
job.resources.cloud = "custom"

first_cluster, job_status = launcher.up(job, "first_cluster")
print(job_status)
second_cluster, second_job_status = launcher.up(job, "second_cluster")
print(second_job_status)

print("Canceling the first job...")
print(launcher.cancel(job_status.id, job.resources.cloud, job_status.cluster))
```

And now let's turn down our clusters:

``` {code-block} python
for cluster in launcher.get_cloud("custom").list_clusters():
    cluster.down()
    print(f"Cluster {cluster.name()} is down. Listing jobs...")
    print(cluster.get_jobs())
```
:::
::::
