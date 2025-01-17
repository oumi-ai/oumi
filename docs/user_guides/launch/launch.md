# Running Code on Clusters

```{toctree}
:maxdepth: 2
:caption: Launch
:hidden:

deploy
custom_cluster
```

In addition to running Oumi locally, you can use the `oumi launch` command in the Oumi CLI to run jobs on remote clusters. It provides a unified interface for running your code, allowing you to seamlessly switch between popular cloud providers and your own custom clusters!

## Overview

The Oumi Launcher operates using three key concepts:

1) `Jobs`: A `job` is a unit of work, such as running training or model evaluation. This can be any script you'd like!
2) `Clusters`: A `cluster` is a set of dedicated hardware upon which `jobs` are run. A `cluster` could be as simple as a cloud VM environment.
3) `Clouds` : A `cloud` is a resource provider that manages `clusters`. These include GCP, AWS, Lambda, Runpod, etc.

When you submit a job to the launcher it will handle queueing your job in the proper cluster's job queue. If your desired Cloud does not have an appropriate cluster for running your job it will try to create one on the fly!

## Setup

The Oumi launcher integrates with SkyPilot to launch jobs on various cloud providers. To run on a cloud GPU cluster, first make sure to have all the dependencies installed for your desired cloud provider:

  ```shell
  pip install oumi[aws]     # For Amazon Web Services
  pip install oumi[azure]   # For Microsoft Azure
  pip install oumi[gcp]     # For Google Cloud Platform
  pip install oumi[lambda]  # For Lambda Cloud
  pip install oumi[runpod]  # For RunPod
  ```

Then, you need to enable your desired cloud provider in SkyPilot. Run `sky check` to check which providers you have enabled, along with instructions on how to enable the ones you don't. More detailed setup instructions can be found in [SkyPilot's documentation](https://skypilot.readthedocs.io/en/latest/getting-started/installation.html#cloud-account-setup).

## Defining a Job

Like most configurable pieces of Oumi, Jobs are defined via YAML configs. In this case, every job is defined by a {class}`~oumi.launcher.JobConfig`.

When creating a job, there are several important fields you should be aware of:
- {attr}`~oumi.launcher.JobConfig.resources`: where you specify resource requirements (cloud to use, GPUs, disk size, etc)  via {class}`oumi.launcher.JobResources`
- {attr}`~oumi.launcher.JobConfig.setup`: an optional script that is run when a cluster is created
- {attr}`~oumi.launcher.JobConfig.run`: the main script to run for your job
- {attr}`~oumi.launcher.JobConfig.working_dir`: the local directory to be copied to the cluster for use during execution.

A sample job is provided below:
````{dropdown} configs/recipes/smollm/sft/135m/quickstart_gcp_job.yaml
```{literalinclude} ../../../configs/recipes/smollm/sft/135m/quickstart_gcp_job.yaml
:language: yaml
```
````

## Core Functionality

Launching jobs remotely is available via both the Oumi CLI and our python API ({mod}`~oumi.launcher`)

`oumi launch` provides you with all the capabilities you need to kickoff and monitor jobs running on remote machines.

We'll cover the most common use case here, which boils down to:

1. Using `oumi launch up` to create a cluster and run a job.
2. Using `oumi launch status` to check the status of your job and cluster.
3. Canceling jobs using `oumi launch cancel`
4. Turning down a cluster manually using `oumi launch down`

For a quick overview of all `oumi launch` commands, see our [CLI Launch Reference](/cli/commands.md#launch)

### Launching Jobs

::::{tab-set}
:::{tab-item} CLI

To launch a job on your desired cloud, run:
```{code-block} shell
oumi launch up --cluster my-cluster -c configs/recipes/smollm/sft/135m/quickstart_gcp_job.yaml
```
This command will create the cluster if it doesn't exist, and then execute the job on it. It can also run the job on an existing cluster with that name.

To launch on the cloud of your choice, use the `--resources.cloud` flag, ex. `--resources.cloud lambda`. Most of our configs run on GCP by default. See {attr}`~oumi.launcher.JobResources.cloud` for all supported clouds, or run:

```{code-block} shell
oumi launch which
```

To return immediately when the job is scheduled and not poll for the job's completion, specify the `--detach` flag:

```{code-block} shell
oumi launch up --cluster my-cluster -c configs/recipes/smollm/sft/135m/quickstart_gcp_job.yaml --detach
```

To find out more about the GPUs available on your cloud provider, you can use skypilot:

```{code-block} shell
sky show-gpus
```

If you made any code changes to the oumi codebase (not including configs), you need to run
`pip install '.'` in the `run` section of the job config to install the
changes on the cluster.
:::

:::{tab-item} Python

To launch a job on your desired cloud, run:
```{code-block} python
import oumi.launcher as launcher

# Read our JobConfig from the YAML file
job_config = launcher.JobConfig.from_yaml(str(Path("configs/recipes/smollm/sft/135m/quickstart_gcp_job.yaml")))
# Start the job
launcher.up(job_config, "your_cluster_name")
```
This command will create the cluster if it doesn't exist, and then execute the job on it. It can also run the job on an existing cluster with that name.

To launch on the cloud of your choice, simply set `job_config.resources.cloud`, ex. `job_config.resources.cloud = "gcp"`. Most of our configs run on GCP by default. See {attr}`~oumi.launcher.JobResources.cloud` for all supported clouds, or run:

```{code-block} python
import oumi.launcher as launcher

# Print all available clouds
print(launcher.which_clouds())
```

To find out more about the GPUs available on your cloud provider, you can use skypilot:

```{code-block} shell
sky show-gpus
```

If you made any code changes to the oumi codebase (not including configs), you need to run
`pip install '.'` in the `run` section of the job config to install the
changes on the cluster.
:::
::::


#### Mount Cloud Storage

You can mount cloud storage containers like GCS or S3 to your job, which maps their remote paths to a directory on your job's disk. This is a fantastic way to write important information (such as data or model checkpoints) to a persistent disk that outlives your cluster's lifetime.

```{tip}
Writing your job's output to cloud storage is recommended for preemptible cloud instances, or jobs outputting a large amount of data like large model checkpoints. Data on local disk will be lost on job preemption, and your job's local disk may not have enough storage for multiple large model checkpoints.
```

For example, to mount your GCS bucket `gs://my-bucket`, add the following to your {py:class}`~oumi.core.configs.JobConfig`:

```yaml
storage_mounts:
  /gcs_dir:
    source: gs://my-bucket
    store: gcs
```

You can now access files in your bucket as if they're on your local disk's file system! For example, `gs://my-bucket/path/to/file` can be accessed in your jobs with `/gcs_dir/path/to/file`.

```{tip}
To improve I/O speeds, prefer using a bucket in the same cloud region as your job!
```

### Check Cluster and Job Status


::::{tab-set}
:::{tab-item} CLI
To quickly check the status of all jobs and clusters, run:

```{code-block} shell
oumi launch status
```

This will return a list of all jobs and clusters you've created across all registered cloud providers.

To further filter this list, you can optionally specify a cloud provider, cluster name, and/or job id. The results will be filtered to only jobs / clusters meeting the specified criteria. For example, the following command will return a list of jobs from all cloud providers running on a cluster named `my-cluster` with a job id of `my-job-id`:

```{code-block} shell
oumi launch status --cluster my-cluster --id my-job-id
```
:::

:::{tab-item} Python
To quickly check the status of all jobs and clusters, run:

```{code-block} python
import oumi.launcher as launcher

status_list = launcher.status()

print(status_list)
```

This will return a list of all jobs and clusters you've created across all registered cloud providers.

To further filter this list, you can optionally specify a cloud provider, cluster name, and/or job id. The results will be filtered to only jobs / clusters meeting the specified criteria. For example, the following command will return a list of jobs from all cloud providers running on a cluster named `my-cluster` with a job id of `my-job-id`:

```{code-block} python
import oumi.launcher as launcher

status_list = launcher.status(cluster="my-cluster", id="my-job-id")

print(status_list)
```
:::
::::



### View Logs

Often you'll want to view logs of running or terminated jobs.
To view the logs of your jobs on clouds supported by SkyPilot, run:

```{code-block} shell
sky logs my-cluster
```

### Cancel Jobs

::::{tab-set}
:::{tab-item} CLI
To cancel a running job without stopping the cluster, run:

```{code-block} shell
oumi launch cancel --cluster my-cluster --cloud gcp --id my-job-id
```

The id of the job can be obtained by running `oumi launch status`.
:::

:::{tab-item} Python
To cancel a running job without stopping the cluster, run:

```{code-block} python
import oumi.launcher as launcher

launcher.cancel(job_id="my-job-id", cloud_name="gcp", cluster_name="my-cluster")
```

The id of the job can be obtained by using `launcher.status()` as in the previous
section.
:::
::::



### Stop/Turn Down Clusters

::::{tab-set}
:::{tab-item} CLI
To stop the cluster when you are done to avoid extra charges, run:

```{code-block} shell
oumi launch stop --cluster my-cluster
```

In addition, the Oumi launcher automatically sets [`idle_minutes_to_autostop`](https://docs.skypilot.co/en/latest/reference/api.html#sky.launch) to 30, i.e. clusters will stop automatically after 30 minutes of no jobs running.

Stopped clusters preserve their disk, and are quicker to initialize than turning up a brand new cluster. Stopped clusters can be automatically restarted by specifying them in an `oumi launch up` command.

To turn down a cluster, which deletes their associated disk and removes them from our list of existing clusters, run:

```{code-block} shell
oumi launch down --cluster my-cluster
```
:::

:::{tab-item} Python
To stop the cluster when you are done to avoid extra charges, run:

```{code-block} python
import oumi.launcher as launcher

launcher.stop(cloud_name="gcp", cluster_name="my-cluster")
```

In addition, Oumi automatically sets [`idle_minutes_to_autostop`](https://docs.skypilot.co/en/latest/reference/api.html#sky.launch) to 30, i.e. clusters will stop automatically after 30 minutes of no jobs running.

Stopped clusters preserve their disk, and are quicker to initialize than turning up a brand new cluster. Stopped clusters can be automatically restarted by specifying them in a `launcher.up(...)` command.

To turn down a cluster, which deletes their associated disk and removes them from our list of existing clusters, run:

```{code-block} python
import oumi.launcher as launcher

launcher.down(cloud_name="gcp", cluster_name="my-cluster")
```
:::
::::
