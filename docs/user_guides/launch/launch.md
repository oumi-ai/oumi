# Launching Jobs

```{toctree}
:maxdepth: 2
:caption: Launch
:hidden:

deploy
remote
custom_cluster
```

In addition to running training locally, you can use the `launch` command in the Oumi CLI to run jobs on remote clusters. It provides a unified interface for running your code, allowing you to seamlessly switch between popular cloud providers and your own custom clusters!

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

## Overview

`oumi launch` provides you with all the capabilities you need to kickoff and monitor jobs running on remote machines.

We'll cover the most common use case here, which boils down to:

1. Using `oumi launch up` to create a cluster and run a job.
2. Using `oumi launch status` to check the status of your job and cluster.
3. Canceling jobs using `oumi launch cancel`
4. Turning down a cluster manually using `oumi launch down`

For a quick overview of all `oumi launch` commands, see our [CLI Launch Reference](/cli/commands.md#launch)

### Launch Jobs

To launch a job on your desired cloud, run:

```shell
oumi launch up --cluster my-cluster -c configs/recipes/smollm/sft/135m/quickstart_gcp_job.yaml
```

This command will create the cluster if it doesn't exist, and then execute the job on it. It can also run the job on an existing cluster with that name.

To launch on the cloud of your choice, use the `--resources.cloud` flag, ex. `--resources.cloud lambda`. Most of our configs run on GCP by default. See [this page](https://oumi.ai/docs/latest/api/oumi.launcher.html#oumi.launcher.JobResources.cloud) for all supported clouds, or run:

```shell
oumi launch which
```

To return immediately when the job is scheduled and not poll for the job's completion, specify the `--detach` flag:

```shell
oumi launch up --cluster my-cluster -c configs/recipes/smollm/sft/135m/quickstart_gcp_job.yaml --detach
```

To find out more about the GPUs available on your cloud provider, you can use skypilot:

```shell
sky show-gpus
```

If you made any code changes to the codebase (not including configs), you need to run
`pip install '.'` in the `run` section of the job config to install the
changes on the cluster.

#### Mount Cloud Storage

You can mount cloud storage like GCS or S3 to your job, which maps their remote paths to a directory on your job's disk. This is a fantastic way to write important information (such as data or model checkpoints) to a persistent disk that outlives your cluster's lifetime.

```{tip}
Writing your job's output to cloud storage is recommended for preemptable cloud instances, or jobs outputting a large amount of data like large model checkpoints. Data on local disk will be lost on job preemption, and your job's local disk may not have enough storage for multiple large model checkpoints.
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
To quickly check the status of all jobs and clusters, run:

```shell
oumi launch status
```

This will return a list of all jobs and clusters you've created across all registered cloud providers.

To further filter this list, you can optionally specify a cloud provider, cluster name, and/or job id. The results will be filtered to only jobs / clusters meeting the specified criteria. For example, the following command will return a list of jobs from all cloud providers running on a cluster named `my-cluster` with a job id of `my-job-id`:

```shell
oumi launch status --cluster my-cluster --id my-job-id
```

### View Logs

Often you'll want to view logs of running or terminated jobs.
To view the logs of your jobs on clouds supported by SkyPilot, run:

```shell
sky logs my-cluster
```

### Cancel Jobs

To cancel a running job without stopping the cluster, run:

```shell
oumi launch cancel --cluster my-cluster --cloud gcp --id 1
```

The id of the job can be obtained by running `oumi launch status`.

### Stop/Turn Down Clusters

To stop the cluster when you are done to avoid extra charges, run:

```shell
oumi launch stop --cluster my-cluster
```

In addition, the Oumi launcher automatically sets [`idle_minutes_to_autostop`](https://docs.skypilot.co/en/latest/reference/api.html#sky.launch) to 30, i.e. clusters will stop automatically after 30 minutes of no jobs running.

Stopped clusters preserve their disk, and are quicker to initialize than turning up a brand new cluster. Stopped clusters can be automatically restarted by specifying them in an `oumi launch up` command.

To turn down a cluster, which deletes their associated disk and removes them from our list of existing clusters, run:

```shell
oumi launch down --cluster my-cluster
```
