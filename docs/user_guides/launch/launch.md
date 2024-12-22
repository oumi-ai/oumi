# Launch

```{toctree}
:maxdepth: 2
:caption: Launch
:hidden:

deploy
remote
```

Oumi launcher allows you to run jobs on remote clusters. It provides a unified interface, allowing you to seamlessly switch between popular cloud providers and your own custom clusters!

## Setup

Oumi launcher integrates with SkyPilot to launch jobs on popular cloud providers. To run on a cloud GPU cluster, first make sure to have all the dependencies installed for your desired cloud provider:

  ```shell
  pip install oumi[aws]     # For Amazon Web Services
  pip install oumi[azure]   # For Microsoft Azure
  pip install oumi[gcp]     # For Google Cloud Platform
  pip install oumi[lambda]  # For Lambda Cloud
  pip install oumi[runpod]  # For RunPod
  ```

Then, you need to enable your desired cloud provider in SkyPilot. Run `sky check` to check which providers you have enabled, along with instructions on how to enable the ones you don't. More detailed setup instructions can be found in [SkyPilot's documentation](https://skypilot.readthedocs.io/en/latest/getting-started/installation.html#cloud-account-setup).

## Overview

To view your existing clusters, run:

```shell
oumi launch status
```

To view available GPUs, run:

```shell
sky show-gpus
```

You can add the `-a` flag to show all GPUs. Example GPUs include `A100` (40GB), `A100-80GB`, and `A100-80GB-SXM`.

### Launch jobs

To launch a job on your desired cloud, run:

```shell
oumi launch up --cluster oumi-cluster -c configs/recipes/smollm/launch/135m_gcp_train_quickstart.yaml
```

This command will create the cluster if it doesn't exist, and then execute the job on it. It can also run the job on an existing cluster with that name.

To launch on the cloud of your choice, use the `--resources.cloud` flag, ex. `--resources.cloud lambda`. Most of our configs run on GCP by default. See [this page](https://oumi.ai/docs/latest/api/oumi.launcher.html#oumi.launcher.JobResources.cloud) for all supported clouds, or run:

```shell
oumi launch which
```

To return immediatly when the job is scheduled and not poll for the job's completion, specify the `--detach` flag.

If you made any code changes to the codebase (not including configs), you need to run
`pip install '.'` in the `run` section of the job config to install the
changes on the cluster.

### View logs

To view the logs of your jobs on clouds supported by SkyPilot, run:

```shell
sky logs oumi-cluster
```

### Cancel jobs

To cancel a running job without stopping the cluster, run:

```shell
oumi launch cancel --cluster oumi-cluster --cloud gcp --id 1
```

The id of the job can be obtained by running `oumi launch status`.

### Stop/turn down clusters

To stop the cluster when you are done to avoid extra charges, run:

```shell
oumi launch stop --cluster oumi-cluster
```

In addition, the Oumi launcher automatically sets [`idle_minutes_to_autostop`](https://docs.skypilot.co/en/latest/reference/api.html#sky.launch) to 30, i.e. clusters will stop automatically after 30 minutes of no jobs running.

Stopped clusters preserve their disk, and are quicker to initialize than turning up a brand new cluster. Stopped clusters can be automatically restarted by specifying them in an `oumi launch up` command.

To turn down a cluster, which deletes their associated disk and removes them from our list of existing clusters, run:

```shell
oumi launch down --cluster oumi-cluster
```
