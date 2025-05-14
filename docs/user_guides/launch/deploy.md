# Deploying a Job

In this tutorial we'll take a working {py:class}`~oumi.core.configs.JobConfig` and deploy it remotely on a cluster of your choice.

This guide dovetails nicely with our [Finetuning Tutorial](https://github.com/oumi-ai/oumi/blob/main/notebooks/Oumi%20-%20Finetuning%20Tutorial.ipynb) where you create your own TrainingConfig and run it locally. Give it a try if you haven't already!


## Launching Your Job

`````{note}
Try using our sample helloworld job for this tutorial:
````{dropdown} configs/examples/misc/hello_world_gcp_job.yaml
```{literalinclude} ../../../configs/examples/misc/hello_world_gcp_job.yaml
:language: yaml
```
````
`````

Let's get started with launching a job! Don't worry about the nitty-gritty&mdash;we'll
address configuring your job in the following sections.

::::{tab-set}
:::{tab-item} CLI
You can easily kick off a job directly from the CLI:
```{code-block} shell
oumi launch up --cluster my-cluster -c configs/examples/misc/hello_world_gcp_job.yaml
```

At any point you can easily change the cloud where your job will run by modifying the job's `resources.cloud` parameter:

```{code-block} shell
oumi launch up --cluster my-cluster -c configs/examples/misc/hello_world_gcp_job.yaml --resources.cloud local
```
:::

:::{tab-item} Python
First let's load your {py:class}`~oumi.core.configs.JobConfig`:
``` {code-block} python
import oumi.launcher as launcher
# Read our JobConfig from the YAML file.
working_dir = "YOUR_WORKING_DIRECTORY" # Specify this value
job_config = launcher.JobConfig.from_yaml(str(Path(working_dir) / "job.yaml"))
```

At any point you can easily change the cloud where your job will run by modifying the job's `resources.cloud` parameter:

``` {code-block} python
# Manually set the cloud to use.
job_config.resources.cloud = "local"
```

Once you have a job config, kicking off your job is simple:

``` {code-block} python
# You can optionally specify a cluster name here. If not specified, a random name will
# be generated. This is also useful for launching multiple jobs on the same cluster.
cluster_name = None

# Launch the job!
cluster, job_status = launcher.up(job_config, cluster_name)
print(f"Job status: {job_status}")
```
:::
::::



Don't worry if you see any errors from the launcher--you may need to configure permissions to run a job on your specified cloud. The error message should provide you with the proper command to run to authenticate (for GCP this is often `gcloud auth application-default login`).

We can quickly check on the status of our job using the `cluster` returned in the previous command:

::::{tab-set}
:::{tab-item} CLI
``` {code-block} shell
oumi launch status
```
:::

:::{tab-item} Python
``` {code-block} python
while job_status and not job_status.done:
    print("Job is running...")
    time.sleep(15)
    job_status = cluster.get_job(job_status.id)

print("Job is done!")
```
:::
::::


Now that we're done with the cluster, let's turn it down to stop billing for non-local clouds.


::::{tab-set}
:::{tab-item} CLI
``` {code-block} shell
oumi launch down --cluster my-cluster
```
:::

:::{tab-item} Python
``` {code-block} python
cluster.down()
```
:::
::::

## Choosing a Cloud
We'll be using the Oumi Launcher to run remote training. To use the launcher, you need to specify which cloud you'd like to run training on.
We'll list the clouds below:

::::{tab-set}
:::{tab-item} CLI
``` {code-block} shell
oumi launch which
```
:::

:::{tab-item} Python
``` {code-block} python
import oumi.launcher as launcher

# Print all available clouds
print(launcher.which_clouds())
```
:::
::::

#### Local Cloud
If you don't have any clouds set up yet, feel free to use the `local` cloud. This will simply execute your job on your current device as if it's a remote cluster. Hardware requirements are ignored for the `local` cloud.

#### Other Providers
Note that to use a cloud you must already have an account registered with that cloud provider.

For example, GCP, RunPod, and Lambda require accounts with billing enabled.

Once you've picked a cloud, move on to the next step.

## Preparing Your JobConfig
Let's get started by creating your {py:class}`~oumi.core.configs.JobConfig`. In the config below, feel free to change `cloud: local` to the cloud you chose in the previous step.

A sample job is provided below:
````{dropdown} job.yaml
```{code-block} yaml
name: job-tutorial
resources:
  cloud: local
  # Accelerators is ignored for the local cloud.
  # This is required for other clouds like GCP, AWS, etc.
  accelerators: A100

# Upload working directory to remote.
# If on the local cloud, we CD into the working directory before running the job.
working_dir: .

envs:
  TEST_ENV_VARIABLE: '"Hello, World!"'
  OUMI_LOGGING_DIR: "deploy_tutorial/logs"

# `setup` will always be executed once when a cluster is created
setup: |
  echo "Running setup..."

run: |
  set -e  # Exit if any command failed.

  echo "$TEST_ENV_VARIABLE"
```
````

## Deploying a Training Config

In our [Finetuning Tutorial](https://github.com/oumi-ai/oumi/blob/main/notebooks/Oumi%20-%20Finetuning%20Tutorial.ipynb), we created and saved a TrainingConfig. We then invoked training by running
```shell
oumi train -c "$tutorial_dir/train.yaml"
```

You can also run that command as a job! Simply update the "run" section of the {py:class}`~oumi.core.configs.JobConfig` with your desired command:


::::{tab-set}
:::{tab-item} CLI
``` {code-block} shell
export PATH_TO_YOUR_TRAIN_CONFIG="deploy_tutorial/train.yaml" # Make sure this exists!
oumi launch up --cluster my-new-cluster -c deploy_tutorial/job.yaml --run "oumi train -c $PATH_TO_YOUR_TRAIN_CONFIG" --setup "pip install oumi"
```
:::

:::{tab-item} Python
``` {code-block} python
working_dir = "YOUR_WORKING_DIRECTORY" # Specify this value
path_to_your_train_config = Path(working_dir) / "train.yaml"  # Make sure this exists!

# Set the `run` command to run your training script.
job_config.run = f'oumi train -c "{path_to_your_train_config}"'
# Make sure we install oumi
job_config.setup = "pip install oumi"
```
:::
::::
