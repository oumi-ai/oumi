# Launch

```{toctree}
:maxdepth: 2
:caption: Judge
:hidden:

deploy
remote
```

To train on a cloud GPU cluster, first make sure to have all the dependencies installed:

For specific cloud providers:

  ```bash
  pip install oumi[aws]     # For Amazon Web Services
  pip install oumi[azure]   # For Microsoft Azure
  pip install oumi[gcp]     # For Google Cloud Platform
  pip install oumi[lambda]  # For Lambda Cloud
  pip install oumi[runpod]  # For RunPod
  ```

Then setup your cloud credentials:

- [Google Cloud](https://github.com/oumi-ai/oumi/wiki/Clouds-Setup)
- [Runpod](https://skypilot.readthedocs.io/en/latest/getting-started/installation.html#runpod)
- [Lambda Labs](https://skypilot.readthedocs.io/en/latest/getting-started/installation.html#lambda-cloud)

Your environment should be ready! Use this to check:

```shell
sky check
```

You can look at the existing clusters with the following command:

```shell
sky status
```

To see the available GPUs, you can use the following command:

```shell
sky show-gpus
```

You can add the `-a` flag to show all GPUs. Example GPUs include `A100` (40GB), `A100-80GB`, and `A100-80GB-SXM`.

To launch a job on the cloud, you can use the following command:

```shell
oumi launch -c oumi-cluster configs/recipes/gpt2/pretraining/sky_job.yaml
```

To launch on the cloud of your choice, use the `--cloud` flag, ex. `--cloud gcp`.

Once you have already launched a job, you can use the following command to execute a job on an existing cluster:

```shell
oumi launch -c oumi-cluster configs/recipes/gpt2/pretraining/sky_job.yaml
```

If you made any code changes to the codebase (not including configs), you need to run
`pip install '.'` in the `run` section of the SkyPilot config to install the
changes on the cluster.

Remember to stop the cluster when you are done to avoid extra charges. You can either do it manually (`sky down oumi-cluster`), or use the following to automatically take it down after 10 minutes of inactivity:

```shell
sky autostop oumi-cluster -i 10
```

Alternatively, you can include `-i 10` into your `sky launch` command (as shown above).
