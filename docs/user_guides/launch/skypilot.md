# Running Jobs with SkyPilot

```{deprecated} 0.1.1
SkyPilot support is deprecated. Please use oumi launcher, which extends SkyPilot with support for oumi specific features.
```

To train on a cloud GPU cluster, first make sure to have all the dependencies installed:

```shell
pip install 'oumi[cloud]'
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
sky launch -c oumi-cluster configs/skypilot/sky_gpt2.yaml
```

To launch on the cloud of your choice, use the `--cloud` flag, ex. `--cloud gcp`.

Once you have already launched a job, you can use the following command to execute a job on an existing cluster:

```shell
sky exec -c oumi-cluster configs/skypilot/sky_gpt2.yaml
```

If you made any code changes to the codebase (not including configs), you need to run
`pip install '.'` in the `run` section of the SkyPilot config to install the
changes on the cluster.

Remember to stop the cluster when you are done to avoid extra charges. You can either do it manually (`sky down oumi-cluster`), or use the following to automatically take it down after 10 minutes of inactivity:

```shell
sky autostop oumi-cluster -i 10
```

Alternatively, you can include `-i 10` into your `sky launch` command (as shown above).
