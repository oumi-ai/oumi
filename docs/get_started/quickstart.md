# Quickstart

## 📋 Prerequisites

Let's start by installing Oumi. You can easily install the latest stable version of Oumi with the following commands:

```bash
pip install oumi

# Optional: If you have an Nvidia or AMD GPU, you can install the GPU dependencies
pip install oumi[gpu]
```

If you need help setting up your environment (python, pip, git, etc), you can find detailed instructions in the {doc}`/development/dev_setup` guide. The {doc}`installation guide </get_started/installation>` offers more details on how to install Oumi for your specific environment and use case.

## 👋 Introduction

Now that we have Oumi installed, let's get started with the basics! We're going to use the `oumi` command-line interface (CLI) to train, evaluate, and run inference with a model.

We'll use a small model (`SmolLM-135M`) so that the examples can run fast on both CPU and GPU. `SmolLM` is a family of state-of-the-art small models with 135M, 360M, and 1.7B parameters, trained on a new high-quality dataset. You can learn more about about them in [this blog post](https://huggingface.co/blog/smollm).

For a full list of recipes, including larger models like Llama 3.2, you can explore the {doc}`recipes page </resources/recipes>`.

## 💻 Oumi CLI

The general structure of Oumi CLI commands is:

```bash
oumi <command> [options]
```

For detailed help on any command, you can use the `--help` option:

```bash
oumi --help            # for general help
oumi <command> --help  # for command-specific help
```

The available commands are:

- `train`
- `evaluate`
- `infer`
- `launch`
- `judge`

Let's go through some examples of each command.

## 📚 Training

You can quickly start training a model using any of existing {doc}`recipes </resources/recipes>` or your own {doc}`custom configs </user_guides/train/configuration>`. The following command will start training using the recipe in `configs/recipes/smollm/sft/135m/quickstart_train.yaml`:

````{dropdown} configs/recipes/smollm/sft/135m/quickstart_train.yaml
```{literalinclude} ../../configs/recipes/smollm/sft/135m/quickstart_train.yaml
:language: yaml
```
````

```bash
oumi train -c configs/recipes/smollm/sft/135m/quickstart_train.yaml
```

Any Oumi command which takes a config path as an argument (`train`, `evaluate`, `infer`, etc.) can override parameters from the command line. See {doc}`/cli/commands` for more details. For example:

```bash
oumi train -c configs/recipes/smollm/sft/135m/quickstart_train.yaml \
  --training.max_steps 20 \
  --training.learning_rate 1e-4 \
  --data.train.datasets[0].shuffle true \
  --training.output_dir output/smollm-135m-sft
```

To run the same recipe on your own dataset (e.g., in our supported JSON or JSONL formats), you can override the dataset name and path. You can try this functionality out by downloading the `alpaca_cleaned` dataset manually via the huggingface CLI, then including that local path in your run.

```bash
huggingface-cli download yahma/alpaca-cleaned --repo-type dataset --local-dir /path/to/local/dataset

oumi train -c configs/recipes/smollm/sft/135m/quickstart_train.yaml \
  --data.train.datasets "[{dataset_name: text_sft, dataset_path: /path/to/local/dataset}]" \
  --training.output_dir output/smollm-135m-sft-custom
```

You can also train on multiple GPUs (make sure to [install the GPU dependencies](/get_started/installation.md#optional-dependencies) if not already installed).

For example, if you have a machine with 4 GPUs, you can run this command to launch a local distributed training run:

```bash
oumi distributed torchrun \
  -m oumi train -c configs/recipes/smollm/sft/135m/quickstart_train.yaml \
  --training.output_dir output/smollm-135m-sft-dist
```

You can also use torchrun directly in standalone mode.

```bash
torchrun --standalone --nproc-per-node 4 --log-dir ./logs \
-m oumi train -c configs/recipes/smollm/sft/135m/quickstart_train.yaml \
--training.output_dir output/smollm-135m-sft-dist
```


## 📊 Evaluation

To evaluate a trained model:

````{dropdown} configs/recipes/smollm/evaluation/135m/quickstart_eval.yaml
```{literalinclude} ../../configs/recipes/smollm/evaluation/135m/quickstart_eval.yaml
:language: yaml
```
````

Using a model downloaded from HuggingFace:

```bash
oumi evaluate -c configs/recipes/smollm/evaluation/135m/quickstart_eval.yaml \
  --model.model_name HuggingFaceTB/SmolLM2-135M-Instruct
```

Or, with our newly trained model saved on disk:

```bash
oumi evaluate -c configs/recipes/smollm/evaluation/135m/quickstart_eval.yaml \
  --model.model_name output/smollm135m.fft
```

If you saved your model to a different directory such as `output/smollm-135m-sft-dist`, you need only change `--model.model_name`.

To explore the benchmarks that our evaluations support, including HuggingFace leaderboards and AlpacaEval, visit our {doc}`evaluation guide </user_guides/evaluate/evaluate>`.

## 🧠 Inference

To run inference with a trained model:

````{dropdown} configs/recipes/smollm/inference/135m_infer.yaml
```{literalinclude} ../../configs/recipes/smollm/inference/135m_infer.yaml
:language: yaml
```
````

Using a model downloaded from HuggingFace:

```bash
oumi infer -c configs/recipes/smollm/inference/135m_infer.yaml \
  --generation.max_new_tokens 40 \
  --generation.temperature 0.7 \
  --interactive
```

Or, with our newly trained model saved on disk:

```bash
oumi infer -c configs/recipes/smollm/inference/135m_infer.yaml \
  --model.model_name output/smollm135m.fft \
  --generation.max_new_tokens 40 \
  --generation.temperature 0.7 \
  --interactive
```

To learn more about running inference locally or remotely (including OpenAI, Google, Anthropic APIs) and leveraging inference engines to parallelize and speed up your jobs, visit our {doc}`inference guide </user_guides/infer/infer>`.

## ☁️ Launching Jobs in the Cloud

So far we have been using Oumi locally. But one of the most exciting and unique Oumi features, compared to similar frameworks, is its integrated ability to launch jobs directly *to the cloud* (GCP, AWS, Azure, etc).

This section of the quickstart is going to be a little different than the others, so please read this next bit carefully before you proceed.

* This tutorial uses GCP; you'll need a [GCP account](https://cloud.google.com/free?hl=en). You can also use other cloud providers, such as AWS, Azure, etc. See {doc}`running jobs remotely </user_guides/launch/launch>` for more details.

````{dropdown} Configuring GCP Account

* In particular, Oumi uses [Skypilot](https://docs.skypilot.co/en/latest/docs/index.html), and the recommended way to use SkyPilot and GCP is with a [GCP service account](https://cloud.google.com/iam/docs/service-account-overview)
* You will need to install Oumi with GCP support: `pip install oumi[gcp]`. Please note that we recommend setting up a different environment for each cloud provider you wish to use.
* Depending on your precise use case, you may also need to install a few other packages from Google

```bash
conda install -c conda-forge google-cloud-sdk -y
conda install -c conda-forge google-api-python-client -y
conda install -c conda-forge google-cloud-storage -y
```

* There are multiple ways to handle credentials with GCP service accounts. We recommend creating a service account key in JSON format, then downloading it to the machine from which you plan to launch the cloud job. After that, you'll need to run a few more setup commands.

```bash
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json
gcloud auth activate-service-account --key-file=$GOOGLE_APPLICATION_CREDENTIALS
gcloud config set project <YOUR_PROJECT>
```

You can now run `sky check` to confirm GCP is enabled.

If you get stuck, please refer to our {doc}`running jobs remotely </user_guides/launch/launch>` section, as well as the documentation for GCP and SkyPilot linked above, for more information.
````

### Launching your first cloud job with Oumi

Once the one-time setup is out of the way, launching a new cloud job with Oumi is very simple.

````{dropdown} configs/recipes/smollm/sft/135m/quickstart_gcp_job.yaml
```{literalinclude} ../../configs/recipes/smollm/sft/135m/quickstart_gcp_job.yaml
:language: yaml
```
````

```bash
oumi launch up -c configs/recipes/smollm/sft/135m/quickstart_gcp_job.yaml
```

To launch an evaluation job:

````{dropdown} configs/recipes/smollm/evaluation/135m/quickstart_gcp_job.yaml
```{literalinclude} ../../configs/recipes/smollm/evaluation/135m/quickstart_gcp_job.yaml
:language: yaml
```
````

```bash
oumi launch up -c configs/recipes/smollm/evaluation/135m/quickstart_gcp_job.yaml
```

After you run one of the above commands, you should see some console output from Oumi which describes how your job is being provisioned and how the cloud installation is proceeding. In particular, your cluster will be assigned a semi-random name such as `sky-7fdd-ab183`, which you should take note of.

After 15 minutes or so, Oumi should tell you that the run is complete.

If you want to see the logs from your cloud run, you can pull them down to your local machine --

```bash
sky logs --sync-down sky-7fdd-ab183
```

**Cloud services can be expensive!** Please keep an eye on your costs, and don't forget to tear down your cluster when you're done with this tutorial.

```bash
sky down sky-7fdd-ab183
```

This command will **destroy your cluster**, including all data on those remote machines, so save your logs and artifacts first!

### 🧭 What's next?

Although this example used GCP, Oumi natively supports a wide range of cloud providers. To explore the Cloud providers that we support, visit {doc}`running jobs remotely </user_guides/launch/launch>`.

## 🔗 Community

⭐ If you like Oumi and you would like to support it, please give it a star on [GitHub](https://github.com/oumi-ai/oumi).

👋 If you are interested in contributing, please read the [Contributor’s Guide](/development/contributing).
