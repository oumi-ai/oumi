# Dev Environment Setup

This guide will help you set up a development environment for contributing to Oumi.

## 1. Set up Oumi

### 1.1 Fork the Oumi repository

You can create a fork of Oumi by clicking the [Fork button](https://github.com/oumi-ai/oumi/fork) in the upper right of the repository. This will create a fork of Oumi associated with your GitHub account.

### 1.2 Clone your fork of the Oumi repository

Now you're ready to clone your fork to your local disk and set up the original repository as a remote:

```shell
git clone git@github.com:<your Github handle>/oumi.git
cd oumi
git remote add upstream https://github.com/oumi-ai/oumi.git
```

### 1.3 Install Miniconda (if not done already)

The simplest way to install Miniconda is to run the following code after cloning the Oumi repository:

```shell
make install-miniconda
```

Alternatively, install Miniconda from the [Anaconda website](https://docs.anaconda.com/free/miniconda/miniconda-install/).

### 1.4 Create a development branch

```{warning}
Do not make changes directly to the `main` branch, even on your fork!
```

Your changes should live on a development branch so you can later create a Pull Request to merge your changes into `main`.

```shell
git checkout -b the-name-of-your-branch
```

### 1.5 Install Oumi package and its dependencies

This command creates a new Conda env, installs relevant packages, and installs pre-commit.

```shell
make setup
```

If you'd like to only run the pre-commits before a push, instead of every commit, you can run:

```shell
pre-commit uninstall
pre-commit install --install-hooks --hook-type pre-push
```

#### 1.5.1 Optional dependencies

Follow [these instructions](../get_started/installation.md#optional-dependencies) to install optional dependencies you may want depending on your use case.

### 1.6 [optional] Add an Oumi alias to your shell

Add the following alias to {.zshrc or .bashrc}:

```shell
alias oumi-conda="cd ~/<YOUR_PATH>/oumi && conda activate oumi"
```

This will change your directory into the Oumi repo and activate the Oumi Conda
environment. Test that this works with:

```shell
source ~/{.zshrc or .bashrc}
oumi-conda
```

## 2. [optional] Set up SkyPilot

The Oumi launcher can be used to launch jobs on remote clusters. Our launcher integrates with SkyPilot to launch jobs on popular cloud providers (GCP, Lambda, etc.). To enable the Oumi launcher to run on your preferred cloud, make sure to follow the setup instructions in our [launch guide](../user_guides/launch/launch.md).

### 2.1 [optional] Force all job configs to do Oumi editable install

If you're making changes to the Oumi codebase which you can't test locally (ex. training of large 8B+ models), you can use the Oumi launcher to launch remote jobs on GPU clusters and test your changes there. By default, most Oumi configs in the [`configs` directory](https://github.com/oumi-ai/oumi/tree/main/configs) install oumi from PyPI (i.e. `pip install oumi`). However, for the remote job to pick up your local changes, you need to install from your local copy of the repo (i.e. `pip install -e .`).

To automate this process, you can set the `OUMI_FORCE_EDITABLE_INSTALL` environment variable to `"true"`. This experimental feature will automatically detect and modify oumi PyPI installs to instead install in editable mode from source using regex.

(optional-set-up-huggingface)=

## 3. [optional] Set up HuggingFace

Oumi integrates with HuggingFace (HF) Hub for access to models and datasets. While most models and datasets are publicly accessible, some like Llama are gated, requiring you to be logged in and be approved for access.

1. [Sign up for HuggingFace](https://huggingface.co/join) if you haven't done so already.
2. Create a [user access token](https://huggingface.co/docs/hub/en/security-tokens). If you only need to read content from the Hub, create a `read` token. If you also plan to push datasets or models to the Hub, create a `write` token.
3. Run the following to log in on your machine, using the token created in the previous step:

   ```shell
   huggingface-cli login
   ```

   This will save your token in the HF cache directory at `~/.cache/huggingface/token`. Oumi jobs mount this file to remote clusters to access gated content there. See [this config](https://github.com/oumi-ai/oumi/blob/535f28b3c93a6423abc247e921a00d2b27de14df/configs/recipes/llama3_1/sft/8b_full/gcp_job.yaml#L19) for an example.

### 3.1 Getting access to Llama

Llama models are gated on HF Hub. To gain access, sign the agreement on your desired Llama model's Hub page. It usually takes a few hours to get access to the model after signing the agreement. There is a separate agreement for each version of Llama:

- [Llama 2](https://huggingface.co/meta-llama/Llama-2-70b-hf)
- [Llama 3](https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct)
- [Llama 3.1](https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct)
- [Llama 3.2](https://huggingface.co/meta-llama/Llama-3.2-90B-Vision-Instruct)
- [Llama 3.3](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct)
- [Llama 4](https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct)

(optional-set-up-weights-and-biases)=

## 4. [optional] Set up Weights and Biases

Oumi integrates with Weights and Biases (WandB) to track the results of training and evaluation runs. Run the following to log in on your machine:

```shell
wandb login
```

This will save your login info at `~/.netrc`. Oumi jobs mount this file to remote clusters to allow them to log to WandB. See [this config](https://github.com/oumi-ai/oumi/blob/535f28b3c93a6423abc247e921a00d2b27de14df/configs/recipes/llama3_1/sft/8b_full/gcp_job.yaml#L16) for an example.

## 5. [optional] Set up VSCode

We recommend using [VSCode](https://code.visualstudio.com/) as the IDE. See our {doc}`/user_guides/train/environments/vscode` guide for recommended setup instructions.

You can also use VSCode to run Jupyter notebooks in the Oumi repository. See our {doc}`/user_guides/train/environments/notebooks` guide for more information.

## 6. [optional] Test your setup

To test that your setup is complete, you can run `oumi launch up -c configs/recipes/llama3_1/sft/8b_lora/gcp_job.yaml --cluster llama8b-lora`. This requires step 4 (SkyPilot GCP), step 5 (HF), step 5.1 (Llama 3.1 access), and step 6 (WandB).

After the training is finished, see the training results by loging into your wandb account.

Please keep an eye on your costs, and don't forget to tear down your cluster when you're done.

```bash
sky down cluster_name
```

or

```bash
oumi launch down --cluster cluster_name
```
