# Dev Environment Setup

This guide will help you set up a development environment for contributing to Oumi. If you already have a working environment, you can skip to [Set up Oumi](#set-up-oumi).

## 1. Install Miniconda

The simplest way to install Miniconda is to first clone the Oumi repository (step 3.1 [below](#clone-the-oumi-repository)), then run:

```shell
make install-miniconda
```

Alternatively, install Miniconda from the [Anaconda website](https://docs.anaconda.com/free/miniconda/miniconda-install/).

## 2. Set up GitHub

### 2.1 Install GitHub CLI

#### 2.1.1 Installation instructions for Windows

We strongly suggest that Windows users set up [WSL](https://learn.microsoft.com/en-us/windows/wsl/) using [these instructions](https://learn.microsoft.com/en-us/windows/wsl/install). Then, proceed to [step 2.1.3](#installation-instructions-for-linux-including-wsl).

#### 2.1.2 Installation instructions for Mac

Install Homebrew (command copied from <https://www.brew.sh>):

```shell
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Then follow "Next steps" (shown after installation) to add `brew` into `.zprofile`:

```shell
brew install gh
```

#### 2.1.3 Installation instructions for Linux, including [WSL](https://learn.microsoft.com/en-us/windows/wsl/)

Use Conda to install the `gh` CLI (command copied from <https://github.com/cli/cli?tab=readme-ov-file#conda>):

```shell
conda install gh --channel conda-forge
```

### 2.2 Authorize GitHub CLI

```shell
gh auth login
```

It is recommended to select "SSH", when asked "What is your preferred protocol for Git operations on this host."

### 2.3 Set your GitHub name and email

```shell
git config --global user.name "YOUR_NAME"
git config --global user.email "YOUR_EMAIL"
```

The name and email will be used to identify your contributions to the Oumi repository. To ensure that commits are attributed to you and appear in your contributions graph, use an email address that is connected to your account on GitHub, or the noreply email address provided to you in your email settings.

You can find more instructions [here](https://docs.github.com/en/account-and-profile/setting-up-and-managing-your-personal-account-on-github/managing-email-preferences/setting-your-commit-email-address).

### 2.4 [optional] Install [Git Credential Manager](https://docs.github.com/en/get-started/getting-started-with-git/about-remote-repositories#cloning-with-https-urls) for authentication management

## 3. Set up Oumi

### 3.1 Clone the Oumi repository

```shell
gh repo clone oumi-ai/oumi
cd oumi
```

### 3.2 Install Oumi package and its dependencies

This command creates a new Conda env, installs relevant packages, and installs pre-commit.

```shell
make setup
```

If you'd like to only run the pre-commits before a push, instead of every commit, you can run:

```shell
pre-commit uninstall
pre-commit install --install-hooks --hook-type pre-push
```

#### 3.2.1 Optional dependencies

Follow [these instructions](../get_started/installation.md#optional-dependencies) to install optional dependencies you may want depending on your use case.

### 3.3 [optional] Add an Oumi alias to your shell

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

## 4. [optional] Set up SkyPilot

The Oumi launcher can be used to launch jobs on remote clusters. Our launcher integrates with SkyPilot to launch jobs on popular cloud providers (GCP, Lambda, etc.). To enable the Oumi launcher to run on your preferred cloud, make sure to follow the setup instructions in our [launch guide](../user_guides/launch/launch.md).

## 5. [optional] Set up HuggingFace

Oumi integrates with HuggingFace (HF) Hub for access to models and datasets. While most models and datasets are publicly accessible, some like Llama are gated, requiring you to be logged in and be approved for access.

1. [Sign up for HuggingFace](https://huggingface.co/join) if you haven't done so already.
2. Create a [user access token](https://huggingface.co/docs/hub/en/security-tokens). If you only need to read content from the Hub, create a `read` token. If you also plan to push datasets or models to the Hub, create a `write` token.
3. Run the following to log in on your machine, using the token created in the previous step:

   ```shell
   huggingface-cli login
   ```

   This will save your token in the HF cache directory at `~/.cache/huggingface/token`. Oumi jobs mount this file to remote clusters to access gated content there. See [this config](https://github.com/oumi-ai/oumi/blob/535f28b3c93a6423abc247e921a00d2b27de14df/configs/recipes/llama3_1/sft/8b_full/gcp_job.yaml#L19) for an example.

### 5.1 Getting access to Llama

Llama models are gated on HF Hub. To gain access, sign the agreement on your desired Llama model's Hub page. It usually takes a few hours to get access to the model after signing the agreement. There is a separate agreement for each version of Llama:

- [Llama 2](https://huggingface.co/meta-llama/Llama-2-70b-hf)
- [Llama 3](https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct)
- [Llama 3.1](https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct)
- [Llama 3.2](https://huggingface.co/meta-llama/Llama-3.2-90B-Vision-Instruct)
- [Llama 3.3](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct)

## 6. [optional] Set up Weights and Biases

Oumi integrates with Weights and Biases (WandB) to track the results of training and evaluation runs. Run the following to log in on your machine:

```shell
wandb login
```

This will save your login info at `~/.netrc`. Oumi jobs mount this file to remote clusters to allow them to log to WandB. See [this config](https://github.com/oumi-ai/oumi/blob/535f28b3c93a6423abc247e921a00d2b27de14df/configs/recipes/llama3_1/sft/8b_full/gcp_job.yaml#L16) for an example.

## 7. [optional] Set up VSCode

We recommend using [VSCode](https://code.visualstudio.com/) as the IDE. See our {doc}`/user_guides/train/environments/vscode` guide for recommended setup instructions.

You can also use VSCode to run Jupyter notebooks in the Oumi repository. See our {doc}`/user_guides/train/environments/notebooks` guide for more information.

## 8. [optional] Test your setup

To test that your setup is complete, you can run `oumi launch up -c configs/recipes/llama3_1/sft/8b_lora/gcp_job.yaml --cluster llama8b-lora`. This requires step 4 (SkyPilot GCP), step 5 (HF), step 5.1 (Llama 3.1 access), and step 6 (WandB).
