# Dev Environment Setup

This guide will help you set up a development environment for contributing to Oumi. If you already have a working environment, you can skip to [Set up Oumi](#set-up-oumi).

## 1. Install Miniconda

   <https://docs.anaconda.com/free/miniconda/miniconda-install/>

## 2. Set up GitHub

### 2.1.0 Installation instructions for Windows

   We strongly suggest that Windows users set up [WSL](https://learn.microsoft.com/en-us/windows/wsl/)

   Follow [these instructions](https://learn.microsoft.com/en-us/windows/wsl/install) to install WSL.

   Next, install conda in your WSL environment:

   ```shell
   mkdir -p ~/miniconda3
   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
   bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
   rm -rf ~/miniconda3/miniconda.sh
   ```

   And reinitialize Conda:

   ```shell
   ~/miniconda3/bin/conda init bash
   ~/miniconda3/bin/conda init zsh
   ```

### 2.1.1 Installation instructions for Mac

   Install Homebrew (the command below was copied from <www.brew.sh>)

   ```shell
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

   Then follow "Next steps" (shown after installation) to add `brew` into `.zprofile`

   ```shell
   brew install gh
   ```

### 2.1.2 Installation instructions for **Linux**, including [WSL](https://learn.microsoft.com/en-us/windows/wsl/)

  Follow <https://github.com/cli/cli?tab=readme-ov-file#conda>

   ```shell
   conda install gh --channel conda-forge
   ```

### 2.2 Authorize Github CLI

   ```shell
   gh auth login
   ```

It is recommended to select "SSH", when asked "What is your preferred protocol for Git operations on this host."

### 2.3 Set your Github name and email

   ```shell
   git config --global user.name "YOUR_NAME"
   git config --global user.email "YOUR_EMAIL"
   ```

The name and email will be used to identify your contributions to the Oumi repository. To ensure that commits are attributed to you and appear in your contributions graph, use an email address that is connected to your account on GitHub, or the noreply email address provided to you in your email settings.

You can find more instructions [here](hhttps://docs.github.com/en/account-and-profile/setting-up-and-managing-your-personal-account-on-github/managing-email-preferences/setting-your-commit-email-address).

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

## 4. [optional] Add an Oumi shortcut in your environment {.zshrc or .bashrc}

   ```shell
   alias oumi-conda="cd ~/<YOUR_PATH>/oumi && conda activate oumi"
   ```

   Ensure that this works with:

   ```shell
   source ~/{.zshrc or .bashrc}
   oumi-conda
   ```
