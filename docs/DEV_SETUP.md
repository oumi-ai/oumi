# Dev Environment Setup

## 1. Install Miniconda

   <https://docs.anaconda.com/free/miniconda/miniconda-install/>

## 2. Set up GitHub

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

### 2.2 Authorize Github CLI (easier when using SSH protocol)

   ```shell
   gh auth login
   ```

### 2.3 Set your Github name and email

   ```shell
   git config --global user.name "YOUR_NAME"
   git config --global user.email YOUR_USERNAME@learning-machines.ai
   ```

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
   alias oumi="cd ~/<YOUR_PATH>/oumi && conda activate oumi"
   ```

   Ensure that this works with:

   ```shell
   source ~/{.zshrc or .bashrc}
   oumi
   ```
