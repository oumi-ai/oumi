# Dev Environment Setup

## 1. Install Miniconda

   <https://docs.anaconda.com/free/miniconda/miniconda-install/>

## 2. Create a new environment for LeMa and activate it

   ```shell
   conda create -n lema python=3.11
   conda activate lema
   ```

## 3. Install GitHub CLI

### 3.1 Instructions for Mac

   Install Homebrew (the command below was copied from <www.brew.sh>)

   ```shell
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

   Then follow "Next steps" (shown after installation) to add `brew` into `.zprofile`

   ```shell
   brew install gh
   ```

### 3.2 Instructions for **Linux**, including [WSL](https://learn.microsoft.com/en-us/windows/wsl/)

  Follow <https://github.com/cli/cli?tab=readme-ov-file#conda>

   ```shell
   conda install gh --channel conda-forge
   ```

## 4. Authorize Github CLI (easier when using SSH protocol)

   ```shell
   gh auth login
   ```

## 5. Set your Github name and email

   ```shell
   git config --global user.name "YOUR_NAME"
   git config --global user.email YOUR_USERNAME@learning-machines.ai
   ```

## 6. Clone the LeMa repository

   ```shell
   gh repo clone openlema/lema
   ```

## 7. Install LeMa package and its dependencies

   ```shell
   cd lema
   pip install -e '.[all]'
   ```

## 8. Install pre-commit hooks

   ```shell
   pre-commit install  # recommended
   ```

   If you'd like to only run the pre-commits before a push, you can use:

   ```shell
   pre-commit install --install-hooks --hook-type pre-push
   ```

   Alternatively, you can run the pre-commit hooks manually with:

   ```shell
   pre-commit run --all-files
   ```

## 9. [optional] Add a LeMa shortcut in your environment {.zshrc or .bashrc}

   ```shell
   alias lema="cd ~/<YOUR_PATH>/lema && conda activate lema"
   ```

   Ensure that this works with:

   ```shell
   source ~/{.zshrc or .bashrc}
   lema
   ```

## 10. [optional] Install [Git Credential Manager](https://docs.github.com/en/get-started/getting-started-with-git/about-remote-repositories#cloning-with-https-urls) for authentication management
