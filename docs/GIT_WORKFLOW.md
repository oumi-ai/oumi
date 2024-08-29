# Simple Git Merge Workflow

This document outlines a `merge` based workflow.

The other option is a `rebase` based workflow, which some people prefer. `rebase` can rewrite history and make it difficult to track changes. The `merge` based workflow is simpler and easier to understand.

## Resources

- Vscode Git integration [intro & documentation](https://code.visualstudio.com/docs/sourcecontrol/intro-to-git)
- GitHub CLI [documentation](https://cli.github.com/manual/)

## Feature development

Note: this is using the `git` cli, but you can also use the GitHub CLI (`gh`) or `vscode` to perform these operations.

1. If not done already, clone the repository:

    ```shell
    git clone https://github.com/openlema/lema
    ```

    OR if you already have cloned the repository, make sure you are in the main branch and pull the latest changes:

    ```shell
    git checkout main
    git pull
    ```

2. Create a new branch for your work:

    ```shell
    git checkout -b my-username/my-new-feature
    ```

3. Make your changes and commit them:

    ```shell
    git add .
    git commit -m "A brief description of the changes"
    ```

4. Push your changes to the remote repository:

    ```shell
    git push
    ```

5. Create a pull request on GitHub and wait for the review and merge.

## Update my branch with the latest changes from the main branch

1. Make sure your local `main` is up to date

    ```shell
    git checkout main
    git pull main
    ```

2. Go back to your feature branch and merge the latest changes from `main`

    ```shell
    git checkout my-username/my-new-feature
    git merge main
    ```

3. Resolve any conflicts if there are any and commit the changes. I personally use the `vscode` editor to resolve conflicts.
4. Push the changes to the remote repository

    ```shell
    git push
    ```
