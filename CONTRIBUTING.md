# Contributing

Thanks for considering contributing to Oumi! This is a community-first effort. We strongly believe that if we all work together, we can ensure a better, safer, and more open future for frontier AI.

We welcome contributions for new models, datasets, algorithms, incremental improvements, bug fixes and anything else that could make Oumi better for all of us!

## Prerequisites

To set up the development environment on your local machine, clone the repository ("git clone https://github.com/oumi-ai/oumi") and run the following commands below inside the 'oumi' folder.

1\. Install the dependencies needed for testing and linting the code:

```bash
pip install -e '.[all]'
```

If you're using a machine with a GPU, you should also install the `gpu` dependencies:

```bash
pip install -e '.[all,gpu]'
```

2\. Configure [pre-commit](https://pre-commit.com/), which automatically formats
code before each commit:

```bash
pre-commit install
```

## Submitting a Contribution

To submit a contribution:

1. [Fork](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo)
a copy of the [Oumi](https://github.com/oumi-ai/oumi) repository into your own account. See 
[Forking a repository](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo#forking-a-repository)
for detailed steps.
2. Clone your fork locally, and add the Oumi repo as a remote repository:

    ```shell
    git clone git@github.com:<github_id>/oumi.git
    cd oumi
    git remote add upstream https://github.com/oumi-ai/oumi.git
    ```

3. Create a branch, and make your proposed changes.

    ```shell
    git checkout -b my-username/my-awesome-new-feature
    ```

4. When you are ready, submit a pull request into the Oumi repository!

## Pull request (PR) guidelines

Basic guidelines that will make your PR easier to review:

- **Title and Description**
  - Please include a concise title and clear PR description.
  - The title should allow someone to understand what the PR changes or does at a glance.
  - The description should allow someone to understand the contents of the PR *without* looking at the code.

- **Testing**
  - Please include tests with your PR!
  - If fixing a bug, add a test that would've caught the bug.
  - If adding a new feature, include unit tests for the new functionality.

- **Code Formatting and Type Checking**
  - Use `pre-commit` to handle formatting and type checking:
  - Ensure you have it installed as described in the [Prerequisites](#prerequisites) section.
  - Run pre-commit hooks before submitting your PR.

## Running Tests

To test your changes locally, run:

```shell
cd ./tests/
pytest -s
```

To run pre-commit hooks manually, run `pre-commit run --all-files`

## Code Style & Typing

See the [Oumi Style Guide](/STYLE_GUIDE.md) for guidelines on how to structure,
and format your code.
