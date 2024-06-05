# Contributing to LeMa

Thanks for considering contributing to LeMa! We welcome contributions for new models,
incremental improvements, datasets, and bug fixes!

TODO: Add info how to reach out to us: email group, Slack, etc.


## Prerequisites

To set up the development environment on your local machine, run the commands below.

1\. Install the dependencies needed for testing and linting the code:

<!--pytest.mark.skip-->
```bash
pip install -e '.[all]'
```

2\. Configure [pre-commit](https://pre-commit.com/), which automatically formats
code before each commit:

<!--pytest.mark.skip-->
```bash
pre-commit install
```

## Submitting a Contribution

To submit a contribution:

1. Fork a copy of the [LeMa](https://github.com/openlema/lema) repository into
your own account.

2. Clone your fork locally, and add the LeMa repo as a remote repository:

<!--pytest.mark.skip-->
```bash
git clone git@github.com:<github_id>/openlema/lema.git
cd ./lema/
git remote add upstream https://github.com/openlema/lema.git
```

3. Create a branch, and make your proposed changes.

<!--pytest.mark.skip-->
```bash
git checkout -b my-awesome-new-feature
```

4. When you are ready, submit a pull request into the LeMa repository!

## Pull request (PR) guidelines

Basic guidelines that will make your PR easier to review:

* Please include a concise title and clear PR description. The title should allow
someone to understand what the PR changes or does at a glance. The description
should allow someone to understand the contents of the PR _without_ looking at the code.
* Include tests. If you are fixing a bug, please add a test that would've caught
the bug. If you are adding a new feature, please add unit tests too.
* `pre-commit` should help you handle formatting and type checking.
Please do make sure you have it installed as described [above](#prerequisites).

## Running Tests

To test your changes locally, run:

* `cd ./tests/; pytest -s`

To run pre-commit hooks manually, run `pre-commit run --all-files`


## Code Style & Typing

See the [LeMa Style Guide](/STYLE_GUIDE.md) for guidelines on how to structure,
and format your code.
