# Contributing

Oumi OSS welcomes any contributions that help make it better for the community: this is a community-first effort. If we all work together, we can ensure a better, more inclusive, safer, and a totally open future for frontier AI. Whether you are an individual contributor or an organization, we invite you to be part of this bold mission to bring frontier AI back in the open. The future of AI is open source, and we can build that together.

Possible contributions include:

* Bug fixes, incremental improvements, and tests, no matter how small
* New features and infrastructure improvements
* Tuning datasets, new ones or existing ones, adapted to the [standardized Oumi OSS format](/resources/datasets/data_formats)
* Benchmarks, new or existing, integrated to [Oumi OSS's evaluation library](/user_guides/evaluate/evaluate)
* Documentation and code readability improvements
* Code review of pull requests
* Tutorials, blog posts, talks, and social media posts that promote Oumi OSS
* Community participation in [GitHub issues](https://github.com/oumi-ai/oumi/issues), [Discord](https://discord.gg/oumi), and [X](https://x.com/Oumi_PBC), to share knowledge and help each other.

If you want to contribute but you are short of ideas or have any questions, reach out (<contact@oumi.ai>) and we can help.

(prerequisites)=

## 📢 Prerequisites

To set up the development environment on your local machine, please follow the steps outlined in the [development setup documentation](/development/dev_setup).

## 📤 Submitting a Contribution

To submit a contribution:

1. [Fork](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo)
a copy of the [Oumi OSS](https://github.com/oumi-ai/oumi) repository into your own account.
See [Forking a repository](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo#forking-a-repository)
for detailed steps.
2. Clone your fork locally, and add the Oumi OSS repo as a remote repository:

    ```shell
    git clone git@github.com:<github_id>/oumi.git
    cd oumi
    git remote add upstream https://github.com/oumi-ai/oumi.git
    ```

3. Create a branch, and make your proposed changes.

    ```shell
    git checkout -b my-username/my-awesome-new-feature
    ```

4. When you are ready, submit a pull request into the Oumi OSS repository!

## 📥 Pull request (PR) guidelines

Basic guidelines that will make your PR easier to review:

* **Title and Description**
  * Please include a concise title and clear PR description.
  * The title should allow someone to understand what the PR changes or does at a glance.
  * The description should allow someone to understand the contents of the PR *without* looking at the code.

* **Testing**
  * Please include tests with your PR!
  * If fixing a bug, add a test that would've caught the bug.
  * If adding a new feature, include unit tests for the new functionality.

* **Code Formatting and Type Checking**
  * Use `pre-commit` to handle formatting and type checking:
  * Ensure you have it installed as described in the [Prerequisites](#prerequisites) section.
  * Run pre-commit hooks before submitting your PR.

## 🏃🏽‍♀️ Running Tests

To test your changes locally, run:

```shell
cd ./tests/
pytest -s -vv
```

To run pre-commit hooks manually, run `pre-commit run --all-files`

## 🎩 Code Style & Typing

See the [Oumi OSS Style Guide](style_guide.md) for guidelines on how to structure, and format your code.

## ©️ Copyright & License Headers

To maintain proper copyright and license notices, please include the header at the top of each source code file.

```python
# Copyright 2025-2026 - Oumi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
```

## 🔗 Becoming a Code Reviewer or Maintainer

Send an email to <contact@oumi.ai> if you would like to become a code reviewer, maintainer or contribute in any other way!

## 🏅 Recognition

Join the Oumi OSS community to be part of defining a better future for open frontier AI. We will recognize top contributors periodically and feature all of them in Oumi OSS's wall of fame.

Also, after you complete your first pull request (no matter how small), you can claim your holographic Oumi sticker! Send an email with title "Oumi Sticker" to <contact@oumi.ai> including your name and full mailing address and we will mail it to you anywhere in the world.
