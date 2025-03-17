# Copyright 2025 - Oumi
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

from typing import Optional

from oumi.core.configs.params.evaluation_params import EvaluationBackend
from oumi.utils.packaging import PackagePrerequisites, check_package_prerequisites

# The `BACKEND_PREREQUISITES` dictionary is 2-levels deep (`dict` of nested `dict`s)
# and contains the list of prerequisites (`PackagePrerequisites`) for each evaluation
# backend. Specifically:
# - The 1st-level key determines the evaluation backend (`EvaluationBackend` Enum).
# - The 2nd-level key is an `str` that signifies either:
#   - The task name of the task to be executed in the backend.
#   - The key `ALL_TASK_PREREQUISITES_KEY`, which returns the aggregate backend
#     package prerequisites, applicable to every task that is executed.
ALL_TASK_PREREQUISITES_KEY = "all_task_prerequisites"
BACKEND_PREREQUISITES: dict[
    EvaluationBackend, dict[str, list[PackagePrerequisites]]
] = {
    EvaluationBackend.LM_HARNESS: {
        ALL_TASK_PREREQUISITES_KEY: [],
        "leaderboard_ifeval": [
            PackagePrerequisites("langdetect"),
            PackagePrerequisites("immutabledict"),
            PackagePrerequisites("nltk", "3.9.1"),
        ],
        "leaderboard_math_hard": [
            PackagePrerequisites("antlr4-python3-runtime", "4.11", "4.11"),
            PackagePrerequisites("sympy", "1.12"),
            PackagePrerequisites("sentencepiece", "0.1.98"),
        ],
    },
    EvaluationBackend.ALPACA_EVAL: {
        ALL_TASK_PREREQUISITES_KEY: [PackagePrerequisites("alpaca_eval")]
    },
    EvaluationBackend.CUSTOM: {ALL_TASK_PREREQUISITES_KEY: []},
}


def check_prerequisites(
    evaluation_backend: EvaluationBackend,
    task_name: Optional[str] = None,
) -> None:
    """Check whether the evaluation backend prerequisites are satisfied.

    Args:
        evaluation_backend: The evaluation backend that the task will run.
        task_name (for LM Harness backend only): The name of the task to run.

    Raises:
        RuntimeError: If the evaluation backend prerequisites are not satisfied.
    """
    # Error message prefixes and suffixes.
    task_reference = f"({task_name}) " if task_name else ""
    runtime_error_prefix = (
        "The current evaluation cannot be launched because the "
        f"{evaluation_backend.value} backend prerequisites for the specific task "
        f"{task_reference}are not satisfied. In order to proceed, the following "
        "package(s) must be installed and have the correct version:\n"
    )
    runtime_error_suffix = (
        "\nNote that you can install all evaluation-related packages with the "
        "following command:\n`pip install oumi[evaluation]`"
    )

    # Per backend prerequisite checks.
    backend_prerequisites_dict = BACKEND_PREREQUISITES[evaluation_backend]
    package_prerequisites_list = backend_prerequisites_dict[ALL_TASK_PREREQUISITES_KEY]
    if task_name and task_name in backend_prerequisites_dict:
        package_prerequisites_list.extend(backend_prerequisites_dict[task_name])
    check_package_prerequisites(
        package_prerequisites=package_prerequisites_list,
        runtime_error_prefix=runtime_error_prefix,
        runtime_error_suffix=runtime_error_suffix,
    )
