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

from oumi.core.configs.params.evaluation_params import EvaluationPlatform
from oumi.utils.packaging import PackagePrerequisites, check_package_prerequisites

# The `PLATFORM_PREREQUISITES` dictionary is 2-levels deep (`dict` of nested `dict`s)
# and contains the list of prerequisites (`PackagePrerequisites`) for each evaluation
# platform. Specifically:
# - The 1st-level key determines the evaluation platform (`EvaluationPlatform` Enum).
# - The 2nd-level key is an `str` that signifies either:
#   - The task name of the task to be executed in the platform.
#   - The key `ALL_TASK_PREREQUISITES_KEY`, which returns the aggregate platform
#     package prerequisites, applicable to every task that is executed.
ALL_TASK_PREREQUISITES_KEY = "all_task_prerequisites"
PLATFORM_PREREQUISITES: dict[
    EvaluationPlatform, dict[str, list[PackagePrerequisites]]
] = {
    EvaluationPlatform.LM_HARNESS: {
        ALL_TASK_PREREQUISITES_KEY: [],
        "leaderboard_ifeval": [
            PackagePrerequisites("langdetect"),
            PackagePrerequisites("immutabledict"),
            PackagePrerequisites("nltk", "3.9.1"),
        ],
        "leaderboard_math_hard": [
            # FIXME: This benchmark is currently NOT compatible with Oumi; MATH
            # requires antlr4 version 4.11, but Oumi's omegaconf (2.3.0) requires
            # antlr4 version 4.9.*. This is a known issue and will be fixed when we
            # upgrade omegaconf to version 2.4.0.
            PackagePrerequisites("antlr4-python3-runtime", "4.11", "4.11"),
            PackagePrerequisites("sympy", "1.12"),
            PackagePrerequisites("sentencepiece", "0.1.98"),
        ],
    },
    EvaluationPlatform.ALPACA_EVAL: {
        ALL_TASK_PREREQUISITES_KEY: [PackagePrerequisites("alpaca_eval")]
    },
}


def check_prerequisites(
    evaluation_platform: EvaluationPlatform,
    task_name: Optional[str] = None,
) -> None:
    """Check whether the evaluation platform prerequisites are satisfied.

    Args:
        evaluation_platform: The evaluation platform that the task will run.
        task_name (for LM Harness platform only): The name of the task to run.

    Raises:
        RuntimeError: If the evaluation platform prerequisites are not satisfied.
    """
    # Error message prefixes and suffixes.
    task_reference = f"({task_name}) " if task_name else ""
    runtime_error_prefix = (
        "The current evaluation cannot be launched because the "
        f"{evaluation_platform.value} platform prerequisites for the specific task "
        f"{task_reference}are not satisfied. In order to proceed, the following "
        "package(s) must be installed and have the correct version:\n"
    )
    runtime_error_suffix = (
        "\nNote that you can install all evaluation-related packages with the "
        "following command:\n`pip install oumi[evaluation]`"
    )

    # Per platform prerequisite checks.
    platform_prerequisites_dict = PLATFORM_PREREQUISITES[evaluation_platform]
    package_prerequisites_list = platform_prerequisites_dict[ALL_TASK_PREREQUISITES_KEY]
    if task_name and task_name in platform_prerequisites_dict:
        package_prerequisites_list.extend(platform_prerequisites_dict[task_name])
    check_package_prerequisites(
        package_prerequisites=package_prerequisites_list,
        runtime_error_prefix=runtime_error_prefix,
        runtime_error_suffix=runtime_error_suffix,
    )
