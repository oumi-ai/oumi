from typing import Optional

from oumi.core.configs.params.evaluation_params import EvaluationPlatform
from oumi.utils.packaging import PackagePrerequisites, check_package_prerequisites


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
        "following command:\n`pip install -e '.[evaluation]'`"
    )

    # Per platform prerequisite checks.
    if evaluation_platform == EvaluationPlatform.LM_HARNESS:
        if task_name == "leaderboard_ifeval":
            check_package_prerequisites(
                [
                    PackagePrerequisites("langdetect"),
                    PackagePrerequisites("immutabledict"),
                    PackagePrerequisites("nltk", "3.9.1", ">="),
                ],
                runtime_error_prefix=runtime_error_prefix,
                runtime_error_suffix=runtime_error_suffix,
            )
        if task_name == "leaderboard_math_hard":
            # FIXME: This benchmark is currently NOT compatible with Oumi; MATH
            # requires antlr4 version 4.11, but Oumi's omegaconf (2.3.0) requires
            # antlr4 version 4.9.*. This is a known issue and will be fixed when we
            # upgrade omegaconf to version 2.4.0.
            check_package_prerequisites(
                [
                    PackagePrerequisites("antlr4-python3-runtime", "4.11", "=="),
                    PackagePrerequisites("sympy", "1.12", ">="),
                    PackagePrerequisites("sentencepiece", "0.1.98", ">="),
                ],
                runtime_error_prefix=runtime_error_prefix,
                runtime_error_suffix=runtime_error_suffix,
            )
    elif evaluation_platform == EvaluationPlatform.ALPACA_EVAL:
        check_package_prerequisites(
            [PackagePrerequisites("alpaca_eval")],
            runtime_error_prefix=runtime_error_prefix,
            runtime_error_suffix=runtime_error_suffix,
        )
