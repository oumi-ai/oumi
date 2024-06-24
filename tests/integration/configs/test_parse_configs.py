import glob
import os
from typing import List

import pytest

from lema.core.types import EvaluationConfig, HardwareException, TrainingConfig


def _is_config_file(path: str) -> bool:
    """Verifies if the path is a yaml file."""
    return os.path.isfile(path) and path.endswith(".yaml")


def _backtrack_on_path(path, n):
    """Goes up n directories in the current path."""
    output_path = path
    for _ in range(n):
        output_path = os.path.dirname(output_path)
    return output_path


def _get_all_config_paths() -> List[str]:
    """Recursively returns all configs in the /configs/lema/ dir of the repo."""
    path_to_current_file = os.path.realpath(__file__)
    repo_root = _backtrack_on_path(path_to_current_file, 4)
    yaml_pattern = os.path.join(repo_root, "configs", "lema", "**", "*.yaml")
    return glob.glob(yaml_pattern, recursive=True)


@pytest.mark.parametrize("config_path", _get_all_config_paths())
def test_parse_configs(config_path: str):
    training_config_error = ""
    eval_config_error = ""
    try:
        _ = TrainingConfig.from_yaml(config_path)
    except (HardwareException, Exception) as exception:
        # Ignore HardwareExceptions.
        if not isinstance(exception, HardwareException):
            training_config_error = str(exception)
    try:
        _ = EvaluationConfig.from_yaml(config_path)
    except (HardwareException, Exception) as exception:
        # Ignore HardwareExceptions.
        if not isinstance(exception, HardwareException):
            eval_config_error = str(exception)
    assert (len(training_config_error) == 0) or (len(eval_config_error) == 0), (
        f"Training config error: {training_config_error} . Evaluation config error: "
        f"{eval_config_error} ."
    )
