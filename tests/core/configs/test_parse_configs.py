import glob
import os
from typing import List

import pytest

from oumi.core.configs import (
    AsyncEvaluationConfig,
    EvaluationConfig,
    InferenceConfig,
    JobConfig,
    TrainingConfig,
)
from oumi.core.types import HardwareException


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
    """Recursively returns all configs in the /configs/oumi/ dir of the repo."""
    path_to_current_file = os.path.realpath(__file__)
    repo_root = _backtrack_on_path(path_to_current_file, 4)
    yaml_pattern = os.path.join(repo_root, "configs", "oumi", "**", "*.yaml")
    return glob.glob(yaml_pattern, recursive=True)


@pytest.mark.parametrize("config_path", _get_all_config_paths())
def test_parse_configs(config_path: str):
    valid_config_classes = [
        AsyncEvaluationConfig,
        EvaluationConfig,
        InferenceConfig,
        JobConfig,
        TrainingConfig,
    ]
    error_messages = []
    for config_class in valid_config_classes:
        try:
            _ = config_class.from_yaml(config_path)
        except (HardwareException, Exception) as exception:
            # Ignore HardwareExceptions.
            if not isinstance(exception, HardwareException):
                error_messages.append(
                    f"Error parsing {config_class.__name__}: {str(exception)}. "
                )
    assert len(error_messages) != len(valid_config_classes), "".join(error_messages)


@pytest.mark.parametrize("config_path", _get_all_config_paths())
def test_parse_configs_from_yaml_and_arg_list(config_path: str):
    valid_config_classes = [
        AsyncEvaluationConfig,
        EvaluationConfig,
        InferenceConfig,
        JobConfig,
        TrainingConfig,
    ]
    error_messages = []
    for config_class in valid_config_classes:
        try:
            _ = config_class.from_yaml_and_arg_list(config_path, [])
        except (HardwareException, Exception) as exception:
            # Ignore HardwareExceptions.
            if not isinstance(exception, HardwareException):
                error_messages.append(
                    f"Error parsing {config_class.__name__}: {str(exception)}. "
                )
    assert len(error_messages) != len(valid_config_classes), "".join(error_messages)
