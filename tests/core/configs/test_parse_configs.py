import glob
import os
from typing import List, Optional, Set

import pytest

from oumi.core.configs import (
    AsyncEvaluationConfig,
    EvaluationConfig,
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


def _get_all_config_paths(exclude_yaml: Optional[Set[str]]) -> List[str]:
    """Recursively returns all configs in the /configs/oumi/ dir of the repo."""
    path_to_current_file = os.path.realpath(__file__)
    repo_root = _backtrack_on_path(path_to_current_file, 4)
    yaml_pattern = os.path.join(repo_root, "configs", "oumi", "**", "*.yaml")
    all_yaml_files = glob.glob(yaml_pattern, recursive=True)
    if exclude_yaml:
        filtered_files = [
            f for f in all_yaml_files if os.path.basename(f) not in exclude_yaml
        ]
    return filtered_files


@pytest.mark.parametrize(
    "config_path", _get_all_config_paths({"fsdp.yaml", "skypilot.yaml"})
)
def test_parse_configs(config_path: str):
    valid_config_classes = [
        TrainingConfig,
        EvaluationConfig,
        AsyncEvaluationConfig,
        JobConfig,
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


@pytest.mark.parametrize(
    "config_path", _get_all_config_paths({"fsdp.yaml", "skypilot.yaml"})
)
def test_parse_configs_from_yaml_and_arg_list(config_path: str):
    valid_config_classes = [
        TrainingConfig,
        EvaluationConfig,
        AsyncEvaluationConfig,
        JobConfig,
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
