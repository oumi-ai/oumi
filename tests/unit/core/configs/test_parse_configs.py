import glob
import os
from typing import Optional

import pytest

from oumi.core.configs import (
    AsyncEvaluationConfig,
    EvaluationConfig,
    InferenceConfig,
    JobConfig,
    TrainingConfig,
)
from oumi.core.types import HardwareException


def _backtrack_on_path(path, n):
    """Goes up n directories in the current path."""
    output_path = path
    for _ in range(n):
        output_path = os.path.dirname(output_path)
    return output_path


def _get_all_config_paths(exclude_yaml_suffixes: Optional[set[str]]) -> list[str]:
    """Recursively returns all configs in the /configs/ dir of the repo."""
    path_to_current_file = os.path.realpath(__file__)
    repo_root = _backtrack_on_path(path_to_current_file, 5)
    yaml_pattern = os.path.join(repo_root, "configs", "**", "*.yaml")
    all_yaml_files = glob.glob(yaml_pattern, recursive=True)
    if exclude_yaml_suffixes:
        exclude_files = []
        for file in all_yaml_files:
            for exclude_yaml in exclude_yaml_suffixes:
                if file.endswith(exclude_yaml):
                    exclude_files.append(file)
                    break
        all_yaml_files = [file for file in all_yaml_files if file not in exclude_files]
    assert len(all_yaml_files) > 0, "No yaml files found to parse."
    return all_yaml_files


@pytest.mark.parametrize(
    "config_path",
    _get_all_config_paths(
        exclude_yaml_suffixes={
            "accelerate.yaml",
        }
    ),
)
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


@pytest.mark.parametrize(
    "config_path",
    _get_all_config_paths(
        {
            "accelerate.yaml",
        }
    ),
)
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
