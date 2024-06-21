import os
from typing import List, Type

import pytest

from lema.core.types import BaseConfig, EvaluationConfig, TrainingConfig


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
    config_dir = os.path.join(repo_root, "configs", "lema")
    return [
        os.path.join(directory, file_name)
        for directory, _, files in os.walk(config_dir)
        for file_name in files
        if _is_config_file(os.path.join(directory, file_name))
    ]


@pytest.mark.parametrize("config_path", _get_all_config_paths())
def test_parse_configs(config_path: str):
    def _can_parse_config(
        config_path: str, config_class: Type[BaseConfig], allowed_errors: List[str]
    ) -> bool:
        try:
            _ = config_class.from_yaml(config_path)
            return True
        except ValueError as exception:
            return any([msg in str(exception) for msg in allowed_errors])

    # Somes checks involve inspecting the user's hardware. Ignore configs that
    # fail for that reason.
    allowed_training_config_errors = ["Flash attention 2"]
    if _can_parse_config(
        config_path,
        TrainingConfig,
        allowed_training_config_errors,
    ) or _can_parse_config(
        config_path,
        EvaluationConfig,
        [],
    ):
        return
    raise Exception(f"Failed to parse `{config_path}`.")
