import os
from typing import List

import pytest

from lema.core.types import EvaluationConfig, TrainingConfig


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
    training_config_error = ""
    eval_config_error = ""
    try:
        _ = TrainingConfig.from_yaml(config_path)
        return
    except Exception as exception:
        # Somes checks involve inspecting the user's hardware. Ignore configs that
        # fail for that reason.
        if "Flash attention 2" in str(exception):
            return
        training_config_error = str(exception)

    try:
        _ = EvaluationConfig.from_yaml(config_path)
        return
    except Exception as exception:
        # Somes checks involve inspecting the user's hardware. Ignore configs that
        # fail for that reason.
        if "Flash attention 2" in str(exception):
            return
        eval_config_error = str(exception)
    raise Exception(
        f"Failed to parse `{config_path}`.\n Training config error: "
        f"{training_config_error} .\n Evaluation config error: "
        f"{eval_config_error} ."
    )
