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

from pathlib import Path
from typing import Optional

from oumi.cli.fetch import fetch
from oumi.core.configs import EvaluationConfig, EvaluationTaskParams, InferenceConfig

FETCH_DIR = Path("~/.oumi/fetch").expanduser()

MODEL_NAME_TO_YAML_FILE_PATH = {
    "gpt-4o": "configs//examples/inference_hosted/gpt_4o.yaml",
    # TODO: Add more models here
}


def get_default_config(
    model_name: str, registered_function_name: str = "my_custom_function"
) -> Optional[EvaluationConfig]:
    """Get the default config for a given model name."""
    # Retrieve the YAML file path for the model name
    if model_name not in MODEL_NAME_TO_YAML_FILE_PATH:
        return None
    yaml_path = MODEL_NAME_TO_YAML_FILE_PATH[model_name]

    # Fetch the YAML file from the specified remote path.
    fetch(config_path=f"oumi://{yaml_path}", output_dir=FETCH_DIR, force=True)

    # Check if the fetched YAML file exists locally.
    local_yaml_path = FETCH_DIR / yaml_path
    if not local_yaml_path.exists():
        raise FileNotFoundError(f"Config file {local_yaml_path} not found.")

    # Load the inference config from the YAML file.
    inference_config = InferenceConfig.from_yaml(local_yaml_path)

    # For evaluation, setting temperature to 0.0 is recommended.
    inference_config.generation.temperature = 0.0

    return EvaluationConfig(
        tasks=[
            EvaluationTaskParams(
                evaluation_backend="custom",
                task_name=registered_function_name,
            )
        ],
        model=inference_config.model,
        generation=inference_config.generation,
        inference_engine=inference_config.engine,
    )
