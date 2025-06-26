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

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests
import yaml
from typing_extensions import Self

from oumi.cli import cli_utils
from oumi.core.configs import BaseConfig
from oumi.core.configs.inference_config import InferenceConfig
from oumi.core.configs.params.judge_params import JudgeParams

JUDGE_CONFIG_REPO_PATH_TEMPLATE = "oumi://configs/projects/judges/{path}.yaml"


@dataclass
class JudgeConfig(BaseConfig):
    """Consolidated configuration for the Judge.

    This class combines the judge parameters (JudgeParams) and inference
    configuration (InferenceConfig) into a single configuration object.

    Example:
        >>> judge_config = JudgeConfig( # doctest: +SKIP
        ...     judge_params=JudgeParams(
        ...         prompt_template="Is this helpful? {question}, {answer}",
        ...         response_format=JudgeResponseFormat.XML,
        ...         judgment_type=JudgeOutputType.BOOL,
        ...         include_explanation=False
        ...     ),
        ...     inference_config=InferenceConfig(
        ...         model=ModelParams(model_name="gpt-4.1"),
        ...         generation=GenerationParams(max_tokens=100),
        ...         engine=InferenceEngineType.OPENAI
        ...     )
        ... )
    """

    judge_params: JudgeParams
    """Parameters for the judge prompt and response format."""

    inference_config: InferenceConfig
    """Configuration for the inference engine and generation parameters."""

    @classmethod
    def from_path(cls, path: str, extra_args: Optional[list[str]] = None) -> Self:
        """Resolve the JudgeConfig from a local or repo path."""

        def _resolve_path(unresolved_path: str) -> Optional[str]:
            try:
                # Attempt to resolve the path using CLI utilities.
                # This will handle both local paths and repo (oumi://) paths.
                resolved_path = str(
                    cli_utils.resolve_and_fetch_config(
                        unresolved_path,
                    )
                )
            except (
                requests.exceptions.RequestException,  # Network/HTTP issues
                yaml.YAMLError,  # YAML parsing errors
                OSError,  # File system operations (includes IOError)
            ):
                # If resolution fails, mask the error and return None.
                return None

            # If resolution succeeds, check if the resolved path exists indeed.
            return resolved_path if Path(resolved_path).exists() else None

        if extra_args is None:
            extra_args = []

        # If `path` is a local or repo path, load JudgeConfig obj from that path.
        # Repo example: path = "oumi://configs/projects/judges/qa/relevance.yaml"
        # Local example: path= "./local_path/relevance.yaml"
        resolved_path = _resolve_path(path)

        # If `path` is a built-in judge name, construct the path from the default
        # repo location, and then load the corresponding JudgeConfig.
        # Example: "qa/relevance" => "oumi://configs/projects/judges/qa/relevance.yaml"
        if not resolved_path:
            resolved_path = _resolve_path(
                JUDGE_CONFIG_REPO_PATH_TEMPLATE.format(path=path)
            )

        if resolved_path:
            try:
                return cls.from_yaml_and_arg_list(resolved_path, extra_args)
            except Exception as e:
                raise ValueError(
                    f"Failed to parse {resolved_path} as JudgeConfig. "
                    f"Please ensure the YAML file contains both 'judge_params' and "
                    f"'inference_config' sections with valid fields. "
                    f"Original error: {e}"
                ) from e
        else:
            raise ValueError(
                f"Could not resolve JudgeConfig from path: {path}. "
                "Please provide a valid local or GitHub repo path."
            )
