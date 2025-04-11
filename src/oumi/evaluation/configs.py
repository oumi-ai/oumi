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

import os
from typing import Optional

from oumi.core.configs import EvaluationConfig
from oumi.core.configs.inference_engine_type import InferenceEngineType
from oumi.core.configs.params.evaluation_params import EvaluationTaskParams
from oumi.core.configs.params.generation_params import GenerationParams
from oumi.core.configs.params.model_params import ModelParams
from oumi.core.configs.params.remote_params import RemoteParams

MAP_ENGINE_TO_API_KEY = {
    InferenceEngineType.OPENAI: "OPENAI_API_KEY",
    InferenceEngineType.GOOGLE_GEMINI: "GEMINI_API_KEY",
    InferenceEngineType.ANTHROPIC: "ANTHROPIC_API_KEY",
}


def _get_gcp_api_url(
    region: Optional[str] = "",
    project_id: Optional[str] = "",
):
    """Get the GCP API URL to query Vertex models."""
    if not region:
        region = os.getenv("REGION") or os.getenv("GCP_REGION") or ""
    if not project_id:
        project_id = os.getenv("PROJECT_ID") or os.getenv("GCP_PROJECT_ID") or ""
    if not region or not project_id:
        raise ValueError(
            "Both `REGION` and `PROJECT_ID` environment variables must be set to query "
            "LLaMMA models on GCP. Please set them in your environment as follows:\n"
            "- os.environ['REGION'] = <your_region>\n"
            "- os.environ['PROJECT_ID'] = <your_project_id>\n"
            f"Current values are: REGION=`{region}` and PROJECT_ID=`{project_id}`"
        )

    return (
        f"https://{region}-aiplatform.googleapis.com/v1beta1/projects/"
        f"{project_id}/locations/{region}/endpoints/openapi/chat/completions"
    )


def _validate_api_key(
    engine: InferenceEngineType,
    model: str = "",
    api_key: Optional[str] = "",
):
    key_name = MAP_ENGINE_TO_API_KEY[engine]

    if api_key:
        os.environ[key_name] = api_key
    else:
        api_key = os.getenv(key_name) or ""

    if not api_key:
        raise ValueError(
            f"API key `{key_name}` is required to run the evaluation task "
            f"for the `{model}` model, which uses the {engine} inference engine. "
            f"Please set it as follows: `os.environ['{key_name}'] = <value>`."
        )


def get_standard_eval_config(
    model: str,
    custom_task_name: str = "custom_evaluation_fn",
    output_dir: str = "output",
    openai_api_key: Optional[str] = "",
    anthropic_api_key: Optional[str] = "",
    gemini_api_key: Optional[str] = "",
    gcp_region: Optional[str] = "",
    gcp_project_id: Optional[str] = "",
) -> EvaluationConfig:
    """Convenience function to retrieve standard (non customized) evaluation configs."""
    temperature = 0.0  # This is the default value for all models except o1 preview.
    api_url = None  # This is the default value for all models except the LLaMAs.
    max_new_tokens = 8192  # This is the max, since all models are SOTA.

    if model == "gpt 4o":
        engine = InferenceEngineType.OPENAI
        model_name = "gpt-4o"
        num_workers = 100
        politeness_policy = 60
        api_key = openai_api_key
    elif model == "gpt 4o mini":
        engine = InferenceEngineType.OPENAI
        model_name = "gpt-4o-mini"
        num_workers = 100
        politeness_policy = 60
        api_key = openai_api_key
    elif model == "o1 preview":
        engine = InferenceEngineType.OPENAI
        model_name = "o1-preview"
        num_workers = 100
        politeness_policy = 60
        temperature = 1.0
        api_key = openai_api_key
    elif model == "claude 3.5 sonnet":
        engine = InferenceEngineType.ANTHROPIC
        model_name = "claude-3-5-sonnet-latest"
        num_workers = 5
        politeness_policy = 65
        api_key = anthropic_api_key
    elif model == "claude 3.5 haiku":
        engine = InferenceEngineType.ANTHROPIC
        model_name = "claude-3-5-haiku-latest"
        num_workers = 5
        politeness_policy = 65
        api_key = anthropic_api_key
    elif model == "gemini 2.0 flash":
        engine = InferenceEngineType.GOOGLE_GEMINI
        model_name = "gemini-2.0-flash"
        num_workers = 10
        politeness_policy = 60
        api_key = gemini_api_key
    elif model == "gemini 1.5 pro":
        engine = InferenceEngineType.GOOGLE_GEMINI
        model_name = "gemini-1.5-pro"
        num_workers = 10
        politeness_policy = 60
        api_key = gemini_api_key
    elif model == "llama 405B":
        engine = InferenceEngineType.GOOGLE_VERTEX
        model_name = "meta/llama-3.1-405b-instruct-maas"
        num_workers = 10
        politeness_policy = 60
        api_key = ""
        api_url = _get_gcp_api_url(gcp_region, gcp_project_id)
    elif model == "llama 70B":
        engine = InferenceEngineType.GOOGLE_VERTEX
        model_name = "meta/llama-3.3-70b-instruct-maas"
        num_workers = 10
        politeness_policy = 60
        api_key = ""
        api_url = _get_gcp_api_url(gcp_region, gcp_project_id)
    else:
        raise ValueError(
            "The `get_evaluation_config` function is a convenience function that "
            "retrieves standard evaluation configs for a limited number of models. "
            f"The model `{model}` is not supported yet. You can easily create your own "
            "evaluation config by using the `EvaluationConfig` class directly. "
            "The models currently supported are: `gpt 4o`, `gpt 4o mini`, `o1 preview`,"
            " `claude 3.5 sonnet`, `claude 3.5 haiku`, `gemini 2.0 flash`, "
            "`gemini 1.5 pro`, `llama 405B`, and `llama 70B`."
        )

    if engine in MAP_ENGINE_TO_API_KEY:
        _validate_api_key(engine, model, api_key)

    return EvaluationConfig(
        tasks=[
            EvaluationTaskParams(
                task_name=custom_task_name,
                evaluation_backend="custom",
            ),
        ],
        model=ModelParams(
            model_name=model_name,
        ),
        generation=GenerationParams(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        ),
        inference_engine=engine,
        inference_remote_params=RemoteParams(
            api_url=api_url,
            num_workers=num_workers,
            politeness_policy=politeness_policy,
        ),
        enable_wandb=False,
        output_dir=output_dir,
    )
