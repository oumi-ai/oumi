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

from dataclasses import dataclass, field
from typing import Optional

from oumi.core.configs.base_config import BaseConfig
from oumi.core.configs.inference_engine_type import InferenceEngineType
from oumi.core.configs.params.generation_params import GenerationParams
from oumi.core.configs.params.model_params import ModelParams
from oumi.core.configs.params.remote_params import RemoteParams

try:
    from oumi_chat.style_params import (  # pyright: ignore[reportMissingImports]
        StyleParams,
    )
except ImportError:
    # If oumi-chat is not installed, provide a minimal StyleParams class
    @dataclass
    class StyleParams:
        """Minimal StyleParams for when oumi-chat is not installed."""

        theme: Optional[str] = None
        force_terminal: Optional[bool] = None
        force_jupyter: Optional[bool] = None
        width: Optional[int] = None
        height: Optional[int] = None
        no_color: bool = False
        legacy_windows: bool = False
        use_emoji: bool = False
        welcome_style: str = "bold green"
        welcome_border_style: str = "green"
        expand_panels: bool = False
        custom_theme: Optional[dict] = None
        user_prompt_style: str = "bold blue"
        status_style: str = "cyan"
        error_style: str = "red"
        error_title_style: str = "bold red"
        error_border_style: str = "red"
        assistant_title_style: str = "bold green"
        assistant_border_style: str = "green"
        assistant_text_style: str = "white"
        assistant_padding: tuple = (0, 1)


@dataclass
class InferenceConfig(BaseConfig):
    model: ModelParams = field(default_factory=ModelParams)
    """Parameters for the model used in inference."""

    generation: GenerationParams = field(default_factory=GenerationParams)
    """Parameters for text generation during inference."""

    input_path: Optional[str] = None
    """Path to the input file containing prompts for text generation.

    The input file should be in JSONL format, where each line is a JSON representation
    of an Oumi `Conversation` object.
    """

    output_path: Optional[str] = None
    """Path to the output file where the generated text will be saved."""

    engine: Optional[InferenceEngineType] = None
    """The inference engine to use for generation.

    Options:

        - NATIVE: Use the native inference engine via a local forward pass.
        - VLLM: Use the vLLM inference engine started locally by oumi.
        - REMOTE_VLLM: Use the external vLLM inference engine.
        - SGLANG: Use the SGLang inference engine.
        - LLAMACPP: Use LlamaCPP inference engine.
        - REMOTE: Use the inference engine for APIs that implement the OpenAI Chat API
          interface.
        - ANTHROPIC: Use the inference engine for Anthropic's API.

    If not specified, the "NATIVE" engine will be used.
    """

    remote_params: Optional[RemoteParams] = None
    """Parameters for running inference against a remote API."""

    style: StyleParams = field(default_factory=StyleParams)
    """Parameters for customizing console styling in interactive inference."""
