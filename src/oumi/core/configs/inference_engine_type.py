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

from enum import Enum


class InferenceEngineType(str, Enum):
    """The supported inference engines."""

    NATIVE = "NATIVE"
    """The native inference engine using a local forward pass."""

    VLLM = "VLLM"
    """The vLLM inference engine started locally by oumi using vLLM library."""

    REMOTE_VLLM = "REMOTE_VLLM"
    """The external vLLM inference engine."""

    SGLANG = "SGLANG"
    """The SGLang inference engine."""

    LAMBDA = "LAMBDA"
    """The Lambda inference engine."""

    LLAMACPP = "LLAMACPP"
    """The LlamaCPP inference engine."""

    REMOTE = "REMOTE"
    """The inference engine for APIs that implement the OpenAI Chat API interface."""

    ANTHROPIC = "ANTHROPIC"
    """The inference engine for Anthropic's API."""

    GOOGLE_VERTEX = "GOOGLE_VERTEX"
    """The inference engine for Google Vertex AI."""

    GOOGLE_GEMINI = "GEMINI"
    """The inference engine for Gemini."""

    DEEPSEEK = "DEEPSEEK"
    """The inference engine for DeepSeek Platform API."""

    PARASAIL = "PARASAIL"
    """The inference engine for Parasail API."""

    TOGETHER = "TOGETHER"
    """The inference engine for Together API."""

    OPENAI = "OPENAI"
    """The inference engine for OpenAI API."""

    SAMBANOVA = "SAMBANOVA"
    """The inference engine for SambaNova API."""
