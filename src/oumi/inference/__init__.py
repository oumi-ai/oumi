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

"""Inference module for the Oumi (Open Universal Machine Intelligence) library.

This module provides various implementations for running model inference.
"""

from oumi.inference.anthropic_inference_engine import AnthropicInferenceEngine
from oumi.inference.deepseek_inference_engine import DeepSeekInferenceEngine
from oumi.inference.gcp_inference_engine import GoogleVertexInferenceEngine
from oumi.inference.gemini_inference_engine import GoogleGeminiInferenceEngine
from oumi.inference.lambda_inference_engine import LambdaInferenceEngine
from oumi.inference.llama_cpp_inference_engine import LlamaCppInferenceEngine
from oumi.inference.native_text_inference_engine import NativeTextInferenceEngine
from oumi.inference.openai_inference_engine import OpenAIInferenceEngine
from oumi.inference.parasail_inference_engine import ParasailInferenceEngine
from oumi.inference.remote_inference_engine import RemoteInferenceEngine
from oumi.inference.remote_vllm_inference_engine import RemoteVLLMInferenceEngine
from oumi.inference.sambanova_inference_engine import SambanovaInferenceEngine
from oumi.inference.sglang_inference_engine import SGLangInferenceEngine
from oumi.inference.together_inference_engine import TogetherInferenceEngine
from oumi.inference.vllm_inference_engine import VLLMInferenceEngine

__all__ = [
    "AnthropicInferenceEngine",
    "DeepSeekInferenceEngine",
    "GoogleGeminiInferenceEngine",
    "GoogleVertexInferenceEngine",
    "LambdaInferenceEngine",
    "LlamaCppInferenceEngine",
    "NativeTextInferenceEngine",
    "OpenAIInferenceEngine",
    "ParasailInferenceEngine",
    "RemoteInferenceEngine",
    "RemoteVLLMInferenceEngine",
    "SambanovaInferenceEngine",
    "SGLangInferenceEngine",
    "TogetherInferenceEngine",
    "VLLMInferenceEngine",
]
