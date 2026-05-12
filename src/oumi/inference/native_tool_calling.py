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

"""Native tool-calling capability allowlist.

Single source of truth for which inference engines round-trip
Conversation.tools <-> Message.tool_calls on the wire.
"""

from oumi.core.configs.inference_engine_type import InferenceEngineType

NATIVE_TOOL_CALLING_ENGINES: frozenset[InferenceEngineType] = frozenset(
    {
        InferenceEngineType.OPENAI,
        InferenceEngineType.ANTHROPIC,
        InferenceEngineType.VLLM,
        InferenceEngineType.REMOTE,
        InferenceEngineType.REMOTE_VLLM,
        InferenceEngineType.SGLANG,
        InferenceEngineType.DEEPSEEK,
        InferenceEngineType.FIREWORKS,
        InferenceEngineType.OPENROUTER,
        InferenceEngineType.CEREBRAS,
        InferenceEngineType.PARASAIL,
        InferenceEngineType.SAMBANOVA,
        InferenceEngineType.TOGETHER,
        InferenceEngineType.GOOGLE_VERTEX,
        InferenceEngineType.HUGGING_FACE_ROUTER,
    }
)


def supports_native_tool_calling(engine_type: InferenceEngineType | None) -> bool:
    """Return True iff the engine supports native tool calling on the wire."""
    return engine_type is not None and engine_type in NATIVE_TOOL_CALLING_ENGINES
