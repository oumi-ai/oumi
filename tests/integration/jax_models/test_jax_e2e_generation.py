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

"""End-to-end generation tests for JAXInferenceEngine.

Tests the full pipeline: tokenize -> prefill -> decode -> detokenize,
using random weights (no model download needed).
"""

import os

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"

import jax  # noqa: E402
import pytest  # noqa: E402

jax.config.update("jax_platforms", "cpu")

from oumi.core.configs import GenerationParams, ModelParams  # noqa: E402
from oumi.core.types.conversation import Conversation, Message, Role  # noqa: E402
from oumi.inference.jax_inference_engine import JAXInferenceEngine  # noqa: E402

pytestmark = pytest.mark.jax


class TestJAXE2EGeneration:
    """End-to-end generation tests using JAXInferenceEngine with random weights."""

    def test_llama3_random_weights_generates_tokens(self):
        """Verify JAXInferenceEngine produces output with random weights."""
        engine = JAXInferenceEngine(
            model_params=ModelParams(
                model_name="meta-llama/Llama-3.1-8B-Instruct",
                load_pretrained_weights=False,
            ),
            generation_params=GenerationParams(max_new_tokens=5),
            max_seq_len=64,
        )
        conversation = Conversation(messages=[Message(role=Role.USER, content="Hello")])
        results = engine.infer_online(input=[conversation])
        assert len(results) == 1
        last_message = results[0].messages[-1]
        assert last_message.role == Role.ASSISTANT
        # With random weights, we should still get some generated text
        assert isinstance(last_message.content, str)
        engine.cleanup()

    def test_multiple_conversations(self):
        """Verify batch processing of multiple conversations."""
        engine = JAXInferenceEngine(
            model_params=ModelParams(
                model_name="meta-llama/Llama-3.1-8B-Instruct",
                load_pretrained_weights=False,
            ),
            generation_params=GenerationParams(max_new_tokens=3),
            max_seq_len=64,
        )
        conversations = [
            Conversation(messages=[Message(role=Role.USER, content="Hello")]),
            Conversation(messages=[Message(role=Role.USER, content="Hi there")]),
        ]
        results = engine.infer_online(input=conversations)
        assert len(results) == 2
        for result in results:
            assert result.messages[-1].role == Role.ASSISTANT
        engine.cleanup()

    def test_cleanup_releases_resources(self):
        """Verify cleanup properly releases model resources."""
        engine = JAXInferenceEngine(
            model_params=ModelParams(
                model_name="meta-llama/Llama-3.1-8B-Instruct",
                load_pretrained_weights=False,
            ),
            generation_params=GenerationParams(max_new_tokens=1),
            max_seq_len=64,
        )
        engine.cleanup()
        assert engine._model_module is None
        assert engine._weights is None
        assert engine._config is None
