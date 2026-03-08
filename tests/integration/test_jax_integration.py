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

"""Integration tests for JAX inference engine."""

from unittest.mock import patch

import pytest

from oumi.core.configs import GenerationParams, ModelParams
from oumi.core.types.conversation import Conversation, Message, Role

pytestmark = pytest.mark.jax


@pytest.fixture
def jax_model_params():
    return ModelParams(
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        load_pretrained_weights=False,
        trust_remote_code=True,
    )


@pytest.fixture
def jax_generation_params():
    return GenerationParams(max_new_tokens=20)


@pytest.fixture
def sample_conversation():
    return Conversation(messages=[Message(role=Role.USER, content="What is 2 + 2?")])


class TestJAXIntegration:
    """Integration tests for JAX inference engine."""

    def test_jax_utils_import(self):
        """JAX utilities should be importable."""
        from oumi.utils.jax_model_utils import (
            load_checkpoint_orbax,
            save_checkpoint_orbax,
            setup_tensor_parallelism,
        )
        from oumi.utils.jax_utils import check_jax_devices, torch_to_jax

        assert callable(torch_to_jax)
        assert callable(check_jax_devices)
        assert callable(setup_tensor_parallelism)
        assert callable(load_checkpoint_orbax)
        assert callable(save_checkpoint_orbax)

    def test_jax_builder_integration(self, jax_model_params):
        """JAX engine should be registered in the builder system."""
        from oumi.builders.inference_engines import build_inference_engine
        from oumi.core.configs import InferenceEngineType

        with (
            patch("oumi.inference.jax_inference_engine.build_tokenizer"),
            patch.object(
                __import__(
                    "oumi.inference.jax_inference_engine",
                    fromlist=["JAXInferenceEngine"],
                ).JAXInferenceEngine,
                "_load_model",
            ),
            patch.object(
                __import__(
                    "oumi.inference.jax_inference_engine",
                    fromlist=["JAXInferenceEngine"],
                ).JAXInferenceEngine,
                "_setup_devices",
            ),
        ):
            engine = build_inference_engine(
                engine_type=InferenceEngineType.JAX,
                model_params=jax_model_params,
            )
            assert engine is not None

    def test_inference_engine_type_has_jax(self):
        """InferenceEngineType enum should include JAX."""
        from oumi.core.configs import InferenceEngineType

        assert hasattr(InferenceEngineType, "JAX")
        assert InferenceEngineType.JAX.value == "JAX"

    def test_jax_engine_in_engine_map(self):
        """JAX engine should be in the ENGINE_MAP."""
        from oumi.builders.inference_engines import ENGINE_MAP
        from oumi.core.configs import InferenceEngineType
        from oumi.inference.jax_inference_engine import JAXInferenceEngine

        assert InferenceEngineType.JAX in ENGINE_MAP
        assert ENGINE_MAP[InferenceEngineType.JAX] == JAXInferenceEngine

    def test_registry_covers_all_architectures(self):
        """Registry should have entries for all major JAX architectures."""
        from oumi.models.experimental.jax_models.registry import (
            list_supported_architectures,
        )

        architectures = list_supported_architectures()
        # Should have at least the 7 model families
        expected = {
            "llama3_jax",
            "llama4_jax",
            "qwen3_jax",
            "deepseek_r1_jax",
            "kimi_k2_jax",
            "gpt_oss_jax",
            "nemotron3_jax",
        }
        assert expected.issubset(set(architectures)), (
            f"Missing architectures: {expected - set(architectures)}"
        )

    @pytest.mark.skipif(
        not pytest.importorskip("jax", reason="JAX not installed"),
        reason="JAX not available",
    )
    def test_jax_device_setup(self):
        """Test that JAX devices can be enumerated."""
        import jax

        devices = jax.devices()
        assert len(devices) > 0

    @pytest.mark.skipif(
        not pytest.importorskip("jax", reason="JAX not installed"),
        reason="JAX not available",
    )
    def test_tensor_parallelism_mesh(self):
        """Test mesh creation for tensor parallelism."""
        from oumi.utils.jax_model_utils import setup_tensor_parallelism

        mesh = setup_tensor_parallelism(1)
        assert mesh is not None
