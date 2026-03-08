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

"""Example: JAX inference with Oumi.

Demonstrates how to use the JAX inference engine for LLM inference on TPU/GPU.

Requirements:
    - Install JAX dependencies: ``pip install oumi[jax]``
    - For Llama models, authenticate: ``huggingface-cli login``
    - TPU or GPU hardware (CPU works for testing but is slow)

Supported models:
    - Llama 3.1 (8B, 70B, 405B)
    - Llama 4 (Scout, Maverick)
    - DeepSeek R1 (native and distilled)
    - Qwen 3 (0.6B to 235B, dense and MoE)
    - Kimi K2
    - GPT-OSS (20B, 120B)
    - Nemotron 3 Nano

Usage:
    python scripts/examples/jax_inference_example.py

    Or via YAML config:
    oumi infer -i -c configs/examples/jax_inference/llama3_basic.yaml

See Also:
    - JAX LLM Examples: https://github.com/jax-ml/jax-llm-examples
    - Oumi inference docs: https://oumi.ai/docs/en/latest/user_guides/infer/infer.html
"""

from oumi.builders.inference_engines import build_inference_engine
from oumi.core.configs import (
    GenerationParams,
    InferenceEngineType,
    ModelParams,
)
from oumi.core.types.conversation import Conversation, Message, Role


def main():
    """Run JAX inference with a Llama model."""
    # Configure model and generation parameters.
    model_params = ModelParams(
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        model_max_length=4096,
        torch_dtype_str="bfloat16",
        load_pretrained_weights=True,
        trust_remote_code=True,
    )

    generation_params = GenerationParams(
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9,
    )

    # Build the JAX inference engine.
    engine = build_inference_engine(
        engine_type=InferenceEngineType.JAX,
        model_params=model_params,
    )

    # Create a conversation and run inference.
    conversation = Conversation(
        messages=[
            Message(
                role=Role.USER,
                content="Explain tensor parallelism in one paragraph.",
            )
        ]
    )

    results = engine.infer(
        input=[conversation],
        generation_params=generation_params,
    )

    # Print the response.
    for result in results:
        assistant_messages = [m for m in result.messages if m.role == Role.ASSISTANT]
        for msg in assistant_messages:
            print(msg.content)

    engine.cleanup()


if __name__ == "__main__":
    main()
