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

Usage:
    # Default (Llama 3.1 8B):
    python scripts/examples/jax_inference_example.py

    # Custom model:
    python scripts/examples/jax_inference_example.py --model Qwen/Qwen3-0.6B

    # Custom prompt:
    python scripts/examples/jax_inference_example.py --prompt "What is JAX?"

    # Or via YAML config:
    oumi infer -i -c configs/examples/jax_inference/llama3_basic.yaml

See Also:
    - JAX LLM Examples: https://github.com/jax-ml/jax-llm-examples
    - Oumi inference docs: https://oumi.ai/docs/en/latest/user_guides/infer/infer.html
"""

import argparse

from oumi.builders.inference_engines import build_inference_engine
from oumi.core.configs import (
    GenerationParams,
    InferenceEngineType,
    ModelParams,
)
from oumi.core.types.conversation import Conversation, Message, Role


def main():
    """Run JAX inference with a specified model."""
    parser = argparse.ArgumentParser(description="JAX inference example with Oumi")
    parser.add_argument(
        "--model",
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="HuggingFace model ID or local checkpoint path",
    )
    parser.add_argument(
        "--prompt",
        default="Explain tensor parallelism in one paragraph.",
        help="Prompt to generate a response for",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=512, help="Max tokens to generate"
    )
    args = parser.parse_args()

    # Configure model and generation parameters.
    model_params = ModelParams(
        model_name=args.model,
        model_max_length=4096,
        torch_dtype_str="bfloat16",
        load_pretrained_weights=True,
        trust_remote_code=True,
    )

    generation_params = GenerationParams(
        max_new_tokens=args.max_new_tokens,
        temperature=0.7,
        top_p=0.9,
    )

    # Build the JAX inference engine.
    engine = build_inference_engine(
        engine_type=InferenceEngineType.JAX,
        model_params=model_params,
    )

    # Create a conversation and run inference.
    conversation = Conversation(messages=[Message(role=Role.USER, content=args.prompt)])

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
