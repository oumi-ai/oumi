#!/usr/bin/env python3
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

"""Example: How to use AWQ quantized models for inference."""

import torch  # type: ignore
from awq import AutoAWQForCausalLM  # type: ignore
from transformers import AutoTokenizer  # type: ignore


def use_local_awq_model():
    """Use a locally quantized AWQ model."""
    # Path to your quantized model (from 'oumi quantize' output)
    model_path = "/home/yuzhang/oumi-quantize/TinyLlama-1.1B-AWQ4bit.pytorch"

    # Load AWQ model
    model = AutoAWQForCausalLM.from_quantized(
        model_path,
        fuse_layers=False,  # Avoid Triton compatibility issues
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Generate response
    prompt = "What is machine learning?"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=500,
            temperature=0.6,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Prompt: {prompt}")
    print(f"Response: {response}")


if __name__ == "__main__":
    print("Example 1: Using local AWQ model")
    print("-" * 50)
    use_local_awq_model()
