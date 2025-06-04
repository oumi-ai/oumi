#!/usr/bin/env python3
"""Test script to understand Phi-3 tokenization behavior."""

from oumi.builders import build_tokenizer
from oumi.core.configs import ModelParams

model_params = ModelParams(
    model_name="microsoft/Phi-3-vision-128k-instruct",
    device_map="cpu",
    trust_remote_code=True,
    chat_template="phi3-instruct",
)
tokenizer = build_tokenizer(model_params)

# Test simple tokenization to understand the exact tokens
response_template = "<|assistant|>"
instruction_template = "<|user|>"

print(
    "Response template tokens:",
    tokenizer.encode(response_template, add_special_tokens=False),
)
print(
    "Instruction template tokens:",
    tokenizer.encode(instruction_template, add_special_tokens=False),
)

# Create a minimal chat template
messages = [
    {"role": "user", "content": "What is this?"},
    {"role": "assistant", "content": "This is a test."},
]

prompt = tokenizer.apply_chat_template(messages, tokenize=False)
print("Full prompt text:")
print(repr(prompt))

tokens = tokenizer.encode(prompt, add_special_tokens=False)
print("Full prompt tokens:", tokens)
print("Decoded tokens:", [tokenizer.decode([t]) for t in tokens])
