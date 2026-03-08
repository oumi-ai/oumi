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

"""Unified CLI for JAX Model Management.

Following jax-llm-examples patterns for download, convert, and run inference.
"""

import argparse
import sys
from pathlib import Path

from oumi.models.experimental.jax_models.manager import JAXModelManager
from oumi.models.experimental.jax_models.registry import (
    get_model_info,
    get_recommended_model,
    get_supported_models,
)


def cmd_list_models(args):
    """List all available JAX models."""
    manager = JAXModelManager(args.cache_dir)
    models = manager.list_available_models()

    print("🤖 Available JAX Models:")
    print("=" * 60)
    for name, info in models.items():
        auth_str = "🔒 Auth Required" if info["requires_auth"] else "🔓 Public"
        size_str = f"{info['size_gb']:.1f}GB" if info["size_gb"] else "Unknown size"

        print(f"\n📦 {name}")
        print(f"   Model ID: {info['model_id']}")
        print(f"   Architecture: {info['architecture']}")
        print(f"   Size: {size_str}")
        print(f"   Access: {auth_str}")
        print(f"   Hardware: {info['recommended_hardware']}")
        print(f"   Description: {info['description']}")
        if info["notes"]:
            print(f"   Notes: {info['notes']}")


def cmd_recommend(args):
    """Recommend a model based on constraints."""
    model_key = get_recommended_model(args.max_size_gb, args.requires_no_auth)

    if model_key:
        model_info = get_model_info(model_key)
        print(f"✨ Recommended Model: {model_info.model_id}")
        print(f"   Key: {model_key}")
        print(f"   Architecture: {model_info.architecture}")
        print(f"   Size: {model_info.size_gb}GB")
        print(f"   Description: {model_info.description}")
        print(f"   Hardware: {model_info.recommended_hardware}")
        if model_info.notes:
            print(f"   Notes: {model_info.notes}")

        # Find model name for CLI commands
        for name, info in get_supported_models().items():
            if info.model_id == model_info.model_id:
                print("\n🚀 To use this model:")
                print(
                    f"   python -m oumi.models.experimental.jax_models.cli run {name}"
                )
                break
    else:
        print("❌ No models match the specified constraints")


def cmd_download(args):
    """Download a model from HuggingFace."""
    manager = JAXModelManager(args.cache_dir)

    try:
        model_dir = manager.download_model(args.model_name, args.force)
        print(f"✅ Download successful: {model_dir}")
    except Exception as e:
        print(f"❌ Download failed: {e}")
        sys.exit(1)


def cmd_convert(args):
    """Convert a downloaded model to JAX format."""
    manager = JAXModelManager(args.cache_dir)

    try:
        jax_dir = manager.convert_model(args.model_name, args.force)
        print(f"✅ Conversion successful: {jax_dir}")
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        sys.exit(1)


def cmd_run(args):
    """Run inference with a JAX model."""
    manager = JAXModelManager(args.cache_dir)

    try:
        # Load model
        print(f"⏳ Loading {args.model_name}...")
        weights, config, tokenizer = manager.load_model(
            args.model_name,
            auto_download=args.auto_download,
            auto_convert=args.auto_convert,
        )

        # Import JAX for inference
        import importlib

        import numpy as np
        from jax import numpy as jnp
        from jax import random
        from jax.sharding import PartitionSpec as P
        from jax.sharding import set_mesh

        # Import the correct model implementation
        from oumi.models.experimental.jax_models.registry import (
            get_implementation_module,
            get_model_info,
        )

        model_info = get_model_info(args.model_name)
        impl_module_path = get_implementation_module(model_info.architecture)
        model_module = importlib.import_module(f"{impl_module_path}.model")

        # Encode input text
        prompts = args.prompt or [
            "Tell me your name",
            "What is the weather like?",
            "Do you like ice cream?",
        ]

        def encode_input(texts, pad_id=0):
            """Simple text encoding - adapt based on model."""
            inputs = []
            for text in texts:
                if hasattr(tokenizer, "apply_chat_template"):
                    # Use chat template if available
                    tokens = tokenizer.apply_chat_template(
                        [{"role": "user", "content": text}]
                    )
                    if "llama" in model_info.architecture:
                        tokens += tokenizer.encode(
                            "<|start_header_id|>assistant<|end_header_id|>"
                        )
                    elif (
                        "qwen" in model_info.architecture
                        or "kimi" in model_info.architecture
                    ):
                        tokens += tokenizer.encode("<|im_start|>assistant")
                    else:
                        tokens += tokenizer.encode("Assistant:")
                else:
                    # Fallback encoding
                    tokens = tokenizer.encode(f"User: {text}\nAssistant:")
                inputs.append(tokens)

            # Pad sequences
            max_len = max(len(x) for x in inputs)
            inputs = [(max_len - len(x)) * [pad_id] + x for x in inputs]
            return np.array(inputs)

        print(f"💭 Prompts: {prompts}")
        input_tokens = encode_input(prompts)
        print(f"✅ Encoded input: {input_tokens.shape}")

        # Run inference
        print("🚀 Running JAX inference...")
        with set_mesh(config.mesh):
            # Initialize cache
            zero_cache = model_module.KVCache.init(
                random.key(1), config, input_tokens.shape[0], config.max_seq_len
            )

            # Prefill
            next_tokens, logits, cache = model_module.prefill(
                input_tokens, weights, zero_cache, config
            )
            curr_tokens = next_tokens.at[:, cache.iter - 1 : cache.iter].get(
                out_sharding=P(None, None)
            )

            # Generate tokens
            tokens_list = []
            for step in range(args.max_new_tokens):
                tokens_list.append(curr_tokens)
                curr_tokens, cache = model_module.decode_step(
                    curr_tokens, weights, cache, config
                )

                if step % 8 == 0:  # Progress indicator
                    print(f"🔄 Generated {step + 1}/{args.max_new_tokens} tokens...")

            # Concatenate all generated tokens
            generated_tokens = np.array(jnp.concatenate(tokens_list, axis=-1))

        # Decode and display responses
        print("\n🎉 Generated Responses:")
        print("=" * 60)
        for i, (prompt, tokens) in enumerate(zip(prompts, generated_tokens)):
            response = tokenizer.decode(tokens)
            print(f"\n💬 Prompt {i + 1}: {prompt}")
            print(f"🤖 Response: {response}")
            print("-" * 40)

    except Exception as e:
        print(f"❌ Inference failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="jax-models", description="Unified JAX Model Management CLI"
    )

    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="Cache directory for models (default: ~/.cache/oumi_jax_models)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # List models command
    list_parser = subparsers.add_parser("list", help="List available models")
    list_parser.set_defaults(func=cmd_list_models)

    # Recommend command
    rec_parser = subparsers.add_parser("recommend", help="Recommend a model")
    rec_parser.add_argument(
        "--max-size-gb", type=float, help="Maximum model size in GB"
    )
    rec_parser.add_argument(
        "--requires-no-auth", action="store_true", help="Only public models"
    )
    rec_parser.set_defaults(func=cmd_recommend)

    # Download command
    dl_parser = subparsers.add_parser("download", help="Download a model")
    dl_parser.add_argument("model_name", help="Model name from registry")
    dl_parser.add_argument("--force", action="store_true", help="Force re-download")
    dl_parser.set_defaults(func=cmd_download)

    # Convert command
    conv_parser = subparsers.add_parser("convert", help="Convert model to JAX")
    conv_parser.add_argument("model_name", help="Model name from registry")
    conv_parser.add_argument("--force", action="store_true", help="Force re-conversion")
    conv_parser.set_defaults(func=cmd_convert)

    # Run command
    run_parser = subparsers.add_parser("run", help="Run inference with model")
    run_parser.add_argument("model_name", help="Model name from registry")
    run_parser.add_argument("--prompt", nargs="+", help="Custom prompts for inference")
    run_parser.add_argument(
        "--max-new-tokens", type=int, default=32, help="Max tokens to generate"
    )
    run_parser.add_argument(
        "--no-auto-download",
        dest="auto_download",
        action="store_false",
        help="Disable auto-download",
    )
    run_parser.add_argument(
        "--no-auto-convert",
        dest="auto_convert",
        action="store_false",
        help="Disable auto-convert",
    )
    run_parser.set_defaults(func=cmd_run)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Call the appropriate command function
    args.func(args)


if __name__ == "__main__":
    main()
