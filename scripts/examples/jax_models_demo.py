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

"""End-to-end JAX model demo: download, convert, and run inference.

This script demonstrates the full JAX model pipeline using oumi's integration
with jax-llm-examples. It can download a model from HuggingFace, convert it
to JAX format, and run inference through the JAXInferenceEngine.

Usage:
    # Quick demo with smallest public model (Qwen3 0.6B):
    python scripts/examples/jax_models_demo.py --quick

    # Specific model:
    python scripts/examples/jax_models_demo.py --model qwen3-0.6b

    # Custom prompt:
    python scripts/examples/jax_models_demo.py --model qwen3-0.6b \
        --prompt "What is JAX?"

    # List available models:
    python scripts/examples/jax_models_demo.py --list

    # Skip download/convert, use existing checkpoint:
    python scripts/examples/jax_models_demo.py --model qwen3-0.6b --skip-setup

    # Only download (don't convert or run):
    python scripts/examples/jax_models_demo.py --model qwen3-0.6b --steps download
"""

import argparse
import sys
from pathlib import Path


def cmd_list():
    """List all available JAX models."""
    from oumi.models.experimental.jax_models.registry import get_supported_models

    models = get_supported_models()
    print("Available JAX Models:")
    print("=" * 70)
    print(f"{'Name':<35s} {'Size':>7s}  {'Auth':>12s}  Architecture")
    print("-" * 70)
    for name, info in models.items():
        size_str = f"{info.size_gb:.1f}GB" if info.size_gb else "?"
        auth_str = "[Auth]" if info.requires_auth else "[Public]"
        print(f"  {name:<33s} {size_str:>7s}  {auth_str:>12s}  {info.architecture}")
    print(f"\n{len(models)} models available.")


def cmd_run(args):
    """Run the full download -> convert -> inference pipeline."""
    from oumi.models.experimental.jax_models.manager import JAXModelManager
    from oumi.models.experimental.jax_models.registry import (
        get_model_info,
        get_recommended_model,
    )

    # Resolve model name
    if args.quick:
        model_name = get_recommended_model(max_size_gb=5.0, requires_no_auth=True)
        if not model_name:
            print("Error: No small public model found in registry.")
            sys.exit(1)
        print(f"Quick mode: selected '{model_name}'")
    elif args.model:
        model_name = args.model
    else:
        print("Error: Specify --model MODEL_NAME or use --quick")
        print("Use --list to see available models.")
        sys.exit(1)

    model_info = get_model_info(model_name)
    if not model_info:
        print(f"Error: Unknown model '{model_name}'. Use --list to see options.")
        sys.exit(1)

    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    manager = JAXModelManager(cache_dir=cache_dir)

    steps = (
        set(args.steps.split(",")) if args.steps else {"download", "convert", "infer"}
    )
    if args.skip_setup:
        steps = {"infer"}

    # Step 1: Download
    if "download" in steps:
        print(f"\n[1/3] Downloading {model_info.model_id}...")
        if model_info.requires_auth:
            print("  Note: This model requires HuggingFace auth.")
            print("  Run: huggingface-cli login")
        try:
            model_dir = manager.download_model(model_name)
            print(f"  Downloaded to: {model_dir}")
        except Exception as e:
            print(f"  Error: {e}")
            sys.exit(1)
    else:
        print("[1/3] Skipping download.")

    # Step 2: Convert
    if "convert" in steps:
        print("\n[2/3] Converting to JAX format...")
        try:
            jax_dir = manager.convert_model(model_name)
            print(f"  Converted to: {jax_dir}")
        except Exception as e:
            print(f"  Error: {e}")
            sys.exit(1)
    else:
        print("[2/3] Skipping conversion.")

    # Step 3: Inference
    if "infer" in steps:
        print("\n[3/3] Running inference...")
        jax_model_dir = manager.get_model_path(model_name, "jax_converted")
        if not jax_model_dir.exists():
            print(f"  Error: JAX model not found at {jax_model_dir}")
            print("  Run without --skip-setup to download and convert first.")
            sys.exit(1)

        from oumi.core.configs import GenerationParams, ModelParams
        from oumi.core.types.conversation import Conversation, Message, Role
        from oumi.inference.jax_inference_engine import JAXInferenceEngine

        prompt = args.prompt or "Explain what JAX is in one sentence."
        max_tokens = args.max_new_tokens

        print(f"  Model: {jax_model_dir}")
        print(f"  Prompt: {prompt}")
        print(f"  Max tokens: {max_tokens}")

        engine = JAXInferenceEngine(
            model_params=ModelParams(model_name=str(jax_model_dir)),
            generation_params=GenerationParams(max_new_tokens=max_tokens),
        )

        conversation = Conversation(messages=[Message(role=Role.USER, content=prompt)])
        results = engine.infer(input=[conversation])

        if results and results[0].messages:
            response = results[0].messages[-1].content
            print(f"\n  Response: {response}")
        else:
            print("  Warning: No response generated.")

        engine.cleanup()
        print("\nDone.")
    else:
        print("[3/3] Skipping inference.")


def main():
    """Entry point for the JAX model demo CLI."""
    parser = argparse.ArgumentParser(
        description="End-to-end JAX model demo: download, convert, and run inference."
    )
    parser.add_argument("--list", action="store_true", help="List available models")
    parser.add_argument("--model", type=str, help="Model name from registry")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Auto-select smallest public model",
    )
    parser.add_argument("--prompt", type=str, help="Custom prompt for inference")
    parser.add_argument(
        "--max-new-tokens", type=int, default=64, help="Max tokens to generate"
    )
    parser.add_argument("--cache-dir", type=str, help="Model cache directory")
    parser.add_argument(
        "--skip-setup",
        action="store_true",
        help="Skip download/convert, assume checkpoint exists",
    )
    parser.add_argument(
        "--steps",
        type=str,
        help="Comma-separated steps to run: download,convert,infer (default: all)",
    )

    args = parser.parse_args()

    if args.list:
        cmd_list()
        return

    cmd_run(args)


if __name__ == "__main__":
    main()
