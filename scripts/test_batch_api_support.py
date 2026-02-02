#!/usr/bin/env python
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

"""Test script to verify batch API support across remote inference engines.

This script tests each remote inference engine to determine if it supports
the OpenAI-compatible batch API (/v1/batches, /v1/files endpoints).

Usage:
    python scripts/test_batch_api_support.py [--engine ENGINE_NAME]

Examples:
    # Test all engines
    python scripts/test_batch_api_support.py

    # Test a specific engine
    python scripts/test_batch_api_support.py --engine openai

    # Test with verbose output
    python scripts/test_batch_api_support.py --verbose

    # Full end-to-end test with polling and result retrieval
    python scripts/test_batch_api_support.py --engine openai --full-test
"""

import argparse
import os
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from oumi.core.configs import InferenceConfig, ModelParams, RemoteParams
from oumi.core.types.conversation import Conversation, Message, Role
from oumi.inference.remote_inference_engine import BatchStatus


class BatchSupport(Enum):
    """Batch API support status."""

    SUPPORTED = "✅ Supported"
    NOT_SUPPORTED = "❌ Not Supported"
    EXPLICITLY_DISABLED = "🚫 Explicitly Disabled"
    UNKNOWN = "❓ Unknown"
    SKIPPED = "⏭️ Skipped (no API key)"
    ERROR = "⚠️ Error"


@dataclass
class EngineTestResult:
    """Result of testing an engine for batch API support."""

    engine_name: str
    status: BatchSupport
    message: str
    error: str | None = None
    batch_id: str | None = None
    full_test_details: dict[str, Any] = field(default_factory=dict)


# Engine configurations with required environment variables and test models
ENGINE_CONFIGS: dict[str, dict[str, Any]] = {
    "openai": {
        "class": "OpenAIInferenceEngine",
        "module": "oumi.inference",
        "env_var": "OPENAI_API_KEY",
        "model": "gpt-4o-mini",
        "expected": BatchSupport.SUPPORTED,
    },
    "anthropic": {
        "class": "AnthropicInferenceEngine",
        "module": "oumi.inference",
        "env_var": "ANTHROPIC_API_KEY",
        "model": "claude-3-5-haiku-latest",
        # Anthropic uses its own Message Batches API (implemented)
        "expected": BatchSupport.SUPPORTED,
    },
    "together": {
        "class": "TogetherInferenceEngine",
        "module": "oumi.inference",
        "env_var": "TOGETHER_API_KEY",
        "model": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        # Together uses purpose="batch-api" and redirect-based upload flow
        "expected": BatchSupport.SUPPORTED,
    },
    "fireworks": {
        "class": "FireworksInferenceEngine",
        "module": "oumi.inference",
        "env_var": "FIREWORKS_API_KEY",
        "env_defaults": {"FIREWORKS_ACCOUNT_ID": "oumi"},
        "model": "accounts/fireworks/models/llama-v3p1-8b-instruct",
        # Fireworks uses /v1/accounts/{account_id}/batchInferenceJobs
        "expected": BatchSupport.SUPPORTED,
    },
    "deepseek": {
        "class": "DeepSeekInferenceEngine",
        "module": "oumi.inference",
        "env_var": "DEEPSEEK_API_KEY",
        "model": "deepseek-chat",
        # DeepSeek doesn't have native batch API
        "expected": BatchSupport.NOT_SUPPORTED,
    },
    "gemini": {
        "class": "GoogleGeminiInferenceEngine",
        "module": "oumi.inference",
        "env_var": "GOOGLE_API_KEY",
        "model": "gemini-1.5-flash",
        # Explicitly disabled in code (but Google recently added support)
        "expected": BatchSupport.EXPLICITLY_DISABLED,
    },
    "vertex": {
        "class": "GoogleVertexInferenceEngine",
        "module": "oumi.inference",
        "env_var": "GOOGLE_APPLICATION_CREDENTIALS",
        "model": "google/gemini-1.5-flash",
        # Vertex has batch prediction but may not use /v1/batches endpoint
        "expected": BatchSupport.UNKNOWN,
    },
    "bedrock": {
        "class": "BedrockInferenceEngine",
        "module": "oumi.inference",
        "env_var": "AWS_REGION",
        "model": "anthropic.claude-3-5-sonnet-20240620-v1:0",
        # Explicitly disabled in code
        "expected": BatchSupport.EXPLICITLY_DISABLED,
    },
    "lambda": {
        "class": "LambdaInferenceEngine",
        "module": "oumi.inference",
        "env_var": "LAMBDA_API_KEY",
        "model": "llama-4-scout-17b-16e-instruct",
        # Lambda Inference API does not support batch processing
        "expected": BatchSupport.NOT_SUPPORTED,
    },
    "sambanova": {
        "class": "SambanovaInferenceEngine",
        "module": "oumi.inference",
        "env_var": "SAMBANOVA_API_KEY",
        "model": "Meta-Llama-3.1-8B-Instruct",
        # SambaNova has batch inference but NOT OpenAI-compatible
        "expected": BatchSupport.NOT_SUPPORTED,
    },
    "parasail": {
        "class": "ParasailInferenceEngine",
        "module": "oumi.inference",
        "env_var": "PARASAIL_API_KEY",
        "model": "meta-llama/Llama-3.2-1B-Instruct",
        # Parasail explicitly supports OpenAI batch API as drop-in replacement
        "expected": BatchSupport.SUPPORTED,
    },
    "openrouter": {
        "class": "OpenRouterInferenceEngine",
        "module": "oumi.inference",
        "env_var": "OPENROUTER_API_KEY",
        "model": "meta-llama/llama-3.2-1b-instruct",
        # OpenRouter does NOT support batch API
        "expected": BatchSupport.NOT_SUPPORTED,
    },
    "remote_vllm": {
        "class": "RemoteVLLMInferenceEngine",
        "module": "oumi.inference",
        "env_var": None,  # Requires custom URL
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        # vLLM server doesn't expose /v1/batches natively
        "expected": BatchSupport.NOT_SUPPORTED,
    },
    "sglang": {
        "class": "SGLangInferenceEngine",
        "module": "oumi.inference",
        "env_var": None,  # Requires custom URL
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        # SGLang batch API is in development, not currently available
        "expected": BatchSupport.NOT_SUPPORTED,
    },
}


def get_engine_class(engine_name: str):
    """Dynamically import and return the engine class."""
    config = ENGINE_CONFIGS[engine_name]
    module = __import__(config["module"], fromlist=[config["class"]])
    return getattr(module, config["class"])


def check_explicitly_disabled(engine_name: str) -> bool:
    """Check if the engine explicitly overrides infer_batch to raise NotImplementedError.

    This checks if the method actually raises NotImplementedError when called,
    not just whether it's overridden (since an override could be an implementation).
    """
    try:
        engine_class = get_engine_class(engine_name)

        # Check if infer_batch is overridden in the class (not inherited)
        if "infer_batch" not in engine_class.__dict__:
            return False

        # Check if the override raises NotImplementedError
        # by inspecting the source code or trying to call it
        import inspect
        source = inspect.getsource(engine_class.infer_batch)
        if "NotImplementedError" in source and "raise" in source:
            return True
        return False
    except Exception:
        return False


def test_engine_batch_support(
    engine_name: str, verbose: bool = False
) -> EngineTestResult:
    """Test if an engine supports the batch API.

    This performs several checks:
    1. Check if infer_batch is explicitly overridden to raise NotImplementedError
    2. If API key is available, attempt to call infer_batch with a test conversation
    3. Analyze the error response to determine support status
    """
    config = ENGINE_CONFIGS[engine_name]

    # Check if explicitly disabled in code
    if check_explicitly_disabled(engine_name):
        return EngineTestResult(
            engine_name=engine_name,
            status=BatchSupport.EXPLICITLY_DISABLED,
            message=f"{config['class']} explicitly overrides infer_batch() to raise NotImplementedError",
        )

    # Check for required environment variable
    env_var = config.get("env_var")
    if env_var and not os.environ.get(env_var):
        return EngineTestResult(
            engine_name=engine_name,
            status=BatchSupport.SKIPPED,
            message=f"Environment variable {env_var} not set",
        )

    # Set default environment variables if not already set
    for env_key, default_value in config.get("env_defaults", {}).items():
        if not os.environ.get(env_key):
            os.environ[env_key] = default_value

    # For engines requiring custom URLs (vLLM, SGLang), skip if no URL provided
    if engine_name in ["remote_vllm", "sglang"]:
        return EngineTestResult(
            engine_name=engine_name,
            status=BatchSupport.SKIPPED,
            message="Requires custom API URL (self-hosted server)",
        )

    # Try to instantiate engine and call infer_batch
    try:
        engine_class = get_engine_class(engine_name)

        # Create engine instance
        model_params = ModelParams(model_name=config["model"])
        remote_params = RemoteParams()

        engine = engine_class(
            model_params=model_params,
            remote_params=remote_params,
        )

        # Create a minimal test conversation
        conversation = Conversation(
            messages=[Message(content="Hello", role=Role.USER)],
            conversation_id="batch-test-1",
        )

        inference_config = InferenceConfig(
            model=model_params,
            remote_params=remote_params,
        )

        if verbose:
            print(f"  Testing {engine_name} with model {config['model']}...")

        # Attempt to call infer_batch
        batch_id = engine.infer_batch([conversation], inference_config)

        # If we get here, batch API is supported!
        return EngineTestResult(
            engine_name=engine_name,
            status=BatchSupport.SUPPORTED,
            message=f"Batch created successfully with ID: {batch_id}",
        )

    except NotImplementedError as e:
        return EngineTestResult(
            engine_name=engine_name,
            status=BatchSupport.EXPLICITLY_DISABLED,
            message="infer_batch() raises NotImplementedError",
            error=str(e),
        )

    except Exception as e:
        error_msg = str(e).lower()

        # Analyze error to determine if it's a "not supported" vs other error
        not_supported_indicators = [
            "404",
            "not found",
            "endpoint not found",
            "invalid url",
            "no route",
            "unknown endpoint",
            "/v1/batches",
            "/v1/files",
        ]

        auth_error_indicators = [
            "401",
            "403",
            "unauthorized",
            "forbidden",
            "invalid api key",
            "authentication",
        ]

        if any(indicator in error_msg for indicator in not_supported_indicators):
            return EngineTestResult(
                engine_name=engine_name,
                status=BatchSupport.NOT_SUPPORTED,
                message="Batch API endpoint not available",
                error=str(e)[:200],
            )
        elif any(indicator in error_msg for indicator in auth_error_indicators):
            return EngineTestResult(
                engine_name=engine_name,
                status=BatchSupport.ERROR,
                message="Authentication error (API key may be invalid)",
                error=str(e)[:200],
            )
        else:
            # Could be supported but failed for another reason
            return EngineTestResult(
                engine_name=engine_name,
                status=BatchSupport.UNKNOWN,
                message="Unexpected error during batch API test",
                error=str(e)[:200],
            )


def test_engine_batch_full(
    engine_name: str,
    verbose: bool = False,
    timeout_seconds: int = 300,
    poll_interval_seconds: int = 5,
) -> EngineTestResult:
    """Full end-to-end test of batch API with polling and result retrieval.

    This test:
    1. Creates a batch job with test conversations
    2. Polls for completion using get_batch_status()
    3. Retrieves and parses results using get_batch_results()
    4. Validates that responses were correctly parsed

    Args:
        engine_name: Name of the engine to test
        verbose: Enable verbose output
        timeout_seconds: Maximum time to wait for batch completion
        poll_interval_seconds: Time between status polls
    """
    config = ENGINE_CONFIGS[engine_name]

    # Check if explicitly disabled in code
    if check_explicitly_disabled(engine_name):
        return EngineTestResult(
            engine_name=engine_name,
            status=BatchSupport.EXPLICITLY_DISABLED,
            message=f"{config['class']} explicitly overrides infer_batch()",
        )

    # Check for required environment variable
    env_var = config.get("env_var")
    if env_var and not os.environ.get(env_var):
        return EngineTestResult(
            engine_name=engine_name,
            status=BatchSupport.SKIPPED,
            message=f"Environment variable {env_var} not set",
        )

    # Set default environment variables if not already set
    for env_key, default_value in config.get("env_defaults", {}).items():
        if not os.environ.get(env_key):
            os.environ[env_key] = default_value

    # For engines requiring custom URLs, skip
    if engine_name in ["remote_vllm", "sglang"]:
        return EngineTestResult(
            engine_name=engine_name,
            status=BatchSupport.SKIPPED,
            message="Requires custom API URL (self-hosted server)",
        )

    test_details: dict[str, Any] = {
        "steps_completed": [],
        "batch_id": None,
        "final_status": None,
        "num_conversations": 0,
        "num_results": 0,
        "sample_response": None,
    }

    try:
        engine_class = get_engine_class(engine_name)

        # Create engine instance
        model_params = ModelParams(model_name=config["model"])
        remote_params = RemoteParams()

        engine = engine_class(
            model_params=model_params,
            remote_params=remote_params,
        )

        # Create test conversations with different prompts
        conversations = [
            Conversation(
                messages=[Message(content="Say 'hello' and nothing else.", role=Role.USER)],
                conversation_id="batch-test-1",
            ),
            Conversation(
                messages=[Message(content="Say 'world' and nothing else.", role=Role.USER)],
                conversation_id="batch-test-2",
            ),
        ]
        test_details["num_conversations"] = len(conversations)

        inference_config = InferenceConfig(
            model=model_params,
            remote_params=remote_params,
        )

        # Step 1: Create batch
        if verbose:
            print(f"  [{engine_name}] Step 1: Creating batch with {len(conversations)} conversations...")

        batch_id = engine.infer_batch(conversations, inference_config)
        test_details["batch_id"] = batch_id
        test_details["steps_completed"].append("create_batch")

        if verbose:
            print(f"  [{engine_name}] Batch created with ID: {batch_id}")

        # Step 2: Poll for completion
        if verbose:
            print(f"  [{engine_name}] Step 2: Polling for completion (timeout: {timeout_seconds}s)...")

        start_time = time.time()
        final_status = None
        poll_count = 0

        while time.time() - start_time < timeout_seconds:
            poll_count += 1
            batch_info = engine.get_batch_status(batch_id)
            final_status = batch_info.status

            if verbose:
                elapsed = int(time.time() - start_time)
                print(f"  [{engine_name}] Poll {poll_count}: status={final_status.value}, elapsed={elapsed}s")

            if final_status == BatchStatus.COMPLETED:
                test_details["steps_completed"].append("poll_completion")
                break
            elif final_status in [BatchStatus.FAILED, BatchStatus.EXPIRED, BatchStatus.CANCELLED]:
                test_details["final_status"] = final_status.value
                return EngineTestResult(
                    engine_name=engine_name,
                    status=BatchSupport.ERROR,
                    message=f"Batch failed with status: {final_status.value}",
                    batch_id=batch_id,
                    full_test_details=test_details,
                )

            time.sleep(poll_interval_seconds)

        if final_status != BatchStatus.COMPLETED:
            test_details["final_status"] = final_status.value if final_status else "timeout"
            return EngineTestResult(
                engine_name=engine_name,
                status=BatchSupport.ERROR,
                message=f"Batch did not complete within {timeout_seconds}s (status: {final_status})",
                batch_id=batch_id,
                full_test_details=test_details,
            )

        test_details["final_status"] = "completed"

        # Step 3: Retrieve results
        if verbose:
            print(f"  [{engine_name}] Step 3: Retrieving batch results...")

        results = engine.get_batch_results(batch_id, conversations)
        test_details["steps_completed"].append("get_results")
        test_details["num_results"] = len(results)

        if verbose:
            print(f"  [{engine_name}] Retrieved {len(results)} results")

        # Step 4: Validate results
        if verbose:
            print(f"  [{engine_name}] Step 4: Validating results...")

        if len(results) != len(conversations):
            return EngineTestResult(
                engine_name=engine_name,
                status=BatchSupport.ERROR,
                message=f"Result count mismatch: expected {len(conversations)}, got {len(results)}",
                batch_id=batch_id,
                full_test_details=test_details,
            )

        # Check that each result has an assistant response
        for i, result in enumerate(results):
            if not result.messages:
                return EngineTestResult(
                    engine_name=engine_name,
                    status=BatchSupport.ERROR,
                    message=f"Result {i} has no messages",
                    batch_id=batch_id,
                    full_test_details=test_details,
                )

            # Find the assistant message (should be the last one)
            assistant_messages = [m for m in result.messages if m.role == Role.ASSISTANT]
            if not assistant_messages:
                return EngineTestResult(
                    engine_name=engine_name,
                    status=BatchSupport.ERROR,
                    message=f"Result {i} has no assistant response",
                    batch_id=batch_id,
                    full_test_details=test_details,
                )

            if i == 0:
                test_details["sample_response"] = assistant_messages[-1].content[:100]

        test_details["steps_completed"].append("validate_results")

        if verbose:
            print(f"  [{engine_name}] ✅ Full test passed!")
            print(f"  [{engine_name}] Sample response: {test_details['sample_response']}")

        return EngineTestResult(
            engine_name=engine_name,
            status=BatchSupport.SUPPORTED,
            message=f"Full test passed: {len(results)} results retrieved and validated",
            batch_id=batch_id,
            full_test_details=test_details,
        )

    except NotImplementedError as e:
        return EngineTestResult(
            engine_name=engine_name,
            status=BatchSupport.EXPLICITLY_DISABLED,
            message="infer_batch() raises NotImplementedError",
            error=str(e),
            full_test_details=test_details,
        )

    except Exception as e:
        error_msg = str(e).lower()

        # Analyze error to determine if it's a "not supported" vs other error
        not_supported_indicators = [
            "404",
            "not found",
            "endpoint not found",
            "invalid url",
            "no route",
            "unknown endpoint",
            "/v1/batches",
            "/v1/files",
        ]

        if any(indicator in error_msg for indicator in not_supported_indicators):
            return EngineTestResult(
                engine_name=engine_name,
                status=BatchSupport.NOT_SUPPORTED,
                message="Batch API endpoint not available",
                error=str(e)[:300],
                full_test_details=test_details,
            )
        else:
            return EngineTestResult(
                engine_name=engine_name,
                status=BatchSupport.ERROR,
                message=f"Error at step: {test_details['steps_completed'][-1] if test_details['steps_completed'] else 'init'}",
                error=str(e)[:300],
                full_test_details=test_details,
            )


def print_results_table(results: list[EngineTestResult]) -> None:
    """Print results as a formatted table."""
    print("\n" + "=" * 100)
    print("BATCH API SUPPORT TEST RESULTS")
    print("=" * 100)
    print(f"{'Engine':<20} {'Status':<25} {'Message':<50}")
    print("-" * 100)

    for result in results:
        status_str = result.status.value
        message = result.message[:47] + "..." if len(result.message) > 50 else result.message
        print(f"{result.engine_name:<20} {status_str:<25} {message:<50}")

    print("=" * 100)

    # Summary
    supported = sum(1 for r in results if r.status == BatchSupport.SUPPORTED)
    not_supported = sum(
        1
        for r in results
        if r.status in [BatchSupport.NOT_SUPPORTED, BatchSupport.EXPLICITLY_DISABLED]
    )
    unknown = sum(
        1
        for r in results
        if r.status in [BatchSupport.UNKNOWN, BatchSupport.SKIPPED, BatchSupport.ERROR]
    )

    print(f"\nSummary: {supported} supported, {not_supported} not supported, {unknown} unknown/skipped")


def print_markdown_table(results: list[EngineTestResult]) -> None:
    """Print results as a markdown table for documentation."""
    print("\n## Batch API Support Matrix\n")
    print("| Engine | Class | Status | Notes |")
    print("|--------|-------|--------|-------|")

    for result in results:
        config = ENGINE_CONFIGS[result.engine_name]
        status_emoji = {
            BatchSupport.SUPPORTED: "✅",
            BatchSupport.NOT_SUPPORTED: "❌",
            BatchSupport.EXPLICITLY_DISABLED: "🚫",
            BatchSupport.UNKNOWN: "❓",
            BatchSupport.SKIPPED: "⏭️",
            BatchSupport.ERROR: "⚠️",
        }[result.status]

        print(
            f"| {result.engine_name} | `{config['class']}` | {status_emoji} | {result.message} |"
        )


def print_full_test_details(results: list[EngineTestResult]) -> None:
    """Print detailed results for full tests."""
    print("\n" + "=" * 80)
    print("FULL TEST DETAILS")
    print("=" * 80)

    for result in results:
        print(f"\n{'─' * 40}")
        print(f"Engine: {result.engine_name}")
        print(f"Status: {result.status.value}")
        print(f"Message: {result.message}")

        if result.batch_id:
            print(f"Batch ID: {result.batch_id}")

        if result.error:
            print(f"Error: {result.error[:200]}")

        if result.full_test_details:
            details = result.full_test_details
            if details.get("steps_completed"):
                print(f"Steps completed: {' → '.join(details['steps_completed'])}")
            if details.get("num_conversations"):
                print(f"Conversations: {details['num_conversations']}")
            if details.get("num_results"):
                print(f"Results retrieved: {details['num_results']}")
            if details.get("sample_response"):
                print(f"Sample response: {details['sample_response'][:80]}...")

    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Test batch API support across remote inference engines"
    )
    parser.add_argument(
        "--engine",
        type=str,
        choices=list(ENGINE_CONFIGS.keys()),
        help="Test a specific engine (default: test all)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )
    parser.add_argument(
        "--markdown", "-m", action="store_true", help="Output as markdown table"
    )
    parser.add_argument(
        "--skip-api-calls",
        action="store_true",
        help="Only check code-level support (no API calls)",
    )
    parser.add_argument(
        "--full-test",
        action="store_true",
        help="Run full end-to-end test with polling and result retrieval",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout in seconds for batch completion (default: 300)",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=5,
        help="Poll interval in seconds (default: 5)",
    )

    args = parser.parse_args()

    # Determine which engines to test
    engines_to_test = [args.engine] if args.engine else list(ENGINE_CONFIGS.keys())

    results: list[EngineTestResult] = []

    print("Testing batch API support for remote inference engines...\n")

    for engine_name in engines_to_test:
        if args.verbose:
            print(f"Testing {engine_name}...")

        if args.skip_api_calls:
            # Only check if explicitly disabled in code
            if check_explicitly_disabled(engine_name):
                result = EngineTestResult(
                    engine_name=engine_name,
                    status=BatchSupport.EXPLICITLY_DISABLED,
                    message="infer_batch() is overridden to raise NotImplementedError",
                )
            else:
                config = ENGINE_CONFIGS[engine_name]
                result = EngineTestResult(
                    engine_name=engine_name,
                    status=config["expected"],
                    message=f"Expected: {config['expected'].value}",
                )
        elif args.full_test:
            # Run full end-to-end test with polling and result retrieval
            result = test_engine_batch_full(
                engine_name,
                verbose=args.verbose,
                timeout_seconds=args.timeout,
                poll_interval_seconds=args.poll_interval,
            )
        else:
            result = test_engine_batch_support(engine_name, verbose=args.verbose)

        results.append(result)

        if args.verbose and result.error:
            print(f"    Error: {result.error}")

    # Print results
    if args.markdown:
        print_markdown_table(results)
    else:
        print_results_table(results)

    # Print detailed results for full tests
    if args.full_test and not args.markdown:
        print_full_test_details(results)

    # Exit with error code if any engines have unexpected results
    unexpected_failures = [
        r
        for r in results
        if r.status == BatchSupport.ERROR
        or (
            ENGINE_CONFIGS[r.engine_name]["expected"] == BatchSupport.SUPPORTED
            and r.status == BatchSupport.NOT_SUPPORTED
        )
    ]

    if unexpected_failures:
        print("\n⚠️  Some engines had unexpected results!")
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
