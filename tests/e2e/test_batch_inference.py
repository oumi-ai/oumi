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

"""End-to-end tests for batch inference APIs.

These tests verify that batch inference works correctly for all supported
inference engines. Each test creates a batch job, polls for completion,
retrieves results, and validates the responses.

Supported engines:
- OpenAI (OPENAI_API_KEY)
- Parasail (PARASAIL_API_KEY)
- Anthropic (ANTHROPIC_API_KEY)
- Together (TOGETHER_API_KEY)
- Fireworks (FIREWORKS_API_KEY, FIREWORKS_ACCOUNT_ID)
"""

import os
import time

import pytest

from oumi.core.configs import GenerationParams, ModelParams
from oumi.core.types.conversation import Conversation, Message, Role
from oumi.inference.remote_inference_engine import BatchStatus


def create_test_conversations() -> list[Conversation]:
    """Create simple test conversations for batch processing."""
    return [
        Conversation(
            messages=[Message(content="Say 'hello' and nothing else.", role=Role.USER)],
        ),
        Conversation(
            messages=[Message(content="Say 'world' and nothing else.", role=Role.USER)],
        ),
    ]


def wait_for_batch_completion(
    engine,
    batch_id: str,
    timeout_seconds: int = 600,
    poll_interval: int = 5,
) -> bool:
    """Poll for batch completion with timeout.

    Args:
        engine: The inference engine instance
        batch_id: The batch job ID to poll
        timeout_seconds: Maximum time to wait for completion
        poll_interval: Time between status checks

    Returns:
        True if batch completed successfully, False otherwise
    """
    start_time = time.time()
    while time.time() - start_time < timeout_seconds:
        status = engine.get_batch_status(batch_id)
        if status.status == BatchStatus.COMPLETED:
            return True
        if status.status in (
            BatchStatus.FAILED,
            BatchStatus.CANCELLED,
            BatchStatus.EXPIRED,
        ):
            return False
        time.sleep(poll_interval)
    return False


# =============================================================================
# OpenAI Batch Tests
# =============================================================================


@pytest.mark.e2e
def test_openai_batch_inference():
    """Test batch inference with OpenAI API."""
    if "OPENAI_API_KEY" not in os.environ:
        pytest.skip("OPENAI_API_KEY is not set")

    from oumi.inference import OpenAIInferenceEngine

    engine = OpenAIInferenceEngine(
        model_params=ModelParams(model_name="gpt-4o-mini"),
        generation_params=GenerationParams(max_new_tokens=10),
    )

    conversations = create_test_conversations()

    # Create batch
    batch_id = engine.infer_batch(conversations)
    assert batch_id is not None
    assert len(batch_id) > 0

    # Wait for completion
    completed = wait_for_batch_completion(engine, batch_id)
    assert completed, f"Batch {batch_id} did not complete in time"

    # Get results
    results = engine.get_batch_results(batch_id, conversations)
    assert len(results) == len(conversations)

    # Validate results
    for result in results:
        assert len(result.messages) > 0
        assert result.messages[-1].role == Role.ASSISTANT
        assert len(result.messages[-1].content) > 0


# =============================================================================
# Parasail Batch Tests
# =============================================================================


@pytest.mark.e2e
def test_parasail_batch_inference():
    """Test batch inference with Parasail API."""
    if "PARASAIL_API_KEY" not in os.environ:
        pytest.skip("PARASAIL_API_KEY is not set")

    from oumi.inference import ParasailInferenceEngine

    engine = ParasailInferenceEngine(
        model_params=ModelParams(model_name="meta-llama/Llama-3.2-1B-Instruct"),
        generation_params=GenerationParams(max_new_tokens=10),
    )

    conversations = create_test_conversations()

    # Create batch
    batch_id = engine.infer_batch(conversations)
    assert batch_id is not None
    assert len(batch_id) > 0

    # Wait for completion
    completed = wait_for_batch_completion(engine, batch_id)
    assert completed, f"Batch {batch_id} did not complete in time"

    # Get results
    results = engine.get_batch_results(batch_id, conversations)
    assert len(results) == len(conversations)

    # Validate results
    for result in results:
        assert len(result.messages) > 0
        assert result.messages[-1].role == Role.ASSISTANT
        assert len(result.messages[-1].content) > 0


# =============================================================================
# Anthropic Batch Tests
# =============================================================================


@pytest.mark.e2e
def test_anthropic_batch_inference():
    """Test batch inference with Anthropic Message Batches API."""
    if "ANTHROPIC_API_KEY" not in os.environ:
        pytest.skip("ANTHROPIC_API_KEY is not set")

    from oumi.inference import AnthropicInferenceEngine

    engine = AnthropicInferenceEngine(
        model_params=ModelParams(model_name="claude-4-5-haiku-latest"),
        generation_params=GenerationParams(max_new_tokens=10),
    )

    conversations = create_test_conversations()

    # Create batch
    batch_id = engine.infer_batch(conversations)
    assert batch_id is not None
    assert len(batch_id) > 0

    # Wait for completion
    completed = wait_for_batch_completion(engine, batch_id)
    assert completed, f"Batch {batch_id} did not complete in time"

    # Get results
    results = engine.get_batch_results(batch_id, conversations)
    assert len(results) == len(conversations)

    # Validate results
    for result in results:
        assert len(result.messages) > 0
        assert result.messages[-1].role == Role.ASSISTANT
        assert len(result.messages[-1].content) > 0


# =============================================================================
# Together Batch Tests
# =============================================================================


@pytest.mark.e2e
def test_together_batch_inference():
    """Test batch inference with Together Batch API."""
    if "TOGETHER_API_KEY" not in os.environ:
        pytest.skip("TOGETHER_API_KEY is not set")

    from oumi.inference import TogetherInferenceEngine

    engine = TogetherInferenceEngine(
        model_params=ModelParams(
            model_name="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
        ),
        generation_params=GenerationParams(max_new_tokens=10),
    )

    conversations = create_test_conversations()

    # Create batch
    batch_id = engine.infer_batch(conversations)
    assert batch_id is not None
    assert len(batch_id) > 0

    # Wait for completion
    completed = wait_for_batch_completion(engine, batch_id)
    assert completed, f"Batch {batch_id} did not complete in time"

    # Get results
    results = engine.get_batch_results(batch_id, conversations)
    assert len(results) == len(conversations)

    # Validate results
    for result in results:
        assert len(result.messages) > 0
        assert result.messages[-1].role == Role.ASSISTANT
        assert len(result.messages[-1].content) > 0


# =============================================================================
# Fireworks Batch Tests
# =============================================================================


@pytest.mark.e2e
def test_fireworks_batch_inference():
    """Test batch inference with Fireworks Batch API."""
    if "FIREWORKS_API_KEY" not in os.environ:
        pytest.skip("FIREWORKS_API_KEY is not set")

    # Set default account ID if not provided
    if "FIREWORKS_ACCOUNT_ID" not in os.environ:
        os.environ["FIREWORKS_ACCOUNT_ID"] = "oumi"

    from oumi.inference import FireworksInferenceEngine

    engine = FireworksInferenceEngine(
        model_params=ModelParams(
            model_name="accounts/fireworks/models/llama-v3p1-8b-instruct"
        ),
        generation_params=GenerationParams(max_new_tokens=10),
    )

    conversations = create_test_conversations()

    # Create batch
    batch_id = engine.infer_batch(conversations)
    assert batch_id is not None
    assert len(batch_id) > 0

    # Wait for completion (Fireworks can take longer)
    completed = wait_for_batch_completion(engine, batch_id, timeout_seconds=900)
    assert completed, f"Batch {batch_id} did not complete in time"

    # Get results
    results = engine.get_batch_results(batch_id, conversations)
    assert len(results) == len(conversations)

    # Validate results
    for result in results:
        assert len(result.messages) > 0
        assert result.messages[-1].role == Role.ASSISTANT
        assert len(result.messages[-1].content) > 0
