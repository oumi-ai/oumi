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

"""AI-powered suggestion generation for analyzer configuration.

This module provides functions to analyze sample conversations and
suggest appropriate analyzers, custom metrics, and tests using an LLM.

Uses OpenAI's structured outputs for guaranteed schema compliance.
"""

import logging
import os
from typing import Any, Literal

from pydantic import BaseModel, Field

from oumi.core.types.conversation import Conversation

logger = logging.getLogger(__name__)

# Maximum characters per conversation to include in prompt
MAX_CONVERSATION_CHARS = 1500
# Maximum total chars for all conversations
MAX_TOTAL_CHARS = 6000


# =============================================================================
# Pydantic Models for Structured Outputs
# =============================================================================


class OutputField(BaseModel):
    """Schema for a custom metric output field."""

    name: str = Field(description="Field name in the output dict")
    type: str = Field(description="Python type: int, float, bool, str, list")
    description: str = Field(default="", description="What this field represents")


class AnalyzerSuggestion(BaseModel):
    """A suggested analyzer configuration."""

    id: str = Field(
        description=(
            "Analyzer ID: length, usefulness, safety, coherence, "
            "factuality, or instruction_following"
        )
    )
    reason: str = Field(description="Why this analyzer is useful for this dataset")
    # Note: params removed - OpenAI structured outputs requires strict schemas
    # The frontend will use default params for each analyzer type


class CustomMetricSuggestion(BaseModel):
    """A suggested custom metric."""

    id: str = Field(description="Unique metric ID (snake_case)")
    description: str = Field(default="", description="What this metric measures")
    function: str = Field(
        description="Python function code: def compute(conversation): ..."
    )
    output_schema: list[OutputField] = Field(
        default_factory=list, description="Output field definitions"
    )
    reason: str = Field(description="Why this metric is useful for this dataset")


class TestSuggestion(BaseModel):
    """A suggested test configuration."""

    id: str = Field(description="Unique test ID (snake_case)")
    type: Literal["threshold", "percentage", "range"] = Field(description="Test type")
    metric: str = Field(description="Metric path: AnalyzerName.field_name")
    title: str = Field(default="", description="Human-readable title")
    description: str = Field(default="", description="What this test checks")
    severity: Literal["low", "medium", "high"] = Field(
        default="medium", description="Test severity"
    )
    reason: str = Field(description="Why this test is important")
    # Threshold test fields
    operator: str | None = Field(
        default=None, description="Comparison operator: <, >, <=, >=, ==, !="
    )
    value: float | int | None = Field(default=None, description="Threshold value")
    # Percentage test fields
    condition: str | None = Field(
        default=None,
        description="Condition for percentage tests: == True, != None, etc.",
    )
    max_percentage: float | None = Field(
        default=None, description="Max % of samples that can fail"
    )
    min_percentage: float | None = Field(
        default=None, description="Min % of samples that must pass"
    )
    # Range test fields
    min_value: float | None = Field(
        default=None, description="Minimum value for range tests"
    )
    max_value: float | None = Field(
        default=None, description="Maximum value for range tests"
    )


class SuggestionsOutput(BaseModel):
    """Structured output from the LLM for suggestions."""

    analyzers: list[AnalyzerSuggestion] = Field(
        default_factory=list, description="List of suggested analyzers"
    )
    custom_metrics: list[CustomMetricSuggestion] = Field(
        default_factory=list, description="List of suggested custom metrics"
    )
    tests: list[TestSuggestion] = Field(
        default_factory=list, description="List of suggested tests"
    )


class SuggestionResponse(BaseModel):
    """Complete suggestion response including potential errors."""

    analyzers: list[AnalyzerSuggestion] = Field(default_factory=list)
    custom_metrics: list[CustomMetricSuggestion] = Field(default_factory=list)
    tests: list[TestSuggestion] = Field(default_factory=list)
    error: str | None = Field(default=None)


def get_analyzer_catalog() -> dict[str, dict[str, Any]]:
    """Get structured information about all available analyzers.

    Returns:
        Dictionary mapping analyzer IDs to their metadata.
    """
    return {
        "length": {
            "name": "Length Analyzer",
            "description": "Computes token, word, and character counts for conversations",
            "metrics": [
                "total_tokens",
                "total_words",
                "total_chars",
                "num_messages",
                "avg_tokens_per_message",
            ],
            "good_for": [
                "Detecting overly long or short conversations",
                "Ensuring responses fit within context windows",
                "Identifying verbose or terse responses",
            ],
            "params": {
                "tiktoken_encoding": "Token encoding to use (default: cl100k_base)",
                "compute_role_stats": "Whether to compute per-role statistics",
            },
        },
        "usefulness": {
            "name": "Usefulness Analyzer",
            "description": "LLM-based evaluation of how useful and helpful responses are",
            "metrics": ["score", "passed", "label", "reasoning"],
            "good_for": [
                "Q&A datasets where response quality matters",
                "Chatbot training data validation",
                "Detecting unhelpful or off-topic responses",
            ],
            "params": {
                "model_name": "LLM model to use for evaluation",
                "api_provider": "API provider (openai or anthropic)",
                "target_scope": "What part to evaluate (conversation, last_turn, etc.)",
            },
        },
        "safety": {
            "name": "Safety Analyzer",
            "description": "Checks content for harmful, dangerous, or inappropriate material",
            "metrics": ["score", "passed", "label", "reasoning"],
            "good_for": [
                "Content moderation datasets",
                "Ensuring training data is safe",
                "Detecting harmful content in responses",
            ],
            "params": {
                "model_name": "LLM model to use for evaluation",
                "api_provider": "API provider (openai or anthropic)",
            },
        },
        "coherence": {
            "name": "Coherence Analyzer",
            "description": "Evaluates logical flow and consistency of conversations",
            "metrics": ["score", "passed", "label", "reasoning"],
            "good_for": [
                "Multi-turn conversations",
                "Detecting contradictions or non-sequiturs",
                "Evaluating conversation flow",
            ],
            "params": {
                "model_name": "LLM model to use for evaluation",
                "api_provider": "API provider (openai or anthropic)",
            },
        },
        "factuality": {
            "name": "Factuality Analyzer",
            "description": "Checks factual accuracy of information in responses",
            "metrics": ["score", "passed", "label", "reasoning"],
            "good_for": [
                "Knowledge-based Q&A datasets",
                "Detecting hallucinations",
                "Validating factual claims",
            ],
            "params": {
                "model_name": "LLM model to use for evaluation",
                "api_provider": "API provider (openai or anthropic)",
            },
        },
        "instruction_following": {
            "name": "Instruction Following Analyzer",
            "description": "Evaluates how well responses follow given instructions",
            "metrics": ["score", "passed", "label", "reasoning"],
            "good_for": [
                "Instruction-tuning datasets",
                "Task completion evaluation",
                "Detecting off-task responses",
            ],
            "params": {
                "model_name": "LLM model to use for evaluation",
                "api_provider": "API provider (openai or anthropic)",
            },
        },
    }


def _truncate_conversation(
    conv: Conversation, max_chars: int = MAX_CONVERSATION_CHARS
) -> str:
    """Format and truncate a conversation for the prompt.

    Args:
        conv: Conversation to format.
        max_chars: Maximum characters to include.

    Returns:
        Formatted conversation string.
    """
    lines = []
    char_count = 0

    for msg in conv.messages:
        role = msg.role.value if hasattr(msg.role, "value") else str(msg.role)
        # Handle content that might be string or list of ContentItem
        raw_content = msg.content
        if isinstance(raw_content, str):
            content = raw_content
        elif raw_content is None:
            content = ""
        else:
            # Handle list of ContentItem - extract text content
            content = str(raw_content)

        # Truncate individual message if needed
        if len(content) > 500:
            content = content[:500] + "..."

        line = f"[{role}]: {content}"
        if char_count + len(line) > max_chars:
            lines.append("... (truncated)")
            break
        lines.append(line)
        char_count += len(line)

    return "\n".join(lines)


def _format_conversations_for_prompt(
    conversations: list[Conversation],
    max_total_chars: int = MAX_TOTAL_CHARS,
) -> str:
    """Format multiple conversations for the LLM prompt.

    Args:
        conversations: List of conversations to format.
        max_total_chars: Maximum total characters.

    Returns:
        Formatted string with all conversations.
    """
    formatted = []
    total_chars = 0

    for i, conv in enumerate(conversations):
        conv_str = _truncate_conversation(conv)
        if total_chars + len(conv_str) > max_total_chars:
            break
        formatted.append(f"=== Conversation {i + 1} ===\n{conv_str}")
        total_chars += len(conv_str)

    return "\n\n".join(formatted)


def build_suggestion_prompt(
    conversations: list[Conversation],
    catalog: dict[str, dict[str, Any]],
) -> tuple[str, str]:
    """Build the system and user prompts for suggestion generation.

    Args:
        conversations: Sample conversations to analyze.
        catalog: Analyzer catalog from get_analyzer_catalog().

    Returns:
        Tuple of (system_prompt, user_prompt).
    """
    # Format analyzer catalog
    catalog_text = ""
    for analyzer_id, info in catalog.items():
        metrics_str = ", ".join(info["metrics"])
        good_for_str = "; ".join(info["good_for"])
        catalog_text += f"""
- **{analyzer_id}** ({info["name"]}): {info["description"]}
  Metrics: {metrics_str}
  Good for: {good_for_str}
"""

    # Note: JSON schema is handled automatically by OpenAI's structured outputs
    # We just need to provide context and guidelines
    system_prompt = (
        """You are an expert data quality analyst helping users configure analyzers for their conversational datasets.

Your task is to analyze sample conversations and recommend:
1. **Analyzers**: Which built-in analyzers would be useful for this data
2. **Custom Metrics**: Any custom Python metrics that could extract useful information
3. **Tests**: Quality tests to run on the analysis results

Available Analyzers:
"""
        + catalog_text
        + """

Test Types:
- **threshold**: Check if a metric exceeds/falls below a value (e.g., total_tokens < 4096)
- **percentage**: Check what % of samples meet a condition (e.g., 95% should have passed == True)
- **range**: Check if a metric falls within a range (e.g., score between 50 and 100)

Guidelines:
- Only suggest analyzers that make sense for the data
- Always suggest "length" analyzer as it's fast and universally useful
- Suggest LLM-based analyzers only if content quality evaluation is relevant
- Custom metrics should extract metadata or compute simple derived values
- For custom metric functions, use: def compute(conversation): ... return {"field": value}
- Tests should catch data quality issues (too long, low quality, etc.)
- Be conservative - don't overwhelm users with too many suggestions (2-4 analyzers, 0-2 custom metrics, 2-4 tests)"""
    )

    # Format sample conversations
    conversations_text = _format_conversations_for_prompt(conversations)

    # Analyze some patterns
    num_conversations = len(conversations)
    avg_messages = sum(len(c.messages) for c in conversations) / max(
        num_conversations, 1
    )
    has_metadata = any(c.metadata for c in conversations)

    metadata_fields = set()
    for conv in conversations:
        if conv.metadata:
            metadata_fields.update(conv.metadata.keys())

    metadata_note = ""
    if metadata_fields:
        metadata_note = f"\nNote: Conversations have metadata fields: {', '.join(sorted(metadata_fields))}"

    user_prompt = f"""Analyze these {num_conversations} sample conversations and suggest appropriate analyzers, custom metrics, and tests.

Dataset Statistics:
- Number of samples shown: {num_conversations}
- Average messages per conversation: {avg_messages:.1f}
- Has metadata: {has_metadata}{metadata_note}

Sample Conversations:
{conversations_text}

Based on this data, what analyzers, custom metrics, and tests would you recommend? Remember to be selective and only suggest what's truly useful for this specific data."""

    return system_prompt, user_prompt


def _call_openai_structured(
    system_prompt: str,
    user_prompt: str,
    model: str = "gpt-4o-mini",
    api_key: str | None = None,
) -> SuggestionResponse:
    """Call OpenAI API with structured outputs for guaranteed schema compliance.

    Uses OpenAI's beta.chat.completions.parse() method which:
    - Guarantees the response matches our Pydantic schema
    - Eliminates JSON parsing errors
    - Provides type-safe access to response fields

    Args:
        system_prompt: System message with context and instructions.
        user_prompt: User message with sample data.
        model: Model to use (must support structured outputs).
        api_key: Optional API key (defaults to OPENAI_API_KEY env var).

    Returns:
        SuggestionResponse with parsed suggestions or error.
    """
    try:
        from openai import OpenAI
    except ImportError:
        return SuggestionResponse(
            error="openai package is required for suggestions. "
            "Install with: pip install openai"
        )

    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return SuggestionResponse(
            error="OpenAI API key not found. Set OPENAI_API_KEY environment variable."
        )

    try:
        client = OpenAI(api_key=api_key)

        # Use structured outputs with Pydantic model
        # This guarantees the response matches our schema exactly
        completion = client.beta.chat.completions.parse(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format=SuggestionsOutput,
            temperature=0.3,  # Lower temperature for more consistent output
            max_tokens=2000,
        )

        # Extract the parsed response
        parsed = completion.choices[0].message.parsed

        if parsed is None:
            # Check if there was a refusal
            refusal = completion.choices[0].message.refusal
            if refusal:
                logger.warning(f"Model refused to generate suggestions: {refusal}")
                return SuggestionResponse(error=f"Model refused: {refusal}")
            return SuggestionResponse(error="No suggestions generated")

        # Convert to SuggestionResponse (which includes error field)
        return SuggestionResponse(
            analyzers=parsed.analyzers,
            custom_metrics=parsed.custom_metrics,
            tests=parsed.tests,
        )

    except Exception as e:
        logger.error(f"OpenAI API call failed: {e}")
        return SuggestionResponse(error=f"API call failed: {e}")


def generate_suggestions(
    dataset_path: str | None = None,
    dataset_name: str | None = None,
    split: str = "train",
    subset: str | None = None,
    sample_count: int = 5,
    model: str = "gpt-4o-mini",
    api_key: str | None = None,
) -> SuggestionResponse:
    """Generate analyzer, metric, and test suggestions for a dataset.

    This is the main entry point for the suggestion system. It loads
    sample conversations, calls the LLM, and returns structured suggestions.

    Args:
        dataset_path: Path to local JSONL dataset file.
        dataset_name: HuggingFace dataset name.
        split: Dataset split (default: "train").
        subset: Dataset subset/config.
        sample_count: Number of sample conversations to analyze.
        model: LLM model to use for suggestions.
        api_key: Optional OpenAI API key.

    Returns:
        SuggestionResponse with analyzers, custom_metrics, and tests.
    """
    # Load sample conversations
    try:
        if dataset_path:
            from oumi.analyze.cli import load_conversations_from_path

            conversations = load_conversations_from_path(dataset_path, sample_count)
        elif dataset_name:
            from oumi.analyze.cli import load_conversations_from_dataset

            conversations = load_conversations_from_dataset(
                dataset_name, split, subset, sample_count
            )
        else:
            return SuggestionResponse(
                error="Either dataset_path or dataset_name is required"
            )

        if not conversations:
            return SuggestionResponse(error="No conversations loaded from dataset")

        logger.info(f"Loaded {len(conversations)} sample conversations for analysis")

    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return SuggestionResponse(error=f"Failed to load dataset: {e}")

    # Get analyzer catalog and build prompt
    catalog = get_analyzer_catalog()
    system_prompt, user_prompt = build_suggestion_prompt(conversations, catalog)

    # Call LLM with structured outputs
    logger.info(f"Calling {model} for suggestions with structured outputs...")
    response = _call_openai_structured(system_prompt, user_prompt, model, api_key)

    if response.error:
        logger.error(f"Suggestion generation failed: {response.error}")
    else:
        logger.info(
            f"Generated {len(response.analyzers)} analyzer, "
            f"{len(response.custom_metrics)} custom metric, "
            f"and {len(response.tests)} test suggestions"
        )

    return response


def suggestion_response_to_dict(response: SuggestionResponse) -> dict[str, Any]:
    """Convert SuggestionResponse to a JSON-serializable dictionary.

    Since SuggestionResponse is a Pydantic model, this uses model_dump()
    for clean serialization with proper handling of nested models.

    Args:
        response: The suggestion response to convert.

    Returns:
        Dictionary suitable for JSON serialization.
    """
    return response.model_dump()
