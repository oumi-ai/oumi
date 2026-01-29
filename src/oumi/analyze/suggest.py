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
            "Analyzer ID: length, quality, usefulness, safety, coherence, "
            "factuality, or instruction_following. Suggest only 2-3 total."
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
    # Percentage test fields - USE POSITIVE CONDITIONS with min_percentage
    condition: str | None = Field(
        default=None,
        description=(
            "Condition for percentage tests. ALWAYS use POSITIVE conditions "
            "(e.g., '== True' for has_alternating_turns) with min_percentage "
            "for clearer failure messages. NEVER use '== False' with max_percentage."
        ),
    )
    max_percentage: float | None = Field(
        default=None,
        description=(
            "Max % of samples matching the condition. Use for BAD metrics "
            "(e.g., max 5% can have total_tokens > 8000)."
        ),
    )
    min_percentage: float | None = Field(
        default=None,
        description=(
            "Min % of samples that must match the condition. Use for GOOD metrics "
            "(e.g., at least 95% should have has_alternating_turns == True)."
        ),
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
        default_factory=list,
        description="List of 2-3 suggested analyzers (always include length and quality)",
    )
    custom_metrics: list[CustomMetricSuggestion] = Field(
        default_factory=list,
        description="0-2 simple custom metrics if needed (no imports, simple logic only)",
    )
    tests: list[TestSuggestion] = Field(
        default_factory=list,
        description="List of 2-4 suggested tests for the most important quality checks",
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
        "quality": {
            "name": "Data Quality Analyzer",
            "description": "Fast, non-LLM quality checks for data validation",
            "metrics": [
                "has_alternating_turns",
                "has_empty_turns",
                "has_invalid_values",
                "fits_4k_context",
                "appears_truncated",
                "has_policy_refusal",
                "has_unbalanced_tags",
                "passes_basic_quality",
            ],
            "good_for": [
                "Detecting malformed conversations (non-alternating turns)",
                "Finding empty or whitespace-only messages",
                "Detecting serialization errors (NaN, null values)",
                "Checking for truncated conversations",
                "Finding policy refusal responses",
                "Checking for unbalanced thinking/code tags",
            ],
            "params": {
                "check_turn_pattern": "Whether to check for alternating turns",
                "check_empty_content": "Whether to check for empty messages",
                "check_truncation": "Whether to check for truncation",
                "check_refusals": "Whether to check for policy refusals",
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
    user_query: str | None = None,
) -> tuple[str, str]:
    """Build the system and user prompts for suggestion generation.

    Args:
        conversations: Sample conversations to analyze.
        catalog: Analyzer catalog from get_analyzer_catalog().
        user_query: Optional user description of their goals and issues to check.

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
        """You are an expert data quality analyst helping users prepare conversational datasets for SFT (Supervised Fine-Tuning) of Large Language Models.

**Primary Goal**: Help users identify and flag problematic examples in their dataset so they can either DROP those examples or FIX them before training. Clean, high-quality training data is essential for effective fine-tuning.

Your task is to analyze sample conversations and recommend:
1. **Analyzers**: Which built-in analyzers would catch relevant issues
2. **Custom Metrics**: Any custom Python metrics to extract useful information or detect specific issues
3. **Tests**: Quality tests that will flag problematic examples for review

Common data quality issues that harm SFT training:
- Malformed conversations (wrong turn order, empty messages)
- Truncated or incomplete responses
- Excessive policy refusals ("I cannot help with that")
- Unbalanced or missing special tokens (think tags, code blocks)
- Too-long conversations that exceed context limits
- Low-quality or unhelpful responses
- Factually incorrect information
- Unsafe or inappropriate content

Available Analyzers:
"""
        + catalog_text
        + """

Test Types:
- **threshold**: For NUMERIC metrics. Check if value exceeds/falls below threshold.
- **percentage**: For BOOLEAN metrics. Check what % of samples meet condition.
- **range**: For NUMERIC metrics. Check if value falls within min/max range.

CRITICAL - Metric Path Format:
- Built-in analyzers: analyzer_id.field_name (e.g., "length.total_tokens", "quality.has_empty_turns")
- Custom metrics: custom_metric_id.field_name (e.g., "my_metric.my_field")
- NEVER use just the metric id without the field name

CRITICAL - Match Test Type to Metric Type:
- Boolean fields (has_empty_turns, passes_basic_quality): use "percentage" test with condition
- Numeric fields (total_tokens, score): use "threshold" test with operator/value
- NEVER use "condition" with numeric metrics or "operator" with boolean metrics

Test Rules for percentage tests (FOLLOW EXACTLY):
- ONLY use min_percentage, NEVER use max_percentage for percentage tests
- Set max_percentage to null always
- Use POSITIVE framing: what percentage SHOULD pass
- For NEGATIVE indicators (has_empty_turns, appears_truncated, has_policy_refusal), use "== False" to check that samples do NOT have the issue
- For POSITIVE indicators (has_alternating_turns, passes_basic_quality), use "== True" to check that samples HAVE the quality

SEMANTICS EXPLANATION:
- "condition: == False, min_percentage: 95" means "At least 95% of samples should NOT have this issue"
- "condition: == True, min_percentage: 95" means "At least 95% of samples SHOULD have this quality"

CORRECT percentage test examples (copy these patterns exactly):
- quality.has_alternating_turns: condition "== True", min_percentage: 95 (95% should have proper turn order)
- quality.has_empty_turns: condition "== False", min_percentage: 100 (100% should NOT have empty turns)
- quality.appears_truncated: condition "== False", min_percentage: 95 (95% should NOT be truncated)
- quality.has_policy_refusal: condition "== False", min_percentage: 95 (95% should NOT have refusals)
- quality.passes_basic_quality: condition "== True", min_percentage: 90 (90% should pass quality)
- custom_metric.has_short_responses: condition "== False", min_percentage: 95 (95% should NOT have short responses)

CORRECT threshold test examples:
- length.total_tokens: type threshold, operator ">", value 8000, max_percentage: 5

FORBIDDEN patterns (will cause errors):
- NEVER set both min_percentage AND max_percentage
- NEVER use max_percentage with percentage tests (only use min_percentage)
- NEVER use "condition" for numeric metrics

Complete test YAML examples to follow:

Example 1 - Check ABSENCE of refusals (negative indicator):
  id: no_policy_refusals
  type: percentage
  metric: quality.has_policy_refusal
  title: "No Policy Refusals"
  description: "Ensures at least 95% of conversations do NOT contain policy refusals."
  condition: "== False"
  min_percentage: 95
  max_percentage: null
  severity: high

Example 2 - Check PRESENCE of good turn order (positive indicator):
  id: proper_turn_order
  type: percentage
  metric: quality.has_alternating_turns
  title: "Proper Turn Order"
  description: "Ensures at least 95% of conversations have proper alternating turns."
  condition: "== True"
  min_percentage: 95
  max_percentage: null
  severity: medium

Example 3 - Check ABSENCE of truncation (negative indicator):
  id: no_truncation
  type: percentage
  metric: quality.appears_truncated
  title: "No Truncated Responses"
  description: "Ensures at least 95% of responses are NOT truncated."
  condition: "== False"
  min_percentage: 95
  max_percentage: null
  severity: high

Custom Metric Examples (keep code simple, NO imports):

Example 1 - Check if assistant ends with punctuation:
```python
def compute(message):
    if str(message.role.value) != "assistant":
        return {"ends_properly": True}
    content = message.content or ""
    ends_properly = content.strip().endswith((".", "!", "?", '"', "'"))
    return {"ends_properly": ends_properly}
```
output_schema: [{"name": "ends_properly", "type": "bool"}]
scope: message

Example 2 - Extract metadata field:
```python
def compute(conversation):
    metadata = conversation.metadata or {}
    label = metadata.get("label", "unknown")
    return {"label": label, "has_label": label != "unknown"}
```
output_schema: [{"name": "label", "type": "str"}, {"name": "has_label", "type": "bool"}]
scope: conversation

Example 3 - Count short assistant responses:
```python
def compute(conversation):
    count = 0
    for m in conversation.messages:
        if str(m.role.value) == "assistant":
            words = len((m.content or "").split())
            if words < 10:
                count += 1
    return {"short_response_count": count, "has_short_responses": count > 0}
```
output_schema: [{"name": "short_response_count", "type": "int"}, {"name": "has_short_responses", "type": "bool"}]
scope: conversation

Example 4 - Use length analyzer results (depends_on):
```python
def compute(conversation, results, index):
    length_result = results["length"][index]
    total_tokens = getattr(length_result, "total_tokens", 0) or 0
    num_messages = getattr(length_result, "num_messages", 1) or 1
    avg_tokens = total_tokens / num_messages
    return {"avg_tokens_per_msg": round(avg_tokens, 1), "is_verbose": avg_tokens > 200}
```
output_schema: [{"name": "avg_tokens_per_msg", "type": "float"}, {"name": "is_verbose", "type": "bool"}]
scope: conversation
depends_on: ["length"]
NOTE: When using depends_on, the function signature is: def compute(conversation, results, index)

Guidelines:
- Suggest 2-4 analyzers total (always include "length" and "quality")
- LLM-based analyzers (usefulness, safety, coherence, etc.) are useful but expensive - suggest 1-2 if relevant
- Custom metrics: only suggest 0-2 if truly needed, keep code simple (NO imports, NO complex logic)
- Suggest 3-6 tests that catch the most important data quality issues
- Tests should have clear thresholds that flag examples needing human review"""
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
        metadata_note = (
            f"\nNote: Conversations have metadata fields: "
            f"{', '.join(sorted(metadata_fields))}"
        )

    # Build user prompt with optional user query
    user_query_section = ""
    if user_query and user_query.strip():
        user_query_section = f"""
**User's Goals and Concerns:**
{user_query.strip()}

Please tailor your suggestions to address these specific concerns while also catching general data quality issues.
"""

    user_prompt = f"""Analyze these {num_conversations} sample conversations and suggest appropriate analyzers, custom metrics, and tests for SFT training data preparation.

Dataset Statistics:
- Number of samples shown: {num_conversations}
- Average messages per conversation: {avg_messages:.1f}
- Has metadata: {has_metadata}{metadata_note}
{user_query_section}
Sample Conversations:
{conversations_text}

Based on this data{" and the user's goals" if user_query else ""}, what analyzers, custom metrics, and tests would you recommend to ensure this dataset is clean and high-quality for fine-tuning?"""

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
            temperature=0.3,
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

        # Normalize tests to use positive conditions for clearer failure messages
        normalized_tests = _normalize_test_suggestions(parsed.tests)

        # Convert to SuggestionResponse (which includes error field)
        return SuggestionResponse(
            analyzers=parsed.analyzers,
            custom_metrics=parsed.custom_metrics,
            tests=normalized_tests,
        )

    except Exception as e:
        logger.error(f"OpenAI API call failed: {e}")
        return SuggestionResponse(error=f"API call failed: {e}")


def _normalize_test_suggestions(tests: list[TestSuggestion]) -> list[TestSuggestion]:
    """Normalize test suggestions to use positive conditions.

    Converts confusing patterns like "== False" with max_percentage to
    the clearer "== True" with min_percentage equivalent.

    Args:
        tests: List of test suggestions from LLM.

    Returns:
        Normalized test suggestions with clearer conditions.
    """
    normalized = []

    for test in tests:
        # Only process percentage tests with the problematic pattern
        if test.type != "percentage" or test.condition is None:
            normalized.append(test)
            continue

        condition = test.condition.strip().lower()

        # Check for the bad pattern: "== false" with max_percentage
        if (
            condition in ("== false", "==false")
            and test.max_percentage is not None
            and test.min_percentage is None
        ):
            # Convert to positive: "== True" with min_percentage = 100 - max_percentage
            new_min_pct = 100.0 - test.max_percentage
            normalized.append(
                TestSuggestion(
                    id=test.id,
                    type=test.type,
                    metric=test.metric,
                    title=test.title,
                    description=test.description,
                    severity=test.severity,
                    reason=test.reason,
                    condition="== True",
                    min_percentage=new_min_pct,
                    max_percentage=None,
                )
            )
            logger.info(
                f"Normalized test '{test.id}': '== False' max={test.max_percentage}% "
                f"-> '== True' min={new_min_pct}%"
            )
        # Check for: "!= true" with max_percentage (same issue)
        elif (
            condition in ("!= true", "!=true")
            and test.max_percentage is not None
            and test.min_percentage is None
        ):
            new_min_pct = 100.0 - test.max_percentage
            normalized.append(
                TestSuggestion(
                    id=test.id,
                    type=test.type,
                    metric=test.metric,
                    title=test.title,
                    description=test.description,
                    severity=test.severity,
                    reason=test.reason,
                    condition="== True",
                    min_percentage=new_min_pct,
                    max_percentage=None,
                )
            )
            logger.info(
                f"Normalized test '{test.id}': '!= True' max={test.max_percentage}% "
                f"-> '== True' min={new_min_pct}%"
            )
        else:
            normalized.append(test)

    return normalized


def generate_suggestions(
    dataset_path: str | None = None,
    dataset_name: str | None = None,
    split: str = "train",
    subset: str | None = None,
    sample_count: int = 1,
    model: str = "gpt-4o-mini",
    api_key: str | None = None,
    user_query: str | None = None,
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
        user_query: Optional user description of goals and issues to check.

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
    system_prompt, user_prompt = build_suggestion_prompt(
        conversations, catalog, user_query
    )

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
