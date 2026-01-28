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
"""

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any

from oumi.core.types.conversation import Conversation

logger = logging.getLogger(__name__)

# Maximum characters per conversation to include in prompt
MAX_CONVERSATION_CHARS = 1500
# Maximum total chars for all conversations
MAX_TOTAL_CHARS = 6000


@dataclass
class AnalyzerSuggestion:
    """A suggested analyzer configuration."""

    id: str
    reason: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class CustomMetricSuggestion:
    """A suggested custom metric."""

    id: str
    function: str
    reason: str
    output_schema: list[dict[str, str]] = field(default_factory=list)
    description: str = ""


@dataclass
class TestSuggestion:
    """A suggested test configuration."""

    id: str
    type: str  # threshold, percentage, range
    metric: str
    reason: str
    title: str = ""
    description: str = ""
    severity: str = "medium"
    operator: str | None = None
    value: float | int | None = None
    condition: str | None = None
    max_percentage: float | None = None
    min_percentage: float | None = None
    min_value: float | None = None
    max_value: float | None = None


@dataclass
class SuggestionResponse:
    """Complete suggestion response from the LLM."""

    analyzers: list[AnalyzerSuggestion] = field(default_factory=list)
    custom_metrics: list[CustomMetricSuggestion] = field(default_factory=list)
    tests: list[TestSuggestion] = field(default_factory=list)
    error: str | None = None


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


def _truncate_conversation(conv: Conversation, max_chars: int = MAX_CONVERSATION_CHARS) -> str:
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
        content = msg.content or ""

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
- **{analyzer_id}** ({info['name']}): {info['description']}
  Metrics: {metrics_str}
  Good for: {good_for_str}
"""

    system_prompt = """You are an expert data quality analyst helping users configure analyzers for their conversational datasets.

Your task is to analyze sample conversations and recommend:
1. **Analyzers**: Which built-in analyzers would be useful for this data
2. **Custom Metrics**: Any custom Python metrics that could extract useful information
3. **Tests**: Quality tests to run on the analysis results

Available Analyzers:
""" + catalog_text + """

Test Types:
- **threshold**: Check if a metric exceeds/falls below a value (e.g., total_tokens < 4096)
- **percentage**: Check what % of samples meet a condition (e.g., 95% should have passed == True)
- **range**: Check if a metric falls within a range (e.g., score between 50 and 100)

Guidelines:
- Only suggest analyzers that make sense for the data
- Always suggest "length" analyzer as it's fast and universally useful
- Suggest LLM-based analyzers only if content quality evaluation is relevant
- Custom metrics should extract metadata or compute simple derived values
- Tests should catch data quality issues (too long, low quality, etc.)
- Be conservative - don't overwhelm users with too many suggestions

Respond ONLY in valid JSON format matching this schema:
{
  "analyzers": [
    {"id": "analyzer_id", "reason": "Why this analyzer is useful for this data", "params": {}}
  ],
  "custom_metrics": [
    {
      "id": "metric_id",
      "description": "What this metric measures",
      "function": "def compute(conversation):\\n    # Python code\\n    return {\\"field\\": value}",
      "output_schema": [{"name": "field", "type": "float", "description": "Field description"}],
      "reason": "Why this metric is useful"
    }
  ],
  "tests": [
    {
      "id": "test_id",
      "type": "threshold|percentage|range",
      "metric": "AnalyzerName.metric_name",
      "title": "Human readable title",
      "description": "What this test checks",
      "severity": "low|medium|high",
      "operator": "< | > | <= | >= | == | !=" (for threshold),
      "value": 1000 (for threshold),
      "condition": "== True" (for percentage),
      "min_percentage": 95.0 (for percentage),
      "max_percentage": 5.0 (for percentage),
      "min_value": 0 (for range),
      "max_value": 100 (for range),
      "reason": "Why this test is important"
    }
  ]
}"""

    # Format sample conversations
    conversations_text = _format_conversations_for_prompt(conversations)

    # Analyze some patterns
    num_conversations = len(conversations)
    avg_messages = sum(len(c.messages) for c in conversations) / max(num_conversations, 1)
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


def parse_suggestion_response(response_text: str) -> SuggestionResponse:
    """Parse and validate the LLM response.

    Args:
        response_text: Raw LLM response text.

    Returns:
        Validated SuggestionResponse.
    """
    try:
        # Try to extract JSON from the response
        # Handle cases where LLM wraps JSON in markdown code blocks
        text = response_text.strip()
        if text.startswith("```"):
            # Remove markdown code blocks
            lines = text.split("\n")
            json_lines = []
            in_block = False
            for line in lines:
                if line.startswith("```"):
                    in_block = not in_block
                    continue
                if in_block or not line.startswith("```"):
                    json_lines.append(line)
            text = "\n".join(json_lines)

        data = json.loads(text)

        # Parse analyzers
        analyzers = []
        for item in data.get("analyzers", []):
            analyzers.append(
                AnalyzerSuggestion(
                    id=item.get("id", ""),
                    reason=item.get("reason", ""),
                    params=item.get("params", {}),
                )
            )

        # Parse custom metrics
        custom_metrics = []
        for item in data.get("custom_metrics", []):
            custom_metrics.append(
                CustomMetricSuggestion(
                    id=item.get("id", ""),
                    function=item.get("function", ""),
                    reason=item.get("reason", ""),
                    output_schema=item.get("output_schema", []),
                    description=item.get("description", ""),
                )
            )

        # Parse tests
        tests = []
        for item in data.get("tests", []):
            tests.append(
                TestSuggestion(
                    id=item.get("id", ""),
                    type=item.get("type", "threshold"),
                    metric=item.get("metric", ""),
                    reason=item.get("reason", ""),
                    title=item.get("title", ""),
                    description=item.get("description", ""),
                    severity=item.get("severity", "medium"),
                    operator=item.get("operator"),
                    value=item.get("value"),
                    condition=item.get("condition"),
                    max_percentage=item.get("max_percentage"),
                    min_percentage=item.get("min_percentage"),
                    min_value=item.get("min_value"),
                    max_value=item.get("max_value"),
                )
            )

        return SuggestionResponse(
            analyzers=analyzers,
            custom_metrics=custom_metrics,
            tests=tests,
        )

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLM response as JSON: {e}")
        return SuggestionResponse(error=f"Failed to parse response: {e}")
    except Exception as e:
        logger.error(f"Error parsing suggestion response: {e}")
        return SuggestionResponse(error=str(e))


def _call_openai(
    system_prompt: str,
    user_prompt: str,
    model: str = "gpt-4o-mini",
    api_key: str | None = None,
) -> str:
    """Call OpenAI API to generate suggestions.

    Args:
        system_prompt: System message.
        user_prompt: User message.
        model: Model to use.
        api_key: Optional API key (defaults to env var).

    Returns:
        LLM response text.

    Raises:
        ValueError: If API key is not available.
        Exception: If API call fails.
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError(
            "openai package is required for suggestions. "
            "Install with: pip install openai"
        )

    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OpenAI API key not found. Set OPENAI_API_KEY environment variable "
            "or pass api_key parameter."
        )

    client = OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,  # Lower temperature for more consistent output
        max_tokens=2000,
    )

    return response.choices[0].message.content or ""


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

    # Call LLM
    try:
        logger.info(f"Calling {model} for suggestions...")
        response_text = _call_openai(system_prompt, user_prompt, model, api_key)
        logger.info("Received suggestions from LLM")
    except ValueError as e:
        # API key missing
        return SuggestionResponse(error=str(e))
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        return SuggestionResponse(error=f"LLM call failed: {e}")

    # Parse response
    return parse_suggestion_response(response_text)


def suggestion_response_to_dict(response: SuggestionResponse) -> dict[str, Any]:
    """Convert SuggestionResponse to a JSON-serializable dictionary.

    Args:
        response: The suggestion response to convert.

    Returns:
        Dictionary suitable for JSON serialization.
    """
    return {
        "analyzers": [
            {"id": a.id, "reason": a.reason, "params": a.params}
            for a in response.analyzers
        ],
        "custom_metrics": [
            {
                "id": m.id,
                "function": m.function,
                "reason": m.reason,
                "output_schema": m.output_schema,
                "description": m.description,
            }
            for m in response.custom_metrics
        ],
        "tests": [
            {
                "id": t.id,
                "type": t.type,
                "metric": t.metric,
                "reason": t.reason,
                "title": t.title,
                "description": t.description,
                "severity": t.severity,
                "operator": t.operator,
                "value": t.value,
                "condition": t.condition,
                "max_percentage": t.max_percentage,
                "min_percentage": t.min_percentage,
                "min_value": t.min_value,
                "max_value": t.max_value,
            }
            for t in response.tests
        ],
        "error": response.error,
    }
