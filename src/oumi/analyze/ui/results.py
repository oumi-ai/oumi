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

"""Results page component for the Analyze web viewer.

This component displays test results grouped by pass/fail status,
with sample conversations showing issues and reasoning.
"""

from typing import Any

import streamlit as st

from oumi.analyze.storage import EvalData


def render_results_table(eval_data: EvalData) -> None:
    """Render the test results page.

    Args:
        eval_data: The eval data to display.
    """
    st.header("Test Results")

    test_results = eval_data.test_results
    analysis_results = eval_data.analysis_results
    conversations = eval_data.conversations or []

    # Test results can be in "results" or "tests" key
    tests = []
    if test_results:
        tests = test_results.get("results", test_results.get("tests", []))

    if not tests:
        st.info(
            "No test results available.\n\n"
            "Add tests to your config to validate analysis results."
        )
        # Show a link to raw results
        if analysis_results:
            with st.expander("View Raw Analysis Results"):
                _render_raw_results(analysis_results)
        return
    if not tests:
        st.info("No tests were configured for this analysis.")
        return

    # Summary metrics
    _render_test_summary(tests)

    st.divider()

    # Separate passed and failed tests
    failed_tests = [t for t in tests if not t.get("passed", False)]
    passed_tests = [t for t in tests if t.get("passed", False)]

    # Failed tests section (expanded by default)
    if failed_tests:
        st.subheader(f"❌ Failed Tests ({len(failed_tests)})")
        for test in failed_tests:
            _render_test_card(
                test, analysis_results, is_failed=True, conversations=conversations
            )

    # Passed tests section (collapsed by default)
    if passed_tests:
        st.subheader(f"✅ Passed Tests ({len(passed_tests)})")
        with st.expander(f"Show {len(passed_tests)} passed tests", expanded=False):
            for test in passed_tests:
                _render_test_card(
                    test, analysis_results, is_failed=False, conversations=conversations
                )


def _render_test_summary(tests: list[dict[str, Any]]) -> None:
    """Render the test summary metrics.

    Args:
        tests: List of test result dictionaries.
    """
    total = len(tests)
    passed = sum(1 for t in tests if t.get("passed", False))
    failed = total - passed
    pass_rate = (passed / total * 100) if total > 0 else 0

    # Count by severity
    high_failures = sum(
        1 for t in tests
        if not t.get("passed", False) and t.get("severity") == "high"
    )
    medium_failures = sum(
        1 for t in tests
        if not t.get("passed", False) and t.get("severity") == "medium"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Pass Rate", f"{pass_rate:.0f}%")

    with col2:
        st.metric("Passed", passed, delta=None)

    with col3:
        delta_color = "inverse" if failed > 0 else "off"
        st.metric("Failed", failed, delta=None)

    with col4:
        if high_failures > 0:
            st.metric("High Severity", high_failures, delta=None)
        elif medium_failures > 0:
            st.metric("Medium Severity", medium_failures, delta=None)
        else:
            st.metric("Issues", 0)


def _render_test_card(
    test: dict[str, Any],
    analysis_results: dict[str, Any],
    is_failed: bool,
    conversations: list[dict[str, Any]] | None = None,
) -> None:
    """Render a single test card with details and sample conversations.

    Args:
        test: Test result dictionary.
        analysis_results: Full analysis results for looking up samples.
        is_failed: Whether this test failed.
        conversations: List of conversation dicts (optional).
    """
    test_id = test.get("test_id", test.get("id", "Unknown"))
    title = test.get("title", test_id)
    severity = test.get("severity", "medium")
    description = test.get("description", "")
    message = test.get("message", "")
    metric = test.get("metric", "")
    actual_value = test.get("actual_value")
    expected_value = test.get("expected_value")
    affected_count = test.get("affected_count")
    total_count = test.get("total_count")
    affected_percentage = test.get("affected_percentage")
    threshold = test.get("threshold")
    sample_indices = test.get("sample_indices", [])

    # Severity styling
    severity_colors = {
        "high": "🔴",
        "medium": "🟡",
        "low": "⚪",
    }
    severity_icon = severity_colors.get(severity, "⚪")

    # Build the card header
    if is_failed:
        header = f"{severity_icon} **{title}**"
    else:
        header = f"✅ **{title}**"

    with st.expander(header, expanded=is_failed):
        # Test details
        col1, col2 = st.columns([2, 1])

        with col1:
            if description:
                st.markdown(f"*{description}*")
            if message:
                st.info(message)

        with col2:
            st.caption(f"**ID:** `{test_id}`")
            st.caption(f"**Severity:** {severity}")
            if metric:
                st.caption(f"**Metric:** `{metric}`")

        # Show statistics
        if affected_count is not None and total_count is not None:
            st.markdown("---")
            stat_cols = st.columns(3)
            with stat_cols[0]:
                st.metric("Affected", f"{affected_count}/{total_count}")
            with stat_cols[1]:
                if affected_percentage is not None:
                    st.metric("Rate", f"{affected_percentage:.1f}%")
            with stat_cols[2]:
                if threshold is not None:
                    st.metric("Threshold", f"{threshold}%")

        # Show actual vs expected if available
        if actual_value is not None or expected_value is not None:
            st.markdown("---")
            val_col1, val_col2 = st.columns(2)
            with val_col1:
                if actual_value is not None:
                    st.metric("Actual", _format_value(actual_value))
            with val_col2:
                if expected_value is not None:
                    st.metric("Expected", _format_value(expected_value))

        # Show sample conversations for failed tests
        if is_failed:
            st.markdown("---")
            # Determine which samples are problematic based on test type
            problematic_indices = _get_problematic_indices(test, analysis_results)
            if problematic_indices:
                st.markdown(f"**Sample Conversations with Issues ({len(problematic_indices)}):**")
                _render_sample_conversations_by_index(
                    problematic_indices,
                    analysis_results,
                    metric,
                    show_all=True,
                    conversations=conversations,
                )
            else:
                st.markdown("**Sample Conversations with Issues:**")
                _render_sample_conversations(
                    test, analysis_results, metric, conversations=conversations
                )


def _render_sample_conversations_by_index(
    sample_indices: list[int],
    analysis_results: dict[str, Any],
    metric: str,
    max_samples: int = 5,
    show_all: bool = False,
    conversations: list[dict[str, Any]] | None = None,
) -> None:
    """Render sample conversations by their indices.

    Args:
        sample_indices: List of sample indices to show.
        analysis_results: Full analysis results.
        metric: The metric that was tested.
        max_samples: Maximum number of samples to show.
        show_all: If True, show all samples without filtering for issues.
        conversations: List of conversation dicts (optional).
    """
    if not sample_indices:
        st.caption("No samples to display.")
        return

    # Parse the metric to get analyzer name
    parts = metric.split(".") if metric else []
    analyzer_name = parts[0] if len(parts) > 1 else None

    # Find the analyzer results to display
    target_results = None
    target_name = None
    if analyzer_name and analyzer_name in analysis_results:
        target_results = analysis_results[analyzer_name]
        target_name = analyzer_name
    else:
        # Try to find any LLM analyzer results (with reasoning)
        for name, results in analysis_results.items():
            if isinstance(results, list) and results:
                if isinstance(results[0], dict) and "reasoning" in results[0]:
                    target_results = results
                    target_name = name
                    break

    if not target_results or not isinstance(target_results, list):
        st.caption("No detailed results available for these samples.")
        return

    # Get samples to show
    samples_to_show = []
    for idx in sample_indices:
        if idx < len(target_results):
            result = target_results[idx]
            if isinstance(result, dict):
                if show_all or _sample_has_issue(result):
                    samples_to_show.append((idx, result))

    if not samples_to_show:
        st.caption("No samples with detailed results found.")
        return

    # Show samples (limit to max_samples)
    for idx, result in samples_to_show[:max_samples]:
        # Get conversation for this index if available
        conv = None
        if conversations and idx < len(conversations):
            conv = conversations[idx]
        _render_single_sample(
            idx, result, metric.split(".")[-1] if metric else "", conversation=conv
        )

    if len(samples_to_show) > max_samples:
        st.caption(f"... and {len(samples_to_show) - max_samples} more samples")


def _sample_has_issue(result: dict[str, Any], score_threshold: int = 50) -> bool:
    """Check if a sample result has an issue.

    Args:
        result: The analysis result dictionary.
        score_threshold: Score below which is considered an issue.

    Returns:
        True if the sample has an issue.
    """
    # Check for explicit failure
    if "passed" in result and result["passed"] is False:
        return True

    # Check for low score
    if "score" in result:
        score = result["score"]
        if isinstance(score, (int, float)) and score < score_threshold:
            return True

    # Check for error
    if "error" in result and result["error"]:
        return True

    return False


def _get_problematic_indices(
    test: dict[str, Any],
    analysis_results: dict[str, Any],
) -> list[int]:
    """Get indices of problematic samples for a failed test.

    For min_percentage tests (threshold): samples NOT in sample_indices are problematic
    For max_percentage tests: samples IN sample_indices are problematic

    Args:
        test: Test result dictionary.
        analysis_results: Full analysis results.

    Returns:
        List of problematic sample indices.
    """
    sample_indices = set(test.get("sample_indices", []))
    total_count = test.get("total_count", 0)
    threshold = test.get("threshold")
    affected_percentage = test.get("affected_percentage")

    # Handle None values
    if threshold is None or affected_percentage is None:
        # Can't determine test type, return sample_indices as-is
        return sorted(sample_indices)

    # Determine test type based on whether we're checking min or max
    # If affected_percentage < threshold, it's likely a min_percentage test
    # If affected_percentage > threshold, it's likely a max_percentage test
    is_min_threshold_test = affected_percentage < threshold

    if is_min_threshold_test:
        # For min_percentage tests that failed:
        # The problematic samples are those NOT in sample_indices
        # (they didn't meet the condition)
        all_indices = set(range(total_count))
        problematic = list(all_indices - sample_indices)
    else:
        # For max_percentage tests that failed:
        # The problematic samples are those IN sample_indices
        # (they exceeded the limit)
        problematic = list(sample_indices)

    return sorted(problematic)


def _render_sample_conversations(
    test: dict[str, Any],
    analysis_results: dict[str, Any],
    metric: str,
    max_samples: int = 5,
    conversations: list[dict[str, Any]] | None = None,
) -> None:
    """Render sample conversations that have issues (failed the quality check).

    Only shows samples with actual issues (low scores, failed, errors).

    Args:
        test: Test result dictionary.
        analysis_results: Full analysis results.
        metric: The metric that was tested.
        max_samples: Maximum number of samples to show.
        conversations: List of conversation dicts (optional).
    """
    # Parse the metric to get analyzer name
    # Metric format is typically "AnalyzerName.field" or just "field"
    parts = metric.split(".")
    analyzer_name = parts[0] if len(parts) > 1 else None
    field_name = parts[-1]

    # Collect samples with issues
    failed_samples = []

    if analyzer_name and analyzer_name in analysis_results:
        analyzer_results = analysis_results[analyzer_name]
        if isinstance(analyzer_results, list):
            for i, result in enumerate(analyzer_results):
                if not isinstance(result, dict):
                    continue

                # Only include samples with actual issues
                if _sample_has_issue(result):
                    failed_samples.append((i, result, field_name))

    # Also check other analyzers if we didn't find the specific one
    if not failed_samples:
        for name, results in analysis_results.items():
            if not isinstance(results, list):
                continue
            for i, result in enumerate(results):
                if isinstance(result, dict) and _sample_has_issue(result):
                    failed_samples.append((i, result, ""))
            if failed_samples:
                break

    if not failed_samples:
        st.caption("No samples with issues found.")
        return

    # Show failed samples
    for i, result, fname in failed_samples[:max_samples]:
        conv = None
        if conversations and i < len(conversations):
            conv = conversations[i]
        _render_single_sample(i, result, fname, conversation=conv)

    if len(failed_samples) > max_samples:
        st.caption(f"... and {len(failed_samples) - max_samples} more samples with issues")


def _render_single_sample(
    index: int,
    result: dict[str, Any],
    field_name: str,
    conversation: dict[str, Any] | None = None,
) -> None:
    """Render a single sample with its analysis result and conversation.

    Args:
        index: Sample index.
        result: The analysis result for this sample.
        field_name: The specific field being tested.
        conversation: The conversation data for this sample (optional).
    """
    with st.container():
        st.markdown(f"**Sample {index + 1}**")

        # Create columns for score/status and reasoning
        col1, col2 = st.columns([1, 3])

        with col1:
            # Show score if available
            if "score" in result:
                score = result["score"]
                if isinstance(score, (int, float)):
                    # Color code the score
                    if score >= 70:
                        st.success(f"Score: {score}")
                    elif score >= 50:
                        st.warning(f"Score: {score}")
                    else:
                        st.error(f"Score: {score}")

            # Show passed status
            if "passed" in result:
                if result["passed"]:
                    st.success("Passed")
                else:
                    st.error("Failed")

            # Show category if available
            if "category" in result and result["category"]:
                st.caption(f"Category: {result['category']}")

            # Show label if available
            if "label" in result and result["label"]:
                st.caption(f"Label: {result['label']}")

        with col2:
            # Show reasoning
            if "reasoning" in result and result["reasoning"]:
                st.markdown("**Reasoning:**")
                st.markdown(f"> {result['reasoning']}")

            # Show error if present
            if "error" in result and result["error"]:
                st.error(f"**Error:** {result['error']}")

        # Show conversation content if available
        if conversation:
            _render_conversation(conversation)

        st.markdown("---")


def _render_conversation(conversation: dict[str, Any]) -> None:
    """Render conversation messages in a chat-like format.

    Args:
        conversation: Conversation dict with messages.
    """
    messages = conversation.get("messages", [])
    if not messages:
        return

    with st.expander("💬 View Conversation", expanded=False):
        for i, msg in enumerate(messages):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")

            # Handle content that might be a list (multimodal)
            if isinstance(content, list):
                text_parts = []
                for part in content:
                    if isinstance(part, dict) and "text" in part:
                        text_parts.append(part["text"])
                    elif isinstance(part, str):
                        text_parts.append(part)
                content = "\n".join(text_parts) if text_parts else str(content)

            content_str = str(content)

            # Set truncation limits by role
            if role == "system":
                truncate_at = 200
            elif role in ("user", "assistant"):
                truncate_at = 500
            else:
                truncate_at = 300

            is_truncated = len(content_str) > truncate_at
            display_content = (
                content_str[:truncate_at] + "..." if is_truncated else content_str
            )

            # Style based on role
            if role == "user":
                st.markdown("**🧑 User:**")
            elif role == "assistant":
                st.markdown("**🤖 Assistant:**")
            elif role == "system":
                st.markdown("**⚙️ System:**")
            else:
                st.markdown(f"**{role}:**")

            # Show truncated content with option to expand
            if is_truncated:
                st.markdown(f"> {display_content}")
                # Use a unique key for each message's expander
                with st.expander("📄 Show full message", expanded=False):
                    st.text_area(
                        label=f"Full {role} message",
                        value=content_str,
                        height=300,
                        disabled=True,
                        label_visibility="collapsed",
                        key=f"full_msg_{id(conversation)}_{i}",
                    )
            else:
                st.markdown(f"> {display_content}")

        # Show metadata if present
        metadata = conversation.get("metadata", {})
        if metadata:
            with st.expander("📋 Metadata", expanded=False):
                st.json(metadata)


def _format_value(value: Any) -> str:
    """Format a value for display.

    Args:
        value: The value to format.

    Returns:
        Formatted string.
    """
    if isinstance(value, float):
        if value == int(value):
            return str(int(value))
        return f"{value:.2f}"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    return str(value)


def _render_raw_results(analysis_results: dict[str, Any]) -> None:
    """Render raw analysis results as JSON.

    Args:
        analysis_results: The analysis results to display.
    """
    # Show selector for analyzer
    analyzer_names = list(analysis_results.keys())
    if not analyzer_names:
        st.info("No analysis results available.")
        return

    selected_analyzer = st.selectbox(
        "Select Analyzer",
        analyzer_names,
        key="raw_results_analyzer",
    )

    if selected_analyzer:
        results = analysis_results[selected_analyzer]
        if isinstance(results, list):
            st.caption(f"Showing {len(results)} results")

            # Show as table if possible
            import pandas as pd
            try:
                df = pd.DataFrame(results)
                st.dataframe(df, use_container_width=True)
            except Exception:
                st.json(results)
        else:
            st.json(results)
