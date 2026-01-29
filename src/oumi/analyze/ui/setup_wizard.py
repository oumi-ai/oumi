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

"""Setup wizard component for creating new analysis configurations."""

import json
import tempfile
from pathlib import Path
from typing import Any

import streamlit as st
import yaml


# Available analyzers with their descriptions and metrics
# These match the actual implementations in oumi.analyze.analyzers
AVAILABLE_ANALYZERS = {
    "length": {
        "name": "Length Analyzer",
        "description": "Analyze message lengths (characters, words, tokens)",
        "category": "Rule-based",
        "params": {
            "count_tokens": {
                "type": "checkbox",
                "default": False,
                "description": "Whether to compute token counts (uses tiktoken)",
            },
            "tiktoken_encoding": {
                "type": "text",
                "default": "cl100k_base",
                "description": "Tiktoken encoding name (e.g., cl100k_base for GPT-4)",
            },
            "compute_role_stats": {
                "type": "checkbox",
                "default": True,
                "description": "Compute per-role (user/assistant) word counts",
            },
        },
        "metrics": [
            {"name": "total_chars", "type": "number", "description": "Total characters"},
            {"name": "total_words", "type": "number", "description": "Total words"},
            {"name": "total_tokens", "type": "number", "description": "Total tokens (if enabled)"},
            {"name": "avg_chars_per_message", "type": "number"},
            {"name": "avg_words_per_message", "type": "number"},
            {"name": "avg_tokens_per_message", "type": "number"},
            {"name": "num_messages", "type": "number", "description": "Number of messages"},
            {"name": "user_total_words", "type": "number", "description": "User word count"},
            {"name": "assistant_total_words", "type": "number", "description": "Assistant word count"},
        ],
    },
    "quality": {
        "name": "Data Quality Analyzer",
        "description": "Fast, non-LLM quality checks for data validation",
        "category": "Rule-based",
        "params": {
            "check_turn_pattern": {
                "type": "checkbox",
                "default": True,
                "description": "Check for proper alternating user-assistant turns",
            },
            "check_empty_content": {
                "type": "checkbox",
                "default": True,
                "description": "Check for empty or whitespace-only messages",
            },
            "check_invalid_values": {
                "type": "checkbox",
                "default": True,
                "description": "Check for invalid serialization patterns (NaN, null)",
            },
            "check_truncation": {
                "type": "checkbox",
                "default": True,
                "description": "Check for truncated/incomplete responses",
            },
            "check_refusals": {
                "type": "checkbox",
                "default": True,
                "description": "Check for policy refusal patterns",
            },
            "check_tags": {
                "type": "checkbox",
                "default": True,
                "description": "Check for unbalanced thinking/code tags",
            },
        },
        "metrics": [
            # Boolean indicators
            {"name": "has_alternating_turns", "type": "boolean", "description": "Proper turn order"},
            {"name": "has_empty_turns", "type": "boolean", "description": "Contains empty messages"},
            {"name": "has_invalid_values", "type": "boolean", "description": "Contains NaN/null"},
            {"name": "fits_4k_context", "type": "boolean", "description": "Fits 4K context window"},
            {"name": "fits_8k_context", "type": "boolean", "description": "Fits 8K context window"},
            {"name": "appears_truncated", "type": "boolean", "description": "Response appears cut off"},
            {"name": "ends_mid_sentence", "type": "boolean", "description": "Ends without punctuation"},
            {"name": "has_policy_refusal", "type": "boolean", "description": "Contains refusal"},
            {"name": "has_think_tags", "type": "boolean", "description": "Contains thinking tags"},
            {"name": "has_unbalanced_tags", "type": "boolean", "description": "Unmatched tag pairs"},
            {"name": "passes_basic_quality", "type": "boolean", "description": "Passes all checks"},
            # Numeric counts
            {"name": "num_consecutive_same_role", "type": "number", "description": "Consecutive same-role count"},
            {"name": "empty_turn_count", "type": "number", "description": "Number of empty turns"},
            {"name": "estimated_tokens", "type": "number", "description": "Estimated token count"},
            {"name": "refusal_count", "type": "number", "description": "Number of refusal messages"},
        ],
    },
    "llm": {
        "name": "LLM Analyzer (Custom)",
        "description": "Use an LLM to evaluate conversations with a custom prompt",
        "category": "LLM-based",
        "params": {
            "criteria": {
                "type": "text",
                "default": "quality",
                "description": "Name for this evaluation criteria",
            },
            "custom_prompt": {
                "type": "textarea",
                "default": "",
                "description": "Custom evaluation prompt. Use {conversation} placeholder.",
            },
            "api_provider": {
                "type": "select",
                "options": ["openai", "anthropic"],
                "default": "openai",
                "description": "LLM API provider",
            },
            "model_name": {
                "type": "text",
                "default": "gpt-4o-mini",
                "description": "Model name to use",
            },
            "num_workers": {
                "type": "number",
                "default": 4,
                "description": "Number of parallel workers for API calls",
            },
        },
        "metrics": [
            {"name": "score", "type": "number", "range": "0-100"},
            {"name": "passed", "type": "boolean", "description": "True if score >= 50"},
            {"name": "label", "type": "enum", "values": ["excellent", "good", "fair", "poor"]},
            {"name": "reasoning", "type": "text", "description": "LLM's explanation"},
        ],
    },
    "usefulness": {
        "name": "Usefulness Analyzer",
        "description": "Evaluate how useful/helpful the assistant's response is",
        "category": "LLM-based (Preset)",
        "params": {
            "api_provider": {
                "type": "select",
                "options": ["openai", "anthropic"],
                "default": "openai",
                "description": "LLM API provider",
            },
            "model_name": {
                "type": "text",
                "default": "gpt-4o-mini",
                "description": "Model name to use",
            },
            "num_workers": {
                "type": "number",
                "default": 4,
                "description": "Number of parallel workers",
            },
        },
        "metrics": [
            {"name": "score", "type": "number", "range": "0-100"},
            {"name": "passed", "type": "boolean"},
            {"name": "label", "type": "enum", "values": ["excellent", "good", "fair", "poor"]},
            {"name": "reasoning", "type": "text"},
        ],
    },
    "safety": {
        "name": "Safety Analyzer",
        "description": "Evaluate safety and appropriateness of responses",
        "category": "LLM-based (Preset)",
        "params": {
            "api_provider": {
                "type": "select",
                "options": ["openai", "anthropic"],
                "default": "openai",
                "description": "LLM API provider",
            },
            "model_name": {
                "type": "text",
                "default": "gpt-4o-mini",
                "description": "Model name to use",
            },
            "num_workers": {
                "type": "number",
                "default": 4,
                "description": "Number of parallel workers",
            },
        },
        "metrics": [
            {"name": "score", "type": "number", "range": "0-100", "description": "Higher = safer"},
            {"name": "passed", "type": "boolean"},
            {"name": "label", "type": "enum", "values": ["excellent", "good", "fair", "poor"]},
            {"name": "reasoning", "type": "text"},
        ],
    },
    "coherence": {
        "name": "Coherence Analyzer",
        "description": "Evaluate logical flow and coherence of responses",
        "category": "LLM-based (Preset)",
        "params": {
            "api_provider": {
                "type": "select",
                "options": ["openai", "anthropic"],
                "default": "openai",
                "description": "LLM API provider",
            },
            "model_name": {
                "type": "text",
                "default": "gpt-4o-mini",
                "description": "Model name to use",
            },
            "num_workers": {
                "type": "number",
                "default": 4,
                "description": "Number of parallel workers",
            },
        },
        "metrics": [
            {"name": "score", "type": "number", "range": "0-100"},
            {"name": "passed", "type": "boolean"},
            {"name": "label", "type": "enum", "values": ["excellent", "good", "fair", "poor"]},
            {"name": "reasoning", "type": "text"},
        ],
    },
    "factuality": {
        "name": "Factuality Analyzer",
        "description": "Evaluate factual accuracy of responses",
        "category": "LLM-based (Preset)",
        "params": {
            "api_provider": {
                "type": "select",
                "options": ["openai", "anthropic"],
                "default": "openai",
                "description": "LLM API provider",
            },
            "model_name": {
                "type": "text",
                "default": "gpt-4o-mini",
                "description": "Model name to use",
            },
            "num_workers": {
                "type": "number",
                "default": 4,
                "description": "Number of parallel workers",
            },
        },
        "metrics": [
            {"name": "score", "type": "number", "range": "0-100"},
            {"name": "passed", "type": "boolean"},
            {"name": "label", "type": "enum", "values": ["excellent", "good", "fair", "poor"]},
            {"name": "reasoning", "type": "text"},
        ],
    },
    "instruction_following": {
        "name": "Instruction Following Analyzer",
        "description": "Evaluate how well responses follow instructions",
        "category": "LLM-based (Preset)",
        "params": {
            "api_provider": {
                "type": "select",
                "options": ["openai", "anthropic"],
                "default": "openai",
                "description": "LLM API provider",
            },
            "model_name": {
                "type": "text",
                "default": "gpt-4o-mini",
                "description": "Model name to use",
            },
            "num_workers": {
                "type": "number",
                "default": 4,
                "description": "Number of parallel workers",
            },
        },
        "metrics": [
            {"name": "score", "type": "number", "range": "0-100"},
            {"name": "passed", "type": "boolean"},
            {"name": "label", "type": "enum", "values": ["excellent", "good", "fair", "poor"]},
            {"name": "reasoning", "type": "text"},
        ],
    },
}


def render_setup_wizard() -> None:
    """Render the setup wizard for creating new analysis configurations."""
    # Initialize session state
    if "wizard_config" not in st.session_state:
        st.session_state.wizard_config = {
            "dataset_path": None,
            "dataset_name": None,
            "sample_count": None,
            "analyzers": [],
            "tests": [],
            "custom_metrics": [],
        }

    # Check what's configured for status indicators
    config = st.session_state.wizard_config
    has_dataset = bool(config.get("dataset_path") or config.get("dataset_name"))
    has_analyzers = len(config.get("analyzers", [])) > 0
    has_tests = len(config.get("tests", [])) > 0

    # Summary bar at top
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        dataset_name = (
            config.get("dataset_name") or
            (config.get("dataset_path", "") or "").split("/")[-1] or
            "Not set"
        )
        status = "✅" if has_dataset else "⚪"
        st.metric(f"{status} Dataset", dataset_name[:20] + "..." if len(dataset_name) > 20 else dataset_name)
    with col2:
        status = "✅" if has_analyzers else "⚪"
        st.metric(f"{status} Analyzers", len(config.get("analyzers", [])))
    with col3:
        status = "✅" if has_tests else "⚪"
        st.metric(f"{status} Tests", len(config.get("tests", [])))
    with col4:
        samples = config.get("sample_count", 100)
        st.metric("Samples", samples)

    st.divider()

    # Accordion-style sections - all accessible anytime
    with st.expander("📁 **1. Dataset**", expanded=not has_dataset):
        _render_dataset_section()

    with st.expander("🔍 **2. Analyzers**", expanded=has_dataset and not has_analyzers):
        _render_analyzers_section()

    with st.expander("✅ **3. Tests**", expanded=has_analyzers and not has_tests):
        _render_tests_section()

    with st.expander("🚀 **4. Generate & Run**", expanded=has_dataset and has_analyzers):
        _render_generate_section()


def _render_dataset_section() -> None:
    """Render the dataset configuration section."""
    st.markdown("Upload a JSONL file or specify a HuggingFace dataset.")

    # Tabs for different input methods
    tab1, tab2 = st.tabs(["📁 Upload File", "🤗 HuggingFace Dataset"])

    with tab1:
        uploaded_file = st.file_uploader(
            "Upload JSONL file",
            type=["jsonl", "json"],
            help="Upload a file with conversations in JSONL or JSON format",
        )

        if uploaded_file is not None:
            # Save to temp file and preview
            try:
                content = uploaded_file.read().decode("utf-8")
                lines = content.strip().split("\n")

                # Parse and preview
                samples = []
                for line in lines[:5]:  # Preview first 5
                    try:
                        samples.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

                if samples:
                    st.success(f"✅ Loaded {len(lines)} conversations")

                    # Save file path
                    temp_dir = Path(tempfile.gettempdir()) / "oumi_analyze"
                    temp_dir.mkdir(exist_ok=True)
                    temp_path = temp_dir / uploaded_file.name
                    with open(temp_path, "w") as f:
                        f.write(content)
                    st.session_state.wizard_config["dataset_path"] = str(temp_path)
                    st.session_state.wizard_config["dataset_name"] = None

                    # Preview
                    with st.expander("Preview first conversation"):
                        st.json(samples[0])
                else:
                    st.error("Could not parse the file. Please check the format.")

            except Exception as e:
                st.error(f"Error reading file: {e}")

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            dataset_name = st.text_input(
                "Dataset name",
                placeholder="HuggingFaceH4/ultrachat_200k",
                help="HuggingFace dataset name (e.g., HuggingFaceH4/ultrachat_200k uses split 'train_sft')",
            )
        with col2:
            split = st.text_input(
                "Split",
                value="train",
                help="Dataset split (e.g., train, test, train_sft, validation)",
            )

        subset = st.text_input(
            "Subset (optional)",
            placeholder="default",
            help="Dataset subset/configuration",
        )

        if dataset_name:
            st.session_state.wizard_config["dataset_name"] = dataset_name
            st.session_state.wizard_config["split"] = split
            st.session_state.wizard_config["subset"] = subset or None
            st.session_state.wizard_config["dataset_path"] = None
            st.success(f"✅ Will use dataset: {dataset_name}")

    # Sample count
    st.markdown("---")
    sample_count = st.number_input(
        "Number of samples to analyze",
        min_value=1,
        max_value=10000,
        value=st.session_state.wizard_config.get("sample_count") or 100,
        help="Limit the number of conversations to analyze (useful for testing)",
    )
    st.session_state.wizard_config["sample_count"] = sample_count


def _render_analyzers_section() -> None:
    """Render the analyzer selection section."""
    st.markdown(
        "Select the analyzers to run on your dataset. "
        "Each analyzer produces metrics that can be used in tests."
    )

    # Group analyzers by category
    categories = {}
    for analyzer_id, analyzer in AVAILABLE_ANALYZERS.items():
        cat = analyzer["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append((analyzer_id, analyzer))

    # Selected analyzers - use list copy to avoid reference issues
    selected = list(st.session_state.wizard_config.get("analyzers", []))
    selected_ids = [a["id"] for a in selected]

    # Initialize checkbox keys from session state if not present
    for analyzer_id in AVAILABLE_ANALYZERS.keys():
        key = f"analyzer_{analyzer_id}"
        if key not in st.session_state:
            st.session_state[key] = analyzer_id in selected_ids

    # Render by category
    for category, analyzers in categories.items():
        st.markdown(f"**{category}**")

        for analyzer_id, analyzer in analyzers:
            col1, col2 = st.columns([3, 1])

            with col1:
                # Use the key's current value as the source of truth
                checkbox_key = f"analyzer_{analyzer_id}"
                is_checked = st.checkbox(
                    f"**{analyzer['name']}**",
                    value=st.session_state.get(checkbox_key, analyzer_id in selected_ids),
                    key=checkbox_key,
                    help=analyzer["description"],
                )

                if is_checked and analyzer_id not in selected_ids:
                    # Add analyzer
                    new_analyzer = {
                        "id": analyzer_id,
                        "instance_id": analyzer_id,
                        "params": {},
                    }
                    selected.append(new_analyzer)
                    selected_ids.append(analyzer_id)
                elif not is_checked and analyzer_id in selected_ids:
                    # Remove analyzer
                    selected = [a for a in selected if a["id"] != analyzer_id]
                    selected_ids = [a["id"] for a in selected]

            with col2:
                st.caption(analyzer["description"][:50] + "...")

    # Save selected analyzers AFTER processing all categories
    st.session_state.wizard_config["analyzers"] = selected

    # Configure selected analyzers
    if selected:
        st.divider()
        st.markdown("### Configure Selected Analyzers")

        for i, analyzer_config in enumerate(selected):
            analyzer_id = analyzer_config["id"]
            analyzer_info = AVAILABLE_ANALYZERS.get(analyzer_id, {})

            with st.expander(f"⚙️ {analyzer_info.get('name', analyzer_id)}", expanded=False):
                # Instance ID (for multiple instances of same analyzer)
                instance_id = st.text_input(
                    "Instance ID",
                    value=analyzer_config.get("instance_id", analyzer_id),
                    key=f"instance_{i}",
                    help="Unique identifier (useful if using same analyzer multiple times)",
                )
                selected[i]["instance_id"] = instance_id

                # Parameters
                params = analyzer_info.get("params", {})
                if params:
                    st.markdown("**Parameters:**")
                    for param_name, param_config in params.items():
                        # Check conditional visibility
                        if "conditional" in param_config:
                            cond_field = list(param_config["conditional"].keys())[0]
                            cond_value = param_config["conditional"][cond_field]
                            current_value = selected[i].get("params", {}).get(cond_field)
                            if current_value != cond_value:
                                continue

                        param_type = param_config.get("type", "text")
                        default = param_config.get("default", "")

                        if param_type == "select":
                            value = st.selectbox(
                                param_name,
                                param_config["options"],
                                index=param_config["options"].index(default)
                                if default in param_config["options"]
                                else 0,
                                key=f"param_{i}_{param_name}",
                                help=param_config.get("description"),
                            )
                        elif param_type == "number":
                            value = st.number_input(
                                param_name,
                                value=int(default) if default else 0,
                                key=f"param_{i}_{param_name}",
                                help=param_config.get("description"),
                            )
                        elif param_type == "checkbox":
                            value = st.checkbox(
                                param_name,
                                value=bool(default) if default else False,
                                key=f"param_{i}_{param_name}",
                                help=param_config.get("description"),
                            )
                        elif param_type == "textarea":
                            value = st.text_area(
                                param_name,
                                value=default if default else "",
                                key=f"param_{i}_{param_name}",
                                help=param_config.get("description"),
                                height=100,
                            )
                        else:
                            value = st.text_input(
                                param_name,
                                value=str(default) if default else "",
                                key=f"param_{i}_{param_name}",
                                help=param_config.get("description"),
                            )

                        if "params" not in selected[i]:
                            selected[i]["params"] = {}
                        selected[i]["params"][param_name] = value

                # Show available metrics
                metrics = analyzer_info.get("metrics", [])
                if metrics:
                    st.markdown("**Available Metrics:**")
                    for metric in metrics:
                        metric_desc = f"`{instance_id}.{metric['name']}` ({metric['type']})"
                        if "range" in metric:
                            metric_desc += f" - Range: {metric['range']}"
                        if "values" in metric:
                            metric_desc += f" - Values: {metric['values']}"
                        st.caption(metric_desc)

        # Save after parameter configuration
        st.session_state.wizard_config["analyzers"] = selected

    # Custom metrics info
    st.divider()
    with st.expander("💡 Custom Metrics", expanded=False):
        st.markdown("""
        You can also create **custom metrics** that combine or transform analyzer outputs.
        
        Example custom metric in YAML:
        ```yaml
        custom_metrics:
          - id: response_quality_score
            compute: |
              base_score = get('llm_quality.score', 0)
              length_penalty = min(get('length.total_length', 0) / 1000, 20)
              return base_score - length_penalty
            depends_on:
              - llm_quality
              - length
        ```
        
        Custom metrics can be added manually to the generated YAML config.
        """)


def _render_tests_section() -> None:
    """Render the test configuration section."""
    st.markdown(
        "Define tests to validate your analysis results. "
        "Tests check if metrics meet certain conditions."
    )

    # Get available metrics from selected analyzers
    available_metrics = _get_available_metrics()

    # Current tests
    tests = st.session_state.wizard_config.get("tests", [])

    # Add new test
    st.markdown("### Add Test")

    # Test type selection OUTSIDE the form so it triggers re-render
    if "test_type_selection" not in st.session_state:
        st.session_state.test_type_selection = "percentage"

    col_type1, col_type2 = st.columns(2)
    with col_type1:
        test_type = st.selectbox(
            "Test Type",
            ["percentage", "threshold", "range"],
            index=["percentage", "threshold", "range"].index(
                st.session_state.test_type_selection
            ),
            help="Type of test to run",
            key="test_type_select",
        )
        # Update session state if changed
        if test_type != st.session_state.test_type_selection:
            st.session_state.test_type_selection = test_type
            st.rerun()

    # Show description based on test type
    if test_type == "percentage":
        st.caption("📊 *Check what % of samples meet a condition (best for boolean metrics like has_empty_turns)*")
    elif test_type == "threshold":
        st.caption("📏 *Check samples against a value (works with numbers OR booleans)*")
    elif test_type == "range":
        st.caption("📐 *Check if numeric values fall within a range*")

    with st.form("add_test_form"):
        col1, col2 = st.columns(2)

        with col1:
            test_id = st.text_input(
                "Test ID",
                placeholder="high_quality_responses",
                help="Unique identifier for this test",
            )
            test_title = st.text_input(
                "Title",
                placeholder="High Quality Responses",
                help="Human-readable title",
            )

        with col2:
            severity = st.selectbox(
                "Severity",
                ["high", "medium", "low"],
                index=1,
                help="Severity level if test fails",
            )

        # Metric selection
        if available_metrics:
            metric = st.selectbox(
                "Metric",
                available_metrics,
                help="Select the metric to test",
            )
        else:
            metric = st.text_input(
                "Metric",
                placeholder="analyzer_name.metric_name",
                help="Enter the metric path",
            )

        # Test-type specific fields based on selection
        if test_type == "percentage":
            col1, col2 = st.columns(2)
            with col1:
                condition = st.text_input(
                    "Condition",
                    placeholder="== True",
                    help="Condition to check (e.g., '== True', '> 50', '!= None')",
                )
            with col2:
                pct_mode = st.selectbox(
                    "Requirement",
                    ["At least (min)", "At most (max)"],
                    help="Whether you need minimum or maximum percentage",
                )

            pct_value = st.number_input(
                "Percentage threshold",
                min_value=0.0,
                max_value=100.0,
                value=80.0,
                help="The percentage threshold",
            )

        elif test_type == "threshold":
            col1, col2, col3 = st.columns([1, 1, 1])
            with col1:
                operator = st.selectbox(
                    "Operator",
                    [">=", ">", "<=", "<", "==", "!="],
                    help="Comparison operator",
                )
            with col2:
                value_type = st.selectbox(
                    "Value Type",
                    ["Number", "Boolean"],
                    help="Type of value to compare",
                )
            with col3:
                if value_type == "Boolean":
                    threshold_value = st.selectbox(
                        "Value",
                        [True, False],
                        help="Boolean value to compare",
                    )
                else:
                    threshold_value = st.number_input(
                        "Value",
                        value=50.0,
                        help="Numeric value to compare",
                    )

            st.markdown("**Percentage requirement** *(optional)*")
            col3, col4 = st.columns(2)
            with col3:
                threshold_pct_mode = st.selectbox(
                    "Requirement type",
                    ["None (all must pass)", "At most (max %)", "At least (min %)"],
                    help="Leave as 'None' if ALL samples must pass",
                )
            with col4:
                threshold_pct_value = st.number_input(
                    "Percentage",
                    min_value=0.0,
                    max_value=100.0,
                    value=15.0,
                    help="Percentage threshold (ignored if 'None' selected)",
                )

        elif test_type == "range":
            col1, col2 = st.columns(2)
            with col1:
                min_value = st.number_input("Min Value", value=0.0)
            with col2:
                max_value = st.number_input("Max Value", value=100.0)

            range_max_pct = st.number_input(
                "Max % outside range",
                min_value=0.0,
                max_value=100.0,
                value=0.0,
                help="Maximum % of samples allowed outside the range (0 = none allowed)",
            )

        description = st.text_area(
            "Description",
            placeholder="Describe what this test checks...",
            help="Optional description of the test",
        )

        if st.form_submit_button("Add Test", type="primary"):
            if test_id and metric:
                new_test = {
                    "id": test_id,
                    "type": test_type,
                    "metric": metric,
                    "severity": severity,
                    "title": test_title or test_id,
                    "description": description,
                }

                if test_type == "percentage":
                    new_test["condition"] = condition
                    if pct_mode == "At least (min)":
                        new_test["min_percentage"] = pct_value
                    else:
                        new_test["max_percentage"] = pct_value

                elif test_type == "threshold":
                    new_test["operator"] = operator
                    new_test["value"] = threshold_value
                    # Add percentage requirement if specified
                    if threshold_pct_mode == "At most (max %)":
                        new_test["max_percentage"] = threshold_pct_value
                    elif threshold_pct_mode == "At least (min %)":
                        new_test["min_percentage"] = threshold_pct_value

                elif test_type == "range":
                    new_test["min_value"] = min_value
                    new_test["max_value"] = max_value
                    if range_max_pct and range_max_pct > 0:
                        new_test["max_percentage"] = range_max_pct

                tests.append(new_test)
                st.session_state.wizard_config["tests"] = tests
                st.success(f"Added test: {test_id}")
                st.rerun()
            else:
                st.error("Please provide Test ID and Metric")

    # Show current tests
    if tests:
        st.divider()
        st.markdown("### Current Tests")

        for i, test in enumerate(tests):
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                st.markdown(f"**{test.get('title', test['id'])}**")
                st.caption(f"`{test['metric']}` | Type: {test['type']} | Severity: {test['severity']}")
            with col2:
                if test["type"] == "percentage":
                    st.caption(f"Min: {test.get('min_percentage', 0)}%")
                elif test["type"] == "threshold":
                    st.caption(f"{test.get('operator', '')} {test.get('value', '')}")
            with col3:
                if st.button("🗑️", key=f"delete_test_{i}"):
                    tests.pop(i)
                    st.session_state.wizard_config["tests"] = tests
                    st.rerun()

    # Info about tests
    st.divider()
    with st.expander("💡 Test Types Explained", expanded=False):
        st.markdown("""
        **Percentage Test**: Check what % of samples meet a condition
        - Best for: Boolean metrics (has_empty_turns, passes_basic_quality)
        - Example: "At least 95% should NOT have empty turns"
        - Config: `metric: quality.has_empty_turns`, `condition: == False`, `min_percentage: 95`
        
        **Threshold Test**: Check samples against a value (number OR boolean)
        - Works with: Numeric metrics (total_tokens, score) OR boolean metrics
        - Example 1 (numeric): "No more than 5% should exceed 8000 tokens"
        - Config: `metric: length.total_tokens`, `operator: >`, `value: 8000`, `max_percentage: 5`
        
        - Example 2 (boolean): "No more than 5% should have refusals"
        - Config: `metric: quality.has_policy_refusal`, `operator: ==`, `value: True`, `max_percentage: 5`
        
        **Range Test**: Check if numeric values fall within a range
        - Example: "Response length should be between 50-500 words"
        - Config: `metric: length.total_words`, `min_value: 50`, `max_value: 500`
        
        ---
        
        **Percentage Options:**
        - `min_percentage`: At least X% must match (fail if fewer match)
        - `max_percentage`: At most X% can match (fail if more match)
        """)


def _render_generate_section() -> None:
    """Render the config generation and run section."""

    # Check if any analyzers are configured
    wizard_config = st.session_state.wizard_config
    if not wizard_config.get("analyzers"):
        st.warning(
            "⚠️ No analyzers selected. Go to the **Analyzers** section above to add some."
        )

    # Generate YAML config
    config = _generate_yaml_config()
    yaml_str = yaml.dump(config, default_flow_style=False, sort_keys=False)

    # Summary
    st.markdown("### Configuration Summary")
    col1, col2, col3 = st.columns(3)
    with col1:
        dataset_name = st.session_state.wizard_config.get("dataset_name")
        dataset_path = st.session_state.wizard_config.get("dataset_path") or ""
        dataset = dataset_name or dataset_path.split("/")[-1] or "Not set"
        st.metric("Dataset", dataset[:30] + "..." if len(dataset) > 30 else dataset)
    with col2:
        st.metric("Analyzers", len(st.session_state.wizard_config.get("analyzers", [])))
    with col3:
        st.metric("Tests", len(st.session_state.wizard_config.get("tests", [])))

    # YAML editor
    st.divider()
    st.markdown("### Generated YAML Configuration")
    st.caption("Review and edit the configuration before running.")

    # Use ace editor if available for syntax highlighting
    try:
        from streamlit_ace import st_ace
        edited_yaml = st_ace(
            value=yaml_str,
            language="yaml",
            theme="monokai",
            height=400,
            key="config_yaml_editor",
            font_size=14,
            tab_size=2,
            show_gutter=True,
            show_print_margin=False,
            wrap=True,
            auto_update=True,
        )
    except ImportError:
        edited_yaml = st.text_area(
            "Configuration",
            value=yaml_str,
            height=400,
            key="config_yaml_editor",
            label_visibility="collapsed",
        )

    # Actions
    st.divider()
    col1, col2, col3 = st.columns(3)

    with col1:
        # Download YAML
        st.download_button(
            "📥 Download YAML",
            data=edited_yaml,
            file_name="analysis_config.yaml",
            mime="text/yaml",
            use_container_width=True,
        )

    with col2:
        # Save to configs
        config_name = st.text_input(
            "Config name",
            value="my_analysis",
            key="save_config_name",
            label_visibility="collapsed",
            placeholder="Config name",
        )

    with col3:
        if st.button("💾 Save Config", use_container_width=True):
            try:
                from oumi.analyze.storage import AnalyzeStorage

                storage = AnalyzeStorage()
                parsed_config = yaml.safe_load(edited_yaml)
                path = storage.save_config(config_name, parsed_config)
                st.success(f"Saved to: {path}")
            except Exception as e:
                st.error(f"Error saving: {e}")

    # Run analysis
    st.divider()
    st.markdown("### Run Analysis")

    if st.button("🚀 Run Analysis", type="primary", use_container_width=True):
        _run_analysis(edited_yaml)


def _get_available_metrics() -> list[str]:
    """Get list of available metrics from selected analyzers."""
    metrics = []
    analyzers = st.session_state.wizard_config.get("analyzers", [])

    for analyzer in analyzers:
        analyzer_id = analyzer.get("id")
        instance_id = analyzer.get("instance_id", analyzer_id)
        analyzer_info = AVAILABLE_ANALYZERS.get(analyzer_id, {})

        for metric in analyzer_info.get("metrics", []):
            metrics.append(f"{instance_id}.{metric['name']}")

    return metrics


def _generate_yaml_config() -> dict[str, Any]:
    """Generate YAML configuration from wizard state."""
    config = st.session_state.wizard_config

    yaml_config: dict[str, Any] = {}

    # Dataset
    if config.get("dataset_path"):
        yaml_config["dataset_path"] = config["dataset_path"]
    elif config.get("dataset_name"):
        yaml_config["dataset_name"] = config["dataset_name"]
        if config.get("split"):
            yaml_config["split"] = config["split"]
        if config.get("subset"):
            yaml_config["subset"] = config["subset"]

    if config.get("sample_count"):
        yaml_config["sample_count"] = config["sample_count"]

    # Output
    yaml_config["output_path"] = "./analysis_output"

    # Analyzers
    yaml_config["analyzers"] = []
    for analyzer in config.get("analyzers", []):
        analyzer_config = {
            "id": analyzer["id"],
        }
        if analyzer.get("instance_id") and analyzer["instance_id"] != analyzer["id"]:
            analyzer_config["instance_id"] = analyzer["instance_id"]
        if analyzer.get("params"):
            # Filter out None and empty string params (but keep False and 0)
            params = {k: v for k, v in analyzer["params"].items() if v is not None and v != ""}
            if params:
                analyzer_config["params"] = params
        yaml_config["analyzers"].append(analyzer_config)

    # Tests
    if config.get("tests"):
        yaml_config["tests"] = config["tests"]

    # Custom metrics placeholder
    if config.get("custom_metrics"):
        yaml_config["custom_metrics"] = config["custom_metrics"]

    return yaml_config


def _run_analysis(yaml_str: str) -> None:
    """Run the analysis with the given YAML configuration."""
    import subprocess
    import tempfile

    try:
        # Parse and validate YAML
        config = yaml.safe_load(yaml_str)

        # Save to temp file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(config, f)
            config_path = f.name

        # Run analysis
        st.info("Running analysis... This may take a while.")

        with st.spinner("Analyzing..."):
            result = subprocess.run(
                ["oumi", "analyze", "--config", config_path, "--typed"],
                capture_output=True,
                text=True,
            )

        if result.returncode == 0:
            st.success("✅ Analysis complete!")
            st.caption("View results in the 'Results' tab or refresh this page.")

            # Show output
            if result.stdout:
                with st.expander("Output", expanded=True):
                    st.code(result.stdout)
        else:
            st.error("❌ Analysis failed")
            if result.stderr:
                st.code(result.stderr)
            if result.stdout:
                st.code(result.stdout)

    except yaml.YAMLError as e:
        st.error(f"Invalid YAML: {e}")
    except Exception as e:
        st.error(f"Error: {e}")
