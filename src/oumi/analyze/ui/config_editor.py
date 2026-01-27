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

"""Config editor component for the Analyze web viewer."""

import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import streamlit as st
import yaml

from oumi.analyze.storage import AnalyzeStorage, EvalData

# Try to import streamlit-ace for syntax highlighting
try:
    from streamlit_ace import st_ace
    HAS_ACE_EDITOR = True
except ImportError:
    HAS_ACE_EDITOR = False


# Example config template
EXAMPLE_CONFIG = '''# Analysis Configuration
# See: https://docs.oumi.ai/analyze for full documentation

# Dataset to analyze
dataset_path: "path/to/your/dataset.jsonl"
# Or use a HuggingFace dataset:
# dataset_name: "HuggingFaceH4/ultrachat_200k"
# split: "train"

# Number of samples to analyze (optional, for testing)
sample_count: 100

# Output directory
output_path: "./analysis_output/my_analysis"

# Analyzers to run
analyzers:
  # Length analyzer - basic statistics
  - id: length

  # LLM-based quality analysis
  - id: llm
    params:
      criteria: usefulness
      model_name: gpt-4o-mini
      num_workers: 4

# Tests to validate results
tests:
  - id: min_quality
    type: threshold
    metric: LLMAnalyzer.score
    operator: ">="
    value: 50
    min_percentage: 80.0
    severity: medium
    title: "Quality threshold"
    description: "At least 80% of samples should have quality >= 50"
'''


def render_config_editor(storage: AnalyzeStorage, eval_data: EvalData | None) -> None:
    """Render the config editor.

    Args:
        storage: The storage instance.
        eval_data: Currently selected eval data (if any).
    """
    st.header("Config Editor")

    # Initialize session state for config
    if "config_yaml" not in st.session_state:
        if eval_data and eval_data.config:
            # Load from current eval
            st.session_state.config_yaml = yaml.dump(
                eval_data.config, default_flow_style=False, sort_keys=False
            )
        else:
            st.session_state.config_yaml = EXAMPLE_CONFIG

    # Template selector
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("Edit your analysis configuration below.")
    with col2:
        if st.button("Load Template", key="load_template"):
            st.session_state.config_yaml = EXAMPLE_CONFIG
            st.rerun()

    # Load from current eval button
    if eval_data and eval_data.config:
        if st.button("Load from Current Eval", key="load_from_eval"):
            st.session_state.config_yaml = yaml.dump(
                eval_data.config, default_flow_style=False, sort_keys=False
            )
            st.rerun()

    # Config editor with syntax highlighting
    if HAS_ACE_EDITOR:
        config_yaml = st_ace(
            value=st.session_state.config_yaml,
            language="yaml",
            theme="monokai",  # Dark theme like Promptfoo
            height=450,
            key="config_ace_editor",
            font_size=14,
            tab_size=2,
            show_gutter=True,
            show_print_margin=False,
            wrap=True,
            auto_update=True,
        )
    else:
        # Fallback to regular text area with some styling
        st.markdown(
            """
            <style>
            .stTextArea textarea {
                font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
                font-size: 14px;
                background-color: #1e1e1e;
                color: #d4d4d4;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        config_yaml = st.text_area(
            "Configuration (YAML)",
            value=st.session_state.config_yaml,
            height=400,
            key="config_editor_area",
            help="Install 'streamlit-ace' for syntax highlighting: pip install streamlit-ace",
        )

    # Update session state
    if config_yaml:
        st.session_state.config_yaml = config_yaml

    # Validation
    config_valid, config_dict, error_msg = _validate_config(config_yaml)

    if not config_valid:
        st.error(f"Invalid YAML: {error_msg}")
    else:
        st.success("Configuration is valid")

        # Show parsed config preview
        with st.expander("Parsed Configuration", expanded=False):
            st.json(config_dict)

    # Action buttons
    st.divider()

    col_run, col_save, col_download = st.columns(3)

    with col_run:
        run_disabled = not config_valid
        if st.button(
            "Run Analysis",
            key="run_analysis",
            type="primary",
            disabled=run_disabled,
            use_container_width=True,
        ):
            if config_dict:
                _run_analysis(config_dict, storage)

    with col_save:
        save_disabled = not config_valid
        if st.button(
            "Save Config",
            key="save_config",
            disabled=save_disabled,
            use_container_width=True,
        ):
            st.session_state.show_save_dialog = True

    with col_download:
        st.download_button(
            "Download YAML",
            data=config_yaml,
            file_name="analyze_config.yaml",
            mime="text/yaml",
            use_container_width=True,
        )

    # Save dialog
    if st.session_state.get("show_save_dialog", False):
        with st.form("save_config_form"):
            config_name = st.text_input(
                "Config name",
                value="my_config",
                help="Name for the saved configuration file",
            )
            col_ok, col_cancel = st.columns(2)
            with col_ok:
                if st.form_submit_button("Save"):
                    if config_dict:
                        path = storage.save_config(config_name, config_dict)
                        st.success(f"Saved to: {path}")
                        st.session_state.show_save_dialog = False
            with col_cancel:
                if st.form_submit_button("Cancel"):
                    st.session_state.show_save_dialog = False
                    st.rerun()


def _validate_config(yaml_str: str) -> tuple[bool, dict[str, Any] | None, str | None]:
    """Validate YAML configuration.

    Args:
        yaml_str: YAML string to validate.

    Returns:
        Tuple of (is_valid, parsed_dict, error_message).
    """
    try:
        config = yaml.safe_load(yaml_str)
        if not isinstance(config, dict):
            return False, None, "Configuration must be a dictionary"

        # Basic validation
        if not config.get("dataset_path") and not config.get("dataset_name"):
            return False, None, "Must specify either 'dataset_path' or 'dataset_name'"

        return True, config, None
    except yaml.YAMLError as e:
        return False, None, str(e)


def _run_analysis(config: dict[str, Any], storage: AnalyzeStorage) -> None:
    """Run analysis with the given config.

    Args:
        config: Configuration dictionary.
        storage: Storage instance.
    """
    # Write config to temp file
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False
    ) as f:
        yaml.dump(config, f, default_flow_style=False)
        config_path = f.name

    st.info(f"Running analysis with config: {config_path}")

    # Show progress
    with st.spinner("Running analysis..."):
        try:
            # Run oumi analyze command
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "oumi",
                    "analyze",
                    "--config",
                    config_path,
                    "--typed",
                ],
                capture_output=True,
                text=True,
                timeout=600,  # 10 minute timeout
            )

            if result.returncode == 0:
                st.success("Analysis completed successfully!")
                st.code(result.stdout, language="text")
                st.info("Refresh the page to see the new results.")
            else:
                st.error("Analysis failed!")
                st.code(result.stderr, language="text")

        except subprocess.TimeoutExpired:
            st.error("Analysis timed out after 10 minutes.")
        except Exception as e:
            st.error(f"Error running analysis: {e}")
        finally:
            # Clean up temp file
            Path(config_path).unlink(missing_ok=True)
