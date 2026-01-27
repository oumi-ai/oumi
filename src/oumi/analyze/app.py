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

"""Main Streamlit app for the Analyze web viewer.

This app provides a web interface for viewing and managing analysis results.

Features:
- Browse and compare past analysis runs
- Interactive results table with filters
- Charts and visualizations
- Config editor to create/edit analysis configs
- Export options (CSV, JSON, YAML)

Usage:
    oumi analyze view
    # or
    streamlit run src/oumi/analyze/app.py
"""

import streamlit as st

# Page config must be first Streamlit command
st.set_page_config(
    page_title="Oumi Analyze",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

from oumi.analyze.storage import AnalyzeStorage
from oumi.analyze.ui.charts import render_charts
from oumi.analyze.ui.config_editor import render_config_editor
from oumi.analyze.ui.exports import render_exports
from oumi.analyze.ui.results import render_results_table
from oumi.analyze.ui.setup_wizard import render_setup_wizard
from oumi.analyze.ui.sidebar import render_sidebar


def _render_editor_with_toggle(
    storage: AnalyzeStorage,
    eval_data,
    key_prefix: str = "main",
) -> None:
    """Render editor with UI/YAML toggle.

    Args:
        storage: Storage instance.
        eval_data: Current eval data (can be None).
        key_prefix: Prefix for session state keys.
    """
    # Initialize toggle state
    toggle_key = f"{key_prefix}_editor_mode"
    if toggle_key not in st.session_state:
        st.session_state[toggle_key] = "ui"  # Default to UI editor

    # Toggle switch
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        mode = st.radio(
            "Editor Mode",
            ["UI Editor", "YAML Editor"],
            index=0 if st.session_state[toggle_key] == "ui" else 1,
            horizontal=True,
            key=f"{key_prefix}_mode_radio",
            label_visibility="collapsed",
        )
        st.session_state[toggle_key] = "ui" if mode == "UI Editor" else "yaml"

    st.divider()

    # Render based on mode
    if st.session_state[toggle_key] == "ui":
        render_setup_wizard()
    else:
        render_config_editor(storage, eval_data)


def main():
    """Main entry point for the Streamlit app."""
    # Initialize storage
    storage = AnalyzeStorage()

    # Render sidebar and get selected eval
    eval_data = render_sidebar(storage)

    # Main content area
    if eval_data is None:
        # No eval selected - show welcome with editor
        st.title("Oumi Analyze")
        st.markdown("Create a new analysis configuration to get started.")

        _render_editor_with_toggle(storage, None, key_prefix="welcome")

        return

    # Show eval name as title
    st.title(f"📊 {eval_data.metadata.name}")

    # Create tabs
    tab_results, tab_charts, tab_config, tab_export = st.tabs([
        "📋 Results",
        "📈 Charts",
        "✏️ Config",
        "📥 Export",
    ])

    with tab_results:
        render_results_table(eval_data)

    with tab_charts:
        render_charts(eval_data)

    with tab_config:
        _render_editor_with_toggle(storage, eval_data, key_prefix="config")

    with tab_export:
        render_exports(eval_data)


if __name__ == "__main__":
    main()
