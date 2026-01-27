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
from oumi.analyze.ui.sidebar import render_sidebar


def main():
    """Main entry point for the Streamlit app."""
    # Initialize storage
    storage = AnalyzeStorage()

    # Render sidebar and get selected eval
    eval_data = render_sidebar(storage)

    # Main content area
    if eval_data is None:
        # No eval selected - show welcome/empty state
        st.title("Oumi Analyze Viewer")
        st.markdown(
            """
            Welcome to the **Oumi Analyze Viewer**!

            This tool helps you:
            - 📊 Browse and compare analysis runs
            - 🔍 Filter and search results
            - 📈 Visualize score distributions and pass rates
            - ✏️ Create and edit analysis configurations
            - 📥 Export results in various formats

            ### Getting Started

            1. Run an analysis first:
               ```bash
               oumi analyze --config your_config.yaml --typed
               ```

            2. Refresh this page to see your results

            Or use the **Config Editor** tab to create a new configuration.
            """
        )

        # Still show config editor for creating new configs
        st.divider()
        tab1, = st.tabs(["Config Editor"])
        with tab1:
            render_config_editor(storage, None)
        return

    # Show eval name as title
    st.title(f"📊 {eval_data.metadata.name}")

    # Create tabs
    tab_results, tab_charts, tab_config, tab_export = st.tabs([
        "📋 Results",
        "📈 Charts",
        "✏️ Config Editor",
        "📥 Export",
    ])

    with tab_results:
        render_results_table(eval_data)

    with tab_charts:
        render_charts(eval_data)

    with tab_config:
        render_config_editor(storage, eval_data)

    with tab_export:
        render_exports(eval_data)


if __name__ == "__main__":
    main()
