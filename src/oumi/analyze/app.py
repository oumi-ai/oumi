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


def main():
    """Main entry point for the Streamlit app."""
    # Initialize storage
    storage = AnalyzeStorage()

    # Render sidebar and get selected eval
    eval_data = render_sidebar(storage)

    # Main content area
    if eval_data is None:
        # No eval selected - show welcome with setup wizard
        st.title("Oumi Analyze")

        # Create tabs for new users
        tab_wizard, tab_config = st.tabs([
            "🚀 Create New Analysis",
            "✏️ YAML Editor",
        ])

        with tab_wizard:
            render_setup_wizard()

        with tab_config:
            st.markdown("### Manual Configuration")
            st.caption("For advanced users who prefer editing YAML directly.")
            render_config_editor(storage, None)

        return

    # Show eval name as title
    st.title(f"📊 {eval_data.metadata.name}")

    # Create tabs
    tab_results, tab_charts, tab_new, tab_config, tab_export = st.tabs([
        "📋 Results",
        "📈 Charts",
        "🚀 New Analysis",
        "✏️ Config Editor",
        "📥 Export",
    ])

    with tab_results:
        render_results_table(eval_data)

    with tab_charts:
        render_charts(eval_data)

    with tab_new:
        render_setup_wizard()

    with tab_config:
        render_config_editor(storage, eval_data)

    with tab_export:
        render_exports(eval_data)


if __name__ == "__main__":
    main()
