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

"""Sidebar component for the Analyze web viewer."""

from datetime import datetime

import streamlit as st

from oumi.analyze.storage import AnalyzeStorage, EvalData, EvalMetadata


def render_sidebar(storage: AnalyzeStorage) -> EvalData | None:
    """Render the sidebar with eval selector and actions.

    Args:
        storage: The AnalyzeStorage instance.

    Returns:
        The selected EvalData or None if no eval is selected.
    """
    st.sidebar.title("Oumi Analyze")

    # Get list of evals
    evals = storage.list_evals()

    if not evals:
        st.sidebar.info(
            "No analysis runs found.\n\n"
            "Run an analysis first:\n"
            "```\n"
            "oumi analyze --config your_config.yaml --typed\n"
            "```"
        )
        return None

    # Eval selector
    st.sidebar.subheader("Select Analysis Run")

    # Format options for display
    options = []
    for e in evals:
        created = _format_timestamp(e.created_at)
        pass_info = ""
        if e.pass_rate is not None:
            pass_pct = e.pass_rate * 100
            pass_info = f" | {pass_pct:.0f}% pass"
        options.append(f"{e.name} ({created}{pass_info})")

    selected_idx = st.sidebar.selectbox(
        "Analysis Run",
        range(len(options)),
        format_func=lambda i: options[i],
        key="eval_selector",
    )

    if selected_idx is None:
        return None

    selected_meta = evals[selected_idx]
    eval_data = storage.load_eval(selected_meta.id)

    if eval_data is None:
        st.sidebar.error(f"Failed to load eval: {selected_meta.id}")
        return None

    # Quick stats
    st.sidebar.divider()
    st.sidebar.subheader("Quick Stats")

    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric("Samples", selected_meta.sample_count)
    with col2:
        st.metric("Analyzers", selected_meta.analyzer_count)

    col3, col4 = st.sidebar.columns(2)
    with col3:
        if selected_meta.pass_rate is not None:
            st.metric("Pass Rate", f"{selected_meta.pass_rate * 100:.1f}%")
        else:
            st.metric("Pass Rate", "N/A")
    with col4:
        st.metric("Tests", f"{selected_meta.tests_passed}/{selected_meta.test_count}")

    # Actions
    st.sidebar.divider()
    st.sidebar.subheader("Actions")

    col_a, col_b = st.sidebar.columns(2)
    with col_a:
        if st.button("Rename", key="rename_btn", use_container_width=True):
            st.session_state.show_rename_dialog = True

    with col_b:
        if st.button("Delete", key="delete_btn", use_container_width=True, type="secondary"):
            st.session_state.show_delete_dialog = True

    # Handle rename dialog
    if st.session_state.get("show_rename_dialog", False):
        with st.sidebar.form("rename_form"):
            new_name = st.text_input("New name", value=selected_meta.name)
            col_ok, col_cancel = st.columns(2)
            with col_ok:
                if st.form_submit_button("Save"):
                    storage.rename_eval(selected_meta.id, new_name)
                    st.session_state.show_rename_dialog = False
                    st.rerun()
            with col_cancel:
                if st.form_submit_button("Cancel"):
                    st.session_state.show_rename_dialog = False
                    st.rerun()

    # Handle delete dialog
    if st.session_state.get("show_delete_dialog", False):
        st.sidebar.warning(f"Delete '{selected_meta.name}'?")
        col_yes, col_no = st.sidebar.columns(2)
        with col_yes:
            if st.button("Yes, delete", type="primary"):
                storage.delete_eval(selected_meta.id)
                st.session_state.show_delete_dialog = False
                st.rerun()
        with col_no:
            if st.button("Cancel"):
                st.session_state.show_delete_dialog = False
                st.rerun()

    # Info section
    st.sidebar.divider()
    st.sidebar.caption(f"**ID:** {selected_meta.id}")
    st.sidebar.caption(f"**Created:** {selected_meta.created_at}")
    if selected_meta.config_path:
        st.sidebar.caption(f"**Config:** {selected_meta.config_path}")

    return eval_data


def _format_timestamp(iso_timestamp: str) -> str:
    """Format an ISO timestamp for display."""
    try:
        dt = datetime.fromisoformat(iso_timestamp)
        now = datetime.now()
        delta = now - dt

        if delta.days == 0:
            if delta.seconds < 3600:
                minutes = delta.seconds // 60
                return f"{minutes}m ago"
            hours = delta.seconds // 3600
            return f"{hours}h ago"
        elif delta.days == 1:
            return "yesterday"
        elif delta.days < 7:
            return f"{delta.days}d ago"
        else:
            return dt.strftime("%b %d")
    except (ValueError, TypeError):
        return iso_timestamp[:10] if len(iso_timestamp) >= 10 else iso_timestamp
