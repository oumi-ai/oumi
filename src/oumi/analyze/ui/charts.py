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

"""Charts component for the Analyze web viewer."""

from typing import Any

import streamlit as st

from oumi.analyze.storage import EvalData


def render_charts(eval_data: EvalData) -> None:
    """Render charts and visualizations.

    Args:
        eval_data: The eval data to display.
    """
    st.header("Charts & Visualizations")

    results = eval_data.analysis_results
    test_results = eval_data.test_results

    # Test results summary
    if test_results:
        _render_test_summary(test_results)

    # Score distributions
    if results:
        _render_score_distributions(results)

    # Pass rate by analyzer
    if results:
        _render_pass_rate_chart(results)


def _render_test_summary(test_results: dict[str, Any]) -> None:
    """Render test results summary cards.

    Args:
        test_results: Test results dict.
    """
    st.subheader("Test Results")

    tests = test_results.get("tests", [])
    if not tests:
        st.info("No tests configured.")
        return

    # Summary metrics
    passed = sum(1 for t in tests if t.get("passed", False))
    failed = len(tests) - passed
    pass_rate = (passed / len(tests) * 100) if tests else 0

    col1, col2, col3 = st.columns(3)
    col1.metric("Pass Rate", f"{pass_rate:.1f}%")
    col2.metric("Passed", passed, delta=None)
    col3.metric("Failed", failed, delta=None)

    # Test details
    st.divider()

    # Failed tests first
    failed_tests = [t for t in tests if not t.get("passed", False)]
    passed_tests = [t for t in tests if t.get("passed", False)]

    if failed_tests:
        st.markdown("**Failed Tests**")
        for test in failed_tests:
            severity = test.get("severity", "medium")
            title = test.get("title", test.get("id", "Unknown"))
            message = test.get("message", "")

            severity_color = {
                "high": "red",
                "medium": "orange",
                "low": "gray",
            }.get(severity, "gray")

            with st.expander(f":{severity_color}[●] {title}", expanded=False):
                st.write(f"**ID:** {test.get('id', 'N/A')}")
                st.write(f"**Severity:** {severity}")
                if message:
                    st.write(f"**Message:** {message}")
                if test.get("description"):
                    st.write(f"**Description:** {test['description']}")
                if test.get("actual_value") is not None:
                    st.write(f"**Actual:** {test['actual_value']}")
                if test.get("expected_value") is not None:
                    st.write(f"**Expected:** {test['expected_value']}")

    if passed_tests:
        with st.expander(f"Passed Tests ({len(passed_tests)})", expanded=False):
            for test in passed_tests:
                title = test.get("title", test.get("id", "Unknown"))
                st.write(f":green[✓] {title}")


def _render_score_distributions(results: dict[str, Any]) -> None:
    """Render score distribution histograms.

    Args:
        results: Analysis results dict.
    """
    try:
        import plotly.express as px
    except ImportError:
        st.warning("Plotly required for charts. Install with: pip install plotly")
        return

    st.subheader("Score Distributions")

    # Collect scores by analyzer
    analyzer_scores: dict[str, list[float]] = {}

    for analyzer_name, analyzer_results in results.items():
        if not isinstance(analyzer_results, list):
            continue

        scores = []
        for result in analyzer_results:
            if isinstance(result, dict) and "score" in result:
                score = result["score"]
                if isinstance(score, (int, float)):
                    scores.append(float(score))

        if scores:
            analyzer_scores[analyzer_name] = scores

    if not analyzer_scores:
        st.info("No score data available for charts.")
        return

    # Create histogram for each analyzer
    cols = st.columns(min(len(analyzer_scores), 2))
    for i, (analyzer_name, scores) in enumerate(analyzer_scores.items()):
        with cols[i % 2]:
            fig = px.histogram(
                x=scores,
                nbins=20,
                title=f"{analyzer_name}",
                labels={"x": "Score", "count": "Count"},
            )
            fig.update_layout(
                showlegend=False,
                height=300,
                margin=dict(l=20, r=20, t=40, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)


def _render_pass_rate_chart(results: dict[str, Any]) -> None:
    """Render pass rate donut chart.

    Args:
        results: Analysis results dict.
    """
    try:
        import plotly.express as px
    except ImportError:
        return

    st.subheader("Pass Rate by Analyzer")

    # Calculate pass rate per analyzer
    data = []
    for analyzer_name, analyzer_results in results.items():
        if not isinstance(analyzer_results, list):
            continue

        total = 0
        passed = 0
        for result in analyzer_results:
            if isinstance(result, dict) and "passed" in result:
                total += 1
                if result["passed"]:
                    passed += 1

        if total > 0:
            data.append({
                "analyzer": analyzer_name,
                "passed": passed,
                "failed": total - passed,
                "pass_rate": passed / total * 100,
            })

    if not data:
        st.info("No pass/fail data available.")
        return

    # Create bar chart
    import pandas as pd
    df = pd.DataFrame(data)

    fig = px.bar(
        df,
        x="analyzer",
        y="pass_rate",
        title="Pass Rate by Analyzer",
        labels={"pass_rate": "Pass Rate (%)", "analyzer": "Analyzer"},
        color="pass_rate",
        color_continuous_scale="RdYlGn",
        range_color=[0, 100],
    )
    fig.update_layout(
        height=350,
        margin=dict(l=20, r=20, t=40, b=20),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)
