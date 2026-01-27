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

"""Export component for the Analyze web viewer."""

import io
import json
from typing import Any

import pandas as pd
import streamlit as st
import yaml

from oumi.analyze.storage import EvalData


def render_exports(eval_data: EvalData) -> None:
    """Render export options.

    Args:
        eval_data: The eval data to export.
    """
    st.header("Export")

    st.markdown(
        "Download your analysis results in various formats for further processing."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Data Exports")

        # CSV export
        csv_data = _results_to_csv(eval_data.analysis_results)
        st.download_button(
            "Download CSV",
            data=csv_data,
            file_name=f"{eval_data.metadata.name}_results.csv",
            mime="text/csv",
            use_container_width=True,
            help="Download results as CSV for spreadsheet analysis",
        )

        # JSON export
        json_data = json.dumps(
            {
                "metadata": eval_data.metadata.model_dump(),
                "results": eval_data.analysis_results,
                "test_results": eval_data.test_results,
            },
            indent=2,
            default=str,
        )
        st.download_button(
            "Download JSON",
            data=json_data,
            file_name=f"{eval_data.metadata.name}_full.json",
            mime="application/json",
            use_container_width=True,
            help="Download full eval data as JSON",
        )

    with col2:
        st.subheader("Configuration")

        # YAML config export
        if eval_data.config:
            yaml_data = yaml.dump(
                eval_data.config, default_flow_style=False, sort_keys=False
            )
            st.download_button(
                "Download Config (YAML)",
                data=yaml_data,
                file_name=f"{eval_data.metadata.name}_config.yaml",
                mime="text/yaml",
                use_container_width=True,
                help="Download the configuration used for this analysis",
            )
        else:
            st.info("No configuration available for this eval.")

        # Test results export
        if eval_data.test_results:
            test_json = json.dumps(eval_data.test_results, indent=2, default=str)
            st.download_button(
                "Download Test Results",
                data=test_json,
                file_name=f"{eval_data.metadata.name}_tests.json",
                mime="application/json",
                use_container_width=True,
                help="Download test results as JSON",
            )

    # Advanced exports
    st.divider()
    st.subheader("Advanced Exports")

    col3, col4 = st.columns(2)

    with col3:
        # Failed tests only
        if eval_data.test_results:
            failed_tests = [
                t
                for t in eval_data.test_results.get("tests", [])
                if not t.get("passed", False)
            ]
            if failed_tests:
                failed_json = json.dumps(failed_tests, indent=2, default=str)
                st.download_button(
                    "Download Failed Tests Only",
                    data=failed_json,
                    file_name=f"{eval_data.metadata.name}_failed_tests.json",
                    mime="application/json",
                    use_container_width=True,
                    help="Export only the failed tests for debugging",
                )

    with col4:
        # Low score samples
        low_score_data = _get_low_score_samples(eval_data.analysis_results, threshold=50)
        if low_score_data:
            low_score_json = json.dumps(low_score_data, indent=2, default=str)
            st.download_button(
                "Download Low Score Samples",
                data=low_score_json,
                file_name=f"{eval_data.metadata.name}_low_scores.json",
                mime="application/json",
                use_container_width=True,
                help="Export samples with score < 50 for review",
            )


def _results_to_csv(results: dict[str, Any]) -> str:
    """Convert analysis results to CSV format.

    Args:
        results: Analysis results dict.

    Returns:
        CSV string.
    """
    rows = []

    for analyzer_name, analyzer_results in results.items():
        if not isinstance(analyzer_results, list):
            continue

        for i, result in enumerate(analyzer_results):
            if isinstance(result, dict):
                row = {
                    "index": i,
                    "analyzer": analyzer_name,
                    **{k: v for k, v in result.items() if not k.startswith("_")},
                }
                rows.append(row)

    if not rows:
        return ""

    df = pd.DataFrame(rows)

    # Convert to CSV
    output = io.StringIO()
    df.to_csv(output, index=False)
    return output.getvalue()


def _get_low_score_samples(
    results: dict[str, Any], threshold: float = 50
) -> list[dict[str, Any]]:
    """Get samples with low scores.

    Args:
        results: Analysis results dict.
        threshold: Score threshold (samples below this are returned).

    Returns:
        List of low-score samples.
    """
    low_score_samples = []

    for analyzer_name, analyzer_results in results.items():
        if not isinstance(analyzer_results, list):
            continue

        for i, result in enumerate(analyzer_results):
            if isinstance(result, dict) and "score" in result:
                score = result.get("score", 100)
                if isinstance(score, (int, float)) and score < threshold:
                    low_score_samples.append({
                        "index": i,
                        "analyzer": analyzer_name,
                        **result,
                    })

    return low_score_samples
