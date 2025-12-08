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

"""HTML report generator for dataset analysis results."""

import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import pandas as pd
from jinja2 import Environment, FileSystemLoader

from oumi.utils.analysis_utils import DistributionType, detect_distribution_type
from oumi.utils.logging import logger

if TYPE_CHECKING:
    from oumi.core.analyze.dataset_analyzer import DatasetAnalyzer
    from oumi.core.analyze.health_score import DatasetHealthScore


class HTMLReportGenerator:
    """Generate interactive HTML reports from dataset analysis results.

    This generator creates self-contained HTML reports with:
    - Dataset overview statistics
    - Interactive Plotly charts for distributions
    - Recommendations section with severity indicators
    - Statistics tables for message and conversation levels

    The generated reports use Plotly for interactive visualizations
    and are fully self-contained (no external dependencies at runtime
    except for Plotly CDN).
    """

    def __init__(
        self,
        *,
        include_charts: bool = True,
        include_tables: bool = True,
        include_recommendations: bool = True,
        include_anomaly_visualization: bool = True,
        include_health_score: bool = True,
        chart_height: int = 400,
        max_charts: int = 10,
        outlier_std_threshold: float = 3.0,
    ):
        """Initialize the HTMLReportGenerator.

        Args:
            include_charts: Whether to include interactive Plotly charts.
                Requires plotly to be installed.
            include_tables: Whether to include statistics tables.
            include_recommendations: Whether to include the recommendations section.
            include_anomaly_visualization: Whether to include scatter plots
                highlighting outliers and anomalies.
            include_health_score: Whether to include the dataset health score.
            chart_height: Height of each chart in pixels.
            max_charts: Maximum number of charts to generate (to avoid huge reports).
            outlier_std_threshold: Standard deviations for outlier highlighting.
        """
        self.include_charts = include_charts
        self.include_tables = include_tables
        self.include_recommendations = include_recommendations
        self.include_anomaly_visualization = include_anomaly_visualization
        self.include_health_score = include_health_score
        self.chart_height = chart_height
        self.max_charts = max_charts
        self.outlier_std_threshold = outlier_std_threshold

        self._plotly_available = self._check_plotly()
        if self.include_charts and not self._plotly_available:
            logger.warning(
                "Plotly is not installed. Charts will not be included in the report. "
                "Install with: pip install 'oumi[analyze_advanced]'"
            )
            self.include_charts = False
            self.include_anomaly_visualization = False

        self._template = self._load_template()

    def _check_plotly(self) -> bool:
        """Check if plotly is available.

        Returns:
            True if plotly is installed, False otherwise.
        """
        try:
            import plotly  # noqa: F401

            return True
        except ImportError:
            return False

    def _load_template(self) -> Any:
        """Load the HTML report Jinja template.

        Returns:
            Loaded Jinja2 template.
        """
        template_dir = Path(__file__).parent / "templates"
        env = Environment(loader=FileSystemLoader(template_dir), autoescape=True)
        return env.get_template("report_template.html.jinja")

    def generate_report(
        self,
        analyzer: "DatasetAnalyzer",
        output_path: Path,
        title: Optional[str] = None,
        health_score: Optional["DatasetHealthScore"] = None,
    ) -> Path:
        """Generate an HTML report with external data files for performance.

        Creates a directory structure with:
        - index.html: Main report page (lightweight)
        - data/recommendations.json: Recommendation samples
        - data/duplicates.json: Duplicate group samples
        - data/clusters.json: Cluster samples
        - data/charts.json: Chart specifications

        Args:
            analyzer: DatasetAnalyzer instance with completed analysis.
            output_path: Path to save the HTML report (file or directory).
            title: Optional custom title for the report.
            health_score: Optional pre-computed health score to include.

        Returns:
            Path to the generated report directory.

        Raises:
            RuntimeError: If analysis has not been run on the analyzer.
        """
        # Get analysis summary
        try:
            summary = analyzer.analysis_summary
        except RuntimeError:
            raise RuntimeError(
                "Analysis has not been run yet. "
                "Please call analyze_dataset() before generating a report."
            )

        # Determine output directory
        if output_path.is_dir():
            output_dir = output_path
        elif output_path.suffix == ".html":
            # If a .html file is specified, use its parent as the directory
            output_dir = output_path.parent
        else:
            output_dir = output_path

        output_dir.mkdir(parents=True, exist_ok=True)
        data_dir = output_dir / "data"
        data_dir.mkdir(exist_ok=True)

        # Prepare full template data
        full_data = self._prepare_template_data(
            analyzer, summary, title, health_score
        )

        # Extract large data to external files
        external_data = self._write_external_data_files(full_data, data_dir)

        # Prepare lightweight template data (with metadata instead of full data)
        template_data = self._prepare_lightweight_template_data(
            full_data, external_data
        )

        # Render template
        html_content = self._template.render(**template_data)

        # Write main HTML file
        output_file = output_dir / "index.html"
        output_file.write_text(html_content, encoding="utf-8")
        logger.info(f"Generated HTML report: {output_file}")
        logger.info(f"External data files written to: {data_dir}")

        return output_dir

    def _write_external_data_files(
        self,
        full_data: dict[str, Any],
        data_dir: Path,
    ) -> dict[str, Any]:
        """Write large data to external JSON files.

        Args:
            full_data: Full template data dictionary.
            data_dir: Directory to write data files.

        Returns:
            Dictionary with metadata about external files.
        """
        external_data: dict[str, Any] = {}

        # Write recommendations data
        if full_data.get("recommendations"):
            recommendations_file = data_dir / "recommendations.json"
            recommendations_file.write_text(
                json.dumps(
                    full_data["recommendations"], indent=2, cls=PlotlyJSONEncoder
                ),
                encoding="utf-8",
            )
            external_data["recommendations"] = {
                "file": "data/recommendations.json",
                "count": len(full_data["recommendations"]),
            }

        # Write duplicates data
        if full_data.get("duplicates"):
            duplicates_file = data_dir / "duplicates.json"
            duplicates_file.write_text(
                json.dumps(full_data["duplicates"], indent=2, cls=PlotlyJSONEncoder),
                encoding="utf-8",
            )
            external_data["duplicates"] = {
                "file": "data/duplicates.json",
                "has_semantic": full_data["duplicates"].get("semantic") is not None,
                "has_fuzzy": full_data["duplicates"].get("fuzzy") is not None,
            }

        # Write clusters data
        if full_data.get("clusters"):
            clusters_file = data_dir / "clusters.json"
            clusters_file.write_text(
                json.dumps(full_data["clusters"], indent=2, cls=PlotlyJSONEncoder),
                encoding="utf-8",
            )
            external_data["clusters"] = {
                "file": "data/clusters.json",
                "has_embedding": (
                    full_data["clusters"].get("embedding_clusters") is not None
                ),
                "has_question_diversity": (
                    full_data["clusters"].get("question_diversity_clusters") is not None
                ),
            }

        # Write charts data
        if full_data.get("charts") or full_data.get("anomaly_charts"):
            charts_data = {
                "charts": full_data.get("charts", []),
                "anomaly_charts": full_data.get("anomaly_charts", []),
            }
            charts_file = data_dir / "charts.json"
            charts_file.write_text(
                json.dumps(charts_data, indent=2, cls=PlotlyJSONEncoder),
                encoding="utf-8",
            )
            external_data["charts"] = {
                "file": "data/charts.json",
                "chart_count": len(full_data.get("charts", [])),
                "anomaly_chart_count": len(full_data.get("anomaly_charts", [])),
            }

        return external_data

    def _prepare_lightweight_template_data(
        self,
        full_data: dict[str, Any],
        external_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Prepare lightweight template data with external file references.

        Args:
            full_data: Full template data dictionary.
            external_data: Metadata about external data files.

        Returns:
            Lightweight template data with file references.
        """
        # Start with basic data that's always small
        template_data = {
            "title": full_data["title"],
            "generated_at": full_data["generated_at"],
            "overview": full_data["overview"],
            "message_stats": full_data.get("message_stats", {}),
            "conversation_stats": full_data.get("conversation_stats", {}),
            "conversation_turns": full_data.get("conversation_turns", {}),
            "health_score": full_data.get("health_score"),
            # External file references
            "external_data": external_data,
            # Summary counts for display (without full data)
            "recommendations_summary": self._get_recommendations_summary(
                full_data.get("recommendations", [])
            ),
            "duplicates_summary": self._get_duplicates_summary(
                full_data.get("duplicates")
            ),
            "clusters_summary": self._get_clusters_summary(
                full_data.get("clusters")
            ),
            "charts_summary": {
                "count": len(full_data.get("charts", [])),
                "anomaly_count": len(full_data.get("anomaly_charts", [])),
            },
        }

        return template_data

    def _get_recommendations_summary(
        self, recommendations: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Get summary of recommendations for lightweight display.

        Args:
            recommendations: List of recommendation dictionaries.

        Returns:
            Summary dictionary with counts by severity.
        """
        summary = {
            "total": len(recommendations),
            "high": 0,
            "medium": 0,
            "low": 0,
            "list": [],  # Renamed from 'items' to avoid Jinja2 dict.items conflict
        }

        for rec in recommendations:
            severity = rec.get("severity", "low")
            if severity == "high":
                summary["high"] += 1
            elif severity == "medium":
                summary["medium"] += 1
            else:
                summary["low"] += 1

            # Include basic info (without conversation samples)
            summary["list"].append({
                "id": rec.get("id", ""),
                "title": rec.get("title", ""),
                "description": rec.get("description", ""),
                "severity": severity,
                "affected_samples": rec.get("affected_samples", 0),
                "metric_name": rec.get("metric_name", ""),
                "has_conversations": bool(rec.get("conversations")),
                "conversation_count": len(rec.get("conversations", [])),
            })

        return summary

    def _get_duplicates_summary(
        self, duplicates: Optional[dict[str, Any]]
    ) -> Optional[dict[str, Any]]:
        """Get summary of duplicates for lightweight display.

        Args:
            duplicates: Duplicates data dictionary.

        Returns:
            Summary dictionary or None.
        """
        if not duplicates:
            return None

        summary: dict[str, Any] = {}

        if duplicates.get("semantic"):
            summary["semantic"] = {
                "total_with_duplicates": duplicates["semantic"].get(
                    "total_with_duplicates", 0
                ),
                "percentage": duplicates["semantic"].get("percentage", 0),
                "num_groups": duplicates["semantic"].get("num_groups", 0),
            }

        if duplicates.get("fuzzy"):
            summary["fuzzy"] = {
                "total_with_duplicates": duplicates["fuzzy"].get(
                    "total_with_duplicates", 0
                ),
                "percentage": duplicates["fuzzy"].get("percentage", 0),
                "num_groups": duplicates["fuzzy"].get("num_groups", 0),
            }

        return summary if summary else None

    def _get_clusters_summary(
        self, clusters: Optional[dict[str, Any]]
    ) -> Optional[dict[str, Any]]:
        """Get summary of clusters for lightweight display.

        Args:
            clusters: Clusters data dictionary.

        Returns:
            Summary dictionary or None.
        """
        if not clusters:
            return None

        summary: dict[str, Any] = {}

        if clusters.get("embedding_clusters"):
            ec = clusters["embedding_clusters"]
            summary["embedding_clusters"] = {
                "total_clusters": ec.get("total_clusters", 0),
                "noise_count": ec.get("noise_count", 0),
            }

        if clusters.get("question_diversity_clusters"):
            qc = clusters["question_diversity_clusters"]
            summary["question_diversity_clusters"] = {
                "total_clusters": qc.get("total_clusters", 0),
                "noise_count": qc.get("noise_count", 0),
                "total_questions": qc.get("total_questions", 0),
                "concentrated_count": qc.get("concentrated_count", 0),
            }

        return summary if summary else None

    def _prepare_template_data(
        self,
        analyzer: "DatasetAnalyzer",
        summary: dict[str, Any],
        title: Optional[str],
        health_score: Optional["DatasetHealthScore"] = None,
    ) -> dict[str, Any]:
        """Prepare data for the HTML template.

        Args:
            analyzer: DatasetAnalyzer instance.
            summary: Analysis summary dictionary.
            title: Optional custom title.
            health_score: Optional health score to include.

        Returns:
            Dictionary of template variables.
        """
        overview = summary.get("dataset_overview", {})
        dataset_name = overview.get("dataset_name", "Unknown Dataset")

        data = {
            "title": title or f"Analysis Report: {dataset_name}",
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "overview": overview,
            "message_stats": summary.get("message_level_summary", {}),
            "conversation_stats": summary.get("conversation_level_summary", {}),
            "conversation_turns": summary.get("conversation_turns", {}),
            "recommendations": [],
            "charts": [],
            "anomaly_charts": [],
            "health_score": None,
            "duplicates": None,
            "clusters": None,
        }

        # Add recommendations if enabled
        if self.include_recommendations:
            recommendations = summary.get("recommendations", [])
            # Enrich recommendations with sample data
            data["recommendations"] = self._enrich_recommendations_with_samples(
                recommendations, analyzer
            )

        # Generate charts if enabled
        if self.include_charts and self._plotly_available:
            data["charts"] = self._generate_charts(analyzer)

        # Generate anomaly visualizations if enabled
        if self.include_anomaly_visualization and self._plotly_available:
            data["anomaly_charts"] = self._generate_anomaly_charts(analyzer)

        # Add health score if provided
        if self.include_health_score and health_score is not None:
            data["health_score"] = health_score.to_dict()

        # Extract duplicate and cluster information
        data["duplicates"] = self._extract_duplicate_data(analyzer)
        data["clusters"] = self._extract_cluster_data(analyzer)

        return data

    def _enrich_recommendations_with_samples(
        self,
        recommendations: list[dict[str, Any]],
        analyzer: "DatasetAnalyzer",
    ) -> list[dict[str, Any]]:
        """Enrich recommendations with full conversation data for display.

        For each problematic sample, retrieves the entire conversation to provide
        context. The problematic message is highlighted within the conversation.

        Args:
            recommendations: List of recommendation dictionaries.
            analyzer: DatasetAnalyzer instance with analysis results.

        Returns:
            List of recommendations enriched with conversation data.
        """
        message_df = analyzer.message_df
        if message_df is None or message_df.empty:
            return recommendations

        # Check if conversation_id column exists
        has_conv_id = "conversation_id" in message_df.columns

        enriched = []
        for idx, rec in enumerate(recommendations):
            rec_copy = dict(rec)
            rec_copy["id"] = f"rec_{idx}"
            sample_indices = rec.get("sample_indices", [])
            sample_indices_set = set(sample_indices)

            if sample_indices:
                # Get unique conversation IDs from the sample indices
                conversations = []
                seen_conv_ids: set[str] = set()

                for sample_idx in sample_indices:
                    if sample_idx not in message_df.index:
                        continue

                    row = message_df.loc[sample_idx]
                    if has_conv_id:
                        conv_id = row.get("conversation_id", f"sample_{sample_idx}")
                    else:
                        # For flat format, each sample is its own "conversation"
                        conv_id = f"sample_{sample_idx}"

                    # Skip if we've already processed this conversation
                    if conv_id in seen_conv_ids:
                        continue

                    # Limit to 20 conversations
                    if len(seen_conv_ids) >= 20:
                        break

                    seen_conv_ids.add(conv_id)

                    # Get all messages in this conversation
                    if has_conv_id:
                        conv_messages = message_df[
                            message_df["conversation_id"] == conv_id
                        ].sort_index()
                    else:
                        # For flat format, just use the single row
                        conv_messages = message_df.loc[[sample_idx]]

                    # Build conversation data with turns
                    turns = []
                    for msg_idx, msg_row in conv_messages.iterrows():
                        # Get text content - try multiple possible column names
                        text = ""
                        if "text_content" in msg_row and msg_row.get("text_content"):
                            text = str(msg_row.get("text_content", ""))
                        elif not has_conv_id:
                            # For flat format, construct text from available columns
                            parts = []
                            for col in ["instruction", "input", "prompt", "question"]:
                                if col in msg_row and msg_row.get(col):
                                    parts.append(f"[{col}]: {msg_row[col]}")
                            for col in ["output", "response", "answer", "completion"]:
                                if col in msg_row and msg_row.get(col):
                                    parts.append(f"[{col}]: {msg_row[col]}")
                            text = "\n".join(parts) if parts else ""

                        turn_data = {
                            "index": int(msg_idx),
                            "role": msg_row.get("role", "sample"),
                            "text": text,
                            "is_flagged": int(msg_idx) in sample_indices_set,
                        }
                        # Add metric value for flagged messages
                        if turn_data["is_flagged"]:
                            metric_name = rec.get("metric_name")
                            if metric_name and metric_name in msg_row:
                                turn_data["metric_value"] = self._format_metric_value(
                                    msg_row[metric_name]
                                )
                        turns.append(turn_data)

                    conversations.append(
                        {
                            "conversation_id": str(conv_id),
                            "turns": turns,
                            "flagged_count": sum(
                                1 for t in turns if t.get("is_flagged")
                            ),
                        }
                    )

                rec_copy["conversations"] = conversations
                rec_copy["total_conversations"] = len(seen_conv_ids)
            else:
                rec_copy["conversations"] = []
                rec_copy["total_conversations"] = 0

            enriched.append(rec_copy)

        return enriched

    def _truncate_text(self, text: str, max_length: int = 300) -> str:
        """Truncate text to a maximum length with ellipsis.

        Args:
            text: Text to truncate.
            max_length: Maximum length before truncation.

        Returns:
            Truncated text with ellipsis if needed.
        """
        if len(text) <= max_length:
            return text
        return text[:max_length].rsplit(" ", 1)[0] + "..."

    def _format_metric_value(self, value: Any) -> str:
        """Format a metric value for display.

        Args:
            value: Metric value to format.

        Returns:
            Formatted string representation.
        """
        import math

        if value is None:
            return "N/A"
        if isinstance(value, float):
            if math.isnan(value) or math.isinf(value):
                return "N/A"
            return f"{value:.2f}"
        elif isinstance(value, bool):
            return "Yes" if value else "No"
        # Handle numpy types
        if hasattr(value, "item"):
            try:
                py_val = value.item()
                if isinstance(py_val, float) and (math.isnan(py_val) or math.isinf(py_val)):
                    return "N/A"
                return f"{py_val:.2f}" if isinstance(py_val, float) else str(py_val)
            except (ValueError, TypeError):
                pass
        return str(value)

    def _extract_duplicate_data(
        self, analyzer: "DatasetAnalyzer"
    ) -> Optional[dict[str, Any]]:
        """Extract duplicate detection data from the analyzer.

        Args:
            analyzer: DatasetAnalyzer instance with analysis results.

        Returns:
            Dictionary with duplicate statistics and sample groups, or None.
        """
        message_df = analyzer.message_df
        if message_df is None or message_df.empty:
            return None

        result: dict[str, Any] = {
            "semantic": None,
            "fuzzy": None,
        }

        # Find semantic duplicate columns
        semantic_dup_cols = [
            col for col in message_df.columns if "has_semantic_duplicate" in col
        ]
        semantic_group_cols = [
            col for col in message_df.columns if "duplicate_group" in col
            and "fuzzy" not in col
        ]

        if semantic_dup_cols and semantic_group_cols:
            dup_col = semantic_dup_cols[0]
            group_col = semantic_group_cols[0]

            total_with_dups = message_df[dup_col].sum()
            total_samples = len(message_df)

            # Get unique duplicate groups (excluding singletons)
            group_counts = message_df[group_col].value_counts()
            dup_groups = group_counts[group_counts > 1]

            # Build sample groups for display (up to 10 groups)
            sample_groups = []
            for group_id, count in list(dup_groups.items())[:10]:
                group_rows = message_df[message_df[group_col] == group_id]
                samples = []
                for idx, row in list(group_rows.iterrows())[:5]:
                    samples.append({
                        "index": int(idx),
                        "text": self._truncate_text(
                            str(row.get("text_content", "")), 200
                        ),
                        "role": row.get("role", "unknown"),
                        "conversation_id": str(row.get("conversation_id", ""))[:20],
                    })
                sample_groups.append({
                    "group_id": int(group_id) if not np.isnan(group_id) else 0,
                    "count": int(count),
                    "samples": samples,
                })

            result["semantic"] = {
                "total_with_duplicates": int(total_with_dups),
                "total_samples": total_samples,
                "percentage": round(100 * total_with_dups / total_samples, 1)
                if total_samples > 0 else 0,
                "num_groups": len(dup_groups),
                "sample_groups": sample_groups,
            }

        # Find fuzzy duplicate columns
        fuzzy_dup_cols = [
            col for col in message_df.columns if "has_fuzzy_duplicate" in col
        ]
        fuzzy_group_cols = [
            col for col in message_df.columns if "fuzzy_duplicate_group" in col
        ]

        if fuzzy_dup_cols and fuzzy_group_cols:
            dup_col = fuzzy_dup_cols[0]
            group_col = fuzzy_group_cols[0]

            total_with_dups = message_df[dup_col].sum()
            total_samples = len(message_df)

            group_counts = message_df[group_col].value_counts()
            dup_groups = group_counts[group_counts > 1]

            sample_groups = []
            for group_id, count in list(dup_groups.items())[:10]:
                group_rows = message_df[message_df[group_col] == group_id]
                samples = []
                for idx, row in list(group_rows.iterrows())[:5]:
                    samples.append({
                        "index": int(idx),
                        "text": self._truncate_text(
                            str(row.get("text_content", "")), 200
                        ),
                        "role": row.get("role", "unknown"),
                        "conversation_id": str(row.get("conversation_id", ""))[:20],
                    })
                sample_groups.append({
                    "group_id": int(group_id) if not np.isnan(group_id) else 0,
                    "count": int(count),
                    "samples": samples,
                })

            result["fuzzy"] = {
                "total_with_duplicates": int(total_with_dups),
                "total_samples": total_samples,
                "percentage": round(100 * total_with_dups / total_samples, 1)
                if total_samples > 0 else 0,
                "num_groups": len(dup_groups),
                "sample_groups": sample_groups,
            }

        # Return None if no duplicate data found
        if result["semantic"] is None and result["fuzzy"] is None:
            return None

        return result

    def _extract_cluster_data(
        self, analyzer: "DatasetAnalyzer"
    ) -> Optional[dict[str, Any]]:
        """Extract clustering data from the analyzer.

        Args:
            analyzer: DatasetAnalyzer instance with analysis results.

        Returns:
            Dictionary with cluster statistics and distribution, or None.
        """
        message_df = analyzer.message_df
        if message_df is None or message_df.empty:
            return None

        result: dict[str, Any] = {
            "embedding_clusters": None,
            "question_diversity_clusters": None,
        }

        # Find embedding cluster columns
        embedding_cluster_cols = [
            col for col in message_df.columns
            if col.endswith("_embedding_cluster")
        ]

        if embedding_cluster_cols:
            cluster_col = embedding_cluster_cols[0]
            cluster_counts = message_df[cluster_col].value_counts().sort_index()

            # Build distribution data with samples
            distribution = []
            for cluster_id, count in cluster_counts.items():
                if cluster_id == -1:
                    label = "Noise"
                else:
                    label = f"Cluster {int(cluster_id)}"

                # Get sample texts for this cluster (up to 5)
                cluster_rows = message_df[message_df[cluster_col] == cluster_id]
                samples = []
                for idx, row in list(cluster_rows.iterrows())[:5]:
                    samples.append({
                        "index": int(idx),
                        "text": str(row.get("text_content", "")),
                        "role": row.get("role", "unknown"),
                        "conversation_id": str(row.get("conversation_id", ""))[:20],
                    })

                distribution.append({
                    "cluster_id": int(cluster_id) if not np.isnan(cluster_id) else -1,
                    "label": label,
                    "count": int(count),
                    "percentage": round(100 * count / len(message_df), 1),
                    "samples": samples,
                })

            result["embedding_clusters"] = {
                "total_clusters": len([c for c in cluster_counts.index if c != -1]),
                "noise_count": int(cluster_counts.get(-1, 0)),
                "distribution": distribution,
            }

        # Find question diversity cluster columns
        question_cluster_cols = [
            col for col in message_df.columns
            if "question_diversity_cluster_id" in col
        ]

        if question_cluster_cols:
            cluster_col = question_cluster_cols[0]
            # Filter to only rows with cluster assignments (user messages)
            clustered_df = message_df[message_df[cluster_col].notna()]

            if len(clustered_df) > 0:
                cluster_counts = clustered_df[cluster_col].value_counts().sort_index()

                distribution = []
                for cluster_id, count in cluster_counts.items():
                    if cluster_id == -1:
                        label = "Noise"
                    else:
                        label = f"Cluster {int(cluster_id)}"

                    # Get sample texts for this cluster (up to 5)
                    cluster_rows = clustered_df[clustered_df[cluster_col] == cluster_id]
                    samples = []
                    for idx, row in list(cluster_rows.iterrows())[:5]:
                        samples.append({
                            "index": int(idx),
                            "text": str(row.get("text_content", "")),
                            "role": row.get("role", "unknown"),
                            "conversation_id": str(row.get("conversation_id", ""))[:20],
                        })

                    distribution.append({
                        "cluster_id": int(cluster_id) if not np.isnan(cluster_id) else -1,
                        "label": label,
                        "count": int(count),
                        "percentage": round(100 * count / len(clustered_df), 1),
                        "samples": samples,
                    })

                # Get concentration info
                concentrated_col = cluster_col.replace("_cluster_id", "_is_concentrated")
                concentrated_count = 0
                if concentrated_col in message_df.columns:
                    concentrated_count = int(
                        message_df[concentrated_col].fillna(False).sum()
                    )

                result["question_diversity_clusters"] = {
                    "total_clusters": len([c for c in cluster_counts.index if c != -1]),
                    "noise_count": int(cluster_counts.get(-1, 0)),
                    "total_questions": len(clustered_df),
                    "concentrated_count": concentrated_count,
                    "distribution": distribution,
                }

        # Return None if no cluster data found
        if (result["embedding_clusters"] is None and
                result["question_diversity_clusters"] is None):
            return None

        return result

    def _generate_charts(self, analyzer: "DatasetAnalyzer") -> list[dict[str, Any]]:
        """Generate Plotly chart specifications for the report.

        Args:
            analyzer: DatasetAnalyzer instance with analysis results.

        Returns:
            List of chart dictionaries with id, title, data, and layout.
        """
        import plotly.graph_objects as go

        charts = []
        chart_count = 0

        message_df = analyzer.message_df
        if message_df is None or message_df.empty:
            return charts

        # Find numeric analysis columns to visualize
        base_columns = {
            "conversation_index",
            "conversation_id",
            "message_index",
            "message_id",
            "role",
            "text_content",
        }

        numeric_cols = [
            col
            for col in message_df.columns
            if col not in base_columns
            and message_df[col].dtype in ["int64", "float64"]
        ]

        # Generate histograms for numeric columns
        for col in numeric_cols:
            if chart_count >= self.max_charts:
                break

            series = message_df[col].dropna()
            if len(series) < 2:
                continue

            # Create a cleaner title
            col_title = col.replace("text_content_", "").replace("_", " ").title()

            # Detect distribution type for enhanced visualization
            dist_result = detect_distribution_type(series)
            is_multimodal = dist_result.distribution_type in (
                DistributionType.BIMODAL,
                DistributionType.MULTIMODAL,
            )

            # Create histogram
            fig = go.Figure(
                data=[
                    go.Histogram(
                        x=series,
                        nbinsx=30,
                        marker_color="#d4a574",
                        opacity=0.85,
                        name="Distribution",
                    )
                ]
            )

            # Add mode annotations for multimodal distributions
            if is_multimodal and dist_result.mode_statistics:
                # Color palette for mode indicators
                mode_colors = ["#6b9bd1", "#7cb97c", "#e0b854", "#d66a6a", "#9b59b6"]

                for i, ms in enumerate(dist_result.mode_statistics):
                    color = mode_colors[i % len(mode_colors)]

                    # Add vertical line at mode mean
                    fig.add_vline(
                        x=ms.mean,
                        line_dash="dash",
                        line_color=color,
                        line_width=2,
                        annotation_text=f"Mode {i+1}: μ={ms.mean:.1f}",
                        annotation_font_color="#a39e93",
                        annotation_font_size=10,
                    )

                    # Add shaded region for ±1 std (subtle rectangle)
                    fig.add_vrect(
                        x0=ms.mean - ms.std,
                        x1=ms.mean + ms.std,
                        fillcolor=color,
                        opacity=0.1,
                        line_width=0,
                    )

                # Update title to indicate multimodal
                col_title = f"{col_title} ({dist_result.num_modes} modes)"

            fig.update_layout(
                title=None,
                xaxis_title=col_title,
                yaxis_title="Count",
                height=self.chart_height,
                margin=dict(l=50, r=30, t=30, b=50),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#a39e93"),
                xaxis=dict(gridcolor="rgba(255,255,255,0.05)", color="#a39e93"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.05)", color="#a39e93"),
                showlegend=is_multimodal,
            )

            chart_data = {
                "id": f"chart_{chart_count}",
                "title": col_title,
                "data": json.dumps(fig.data, cls=PlotlyJSONEncoder),
                "layout": json.dumps(fig.layout, cls=PlotlyJSONEncoder),
            }

            # Add mode metadata for multimodal distributions
            if is_multimodal:
                chart_data["distribution_type"] = dist_result.distribution_type.value
                chart_data["num_modes"] = dist_result.num_modes
                chart_data["mode_statistics"] = [
                    {
                        "mode_id": ms.mode_id,
                        "mean": ms.mean,
                        "std": ms.std,
                        "count": ms.count,
                        "weight": ms.weight,
                    }
                    for ms in dist_result.mode_statistics
                ]

            charts.append(chart_data)
            chart_count += 1

        # Add role distribution pie chart if available
        if "role" in message_df.columns and chart_count < self.max_charts:
            role_counts = message_df["role"].value_counts()

            fig = go.Figure(
                data=[
                    go.Pie(
                        labels=role_counts.index.tolist(),
                        values=role_counts.values.tolist(),
                        hole=0.45,
                        marker_colors=["#d4a574", "#6b9bd1", "#7cb97c", "#e0b854"],
                        textfont=dict(color="#e8e6e3"),
                    )
                ]
            )

            fig.update_layout(
                title=None,
                height=self.chart_height,
                margin=dict(l=30, r=30, t=30, b=30),
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#a39e93"),
                legend=dict(font=dict(color="#a39e93")),
            )

            charts.append(
                {
                    "id": f"chart_{chart_count}",
                    "title": "Role Distribution",
                    "data": json.dumps(fig.data, cls=PlotlyJSONEncoder),
                    "layout": json.dumps(fig.layout, cls=PlotlyJSONEncoder),
                }
            )

        return charts


    def _generate_anomaly_charts(
        self, analyzer: "DatasetAnalyzer"
    ) -> list[dict[str, Any]]:
        """Generate scatter plots highlighting outliers and anomalies.

        Creates visualizations that highlight outlier samples to help users
        identify problematic data points for review.

        Args:
            analyzer: DatasetAnalyzer instance with analysis results.

        Returns:
            List of chart dictionaries with id, title, data, and layout.
        """
        import plotly.graph_objects as go

        charts = []
        chart_count = 0
        max_anomaly_charts = 8  # Includes outliers + quality + IFD
        # Reserve slots for specific important charts (quality, IFD)
        max_outlier_charts = max_anomaly_charts - 2

        message_df = analyzer.message_df
        if message_df is None or message_df.empty:
            return charts

        # Find numeric analysis columns
        base_columns = {
            "conversation_index",
            "conversation_id",
            "message_index",
            "message_id",
            "role",
            "text_content",
        }

        numeric_cols = [
            col
            for col in message_df.columns
            if col not in base_columns
            and message_df[col].dtype in ["int64", "float64"]
        ]

        for col in numeric_cols:
            if chart_count >= max_outlier_charts:
                break

            series = message_df[col].dropna()
            if len(series) < 10:
                continue

            mean = series.mean()
            std = series.std()

            if std == 0:
                continue

            # Calculate outlier bounds
            upper_bound = mean + (self.outlier_std_threshold * std)
            lower_bound = mean - (self.outlier_std_threshold * std)

            # Identify outliers
            is_outlier = (series > upper_bound) | (series < lower_bound)
            outlier_count = is_outlier.sum()

            if outlier_count == 0:
                continue

            # Create scatter plot with outliers highlighted
            col_title = col.replace("text_content_", "").replace("_", " ").title()

            # Normal points
            normal_mask = ~is_outlier
            normal_x = list(range(len(series[normal_mask])))
            normal_y = series[normal_mask].tolist()

            # Outlier points
            outlier_indices = series[is_outlier].index.tolist()
            outlier_y = series[is_outlier].tolist()

            fig = go.Figure()

            # Add normal points
            fig.add_trace(
                go.Scatter(
                    x=normal_x,
                    y=normal_y,
                    mode="markers",
                    name="Normal",
                    marker=dict(color="#6b9bd1", size=6, opacity=0.6),
                    hovertemplate="Index: %{x}<br>Value: %{y:.2f}<extra></extra>",
                )
            )

            # Add outlier points
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(outlier_y))),
                    y=outlier_y,
                    mode="markers",
                    name=f"Outliers ({outlier_count})",
                    marker=dict(color="#d66a6a", size=10, symbol="x"),
                    hovertemplate=(
                        "Index: %{customdata}<br>Value: %{y:.2f}<extra></extra>"
                    ),
                    customdata=outlier_indices,
                )
            )

            # Add threshold lines
            fig.add_hline(
                y=upper_bound,
                line_dash="dash",
                line_color="#e0b854",
                annotation_text=f"Upper bound ({upper_bound:.1f})",
                annotation_font_color="#a39e93",
            )
            fig.add_hline(
                y=lower_bound,
                line_dash="dash",
                line_color="#e0b854",
                annotation_text=f"Lower bound ({lower_bound:.1f})",
                annotation_font_color="#a39e93",
            )
            fig.add_hline(
                y=mean,
                line_dash="dot",
                line_color="#7cb97c",
                annotation_text=f"Mean ({mean:.1f})",
                annotation_font_color="#a39e93",
            )

            fig.update_layout(
                title=None,
                xaxis_title="Sample Index",
                yaxis_title=col_title,
                height=self.chart_height,
                margin=dict(l=50, r=30, t=30, b=50),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#a39e93"),
                xaxis=dict(gridcolor="rgba(255,255,255,0.05)", color="#a39e93"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.05)", color="#a39e93"),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1,
                    font=dict(color="#a39e93"),
                ),
            )

            charts.append(
                {
                    "id": f"anomaly_chart_{chart_count}",
                    "title": f"Outliers in {col_title}",
                    "data": json.dumps(fig.data, cls=PlotlyJSONEncoder),
                    "layout": json.dumps(fig.layout, cls=PlotlyJSONEncoder),
                    "outlier_count": outlier_count,
                    "total_count": len(series),
                }
            )
            chart_count += 1

        # Add quality score distribution if available (reserved slot)
        quality_cols = [col for col in message_df.columns if "quality_score" in col]
        if quality_cols:
            col = quality_cols[0]
            scores = message_df[col].dropna()

            if len(scores) > 0:
                # Create histogram with low-quality samples highlighted
                low_quality_mask = scores < 0.5
                high_quality_mask = scores >= 0.5

                fig = go.Figure()

                fig.add_trace(
                    go.Histogram(
                        x=scores[high_quality_mask].tolist(),
                        name="Good Quality (≥0.5)",
                        marker_color="#7cb97c",
                        opacity=0.75,
                        nbinsx=20,
                    )
                )

                fig.add_trace(
                    go.Histogram(
                        x=scores[low_quality_mask].tolist(),
                        name=f"Low Quality ({low_quality_mask.sum()})",
                        marker_color="#d66a6a",
                        opacity=0.75,
                        nbinsx=20,
                    )
                )

                fig.update_layout(
                    title=None,
                    xaxis_title="Quality Score",
                    yaxis_title="Count",
                    height=self.chart_height,
                    barmode="overlay",
                    margin=dict(l=50, r=30, t=30, b=50),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#a39e93"),
                    xaxis=dict(gridcolor="rgba(255,255,255,0.05)", color="#a39e93"),
                    yaxis=dict(gridcolor="rgba(255,255,255,0.05)", color="#a39e93"),
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1,
                        font=dict(color="#a39e93"),
                    ),
                )

                charts.append(
                    {
                        "id": f"anomaly_chart_{chart_count}",
                        "title": "Quality Score Distribution",
                        "data": json.dumps(fig.data, cls=PlotlyJSONEncoder),
                        "layout": json.dumps(fig.layout, cls=PlotlyJSONEncoder),
                        "outlier_count": int(low_quality_mask.sum()),
                        "total_count": len(scores),
                    }
                )
                chart_count += 1

        # Add IFD score distribution if available (reserved slot)
        ifd_cols = [col for col in message_df.columns if "ifd_score" in col]
        if ifd_cols:
            col = ifd_cols[0]
            scores = pd.to_numeric(message_df[col], errors="coerce").dropna()
            # Filter out invalid values (<=0, inf) for log scale compatibility
            scores = scores[scores > 0]
            scores = scores[np.isfinite(scores)]

            if len(scores) > 0:
                # Create histogram with problematic samples (IFD < 1) highlighted
                low_ifd_mask = scores < 1.0
                good_ifd_mask = (scores >= 1.0) & (scores <= 10.0)
                high_ifd_mask = scores > 10.0

                fig = go.Figure()

                # High IFD samples (valuable)
                if high_ifd_mask.sum() > 0:
                    fig.add_trace(
                        go.Histogram(
                            x=scores[high_ifd_mask].tolist(),
                            name=f"High IFD >10 ({high_ifd_mask.sum()})",
                            marker_color="#6b9bd1",
                            opacity=0.75,
                            nbinsx=20,
                        )
                    )

                # Good IFD samples (normal)
                if good_ifd_mask.sum() > 0:
                    fig.add_trace(
                        go.Histogram(
                            x=scores[good_ifd_mask].tolist(),
                            name=f"Good IFD 1-10 ({good_ifd_mask.sum()})",
                            marker_color="#7cb97c",
                            opacity=0.75,
                            nbinsx=20,
                        )
                    )

                # Low IFD samples (problematic)
                if low_ifd_mask.sum() > 0:
                    fig.add_trace(
                        go.Histogram(
                            x=scores[low_ifd_mask].tolist(),
                            name=f"Low IFD <1 ({low_ifd_mask.sum()})",
                            marker_color="#d66a6a",
                            opacity=0.75,
                            nbinsx=20,
                        )
                    )

                # Only proceed if we have at least one trace
                if fig.data:
                    # Add vertical line at IFD = 1.0 (threshold)
                    fig.add_vline(
                        x=1.0,
                        line_dash="dash",
                        line_color="#e0b854",
                        annotation_text="IFD = 1.0 (threshold)",
                        annotation_font_color="#a39e93",
                    )

                    fig.update_layout(
                        title=None,
                        xaxis_title="IFD Score (log scale)",
                        yaxis_title="Count",
                        height=self.chart_height,
                        barmode="overlay",
                        margin=dict(l=50, r=30, t=30, b=50),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#a39e93"),
                        xaxis=dict(
                            gridcolor="rgba(255,255,255,0.05)",
                            color="#a39e93",
                            type="log",  # Log scale for IFD
                        ),
                        yaxis=dict(
                            gridcolor="rgba(255,255,255,0.05)", color="#a39e93"
                        ),
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1,
                            font=dict(color="#a39e93"),
                        ),
                    )

                    charts.append(
                        {
                            "id": f"anomaly_chart_{chart_count}",
                            "title": "IFD Score Distribution",
                            "data": json.dumps(sanitize_for_json(fig.data)),
                            "layout": json.dumps(sanitize_for_json(fig.layout)),
                            "outlier_count": int(low_ifd_mask.sum()),
                            "total_count": len(scores),
                        }
                    )

        return charts


def sanitize_for_json(obj: Any) -> Any:
    """Recursively sanitize data to replace NaN/Inf with None for JSON.

    Args:
        obj: Object to sanitize.

    Returns:
        Sanitized object safe for JSON serialization.
    """
    import math

    if obj is None:
        return None
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    elif isinstance(obj, (np.floating, np.integer)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj) if isinstance(obj, np.floating) else int(obj)
    elif isinstance(obj, np.ndarray):
        return sanitize_for_json(obj.tolist())
    elif isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        result = [sanitize_for_json(item) for item in obj]
        return tuple(result) if isinstance(obj, tuple) else result
    elif hasattr(obj, "to_plotly_json"):
        return sanitize_for_json(obj.to_plotly_json())
    return obj


class PlotlyJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for Plotly objects."""

    def default(self, obj):
        """Encode Plotly objects to JSON-serializable format."""
        # Handle numpy types
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            # Handle NaN and Inf values which are not valid JSON
            if np.isnan(obj):
                return None
            elif np.isinf(obj):
                return None
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()

        # Handle Plotly objects
        if hasattr(obj, "to_plotly_json"):
            return obj.to_plotly_json()

        return super().default(obj)
