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
from jinja2 import Environment, FileSystemLoader

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
        """Generate an HTML report and save to file.

        Args:
            analyzer: DatasetAnalyzer instance with completed analysis.
            output_path: Path to save the HTML report (file or directory).
            title: Optional custom title for the report.
            health_score: Optional pre-computed health score to include.

        Returns:
            Path to the generated report file.

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

        # Determine output file path
        if output_path.is_dir():
            output_file = output_path / "analysis_report.html"
        else:
            output_file = output_path
            output_file.parent.mkdir(parents=True, exist_ok=True)

        # Prepare template data
        template_data = self._prepare_template_data(
            analyzer, summary, title, health_score
        )

        # Render template
        html_content = self._template.render(**template_data)

        # Write to file
        output_file.write_text(html_content, encoding="utf-8")
        logger.info(f"Generated HTML report: {output_file}")

        return output_file

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

        return data

    def _enrich_recommendations_with_samples(
        self,
        recommendations: list[dict[str, Any]],
        analyzer: "DatasetAnalyzer",
    ) -> list[dict[str, Any]]:
        """Enrich recommendations with actual sample data for display.

        Args:
            recommendations: List of recommendation dictionaries.
            analyzer: DatasetAnalyzer instance with analysis results.

        Returns:
            List of recommendations enriched with sample data.
        """
        message_df = analyzer.message_df
        if message_df is None or message_df.empty:
            return recommendations

        enriched = []
        for idx, rec in enumerate(recommendations):
            rec_copy = dict(rec)
            rec_copy["id"] = f"rec_{idx}"
            sample_indices = rec.get("sample_indices", [])

            if sample_indices:
                samples = []
                for sample_idx in sample_indices[:20]:  # Limit to 20 samples
                    if sample_idx in message_df.index:
                        row = message_df.loc[sample_idx]
                        sample_data = {
                            "index": int(sample_idx),
                            "conversation_id": row.get("conversation_id", "N/A"),
                            "role": row.get("role", "N/A"),
                            "text_preview": self._truncate_text(
                                str(row.get("text_content", "")), max_length=300
                            ),
                            "text_full": str(row.get("text_content", "")),
                        }
                        # Add metric value if available
                        metric_name = rec.get("metric_name")
                        if metric_name and metric_name in row:
                            sample_data["metric_value"] = self._format_metric_value(
                                row[metric_name]
                            )
                        samples.append(sample_data)
                rec_copy["samples"] = samples
            else:
                rec_copy["samples"] = []

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
        if isinstance(value, float):
            return f"{value:.2f}"
        elif isinstance(value, bool):
            return "Yes" if value else "No"
        return str(value)

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

            # Create a cleaner title
            col_title = col.replace("text_content_", "").replace("_", " ").title()

            # Create histogram
            fig = go.Figure(
                data=[
                    go.Histogram(
                        x=message_df[col].dropna(),
                        nbinsx=30,
                        marker_color="#d4a574",
                        opacity=0.85,
                    )
                ]
            )

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
            )

            charts.append(
                {
                    "id": f"chart_{chart_count}",
                    "title": col_title,
                    "data": json.dumps(fig.data, cls=PlotlyJSONEncoder),
                    "layout": json.dumps(fig.layout, cls=PlotlyJSONEncoder),
                }
            )
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
        max_anomaly_charts = 5  # Limit anomaly charts

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
            if chart_count >= max_anomaly_charts:
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

        # Add quality score distribution if available
        quality_cols = [col for col in message_df.columns if "quality_score" in col]
        if quality_cols and chart_count < max_anomaly_charts:
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

        return charts


class PlotlyJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for Plotly objects."""

    def default(self, obj):
        """Encode Plotly objects to JSON-serializable format."""
        # Handle numpy types
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()

        # Handle Plotly objects
        if hasattr(obj, "to_plotly_json"):
            return obj.to_plotly_json()

        return super().default(obj)
