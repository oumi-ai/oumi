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

"""Statistical outlier analyzer for detecting outliers in numeric columns."""

from typing import Optional

import numpy as np
import pandas as pd

from oumi.core.analyze.sample_analyzer import SampleAnalyzer
from oumi.core.registry import register_sample_analyzer


@register_sample_analyzer("statistical_outlier")
class StatisticalOutlierAnalyzer(SampleAnalyzer):
    """Analyzer that detects statistical outliers in numeric columns.

    Uses z-score and IQR methods to identify outliers.
    """

    def __init__(
        self,
        *,
        zscore_threshold: float = 3.0,
        iqr_multiplier: float = 1.5,
        columns: Optional[list[str]] = None,
        tokenizer=None,
    ):
        """Initialize the StatisticalOutlierAnalyzer.

        Args:
            zscore_threshold: Z-score threshold for outlier detection.
            iqr_multiplier: IQR multiplier for outlier detection.
            columns: Specific columns to analyze. If None, analyzes all numeric columns.
            tokenizer: Optional tokenizer (not used by this analyzer).
        """
        self.zscore_threshold = zscore_threshold
        self.iqr_multiplier = iqr_multiplier
        self.columns = columns

    def analyze_sample(
        self,
        df: pd.DataFrame,
        schema: Optional[dict] = None,
    ) -> pd.DataFrame:
        """Analyze numeric columns for outliers.

        Args:
            df: Input DataFrame with numeric columns.
            schema: Column schema (optional, will auto-detect numeric columns).

        Returns:
            DataFrame with added columns per numeric column:
            - {column}_zscore: Z-score value
            - {column}_percentile: Percentile rank (0-100)
            - {column}_is_outlier_zscore: Outlier by z-score method
            - {column}_is_outlier_iqr: Outlier by IQR method
        """
        result_df = df.copy()

        # Determine which columns to analyze
        if self.columns:
            numeric_columns = [c for c in self.columns if c in df.columns]
        else:
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

        if not numeric_columns:
            return result_df  # No numeric columns to analyze

        for column in numeric_columns:
            values = df[column].astype(float)

            # Z-score
            mean = values.mean()
            std = values.std()
            if std > 0:
                zscores = (values - mean) / std
            else:
                zscores = pd.Series(0.0, index=values.index)

            result_df[f"{column}_zscore"] = zscores
            result_df[f"{column}_is_outlier_zscore"] = zscores.abs() > self.zscore_threshold

            # Percentile rank
            result_df[f"{column}_percentile"] = values.rank(pct=True) * 100

            # IQR method
            q1 = values.quantile(0.25)
            q3 = values.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - self.iqr_multiplier * iqr
            upper_bound = q3 + self.iqr_multiplier * iqr
            result_df[f"{column}_is_outlier_iqr"] = (values < lower_bound) | (
                values > upper_bound
            )

        return result_df, {}
