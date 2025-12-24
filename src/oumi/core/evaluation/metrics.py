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

from collections.abc import Callable

import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import f1_score as sklearn_f1_score
from sklearn.utils import resample

from oumi.core.types.exceptions import InvalidParameterValueError, MissingParameterError


class Metric:
    """Base class for all metrics."""

    name: str | None = ""
    """The name of the metric."""

    value: float | None = None
    """The value of the metric."""

    ci: float | None = None
    """The confidence interval of the metric."""

    def __init__(
        self,
        name: str | None = "",
        value: float | None = None,
        ci: float | None = None,
    ) -> None:
        """Initialize a Metric instance."""
        self.name = name
        self.value = value
        self.ci = ci

    def __str__(self):
        """Return a string representation of the Metric instance."""
        return f"Metric(name={self.name}, value={self.value}, ci={self.ci})"


def bootstrap(
    y_true: list[int],
    y_pred: list[int],
    metric_fn: Callable[..., float],
    alpha: float = 0.95,
    n_iter: int = 1000,
    sample_prop=1.0,
) -> tuple[float, float]:
    """Perform bootstrap resampling to calculate confidence intervals for a metric.

    Args:
        y_true (list): True labels.
        y_pred (list): Predicted labels.
        metric_fn (callable): A function that computes a performance metric. The
            required signature for this function is:
            `metric_fn(y_true: list[int], y_pred: list[int]) -> float`.
        alpha (float): Confidence level (default is 0.95 for 95% confidence interval).
        n_iter (int): Number of bootstrap iterations (default is 1000).
        sample_prop (float): Proportion of the data to sample in each iteration
            (default 1.0, meaning same size as original).

    Returns:
        tuple: Midpoint and half-width of the confidence interval.
    """
    # Ensure that our inputs are correct.
    if not y_true or not y_pred:
        raise MissingParameterError(
            "`y_true` and `y_pred` must not be empty. "
            f"Got y_true length={len(y_true) if y_true else 0}, "
            f"y_pred length={len(y_pred) if y_pred else 0}."
        )
    if len(y_true) != len(y_pred):
        raise InvalidParameterValueError(
            f"`y_true` and `y_pred` must have the same length. "
            f"Got y_true length={len(y_true)}, y_pred length={len(y_pred)}."
        )
    if not (0 < alpha < 1):
        raise InvalidParameterValueError(
            f"`alpha` must be between 0 and 1 (exclusive). Got alpha={alpha}."
        )
    if n_iter <= 0:
        raise InvalidParameterValueError(
            f"`n_iter` must be a positive integer. Got n_iter={n_iter}."
        )
    if not (0 < sample_prop <= 1):
        raise InvalidParameterValueError(
            f"`sample_prop` must be between 0 (exclusive) and 1 (inclusive). "
            f"Got sample_prop={sample_prop}."
        )

    # Combine the true values and predictions into paired data points.
    data = list(zip(y_true, y_pred))

    # Determine the sample size for each bootstrap iteration.
    n_size = int(len(data) * sample_prop)

    # Run bootstrap.
    scores = []  # Metric scores calculated during the bootstrap iterations.
    for _ in range(n_iter):
        samples = resample(data, n_samples=n_size)
        if not samples:  # samples should never be None.
            continue
        labels = [sample[0] for sample in samples]
        predictions = [sample[1] for sample in samples]
        score = metric_fn(y_true=labels, y_pred=predictions)
        scores.append(score)

    # Calculate the confidence interval.
    lower_percentile = ((1.0 - alpha) / 2.0) * 100
    upper_percentile = (alpha + ((1.0 - alpha) / 2.0)) * 100

    lower_bound = max(0.0, float(np.percentile(scores, lower_percentile)))
    upper_bound = min(1.0, float(np.percentile(scores, upper_percentile)))
    midpoint = (upper_bound + lower_bound) / 2.0

    return midpoint, upper_bound - midpoint


def f1_score(
    y_true: list[int],
    y_pred: list[int],
    average: str = "binary",
    pos_label: int = 1,
    populate_ci: bool = True,
    alpha: float = 0.95,
    n_iter: int = 1000,
    sample_prop=1.0,
) -> Metric:
    """Calculate the F1 score and its confidence interval."""
    # Ensure that our inputs are correct.
    if not y_true or not y_pred:
        raise MissingParameterError(
            "`y_true` and `y_pred` must not be empty. "
            f"Got y_true length={len(y_true) if y_true else 0}, "
            f"y_pred length={len(y_pred) if y_pred else 0}."
        )
    if len(y_true) != len(y_pred):
        raise InvalidParameterValueError(
            f"`y_true` and `y_pred` must have the same length. "
            f"Got y_true length={len(y_true)}, y_pred length={len(y_pred)}."
        )
    valid_averages = ["micro", "macro", "samples", "weighted", "binary"]
    if average not in valid_averages:
        raise InvalidParameterValueError(
            f"Invalid value for `average`: '{average}'. "
            f"Must be one of: {valid_averages}."
        )
    if not (0 < alpha < 1):
        raise InvalidParameterValueError(
            f"`alpha` must be between 0 and 1 (exclusive). Got alpha={alpha}."
        )
    if n_iter <= 0:
        raise InvalidParameterValueError(
            f"`n_iter` must be a positive integer. Got n_iter={n_iter}."
        )
    if not (0 < sample_prop <= 1):
        raise InvalidParameterValueError(
            f"`sample_prop` must be between 0 (exclusive) and 1 (inclusive). "
            f"Got sample_prop={sample_prop}."
        )

    def f1_fn(y_true: list[int], y_pred: list[int]) -> float:
        return float(
            sklearn_f1_score(
                y_true=y_true,
                y_pred=y_pred,
                pos_label=pos_label,
                average=average,
            )
        )

    # Calculate F1 score
    if populate_ci:
        f1, ci = bootstrap(
            y_true=y_true,
            y_pred=y_pred,
            metric_fn=f1_fn,
            alpha=alpha,
            n_iter=n_iter,
            sample_prop=sample_prop,
        )
    else:
        f1 = f1_fn(
            y_true=y_true,
            y_pred=y_pred,
        )
        ci = None

    return Metric(name="F1 Score", value=f1, ci=ci)


def bacc_score(
    y_true: list[int],
    y_pred: list[int],
    populate_ci: bool = True,
    alpha: float = 0.95,
    n_iter: int = 1000,
    sample_prop=1.0,
) -> Metric:
    """Calculate the balanced accuracy score."""
    # Ensure that our inputs are correct.
    if not y_true or not y_pred:
        raise MissingParameterError(
            "`y_true` and `y_pred` must not be empty. "
            f"Got y_true length={len(y_true) if y_true else 0}, "
            f"y_pred length={len(y_pred) if y_pred else 0}."
        )
    if len(y_true) != len(y_pred):
        raise InvalidParameterValueError(
            f"`y_true` and `y_pred` must have the same length. "
            f"Got y_true length={len(y_true)}, y_pred length={len(y_pred)}."
        )
    if not (0 < alpha < 1):
        raise InvalidParameterValueError(
            f"`alpha` must be between 0 and 1 (exclusive). Got alpha={alpha}."
        )
    if n_iter <= 0:
        raise InvalidParameterValueError(
            f"`n_iter` must be a positive integer. Got n_iter={n_iter}."
        )
    if not (0 < sample_prop <= 1):
        raise InvalidParameterValueError(
            f"`sample_prop` must be between 0 (exclusive) and 1 (inclusive). "
            f"Got sample_prop={sample_prop}."
        )

    if populate_ci:
        # Calculate balanced accuracy score with bootstrap
        bacc, ci = bootstrap(
            y_true=y_true,
            y_pred=y_pred,
            metric_fn=balanced_accuracy_score,
            alpha=alpha,
            n_iter=n_iter,
            sample_prop=sample_prop,
        )
    else:
        bacc = balanced_accuracy_score(
            y_true=y_true,
            y_pred=y_pred,
        )
        ci = None

    return Metric(name="Balanced Accuracy", value=bacc, ci=ci)
