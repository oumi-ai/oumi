"""Based on MFU from PaLM paper: https://arxiv.org/pdf/2204.02311."""

from typing import Optional, Union

import transformers

from oumi.core.callbacks.base_trainer_callback import BaseTrainerCallback
from oumi.core.configs import TrainingParams
from oumi.utils.logging import logger

_LOGS_KWARG = "logs"
_METRICS_KWARG = "metrics"


class AggregateMetricCallback(BaseTrainerCallback):
    """Trainer callback to calculate the MFU of the model during training.

    Should be compatible with all trainers that inherit from transformers.Trainer.
    """

    def __init__(
        self,
        num_datasets,
        metrics=["loss", "balanced_accuracy", "f1_score", "pr_auc", "roc_auc"],
    ):
        """Initialize the MfuTrainerCallback.

        Args:
            metric_name: Name of the metric to aggregate
        """
        self._num_datasets = num_datasets
        self._splits = ["eval", "test"]
        self._metrics = metrics
        self._metric_values = {}
        for s in self._splits:
            for m in self._metrics:
                self._metric_values[(m, s)] = {}

    def _get_agg_metric_name(self, metric_name: str, split="eval") -> str:
        return f"{split}_avg_{metric_name}"

    def on_evaluate(
        self,
        args: Union[transformers.TrainingArguments, TrainingParams],
        state: Optional[transformers.TrainerState] = None,
        control: Optional[transformers.TrainerControl] = None,
        **kwargs,
    ):
        log = state.log_history[-1]
        for s in self._splits:
            for m in self._metrics:
                for k in log:
                    key = (m, s)
                    if m in k and s in k:
                        self._metric_values[key][k] = log[k]
                        break
 
        for s in self._splits:
            for m in self._metrics:
                key = (m, s)
                metric_values = list(self._metric_values[key].values())
                if len(metric_values) == self._num_datasets:
                    mean = sum(metric_values) / len(metric_values)
                    agg_metric_name = self._get_agg_metric_name(m, s)
                    kwargs[_METRICS_KWARG][agg_metric_name] = mean
                    logger.info(
                        f"{agg_metric_name}: {mean}, {len(metric_values)}, {str(self._metric_values[key])}"
                    )

    def on_log(
        self,
        args: Union[transformers.TrainingArguments, TrainingParams],
        state: Optional[transformers.TrainerState] = None,
        control: Optional[transformers.TrainerControl] = None,
        **kwargs,
    ):
        for s in self._splits:
            for m in self._metrics:
                key = (m, s)
                metric_values = list(self._metric_values[key].values())
                if len(metric_values) == self._num_datasets:
                    mean = sum(metric_values) / len(metric_values)
                    agg_metric_name = self._get_agg_metric_name(m, s)
                    kwargs[_LOGS_KWARG][agg_metric_name] = mean
                    self._metric_values[key] = {}
