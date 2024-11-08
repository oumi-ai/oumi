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

    def __init__(self, num_datasets, metric_name="balanced_accuracy"):
        """Initialize the MfuTrainerCallback.

        Args:
            metric_name: Name of the metric to aggregate
        """
        self._num_datasets = num_datasets
        self._metric_name = metric_name
        self._agg_metric_name = "eval_avg_" + self._metric_name
        self._metric_values = {}

    def on_evaluate(
        self,
        args: Union[transformers.TrainingArguments, TrainingParams],
        state: Optional[transformers.TrainerState] = None,
        control: Optional[transformers.TrainerControl] = None,
        **kwargs,
    ):
        try:
            log = state.log_history[-1]
            for k in log:
                if self._metric_name in k and "eval" in k:
                    self._metric_values[k] = log[k]
                    break

            if len(self._metric_values.keys()) == self._num_datasets:
                mean = sum(self._metric_values.values()) / len(
                    self._metric_values.values()
                )
                kwargs[_METRICS_KWARG][self._agg_metric_name] = mean
        except:
            return

    def on_log(
        self,
        args: Union[transformers.TrainingArguments, TrainingParams],
        state: Optional[transformers.TrainerState] = None,
        control: Optional[transformers.TrainerControl] = None,
        **kwargs,
    ):
        if len(self._metric_values.keys()) == self._num_datasets:
            mean = sum(self._metric_values.values()) / len(self._metric_values.values())
            kwargs[_LOGS_KWARG][self._agg_metric_name] = mean
            logger.info(
                f"{self._agg_metric_name}: {mean}, {len(self._metric_values.keys())}, {str(self._metric_values)}"
            )
            self._metric_values = {}
