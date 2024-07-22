from typing import Callable, Optional

from lema.core.registry import REGISTRY
from lema.core.types import TrainingParams


def build_metrics_function(config: TrainingParams) -> Optional[Callable]:
    """Builds the metrics function."""
    metrics_function = None
    if config.metrics_function:
        metrics_function = REGISTRY.get_metrics_function(config.metrics_function)
        if not metrics_function:
            raise KeyError(
                f"metrics_function `{config.metrics_function}` "
                "was not found in the registry."
            )

    return metrics_function
