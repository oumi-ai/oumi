from oumi.core.datasets.base_dpo_processor import BaseExperimentalDpoPreprocessor
from oumi.core.registry import register_dataset


@register_dataset("mlabonne/orpo-dpo-mix-40k")
class ORPOExperimentalDpoPreprocessor(BaseExperimentalDpoPreprocessor):
    pass
