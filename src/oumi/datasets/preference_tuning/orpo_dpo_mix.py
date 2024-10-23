from oumi.core.datasets import BaseExperimentalDpoDataset
from oumi.core.registry import register_dataset


@register_dataset("mlabonne/orpo-dpo-mix-40k")
class OrpoDpoMix40kDataset(BaseExperimentalDpoDataset):
    """Preprocess the ORPO dataset for DPO.

    A dataset designed for ORPO (Offline Reinforcement Learning for Preference
    Optimization) or DPO (Direct Preference Optimization) training.

    This dataset is a combination of high-quality DPO datasets, including:
    - Capybara-Preferences
    - distilabel-intel-orca-dpo-pairs
    - ultrafeedback-binarized-preferences-cleaned
    - distilabel-math-preference-dpo
    - toxic-dpo-v0.2
    - prm_dpo_pairs_cleaned
    - truthy-dpo-v0.1

    Rule-based filtering was applied to remove 'gptisms' in the chosen answers.

    Data Fields:
        - source: string
        - chosen: list of dictionaries with 'content' and 'role' fields
        - rejected: list of dictionaries with 'content' and 'role' fields
        - prompt: string
        - question: string

    See Also:
        For more information on how to use this dataset, refer to:
        - Blog post: https://huggingface.co/blog/mlabonne/orpo-llama-3
        - Huggingface hub: https://huggingface.co/datasets/mlabonne/orpo-dpo-mix-40k
    """

    default_dataset = "mlabonne/orpo-dpo-mix-40k"
