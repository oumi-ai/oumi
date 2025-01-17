"""OpenO1 synthetic reasoning SFT dataset."""

from oumi.core.registry.registry import register_dataset
from oumi.datasets.sft.prompt_response import PromptResponseDataset


@register_dataset("O1-OPEN/OpenO1-SFT")
class OpenO1SFTDataset(PromptResponseDataset):
    """Synthetic reasoning SFT dataset."""

    default_dataset = "O1-OPEN/OpenO1-SFT"

    def __init__(
        self,
        **kwargs,
    ) -> None:
        """Initializes a dataset for OpenO1SFT from HuggingFace."""
        super().__init__(
            hf_dataset_path="O1-OPEN/OpenO1-SFT",
            prompt_column="instruction",
            response_column="output",
            **kwargs,
        )
