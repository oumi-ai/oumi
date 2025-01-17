"""Supervised fine-tuning datasets module."""

from oumi.datasets.sft.alpaca import AlpacaDataset
from oumi.datasets.sft.aya import AyaDataset
from oumi.datasets.sft.chatqa import ChatqaDataset, ChatqaTatqaDataset
from oumi.datasets.sft.chatrag_bench import ChatRAGBenchDataset
from oumi.datasets.sft.dolly import ArgillaDollyDataset
from oumi.datasets.sft.magpie import ArgillaMagpieUltraDataset, MagpieProDataset
from oumi.datasets.sft.openo1_sft import OpenO1SFTDataset
from oumi.datasets.sft.prompt_response import PromptResponseDataset
from oumi.datasets.sft.sft_jsonlines import TextSftJsonLinesDataset
from oumi.datasets.sft.ultrachat import UltrachatH4Dataset

__all__ = [
    "AlpacaDataset",
    "ArgillaDollyDataset",
    "ArgillaMagpieUltraDataset",
    "AyaDataset",
    "ChatqaDataset",
    "ChatqaTatqaDataset",
    "ChatRAGBenchDataset",
    "MagpieProDataset",
    "OpenO1SFTDataset",
    "PromptResponseDataset",
    "TextSftJsonLinesDataset",
    "UltrachatH4Dataset",
]
