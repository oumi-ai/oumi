from oumi.core.datasets import BasePretrainingDataset
from oumi.core.registry import register_dataset


@register_dataset("PleIAs/YouTube-Commons")
class YouTubeCommonsDataset(BasePretrainingDataset):
    """YouTube-Commons Dataset.

    This dataset is a collection of audio transcripts from 2,063,066 videos shared on
    YouTube under a CC-By license. It contains 22,709,724 original and automatically
    translated transcripts from 3,156,703 videos (721,136 individual channels),
    representing nearly 45 billion words.

    The corpus is multilingual, with a majority of English-speaking content (71%) for
    original languages. Automated translations are provided for nearly all videos in
    English, French, Spanish, German, Russian, Italian, and Dutch.

    This dataset aims to expand the availability of conversational data for research
    in AI, computational social science, and digital humanities.

    See Also:
        - Hugging Face Hub: https://huggingface.co/datasets/PleIAs/YouTube-Commons

    Data Fields:
        - video_id: string
        - video_link: string
        - title: string
        - text: string
        - channel: string
        - channel_id: string
        - date: string
        - license: string
        - original_language: string
        - source_language: string
        - transcription_language: string
        - word_count: int64
        - character_count: int64

    Note:
        The text can be used for training models and republished for reproducibility
        purposes. In accordance with the CC-By license, every YouTube channel is fully
        credited.

    Note:
        This dataset is licensed under CC-BY-4.0.
    """

    default_dataset = "PleIAs/YouTube-Commons"
