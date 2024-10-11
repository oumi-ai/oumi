from oumi.core.datasets import BasePretrainingIterableDataset
from oumi.core.registry import register_dataset


@register_dataset("EleutherAI/pile")
class PileV1Dataset(BasePretrainingIterableDataset):
    """The Pile: An 825 GiB diverse, open source language modeling dataset.

    The Pile is a large-scale English language dataset consisting of 22 smaller,
    high-quality datasets combined together. It is designed for training large
    language models and supports various natural language processing tasks.

    Key Features:
    - 825 GiB of diverse text data
    - Primarily in English
    - Supports text generation and fill-mask tasks
    - Includes various subsets like enron_emails, europarl, free_law, etc.

    Homepage: https://pile.eleuther.ai/
    HuggingFace hub: https://huggingface.co/datasets/EleutherAI/pile

    Data Fields:
    - text (str): The main text content.
    - meta (dict): Metadata about the instance, including 'pile_set_name'.

    Subsets:
        - all
        - enron_emails
        - europarl
        - free_law
        - hacker_news
        - nih_exporter
        - pubmed
        - pubmed_central
        - ubuntu_irc
        - uspto
        - github

    Dataset Splits:
    - train
    - validation
    - test

    References:
        @article{gao2020pile,
            title={The Pile: An 800{GB} dataset of diverse text for language modeling},
            author={Gao, Leo and Biderman, Stella and Black, Sid and Golding,
                Laurence and Hoppe, Travis and Foster, Charles and Phang, Jason and He,
                Horace and Thite, Anish and Nabeshima, Noa and others},
            journal={arXiv preprint arXiv:2101.00027},
            year={2020}
        }

        @article{biderman2022datasheet,
            title={Datasheet for the pile},
            author={Biderman, Stella and Bicheno, Kieran and Gao, Leo},
            journal={arXiv preprint arXiv:2201.07311},
            year={2022}
        }

    Note:
        This dataset contains text from various sources and may include
        personal or sensitive information. Users should consider potential biases
        and limitations when using this dataset.
    """
