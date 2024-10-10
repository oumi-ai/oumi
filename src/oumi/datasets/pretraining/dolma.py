from oumi.core.datasets import BasePretrainingIterableDataset
from oumi.core.registry import register_dataset


@register_dataset("allenai/dolma")
class DolmaDataset(BasePretrainingIterableDataset):
    """Dolma: A dataset of 3 trillion tokens from diverse web content.

    Dolma is a large-scale dataset containing approximately 3 trillion tokens
    sourced from various web content, academic publications, code, books, and
    encyclopedic materials. It is designed for language modeling tasks and
    casual language model training.

    The dataset is available in multiple versions, with v1.7 being the latest
    release used to train OLMo 7B-v1.7. It includes data from sources such as
    Common Crawl, Refined Web, StarCoder, C4, Reddit, Semantic Scholar, arXiv,
    StackExchange, and more.

    For more information, see:
    - Manuscript and Data Sheet: https://arxiv.org/abs/2402.00159
    - GitHub project: https://github.com/allenai/dolma
    - Hugging Face Hub: https://huggingface.co/datasets/allenai/dolma

    Note:
        The dataset is released under the ODC-BY license. Users are bound by
        the license agreements and terms of use of the original data sources.

    Data Fields:
        The specific data fields are not provided in the README.
        - id (str): Unique identifier for the data entry.
        - text (str): The main content of the data entry.
        - added (str, optional): Timestamp indicating when the entry was added
        to the dataset.
        - created (str, optional): Timestamp indicating when the original content
        was created.
        - source (str, optional): Information about the origin or source of the
        data.

    Citation:
        @article{dolma,
            title = {{Dolma: an Open Corpus of Three Trillion Tokens for Language Model
              Pretraining Research}},
            author={
                Luca Soldaini and Rodney Kinney and Akshita Bhagia and Dustin Schwenk
                and David Atkinson and Russell Authur and Ben Bogin and Khyathi Chandu
                and Jennifer Dumas and Yanai Elazar and Valentin Hofmann
                and Ananya Harsh Jha and Sachin Kumar and Li Lucy and Xinxi Lyu
                and Nathan Lambert and Ian Magnusson and Jacob Morrison
                and Niklas Muennighoff and Aakanksha Naik and Crystal Nam
                and Matthew E. Peters and Abhilasha Ravichander and Kyle Richardson
                and Zejiang Shen and Emma Strubell and Nishant Subramani
                and Oyvind Tafjord and Pete Walsh and Luke Zettlemoyer
                and Noah A. Smith and Hannaneh Hajishirzi and Iz Beltagy
                and Dirk Groeneveld and Jesse Dodge and Kyle Lo
            },
        year = {2024},
        journal={arXiv preprint},
        }
    """
