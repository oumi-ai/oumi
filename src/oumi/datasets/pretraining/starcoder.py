from oumi.core.datasets import BasePretrainingDataset
from oumi.core.registry import register_dataset


@register_dataset("bigcode/starcoderdata")
class StarCoderDataset(BasePretrainingDataset):
    """StarCoder Training Dataset used for training StarCoder and StarCoderBase models.

    This dataset contains 783GB of code in 86 programming languages, including 54GB
    of GitHub Issues, 13GB of Jupyter notebooks in scripts and text-code pairs, and
    32GB of GitHub commits, totaling approximately 250 Billion tokens.

    The dataset is a cleaned, decontaminated, and near-deduplicated version of
    The Stack dataset, with PII removed. It includes various programming languages,
    GitHub issues, Jupyter Notebooks, and GitHub commits.

    Data Fields:
        id: str
        content: str
        max_stars_repo_path: str
        max_stars_repo_name: int
        max_stars_count: str

    See Also:
        - Huggingface hub: https://huggingface.co/datasets/bigcode/starcoderdata

    Note:
        GitHub issues, GitHub commits, and Jupyter notebooks subsets have different
        columns from the rest. It's recommended to load programming languages separately
        from these categories:
        - jupyter-scripts-dedup-filtered
        - jupyter-structured-clean-dedup
        - github-issues-filtered-structured
        - git-commits-cleaned

    Subsets (See dataset for full list):
        - python
        - javascript
        - assembly
        - awk
        - git-commits-cleaned
        - github-issues-filtered-structured
        - ...

    Warning:
        Not all subsets have the same format, in particular:
        - jupyter-scripts-dedup-filtered
        - jupyter-structured-clean-dedup
        - github-issues-filtered-structured
        - git-commits-cleaned
    """

    default_dataset = "bigcode/starcoderdata"
    _data_column = "content"
