from oumi.core.datasets import BasePretrainingIterableDataset
from oumi.core.registry import register_dataset


@register_dataset("bigcode/the-stack")
class TheStackDataset(BasePretrainingIterableDataset):
    """A dataset containing over 6TB of permissively-licensed source code files.

    The Stack was created as part of the BigCode Project, an open scientific
    collaboration working on the responsible development of Large Language Models
    for Code (Code LLMs). It serves as a pre-training dataset for Code LLMs,
    enabling the synthesis of programs from natural language descriptions and
    other code snippets, and covers 358 programming languages.

    The dataset contains code in multiple natural languages, primarily found in
    comments and docstrings. It supports tasks such as code completion,
    documentation generation, and auto-completion of code snippets.

    See Also:
        - Huggingface hub: https://huggingface.co/datasets/bigcode/the-stack
        - Homepage: https://www.bigcode-project.org/
        - Repository: https://github.com/bigcode-project
        - Paper: https://arxiv.org/abs/2211.15533

    Data Fields:
        - content (string): The content of the file.
        - size (integer): Size of the uncompressed file.
        - lang (string): The programming language.
        - ext (string): File extension.
        - avg_line_length (float): The average line-length of the file.
        - max_line_length (integer): The maximum line-length of the file.
        - alphanum_fraction (float): The fraction of alphanumeric characters.
        - hexsha (string): Unique git hash of file.
        - max_{stars|forks|issues}_repo_path (string): Path to file in repo.
        - max_{stars|forks|issues}_repo_name (string): Name of repo.
        - max_{stars|forks|issues}_repo_head_hexsha (string): Hexsha of repo head.
        - max_{stars|forks|issues}_repo_licenses (string): Licenses in repository.
        - max_{stars|forks|issues}_count (integer): Number of stars/forks/issues.
        - max_{stars|forks|issues}_repo_{stars|forks|issues}_min_datetime (string):
          First timestamp of a stars/forks/issues event.
        - max_{stars|forks|issues}_repo_{stars|forks|issues}_max_datetime (string):
          Last timestamp of a stars/forks/issues event.
    """

    _data_column = "content"
