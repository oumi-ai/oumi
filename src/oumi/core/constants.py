"""Common constants used in oumi codebase."""

from typing import Final

# Tokens with this label value don't contribute to the loss computation.
# For example, this can be `PAD`, or image tokens. `-100` is the PyTorch convention.
# Refer to the `ignore_index` parameter of `torch.nn.CrossEntropyLoss()`
# for more details.
LABEL_IGNORE_INDEX: Final[int] = -100
