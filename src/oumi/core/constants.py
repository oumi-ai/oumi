# Copyright 2025 - Oumi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Common constants used in oumi codebase."""

from typing import Final

# Tokens with this label value don't contribute to the loss computation.
# For example, this can be `PAD`, or image tokens. `-100` is the PyTorch convention.
# Refer to the `ignore_index` parameter of `torch.nn.CrossEntropyLoss()`
# for more details.
LABEL_IGNORE_INDEX: Final[int] = -100
