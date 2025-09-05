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


from oumi.core.datasets import BaseExperimentalKtoDataset
from oumi.core.registry import register_dataset


@register_dataset("trl-lib/kto-mix-14k")
class KtoMix14kDataset(BaseExperimentalKtoDataset):
    """Preprocess the KTO dataset.

    A KTO-formatted version of argilla/dpo-mix-7k designed for Kahneman-Tversky
    Optimization training. This dataset provides binary preference data for
    training language models with human preferences.

    Data Fields:
        - prompt: List of message dictionaries with a single user message
          Example: [{"content": "Question text", "role": "assistant"}]
        - completion: List of message dictionaries with a single assistant message
          Example: [{"content": "Answer text", "role": "assistant"}]
        - label: boolean (True for desirable, False for undesirable)

    See Also:
        For more information on how to use this dataset, refer to:
        - Huggingface hub: https://huggingface.co/datasets/trl-lib/kto-mix-14k
        - KTO documentation: https://huggingface.co/docs/trl/main/en/kto_trainer
    """

    default_dataset = "trl-lib/kto-mix-14k"
