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

from oumi.core.datasets import BaseKtoDataset
from oumi.core.registry import register_dataset


@register_dataset("mlabonne/kto-mix-40k")
class KtoMix40kDataset(BaseKtoDataset):
    """Preprocess the KTO dataset.

    A dataset designed for KTO (Kahneman-Tversky Optimization) training.
    This dataset is a combination of high-quality datasets with binary feedback,
    including:
    - Capybara-Preferences (converted to binary)
    - distilabel-intel-orca-dpo-pairs (converted to binary)
    - ultrafeedback-binarized-preferences-cleaned
    - distilabel-math-preference-dpo (converted to binary)
    - toxic-dpo-v0.2 (converted to binary)
    - prm_dpo_pairs_cleaned (converted to binary)
    - truthy-dpo-v0.1 (converted to binary)

    Rule-based filtering was applied to remove 'gptisms' in the desirable answers.

    Data Fields:
        - source: string
        - prompt: string
        - response: string
        - label: boolean (True for desirable, False for undesirable)

    See Also:
        For more information on how to use this dataset, refer to:
        - Paper: https://arxiv.org/pdf/2402.01306
        - Huggingface hub: https://huggingface.co/docs/trl/main/en/kto_trainer
    """

    default_dataset = "mlabonne/kto-mix-40k" 