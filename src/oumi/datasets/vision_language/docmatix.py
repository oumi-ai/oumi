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


from oumi.core.registry import register_dataset
from oumi.datasets.vision_language.the_cauldron import TheCauldronDataset


@register_dataset("HuggingFaceM4/Docmatix")
class DocmatixDataset(TheCauldronDataset):
    """Dataset class for the `HuggingFaceM4/Docmatix` dataset.

    The dataset has the same data layout and format as `HuggingFaceM4/the_cauldron`
    (hence it's defined as a sub-class) but the underlying data is different.
    Unlike `HuggingFaceM4/the_cauldron`, the dataset contains many multi-image examples,
    and fewer subsets.

    Be aware that 'HuggingFaceM4/Docmatix' is a very large dataset (~0.5TB) that
    requires a lot of Internet bandwidth to download, and a lot of disk space to store,
    so only use it if you know what you're doing.

    Using the 'Docmatix' dataset in Oumi should become easier after streaming support
    is supported for custom Oumi datasets (OPE-1021).
    """

    default_dataset = "HuggingFaceM4/Docmatix"
