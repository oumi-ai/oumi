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

from trl import DPOTrainer


class TrlDpoTrainer(DPOTrainer):
    """Light wrapper around the DPOTrainer to handle vision models."""

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        """Initializes the TrlDpoTrainer."""
        super().__init__(*args, **kwargs)

    def _prepare_dataset(self, dataset, processing_class, args, dataset_name):
        """Prepare the dataset for training."""
        # Skip the dataset preparation since the dataset is already prepared.
        return dataset
