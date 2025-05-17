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

import abc
from typing import Optional

import PIL.Image
import transformers


class BaseImageProcessor(abc.ABC):
    """Base class for oumi image processors."""

    @abc.abstractmethod
    def __call__(
        self,
        images: list[PIL.Image.Image],
        *,
        return_tensors: Optional[str] = "pt",
    ) -> transformers.BatchFeature:
        """Extracts image features.

        Args:
            images: A list of input images.
            return_tensors: The format of returned tensors.

        Returns:
            transformers.BatchFeature: The model-specific input features.
        """
        raise NotImplementedError
