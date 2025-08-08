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

"""Kernel optimization utilities for the Oumi framework.

This module provides utilities for detecting and applying optimized kernels
from the HuggingFace kernels ecosystem to improve model performance.
"""

from oumi.core.kernels.detection import (
    is_flash_attn3_kernel_available,
    is_kernels_available,
)
from oumi.core.kernels.model_enhancer import enhance_model_with_kernels

__all__ = [
    "is_kernels_available",
    "is_flash_attn3_kernel_available",
    "enhance_model_with_kernels",
]
