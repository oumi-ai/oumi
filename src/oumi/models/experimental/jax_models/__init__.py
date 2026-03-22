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

"""JAX Models - Unified platform for JAX-based language models.

Based on jax-llm-examples with integrated download, conversion, and inference.
"""

from oumi.models.experimental.jax_models.manager import JAXModelManager
from oumi.models.experimental.jax_models.registry import (
    JAXModelInfo,
    get_model_info,
    get_recommended_model,
    get_supported_models,
    list_supported_architectures,
)

__all__ = [
    "JAXModelManager",
    "get_supported_models",
    "get_model_info",
    "get_recommended_model",
    "list_supported_architectures",
    "JAXModelInfo",
]
