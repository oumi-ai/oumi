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

"""Types module for the Oumi (Open Universal Machine Intelligence) library.

This module provides custom types and exceptions used throughout the Oumi framework.

Exceptions:
    :class:`OumiError`: Base exception for all Oumi errors.
    :class:`ConfigurationError`: Exception raised for invalid configuration.
    :class:`ModelError`: Exception raised for model-related errors.
    :class:`ModelNotFoundError`: Exception raised when a model cannot be found.
    :class:`DatasetError`: Exception raised for dataset-related errors.
    :class:`InferenceError`: Exception raised for errors during inference.
    :class:`TrainingError`: Exception raised for errors during training.
    :class:`HardwareException`: Exception raised for hardware-related errors.

Note:
    This module is part of the core Oumi framework and is used across various
    components to ensure consistent error handling and type definitions.
"""

from oumi.core.types.conversation import (
    ContentItem,
    ContentItemCounts,
    Conversation,
    FinishReason,
    Message,
    Role,
    TemplatedMessage,
    Type,
)
from oumi.core.types.tool_call import (
    FunctionCall,
    FunctionDefinition,
    JSONSchema,
    ToolCall,
    ToolDefinition,
    ToolResult,
    ToolType,
)
from oumi.exceptions import (
    ConfigurationError,
    DatasetError,
    HardwareException,
    InferenceError,
    ModelError,
    ModelNotFoundError,
    OumiError,
    TrainingError,
)

__all__ = [
    # Exceptions
    "ConfigurationError",
    "DatasetError",
    "HardwareException",
    "InferenceError",
    "ModelError",
    "ModelNotFoundError",
    "OumiError",
    "TrainingError",
    # Conversation types
    "ContentItem",
    "ContentItemCounts",
    "Conversation",
    "FinishReason",
    "FunctionCall",
    "FunctionDefinition",
    "JSONSchema",
    "Message",
    "Role",
    "TemplatedMessage",
    "ToolCall",
    "ToolDefinition",
    "ToolResult",
    "ToolType",
    "Type",
]
