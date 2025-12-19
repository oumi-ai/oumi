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

"""Format converters for transforming data formats to Conversation objects.

This module provides converters that transform various dataset formats
(Oumi, Alpaca, ShareGPT, Langfuse, etc.) into the standard Conversation format.
"""

from oumi.core.converters.format_converters import (
    ConverterFn,
    auto_detect_converter,
    convert_alpaca,
    convert_conversations,
    convert_langchain,
    convert_langfuse,
    convert_opentelemetry,
    convert_oumi,
    convert_sharegpt,
    create_alpaca_converter,
    get_converter,
)

__all__ = [
    "ConverterFn",
    "auto_detect_converter",
    "convert_alpaca",
    "convert_conversations",
    "convert_langchain",
    "convert_langfuse",
    "convert_opentelemetry",
    "convert_oumi",
    "convert_sharegpt",
    "create_alpaca_converter",
    "get_converter",
]
