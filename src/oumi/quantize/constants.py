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

"""Constants and mappings for quantization methods."""

# Supported quantization methods
SUPPORTED_METHODS = [
    # AWQ methods
    "awq_q4_0",
    "awq_q4_1",
    "awq_q8_0",
    "awq_f16",
    # BitsAndBytes methods
    "bnb_4bit",
    "bnb_8bit",
    # Direct GGUF methods
    "q4_0",
    "q4_1",
    "q5_0",
    "q5_1",
    "q8_0",
    "f16",
    "f32",
]

# Supported output formats
SUPPORTED_OUTPUT_FORMATS = ["gguf", "safetensors", "pytorch"]


# Size units for formatting
SIZE_UNITS = ["B", "KB", "MB", "GB", "TB", "PB"]


# Common file extensions for model files
MODEL_FILE_EXTENSIONS = [".safetensors", ".bin", ".pth"]
