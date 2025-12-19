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
    # llm_compressor methods
    "llmc_W4A16",
    "llmc_W4A16_ASYM",
    "llmc_W8A16",
    "llmc_W8A8_INT",
    "llmc_W8A8_FP8",
    "llmc_FP8_BLOCK",
    # BitsAndBytes methods
    "bnb_4bit",
    "bnb_8bit",
    # Legacy AWQ aliases (map to llm_compressor methods)
    "awq_q4_0",
    "awq_q4_1",
    "awq_q8_0",
]

# Mapping from legacy AWQ method names to new llm_compressor methods
METHOD_ALIASES = {
    "awq_q4_0": "llmc_W4A16",
    "awq_q4_1": "llmc_W4A16_ASYM",
    "awq_q8_0": "llmc_W8A16",
}


def resolve_method_alias(method: str) -> str:
    """Resolve a method name, converting legacy aliases to canonical names."""
    return METHOD_ALIASES.get(method, method)

# Mapping from method to llm_compressor modifier and scheme
LLMC_METHOD_CONFIG = {
    "llmc_W4A16": {
        "modifier": "AWQModifier",
        "scheme": "W4A16",
        "requires_smoothquant": False,
    },
    "llmc_W4A16_ASYM": {
        "modifier": "AWQModifier",
        "scheme": "W4A16_ASYM",
        "requires_smoothquant": False,
    },
    "llmc_W8A16": {
        "modifier": "GPTQModifier",
        "scheme": "W8A16",
        "requires_smoothquant": False,
    },
    "llmc_W8A8_INT": {
        "modifier": "GPTQModifier",
        "scheme": "W8A8",
        "requires_smoothquant": True,
    },
    "llmc_W8A8_FP8": {
        "modifier": "QuantizationModifier",
        "scheme": "FP8_DYNAMIC",
        "requires_smoothquant": False,
    },
    "llmc_FP8_BLOCK": {
        "modifier": "QuantizationModifier",
        "scheme": "FP8_BLOCK",
        "requires_smoothquant": False,
    },
}

# Supported output formats
SUPPORTED_OUTPUT_FORMATS = ["safetensors"]


# Size units for formatting
SIZE_UNITS = ["B", "KB", "MB", "GB", "TB", "PB"]
