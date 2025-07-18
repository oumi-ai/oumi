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

# AWQ method to GGUF type mapping
AWQ_TO_GGUF_METHOD_MAP = {
    "awq_q4_0": "q4_0",
    "awq_q4_1": "q4_1",
    "awq_q8_0": "q8_0",
    "awq_f16": "f16",
}

# GGUF quantization type constants
GGUF_QUANTIZATION_MAP = {
    "q4_0": 2,  # LLAMA_FTYPE_MOSTLY_Q4_0
    "q4_1": 3,  # LLAMA_FTYPE_MOSTLY_Q4_1
    "q5_0": 8,  # LLAMA_FTYPE_MOSTLY_Q5_0
    "q5_1": 9,  # LLAMA_FTYPE_MOSTLY_Q5_1
    "q8_0": 7,  # LLAMA_FTYPE_MOSTLY_Q8_0
}

# Size units for formatting
SIZE_UNITS = ["B", "KB", "MB", "GB", "TB", "PB"]

# AWQ configuration defaults
AWQ_DEFAULTS = {
    "calibration_dataset": "pileval",
    "calibration_split": "train",
    "calibration_text_column": "text",
    "max_calibration_seq_len": 512,
    "duo_scaling": True,
    "apply_clip": True,
    "n_parallel_calib_samples": None,
}

# Model size estimates (in bytes)
MODEL_SIZE_ESTIMATES = {
    "tinyllama-1.1b": 2_200_000_000,  # TinyLlama 1.1B ~2.2GB in fp16
}

# Common file chunk size for processing
CHUNK_SIZE = 1024 * 1024  # 1MB chunks

# Mock model sizes for simulation mode
MOCK_MODEL_SIZES = {
    "small": 30 * 1024 * 1024,  # 30MB for small models
    "7b": 14 * 1024 * 1024 * 1024,  # 14GB for 7B models (fp16)
    "13b": 26 * 1024 * 1024 * 1024,  # 26GB for 13B models
    "70b": 140 * 1024 * 1024 * 1024,  # 140GB for 70B models
    "default": 100 * 1024 * 1024,  # 100MB default
}

# Common file extensions for model files
MODEL_FILE_EXTENSIONS = [".safetensors", ".bin", ".pth"]

# Tokenizer files to copy during conversion
TOKENIZER_FILES = [
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "generation_config.json",
]

# GGUF file format constants
GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3
GGUF_TYPE_STRING = 8
GGUF_TYPE_FLOAT32 = 0
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_ARRAY = 9
