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

"""Helper functions for the onboard wizard."""

from .file_helpers import detect_files_in_directory, analyze_file_purposes
from .input_helpers import detect_input_source, fallback_input_detection
from .output_helpers import suggest_quality_criteria
from .task_helpers import (
    ExtractedUseCase,
    analyze_task_from_files,
    derive_task_name,
    extract_use_case_from_documents,
    generate_system_prompt,
    infer_task_type,
)

__all__ = [
    "analyze_file_purposes",
    "analyze_task_from_files",
    "derive_task_name",
    "detect_files_in_directory",
    "detect_input_source",
    "extract_use_case_from_documents",
    "ExtractedUseCase",
    "fallback_input_detection",
    "generate_system_prompt",
    "infer_task_type",
    "suggest_quality_criteria",
]
