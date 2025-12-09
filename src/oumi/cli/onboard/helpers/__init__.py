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
from .input_helpers import (
    clarify_template_variables,
    detect_input_source,
    fallback_input_detection,
)
from .output_helpers import merge_criteria, suggest_quality_criteria
from .task_helpers import (
    ExtractedUseCase,
    analyze_task_from_files,
    derive_task_name,
    detect_all_elements,
    detect_labeled_examples,
    detect_unlabeled_prompts,
    extract_evaluation_criteria,
    extract_use_case_from_documents,
    extract_user_prompt_template,
    generate_system_prompt,
    identify_seed_columns,
    infer_task_type,
)

__all__ = [
    "analyze_file_purposes",
    "analyze_task_from_files",
    "derive_task_name",
    "detect_all_elements",
    "detect_files_in_directory",
    "detect_input_source",
    "detect_labeled_examples",
    "detect_unlabeled_prompts",
    "extract_evaluation_criteria",
    "extract_use_case_from_documents",
    "extract_user_prompt_template",
    "ExtractedUseCase",
    "fallback_input_detection",
    "clarify_template_variables",
    "generate_system_prompt",
    "identify_seed_columns",
    "infer_task_type",
    "merge_criteria",
    "suggest_quality_criteria",
]
