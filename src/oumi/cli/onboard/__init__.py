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

"""Onboard wizard module for customer onboarding."""

from .cache import (
    compute_file_hash,
    display_cache_summary,
    get_cache_path,
    load_wizard_cache,
    open_cache_for_editing,
    prompt_cache_action,
    save_wizard_cache,
)
from .cli import analyze, generate, templates, wizard
from .dataclasses import (
    GOAL_CHOICES,
    INPUT_FORMATS,
    JUDGE_TYPE_CHOICES,
    SUPPORTED_EXTENSIONS,
    SYNTH_GOAL_CHOICES,
    TASK_TYPES,
    InputSpec,
    OutputSpec,
    TaskSpec,
    WizardState,
)
from .helpers import (
    analyze_file_purposes,
    analyze_task_from_files,
    detect_files_in_directory,
    detect_input_source,
    fallback_input_detection,
    generate_system_prompt,
    suggest_quality_criteria,
)
from .wizard_steps import (
    wizard_step_generate,
    wizard_step_inputs,
    wizard_step_outputs,
    wizard_step_task,
)

__all__ = [
    # CLI commands
    "analyze",
    "generate",
    "templates",
    "wizard",
    # Dataclasses
    "InputSpec",
    "OutputSpec",
    "TaskSpec",
    "WizardState",
    # Constants
    "GOAL_CHOICES",
    "INPUT_FORMATS",
    "JUDGE_TYPE_CHOICES",
    "SUPPORTED_EXTENSIONS",
    "SYNTH_GOAL_CHOICES",
    "TASK_TYPES",
    # Cache functions
    "compute_file_hash",
    "display_cache_summary",
    "get_cache_path",
    "load_wizard_cache",
    "open_cache_for_editing",
    "prompt_cache_action",
    "save_wizard_cache",
    # Helper functions
    "analyze_file_purposes",
    "analyze_task_from_files",
    "detect_files_in_directory",
    "detect_input_source",
    "fallback_input_detection",
    "generate_system_prompt",
    "suggest_quality_criteria",
    # Wizard steps
    "wizard_step_generate",
    "wizard_step_inputs",
    "wizard_step_outputs",
    "wizard_step_task",
]
