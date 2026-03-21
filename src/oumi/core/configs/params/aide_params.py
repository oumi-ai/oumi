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

"""Configuration parameters for AIDE agentic code optimization.

AIDE (AI-Driven Exploration) uses LLM-powered tree search to iteratively
generate, test, and refine code solutions. Unlike traditional hyperparameter
tuning (Optuna-based ``TuningParams``), AIDE operates in *code space* — it
can modify training configs, reward functions, evaluation logic, and full
training pipelines.

See Also:
    - :class:`oumi.core.configs.params.tuning_params.TuningParams`
    - :class:`oumi.core.configs.aide_config.AideConfig`
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from oumi.core.configs.params.base_params import BaseParams
from oumi.core.configs.params.telemetry_params import TelemetryParams


class AideOptimizationSurface(Enum):
    """The type of code artifact that AIDE optimizes.

    Each surface type changes the prompts and constraints given to the
    AIDE agent, guiding it to generate the appropriate kind of code.
    """

    CONFIG_SEARCH = "config_search"
    """Generate and modify Oumi TrainingConfig YAML parameters."""

    REWARD_FUNCTION = "reward_function"
    """Generate reward functions for GRPO/RLHF training."""

    EVAL_FUNCTION = "eval_function"
    """Generate custom evaluation functions."""

    FULL_PIPELINE = "full_pipeline"
    """Generate complete training pipeline scripts using Oumi's API."""


@dataclass
class AideLLMParams(BaseParams):
    """LLM configuration for AIDE's code generation or feedback evaluation.

    AIDE uses two separate LLM calls per step: one for code generation and one
    for evaluating the execution results. Each can use a different model.
    """

    model: str = "o4-mini"
    """The LLM model identifier.

    Supported providers are auto-detected from the model name:
    - ``gpt-*``, ``o*``: OpenAI
    - ``claude-*``: Anthropic
    - ``gemini-*``: Google Gemini
    - Others: OpenRouter
    """

    temperature: float = 0.5
    """Sampling temperature for the LLM. Higher values increase diversity."""


@dataclass
class AideSearchParams(BaseParams):
    """Tree search hyperparameters controlling AIDE's exploration strategy.

    AIDE uses a three-phase search policy:

    1. **Drafting**: Generate ``num_drafts`` initial solutions from scratch.
    2. **Debugging**: With probability ``debug_prob``, fix a buggy leaf node.
    3. **Improving**: Otherwise, improve the best-performing solution.
    """

    num_drafts: int = 5
    """Number of independent initial solutions to draft before improving.

    More drafts increase diversity but delay the improvement phase.
    """

    debug_prob: float = 0.5
    """Probability of attempting to debug a buggy node vs. improving the best.

    Set higher (e.g. 0.8) if generated code frequently has bugs.
    """

    max_debug_depth: int = 3
    """Maximum consecutive debug attempts on a single buggy branch.

    Prevents the agent from spending too long on an unfixable approach.
    """

    def __post_init__(self):
        """Validates search parameters."""
        if self.num_drafts < 1:
            raise ValueError(f"num_drafts must be >= 1, got {self.num_drafts}")
        if not 0.0 <= self.debug_prob <= 1.0:
            raise ValueError(
                f"debug_prob must be between 0.0 and 1.0, got {self.debug_prob}"
            )
        if self.max_debug_depth < 0:
            raise ValueError(
                f"max_debug_depth must be >= 0, got {self.max_debug_depth}"
            )


@dataclass
class AideExecParams(BaseParams):
    """Execution sandbox settings for running AIDE-generated code.

    Each generated solution runs in an isolated subprocess with timeout
    enforcement and stdout/stderr capture.
    """

    timeout: int = 3600
    """Maximum wall-clock seconds for each generated script to run.

    Set lower for quick iteration (e.g. 600 for config search),
    higher for full training runs (e.g. 7200).
    """

    format_tb_ipython: bool = False
    """Use IPython-style tracebacks for richer error formatting."""

    agent_file_name: str = "runfile.py"
    """Filename for the generated script written to the workspace."""


@dataclass
class AideParams(BaseParams):
    """Top-level AIDE agent parameters.

    Controls the search loop, LLM selection, optimization target, output
    directories, and telemetry. This parallels :class:`TuningParams` for
    Optuna-based tuning.
    """

    steps: int = 20
    """Number of search iterations (draft/debug/improve cycles).

    Each step generates one candidate solution and evaluates it.
    """

    surface: AideOptimizationSurface = AideOptimizationSurface.CONFIG_SEARCH
    """Which type of code artifact AIDE should optimize.

    See :class:`AideOptimizationSurface` for options.
    """

    code_llm: AideLLMParams = field(default_factory=AideLLMParams)
    """LLM settings for code generation (drafting, improving, debugging)."""

    feedback_llm: AideLLMParams = field(
        default_factory=lambda: AideLLMParams(model="gpt-5-mini")
    )
    """LLM settings for evaluating execution results and extracting metrics."""

    search: AideSearchParams = field(default_factory=AideSearchParams)
    """Tree search hyperparameters."""

    execution: AideExecParams = field(default_factory=AideExecParams)
    """Execution sandbox settings."""

    target_metric: str = "eval_loss"
    """Name of the metric to optimize.

    This should match a metric name printed by the generated code or
    returned by Oumi's evaluation pipeline.
    """

    target_direction: str = "minimize"
    """Direction of optimization: ``"minimize"`` or ``"maximize"``.

    For loss metrics use ``"minimize"``, for accuracy use ``"maximize"``.
    """

    output_dir: str = "output/aide"
    """Directory for AIDE run outputs (journal, best solution, reports)."""

    workspace_dir: str = "workspaces/aide"
    """Working directory for AIDE's generated scripts and temporary files."""

    log_level: str = "info"
    """The logging level for the main Oumi logger during AIDE runs.

    Possible values are "debug", "info", "warning", "error", "critical".
    """

    generate_report: bool = True
    """Whether to generate a markdown report summarizing the search."""

    report_llm: AideLLMParams = field(
        default_factory=lambda: AideLLMParams(model="gpt-4.1", temperature=1.0)
    )
    """LLM settings for report generation."""

    telemetry: TelemetryParams = field(default_factory=TelemetryParams)
    """Parameters for telemetry.

    This field contains telemetry configuration options.
    """

    def __post_init__(self):
        """Validates AIDE parameters."""
        if self.target_direction not in ("minimize", "maximize"):
            raise ValueError(
                f"target_direction must be 'minimize' or 'maximize', "
                f"got '{self.target_direction}'"
            )
        if self.steps < 1:
            raise ValueError(f"steps must be >= 1, got {self.steps}")

        # Validate log level
        valid_log_levels = {"debug", "info", "warning", "error", "critical"}
        if self.log_level not in valid_log_levels:
            raise ValueError(
                f"Invalid log_level: {self.log_level}. Choose from {valid_log_levels}."
            )

    @property
    def telemetry_dir(self) -> Path | None:
        """Returns the telemetry stats output directory."""
        result: Path | None = None
        if self.telemetry.telemetry_dir:
            result = Path(self.telemetry.telemetry_dir)

        if self.output_dir:
            output_dir = Path(self.output_dir)
            # If `telemetry.telemetry_dir` is relative, then treat it
            # as a sub-directory of `output_dir`.
            if result and not result.is_absolute():
                result = output_dir / result

        return result
