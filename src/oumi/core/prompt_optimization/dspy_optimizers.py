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

"""DSPy-based prompt optimizers using registry pattern."""

from collections.abc import Callable
import sys
from dataclasses import dataclass
from typing import Any

from oumi.core.configs.prompt_config import PromptOptimizationConfig
from oumi.core.prompt_optimization.base import BaseOptimizer, OptimizationResult
from oumi.utils.logging import logger

_DEFAULT_MAX_ERRORS = 10
_MIPRO_INIT_TEMPERATURE = 0.7
_MIPRO_MINIBATCH_EVAL_STEPS = 10
_MIPRO_MAX_MINIBATCH_SIZE = 25
_MIPRO_MIN_MINIBATCH_SIZE = 4
_GEPA_DEPTH = 3
_MAX_CANDIDATE_PROGRAMS_TO_RETURN = 20
_PROGRAM_STRING_LIMIT = 500
_PROMPT_PREVIEW_LIMIT = 200
_DEFAULT_ESTIMATED_SCORE = 0.5


@dataclass(frozen=True)
class _OptimizerSpec:
    """Internal specification for a DSPy optimizer type."""

    name: str
    """Display name for the optimizer."""

    create_fn: Callable
    """Factory function: (config, metric, num_examples) -> optimizer."""

    compile_fn: Callable
    """Compilation function: (optimizer, program, train, val, config) -> program."""

    requires_metric: bool = False
    """Whether metric_fn is required (raises error if None)."""

    uses_gepa_feedback: bool = False
    """Whether to use GEPA's extended metric signature."""

    extracts_demos: bool = False
    """Whether this optimizer extracts demos (vs prompts)."""

    extract_metadata_fn: Callable | None = None
    """Optional function to extract optimizer-specific metadata."""


def _create_mipro(config: PromptOptimizationConfig, metric, num_examples: int):
    """Create MIPROv2 optimizer."""
    from dspy.teleprompt import MIPROv2

    try:
        return MIPROv2(
            metric=metric,
            auto=None,
            num_candidates=config.optimization.num_trials,
            init_temperature=_MIPRO_INIT_TEMPERATURE,
            max_errors=config.optimization.max_errors or _DEFAULT_MAX_ERRORS,
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to initialize MIPROv2: {e}. "
            f"Please ensure you have dspy-ai>=2.7 installed."
        ) from e


def _compile_mipro(optimizer, program, train_examples, val_examples, config):
    """Run MIPROv2 compilation with minibatch settings."""
    val_size = len(val_examples)
    minibatch_size = min(
        _MIPRO_MAX_MINIBATCH_SIZE, max(_MIPRO_MIN_MINIBATCH_SIZE, val_size // 2)
    )
    requires_permission_to_run = bool(sys.stdin and sys.stdin.isatty())

    try:
        return optimizer.compile(
            program,
            trainset=train_examples,
            num_trials=config.optimization.num_trials,
            max_bootstrapped_demos=config.optimization.max_bootstrapped_demos,
            max_labeled_demos=config.optimization.max_labeled_demos,
            minibatch_size=minibatch_size,
            minibatch_full_eval_steps=_MIPRO_MINIBATCH_EVAL_STEPS,
            requires_permission_to_run=requires_permission_to_run,
        )
    except Exception as e:
        raise RuntimeError(f"MIPROv2 compilation failed: {e}") from e


def _extract_mipro_metadata(optimizer, optimized_program, result: OptimizationResult):
    """Extract MIPROv2-specific metadata (candidate programs)."""
    if not hasattr(optimizer, "candidate_programs"):
        return

    try:
        candidates = []
        for idx, (prog, score) in enumerate(
            optimizer.candidate_programs[:_MAX_CANDIDATE_PROGRAMS_TO_RETURN]
        ):
            prog_str = str(getattr(prog.predictor, "signature", str(prog)))
            candidates.append(
                {
                    "index": idx,
                    "program": prog_str[:_PROGRAM_STRING_LIMIT],
                    "score": float(score) if score is not None else 0.0,
                }
            )
        result.candidate_programs = candidates
        result.metadata["num_candidates_explored"] = len(optimizer.candidate_programs)
    except Exception as e:
        logger.warning(f"Failed to extract candidate programs: {e}")


def _create_gepa(config: PromptOptimizationConfig, metric, num_examples: int):
    """Create GEPA optimizer."""
    try:
        from dspy.teleprompt import GEPA  # type: ignore[attr-defined]
    except ImportError as e:
        raise ImportError(
            "GEPA optimizer is not available. "
            "Please upgrade with: pip install --upgrade 'dspy-ai>=2.7'"
        ) from e

    try:
        return GEPA(
            metric=metric,
            breadth=config.optimization.num_trials,
            depth=_GEPA_DEPTH,
            max_errors=config.optimization.max_errors,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to initialize GEPA: {e}.") from e


def _compile_standard(optimizer, program, train_examples, val_examples, config):
    """Standard compilation - used by GEPA and Bootstrap."""
    try:
        return optimizer.compile(program, trainset=train_examples)
    except Exception as e:
        raise RuntimeError(f"Compilation failed: {e}") from e


def _extract_gepa_metadata(optimizer, optimized_program, result: OptimizationResult):
    """Extract GEPA-specific metadata (detailed results)."""
    if not hasattr(optimized_program, "detailed_results"):
        return

    try:
        gepa_results = optimized_program.detailed_results
        if hasattr(gepa_results, "to_dict"):
            result.detailed_results = gepa_results.to_dict()
        else:
            result.detailed_results = {"raw_results": str(gepa_results)}
    except Exception as e:
        logger.warning(f"Failed to extract GEPA detailed results: {e}")


def _create_bootstrap(config: PromptOptimizationConfig, metric, num_examples: int):
    """Create BootstrapFewShot optimizer."""
    from dspy.teleprompt import BootstrapFewShot

    try:
        return BootstrapFewShot(
            metric=metric,
            max_bootstrapped_demos=min(
                config.optimization.max_bootstrapped_demos, num_examples
            ),
            max_labeled_demos=min(config.optimization.max_labeled_demos, num_examples),
            max_errors=config.optimization.max_errors or _DEFAULT_MAX_ERRORS,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to initialize BootstrapFewShot: {e}.") from e


def _create_bootstrap_optuna(
    config: PromptOptimizationConfig, metric, num_examples: int
):
    """Create BootstrapFewShotWithOptuna optimizer."""
    try:
        from dspy.teleprompt import BootstrapFewShotWithOptuna
    except ImportError as e:
        raise ImportError(
            "BootstrapFewShotWithOptuna is not available. "
            "Please upgrade with: pip install --upgrade 'dspy-ai>=2.7'"
        ) from e

    try:
        import optuna  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "Optuna is required for this optimizer. Install with: pip install optuna"
        ) from e

    try:
        return BootstrapFewShotWithOptuna(
            metric=metric,
            max_bootstrapped_demos=min(
                config.optimization.max_bootstrapped_demos, num_examples
            ),
            max_labeled_demos=min(config.optimization.max_labeled_demos, num_examples),
            num_trials=config.optimization.num_trials,  # type: ignore[call-arg]
            max_errors=config.optimization.max_errors,  # type: ignore[call-arg]
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to initialize BootstrapFewShotWithOptuna: {e}."
        ) from e


def _compile_optuna(optimizer, program, train_examples, val_examples, config):
    """Compilation for Optuna optimizer (requires valset)."""
    try:
        return optimizer.compile(program, trainset=train_examples, valset=val_examples)
    except Exception as e:
        raise RuntimeError(f"BootstrapFewShotWithOptuna compilation failed: {e}") from e


def _extract_optuna_metadata(optimizer, optimized_program, result: OptimizationResult):
    """Extract Optuna-specific metadata (best hyperparameters)."""
    if not hasattr(optimizer, "best_params"):
        return

    try:
        result.optimized_hyperparameters = optimizer.best_params
    except Exception as e:
        logger.warning(f"Failed to extract optimized hyperparameters: {e}")


# =============================================================================
# Optimizer Registry
# =============================================================================

_OPTIMIZER_REGISTRY: dict[str, _OptimizerSpec] = {
    "mipro": _OptimizerSpec(
        name="MIPROv2",
        create_fn=_create_mipro,
        compile_fn=_compile_mipro,
        extract_metadata_fn=_extract_mipro_metadata,
    ),
    "gepa": _OptimizerSpec(
        name="GEPA",
        create_fn=_create_gepa,
        compile_fn=_compile_standard,
        requires_metric=True,
        uses_gepa_feedback=True,
        extract_metadata_fn=_extract_gepa_metadata,
    ),
    "bootstrap": _OptimizerSpec(
        name="BootstrapFewShot",
        create_fn=_create_bootstrap,
        compile_fn=_compile_standard,
        extracts_demos=True,
    ),
    "optuna": _OptimizerSpec(
        name="BootstrapFewShotWithOptuna",
        create_fn=_create_bootstrap_optuna,
        compile_fn=_compile_optuna,
        requires_metric=True,
        extracts_demos=True,
        extract_metadata_fn=_extract_optuna_metadata,
    ),
}


class DSPyOptimizer(BaseOptimizer):
    """DSPy-based prompt optimizer using registry pattern.

    This class provides a unified interface to all DSPy optimization algorithms.
    The specific optimizer is selected via config.optimization.optimizer.

    Supported optimizers:
        - mipro: MIPROv2 for instruction and demo optimization
        - gepa: GEPA for reflective prompt evolution
        - bootstrap: BootstrapFewShot for few-shot example selection
        - optuna: BootstrapFewShot with Bayesian hyperparameter optimization
    """

    def __init__(
        self,
        config: PromptOptimizationConfig,
        metric_fn: Callable[[list[str], list[str]], float] | None = None,
    ):
        """Initialize the optimizer.

        Args:
            config: Configuration for prompt optimization.
            metric_fn: Optional custom metric function.

        Raises:
            ImportError: If DSPy is not installed.
            ValueError: If optimizer requires metric_fn but none provided.
            KeyError: If optimizer name is not in registry.
        """
        super().__init__(config, metric_fn)
        self._check_dspy_available()

        optimizer_name = config.optimization.optimizer
        if optimizer_name not in _OPTIMIZER_REGISTRY:
            raise KeyError(
                f"Unknown optimizer: {optimizer_name}. "
                f"Available: {list(_OPTIMIZER_REGISTRY.keys())}"
            )

        self._spec = _OPTIMIZER_REGISTRY[optimizer_name]

        if self._spec.requires_metric and metric_fn is None:
            raise ValueError(f"{self._spec.name} requires a metric_fn")

    def _check_dspy_available(self) -> None:
        """Check if DSPy is available."""
        try:
            import dspy  # noqa: F401
        except ImportError:
            raise ImportError(
                "DSPy is required. "
                "Install with: pip install 'oumi[prompt-optimization]'"
            )

    def get_optimizer_name(self) -> str:
        """Get the display name of the optimizer."""
        return self._spec.name

    def optimize(
        self,
        train_data: list[dict[str, Any]],
        val_data: list[dict[str, Any]],
        initial_prompt: str | None = None,
    ) -> OptimizationResult:
        """Optimize prompts using the configured DSPy optimizer.

        Args:
            train_data: Training dataset.
            val_data: Validation dataset.
            initial_prompt: Optional initial prompt.

        Returns:
            OptimizationResult with optimized prompts/demos and metadata.
        """
        import dspy

        from oumi.core.prompt_optimization.dspy_integration import OumiDSPyBridge

        # Setup
        self._log_progress(f"Starting {self._spec.name} optimization...")
        self._log_progress(
            f"Training: {len(train_data)}, Validation: {len(val_data)} examples"
        )

        bridge = OumiDSPyBridge(self.config, self.metric_fn)
        bridge.setup_dspy()

        # Convert datasets
        self._log_progress("Converting datasets to DSPy format...")
        train_examples = bridge.create_dspy_dataset(train_data)
        val_examples = bridge.create_dspy_dataset(val_data)
        self._validate_datasets(train_examples, val_examples)

        # Create program and metric
        program = bridge.create_simple_program("question -> answer")
        metric = bridge.create_metric(
            self.metric_fn,  # type: ignore[arg-type]
            support_gepa_feedback=self._spec.uses_gepa_feedback,
        )

        # Create and run optimizer
        self._log_progress(f"Configuring {self._spec.name}...")
        optimizer = self._spec.create_fn(self.config, metric, len(train_examples))

        self._log_progress("Running optimization (this may take a while)...")
        optimized_program = self._spec.compile_fn(
            optimizer, program, train_examples, val_examples, self.config
        )

        # Evaluate
        final_score, val_scores, failed_examples, eval_stats = (
            self._evaluate_with_skip_option(optimized_program, val_examples, metric)
        )

        # Extract results
        optimized_prompt = self._extract_prompt(optimized_program, initial_prompt)
        optimized_demos = self._extract_demos(optimized_program)

        self._log_progress(
            f"Complete! Score: {final_score:.4f} "
            f"({len(val_scores)} examples, {eval_stats.get_elapsed_time():.1f}s)"
        )

        # Build result
        result = OptimizationResult(
            optimized_prompt=initial_prompt or ""
            if self._spec.extracts_demos
            else optimized_prompt,
            optimized_demos=optimized_demos,
            optimized_hyperparameters={},
            final_score=final_score,
            training_history=[],
            num_trials=self.config.optimization.num_trials,
            metadata={
                "optimizer": self._spec.name.lower(),
                "status": "completed",
                "dspy_version": dspy.__version__,
                "validation_examples_evaluated": len(val_scores),
                "failed_examples": failed_examples,
            },
            optimization_stats=eval_stats,
        )

        # Extract optimizer-specific metadata
        if self._spec.extract_metadata_fn:
            self._spec.extract_metadata_fn(optimizer, optimized_program, result)

        return result

    def _validate_datasets(self, train_examples: list, val_examples: list) -> None:
        """Validate that datasets converted successfully."""
        if not train_examples:
            raise RuntimeError(
                "No training examples converted. Check your dataset format."
            )
        if not val_examples:
            raise RuntimeError(
                "No validation examples converted. Check your dataset format."
            )

    def _evaluate_with_skip_option(
        self, optimized_program, val_examples: list, metric
    ) -> tuple[float, list[float], int, Any]:
        """Evaluate program, optionally skipping based on config."""
        if not self.config.optimization.skip_final_eval:
            self._log_progress("Evaluating on validation set...")
            return self._evaluate_program_on_dataset(
                optimized_program, val_examples, metric, "Validation"
            )

        self._log_progress("Skipping final evaluation (using DSPy's internal eval).")
        return self._evaluate_program_on_dataset(
            optimized_program,
            val_examples,
            metric,
            "Validation",
            skip=True,
            estimated_score=_DEFAULT_ESTIMATED_SCORE,
        )

    def _extract_demos(self, optimized_program) -> list[dict[str, Any]]:
        """Extract demonstrations from optimized program."""
        demos = []
        try:
            predictor = optimized_program.predictor
            if hasattr(predictor, "predict"):
                predictor = predictor.predict

            if hasattr(predictor, "demos") and predictor.demos:
                for demo in predictor.demos:
                    demo_dict = self._demo_to_dict(demo)
                    if demo_dict:
                        demos.append(demo_dict)
                self._log_progress(f"Extracted {len(demos)} demonstrations")
        except Exception as e:
            logger.warning(f"Failed to extract demonstrations: {e}")

        return demos

    def _demo_to_dict(self, demo) -> dict[str, Any]:
        """Convert a DSPy demo object to a dictionary."""
        result = {}
        for name in dir(demo):
            if name.startswith("_"):
                continue
            try:
                value = getattr(demo, name, None)
                if value is not None and not callable(value):
                    if isinstance(value, str | int | float | bool | list | dict):
                        result[name] = value
            except Exception:
                pass
        return result

    def _extract_prompt(self, optimized_program, initial_prompt: str | None) -> str:
        """Extract optimized prompt/instructions from program."""
        optimized_prompt = initial_prompt or ""
        verbose = self.config.optimization.verbose

        if verbose:
            self._log_progress("=" * 50)
            self._log_progress("PROMPT EXTRACTION")

        try:
            predictor = optimized_program.predictor
            if hasattr(predictor, "predict"):
                predictor = predictor.predict

            if hasattr(predictor, "signature"):
                signature = predictor.signature
                if hasattr(signature, "instructions"):
                    extracted = signature.instructions
                    if extracted and extracted.strip():
                        optimized_prompt = extracted
                        if verbose:
                            self._log_progress(
                                f"Extracted: {optimized_prompt[:_PROMPT_PREVIEW_LIMIT]}"
                            )
                    else:
                        logger.warning("Extracted instructions empty, using initial")
                elif hasattr(signature, "__doc__") and signature.__doc__:
                    optimized_prompt = signature.__doc__

        except Exception as e:
            logger.error(f"Failed to extract prompt: {e}", exc_info=True)

        if verbose:
            self._log_progress("=" * 50)

        if not optimized_prompt or not optimized_prompt.strip():
            optimized_prompt = (
                "Given the task inputs, produce accurate and helpful outputs."
            )
            logger.warning(f"Using default prompt: {optimized_prompt}")

        return optimized_prompt


class MiproOptimizer(DSPyOptimizer):
    """MIPROv2 optimizer. Alias for DSPyOptimizer with optimizer='mipro'."""

    def __init__(
        self,
        config: PromptOptimizationConfig,
        metric_fn: Callable[[list[str], list[str]], float] | None = None,
    ):
        """Initialize MiproOptimizer."""
        config.optimization.optimizer = "mipro"
        super().__init__(config, metric_fn)


class GepaOptimizer(DSPyOptimizer):
    """GEPA optimizer. Alias for DSPyOptimizer with optimizer='gepa'."""

    def __init__(
        self,
        config: PromptOptimizationConfig,
        metric_fn: Callable[[list[str], list[str]], float] | None = None,
    ):
        """Initialize GepaOptimizer."""
        config.optimization.optimizer = "gepa"
        super().__init__(config, metric_fn)


class BootstrapFewShotOptimizer(DSPyOptimizer):
    """BootstrapFewShot optimizer.

    Alias for DSPyOptimizer with optimizer='bootstrap'.
    """

    def __init__(
        self,
        config: PromptOptimizationConfig,
        metric_fn: Callable[[list[str], list[str]], float] | None = None,
    ):
        """Initialize BootstrapFewShotOptimizer."""
        config.optimization.optimizer = "bootstrap"
        super().__init__(config, metric_fn)


class BootstrapFewShotWithOptunaOptimizer(DSPyOptimizer):
    """BootstrapFewShot with Optuna.

    Alias for DSPyOptimizer with optimizer='optuna'.
    """

    def __init__(
        self,
        config: PromptOptimizationConfig,
        metric_fn: Callable[[list[str], list[str]], float] | None = None,
    ):
        """Initialize BootstrapFewShotWithOptunaOptimizer."""
        config.optimization.optimizer = "optuna"
        super().__init__(config, metric_fn)
