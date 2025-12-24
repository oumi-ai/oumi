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

from abc import abstractmethod
from collections.abc import Callable
from typing import Any

from oumi.core.configs.prompt_config import PromptOptimizationConfig
from oumi.core.prompt_optimization.base import BaseOptimizer, OptimizationResult
from oumi.utils.logging import logger


class DSPyOptimizer(BaseOptimizer):
    """Base class for DSPy-based optimizers."""

    def __init__(
        self,
        config: PromptOptimizationConfig,
        metric_fn: Callable[[list[str], list[str]], float] | None = None,
    ):
        """Initialize the DSPy optimizer.

        Args:
            config: Configuration for prompt optimization.
            metric_fn: Optional custom metric function.
        """
        super().__init__(config, metric_fn)
        self._check_dspy_available()

    def _check_dspy_available(self) -> None:
        """Check if DSPy is available."""
        try:
            import dspy  # noqa: F401
        except ImportError:
            raise ImportError(
                "DSPy is required for this optimizer. "
                "Install it with: pip install 'oumi[prompt-optimization]'"
            )

    def optimize(
        self,
        train_data: list[dict[str, Any]],
        val_data: list[dict[str, Any]],
        initial_prompt: str | None = None,
    ) -> OptimizationResult:
        """Template method for optimization flow.

        Args:
            train_data: Training dataset.
            val_data: Validation dataset.
            initial_prompt: Optional initial prompt.

        Returns:
            OptimizationResult with optimized prompts and metadata.
        """
        import dspy

        from oumi.core.prompt_optimization.dspy_bridge import OumiDSPyBridge

        # Setup phase
        self._log_progress(f"Starting {self.get_optimizer_name()} optimization...")
        self._log_progress(
            f"Training examples: {len(train_data)}, "
            f"Validation examples: {len(val_data)}"
        )

        bridge = OumiDSPyBridge(self.config, self.metric_fn)
        self._log_progress("Initializing DSPy with Oumi inference engine...")
        bridge.setup_dspy()

        # Convert datasets
        self._log_progress("Converting datasets to DSPy format...")
        train_examples = bridge.create_dspy_dataset(train_data)
        val_examples = bridge.create_dspy_dataset(val_data)
        self._validate_datasets(train_examples, val_examples)

        # Create program and metric
        self._log_progress("Creating DSPy program...")
        program = bridge.create_simple_program("question -> answer")
        metric = self._create_metric(bridge)

        # Run optimizer-specific compilation
        optimizer_instance = self._create_optimizer(metric, len(train_examples))
        optimized_program = self._run_compilation(
            optimizer_instance, program, train_examples, val_examples
        )

        # Evaluation phase
        final_score, val_scores, failed_examples, eval_stats = (
            self._evaluate_with_skip_option(optimized_program, val_examples, metric)
        )

        # Extract results
        optimized_prompt = self._extract_prompt_from_program(
            optimized_program, initial_prompt
        )
        optimized_demos = self._extract_demos_from_program(optimized_program)

        self._log_progress(
            f"Optimization complete! Final score: {final_score:.4f} "
            f"(evaluated on {len(val_scores)} examples in "
            f"{eval_stats.get_elapsed_time():.1f}s)"
        )

        # Build result with optimizer-specific metadata
        return self._build_result(
            optimized_prompt=optimized_prompt,
            optimized_demos=optimized_demos,
            final_score=final_score,
            val_scores=val_scores,
            failed_examples=failed_examples,
            eval_stats=eval_stats,
            dspy_version=dspy.__version__,
            optimizer_instance=optimizer_instance,
            initial_prompt=initial_prompt,
        )

    def _validate_datasets(self, train_examples: list, val_examples: list) -> None:
        """Validate converted datasets.

        Args:
            train_examples: Converted training examples.
            val_examples: Converted validation examples.

        Raises:
            RuntimeError: If datasets are empty.
        """
        if not train_examples:
            raise RuntimeError(
                "No training examples could be converted to DSPy format. "
                "Please check your dataset format."
            )
        if not val_examples:
            raise RuntimeError(
                "No validation examples could be converted to DSPy format. "
                "Please check your dataset format."
            )

    def _create_metric(self, bridge):
        """Create the metric function for this optimizer.

        Args:
            bridge: OumiDSPyBridge instance.

        Returns:
            DSPy-compatible metric function.
        """
        return bridge.create_metric(self.metric_fn)  # type: ignore[arg-type]

    @abstractmethod
    def _create_optimizer(self, metric, num_train_examples: int):
        """Create the DSPy optimizer instance.

        Args:
            metric: The metric function.
            num_train_examples: Number of training examples.

        Returns:
            The configured DSPy optimizer.
        """
        pass

    @abstractmethod
    def _run_compilation(
        self, optimizer, program, train_examples: list, val_examples: list
    ):
        """Run the optimizer compilation.

        Args:
            optimizer: The DSPy optimizer instance.
            program: The DSPy program to optimize.
            train_examples: Training examples.
            val_examples: Validation examples.

        Returns:
            The optimized program.
        """
        pass

    def _build_result(
        self,
        optimized_prompt: str,
        optimized_demos: list[dict[str, Any]],
        final_score: float,
        val_scores: list[float],
        failed_examples: int,
        eval_stats,
        dspy_version: str,
        optimizer_instance,
        initial_prompt: str | None,
    ) -> OptimizationResult:
        """Build the optimization result.

        Subclasses can override to add optimizer-specific metadata.
        """
        return OptimizationResult(
            optimized_prompt=optimized_prompt,
            optimized_demos=optimized_demos,
            optimized_hyperparameters={},
            final_score=final_score,
            training_history=[],
            num_trials=self.config.optimization.num_trials,
            metadata={
                "optimizer": self.get_optimizer_name().lower(),
                "status": "completed",
                "dspy_version": dspy_version,
                "validation_examples_evaluated": len(val_scores),
                "failed_examples": failed_examples,
            },
            optimization_stats=eval_stats,
        )

    def _evaluate_with_skip_option(
        self, optimized_program, val_examples: list, metric
    ) -> tuple[float, list[float], int, Any]:
        """Evaluate program with optional skip based on config.

        Args:
            optimized_program: The optimized DSPy program.
            val_examples: Validation examples.
            metric: The metric function.

        Returns:
            Tuple of (final_score, val_scores, failed_examples, eval_stats).
        """
        if not self.config.optimization.skip_final_eval:
            self._log_progress("Evaluating optimized program on validation set...")
            return self._evaluate_program_on_dataset(
                optimized_program,
                val_examples,
                metric,
                "Evaluating on validation set",
            )

        self._log_progress(
            "Skipping final validation evaluation to save time/cost. "
            "DSPy's internal evaluation was used during optimization."
        )
        return self._evaluate_program_on_dataset(
            optimized_program,
            val_examples,
            metric,
            "Evaluating on validation set",
            skip=True,
            estimated_score=0.5,
        )

    def _extract_demos_from_program(self, optimized_program) -> list[dict[str, Any]]:
        """Extract demonstrations from an optimized DSPy program.

        Args:
            optimized_program: The optimized DSPy program.

        Returns:
            List of demo dictionaries with all fields extracted.
        """
        optimized_demos = []
        try:
            predictor = optimized_program.predictor
            if hasattr(predictor, "predict"):
                predictor = predictor.predict

            if hasattr(predictor, "demos") and predictor.demos:
                for demo in predictor.demos:
                    demo_dict = self._extract_demo_fields(demo)
                    if demo_dict:
                        optimized_demos.append(demo_dict)

                self._log_progress(f"Extracted {len(optimized_demos)} demonstrations")
            else:
                self._log_progress("No demonstrations found in optimized program")
        except Exception as e:
            logger.warning(f"Failed to extract demonstrations: {e}")

        return optimized_demos

    def _extract_demo_fields(self, demo) -> dict[str, Any]:
        """Extract fields from a single demo object.

        Args:
            demo: DSPy demo object.

        Returns:
            Dictionary of extracted fields.
        """
        demo_dict = {}
        for field_name in dir(demo):
            if field_name.startswith("_"):
                continue
            try:
                value = getattr(demo, field_name, None)
                if value is not None and not callable(value):
                    if isinstance(value, str | int | float | bool | list | dict):
                        demo_dict[field_name] = value
            except Exception:
                pass
        return demo_dict

    def _extract_prompt_from_program(
        self, optimized_program, initial_prompt: str | None = None
    ) -> str:
        """Extract the optimized prompt/instructions from a DSPy program.

        Args:
            optimized_program: The optimized DSPy program.
            initial_prompt: The initial prompt to use as fallback.

        Returns:
            The optimized prompt string.
        """
        optimized_prompt = initial_prompt or ""
        verbose = self.config.optimization.verbose

        if verbose:
            self._log_progress("=" * 60)
            self._log_progress("PROMPT EXTRACTION")
            self._log_progress(f"Initial prompt: {repr(initial_prompt)}")

        try:
            predictor = optimized_program.predictor
            if verbose:
                self._log_progress(f"Predictor type: {type(predictor)}")

            if hasattr(predictor, "predict"):
                if verbose:
                    self._log_progress("ChainOfThought detected, unwrapping")
                predictor = predictor.predict

            if hasattr(predictor, "signature"):
                signature = predictor.signature
                if hasattr(signature, "instructions"):
                    extracted = signature.instructions
                    if extracted and len(extracted.strip()) > 0:
                        optimized_prompt = extracted
                        if verbose:
                            self._log_progress("Successfully extracted instructions")
                            self._log_progress(f"Preview: {optimized_prompt[:200]}")
                    else:
                        logger.warning(
                            "Extracted instructions are empty, keeping initial prompt"
                        )
                else:
                    logger.warning("Signature has no 'instructions' attribute")
                    if hasattr(signature, "__doc__") and signature.__doc__:
                        optimized_prompt = signature.__doc__
            else:
                logger.warning("Predictor has no 'signature' attribute")

        except Exception as e:
            logger.error(f"Failed to extract optimized prompt: {e}", exc_info=True)

        if verbose:
            self._log_progress(f"Final prompt ({len(optimized_prompt)} chars)")
            self._log_progress("=" * 60)

        if not optimized_prompt or len(optimized_prompt.strip()) == 0:
            default_prompt = (
                "Given the task inputs, produce accurate and helpful outputs."
            )
            logger.warning(
                f"Optimized prompt is empty. Using default: {default_prompt}"
            )
            optimized_prompt = default_prompt

        return optimized_prompt


class MiproOptimizer(DSPyOptimizer):
    """MIPRO (Multi-prompt Instruction PRoposal Optimizer) wrapper."""

    def get_optimizer_name(self) -> str:
        """Get the optimizer name."""
        return "MIPROv2"

    def _create_optimizer(self, metric, num_train_examples: int):
        """Create MIPROv2 optimizer."""
        from dspy.teleprompt import MIPROv2

        self._log_progress(
            f"Configuring MIPROv2 with {self.config.optimization.num_trials} trials..."
        )

        max_errors = self.config.optimization.max_errors
        if max_errors is not None:
            self._log_progress(f"Setting max_errors={max_errors} for fault tolerance")

        try:
            return MIPROv2(  # type: ignore[call-arg]
                metric=metric,
                auto=None,
                num_candidates=self.config.optimization.num_trials,
                init_temperature=0.7,
                max_errors=max_errors if max_errors is not None else 10,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize MIPROv2 optimizer: {e}. "
                f"Please ensure you have dspy-ai>=2.7 installed."
            ) from e

    def _run_compilation(self, optimizer, program, train_examples, val_examples):
        """Run MIPROv2 compilation."""
        self._log_progress("Running optimization (this may take a while)...")
        try:
            val_size = len(val_examples)
            minibatch_size = min(25, max(4, val_size // 2))

            return optimizer.compile(
                program,
                trainset=train_examples,
                num_trials=self.config.optimization.num_trials,
                max_bootstrapped_demos=self.config.optimization.max_bootstrapped_demos,
                max_labeled_demos=self.config.optimization.max_labeled_demos,
                minibatch_size=minibatch_size,
                minibatch_full_eval_steps=10,
            )
        except Exception as e:
            raise RuntimeError(
                f"Optimization failed during MIPROv2 compilation: {e}"
            ) from e

    def _build_result(self, optimizer_instance, **kwargs) -> OptimizationResult:
        """Build result with MIPROv2-specific metadata."""
        result = super()._build_result(optimizer_instance=optimizer_instance, **kwargs)

        # Extract candidate programs if available
        if hasattr(optimizer_instance, "candidate_programs"):
            try:
                candidates = []
                for idx, (prog, score) in enumerate(
                    optimizer_instance.candidate_programs[:20]  # type: ignore[attr-defined]
                ):
                    prog_str = str(getattr(prog.predictor, "signature", str(prog)))
                    candidates.append(
                        {
                            "index": idx,
                            "program": prog_str[:500],
                            "score": float(score) if score is not None else 0.0,
                        }
                    )
                result.candidate_programs = candidates
                result.metadata["num_candidates_explored"] = len(
                    optimizer_instance.candidate_programs  # type: ignore[attr-defined]
                )
                self._log_progress(
                    f"MIPROv2 explored {len(optimizer_instance.candidate_programs)} "  # type: ignore[attr-defined]
                    f"candidate programs."
                )
            except Exception as e:
                logger.warning(f"Failed to extract candidate programs: {e}")

        return result


class GepaOptimizer(DSPyOptimizer):
    """GEPA (Genetic-Pareto) optimizer wrapper."""

    def get_optimizer_name(self) -> str:
        """Get the optimizer name."""
        return "GEPA"

    def _create_metric(self, bridge):
        """Create GEPA-compatible metric with feedback support."""
        if self.metric_fn is None:
            raise ValueError("metric_fn is required for GEPA optimization")
        return bridge.create_metric(self.metric_fn, support_gepa_feedback=True)

    def _create_optimizer(self, metric, num_train_examples: int):
        """Create GEPA optimizer."""
        try:
            from dspy.teleprompt import GEPA  # type: ignore[attr-defined]
        except ImportError as e:
            raise ImportError(
                "GEPA optimizer is not available in your DSPy version. "
                "Please upgrade with: pip install --upgrade 'dspy-ai>=2.7' "
                "or use a different optimizer like 'mipro' or 'bootstrap'."
            ) from e

        self._log_progress(
            f"Configuring GEPA with breadth={self.config.optimization.num_trials}..."
        )

        max_errors = self.config.optimization.max_errors
        if max_errors is not None:
            self._log_progress(f"Setting max_errors={max_errors} for fault tolerance")

        try:
            return GEPA(  # type: ignore[call-arg]
                metric=metric,
                breadth=self.config.optimization.num_trials,
                depth=3,
                max_errors=max_errors,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize GEPA optimizer: {e}.") from e

    def _run_compilation(self, optimizer, program, train_examples, val_examples):
        """Run GEPA compilation."""
        self._log_progress("Running GEPA optimization (this may take a while)...")
        try:
            return optimizer.compile(program, trainset=train_examples)
        except Exception as e:
            raise RuntimeError(
                f"Optimization failed during GEPA compilation: {e}"
            ) from e

    def _build_result(self, optimizer_instance, **kwargs) -> OptimizationResult:
        """Build result with GEPA-specific metadata."""
        result = super()._build_result(optimizer_instance=optimizer_instance, **kwargs)

        # Extract detailed GEPA results if available
        optimized_program = kwargs.get("optimized_program")
        if optimized_program and hasattr(optimized_program, "detailed_results"):
            try:
                gepa_results = optimized_program.detailed_results
                if hasattr(gepa_results, "to_dict"):
                    result.detailed_results = gepa_results.to_dict()
                    self._log_progress(
                        f"GEPA explored {len(result.detailed_results.get('candidates', []))} "
                        f"candidates."
                    )
                else:
                    result.detailed_results = {"raw_results": str(gepa_results)}
            except Exception as e:
                logger.warning(f"Failed to extract GEPA detailed results: {e}")

        return result


class BootstrapFewShotOptimizer(DSPyOptimizer):
    """BootstrapFewShot optimizer wrapper."""

    def get_optimizer_name(self) -> str:
        """Get the optimizer name."""
        return "BootstrapFewShot"

    def _create_optimizer(self, metric, num_train_examples: int):
        """Create BootstrapFewShot optimizer."""
        from dspy.teleprompt import BootstrapFewShot

        max_demos = min(
            self.config.optimization.max_bootstrapped_demos, num_train_examples
        )
        max_labeled = min(
            self.config.optimization.max_labeled_demos, num_train_examples
        )

        self._log_progress(
            f"Configuring BootstrapFewShot with max_bootstrapped_demos={max_demos}, "
            f"max_labeled_demos={max_labeled}..."
        )

        max_errors = self.config.optimization.max_errors
        if max_errors is not None:
            self._log_progress(f"Setting max_errors={max_errors} for fault tolerance")

        try:
            return BootstrapFewShot(
                metric=metric,
                max_bootstrapped_demos=max_demos,
                max_labeled_demos=max_labeled,
                max_errors=max_errors if max_errors is not None else 10,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize BootstrapFewShot optimizer: {e}."
            ) from e

    def _run_compilation(self, optimizer, program, train_examples, val_examples):
        """Run BootstrapFewShot compilation."""
        self._log_progress("Running BootstrapFewShot (selecting best examples)...")
        try:
            return optimizer.compile(program, trainset=train_examples)
        except Exception as e:
            raise RuntimeError(
                f"Optimization failed during BootstrapFewShot compilation: {e}"
            ) from e

    def _build_result(self, optimizer_instance, **kwargs) -> OptimizationResult:
        """Build result for BootstrapFewShot."""
        result = super()._build_result(optimizer_instance=optimizer_instance, **kwargs)
        result.metadata["num_demos"] = len(kwargs.get("optimized_demos", []))
        result.num_trials = kwargs.get("num_train_examples", result.num_trials)

        # Use initial prompt since bootstrap doesn't optimize instructions
        initial_prompt = kwargs.get("initial_prompt")
        if initial_prompt:
            result.optimized_prompt = initial_prompt

        self._log_progress(f"Selected {result.metadata['num_demos']} examples.")
        return result


class BootstrapFewShotWithOptunaOptimizer(DSPyOptimizer):
    """BootstrapFewShot with Bayesian optimization (Optuna) wrapper."""

    def get_optimizer_name(self) -> str:
        """Get the optimizer name."""
        return "BootstrapFewShotWithOptuna"

    def _create_metric(self, bridge):
        """Create metric for Optuna optimizer."""
        if self.metric_fn is None:
            raise ValueError("metric_fn is required for BootstrapFewShotWithOptuna")
        return bridge.create_metric(self.metric_fn)

    def _create_optimizer(self, metric, num_train_examples: int):
        """Create BootstrapFewShotWithOptuna optimizer."""
        try:
            from dspy.teleprompt import BootstrapFewShotWithOptuna
        except ImportError as e:
            raise ImportError(
                "BootstrapFewShotWithOptuna optimizer is not available. "
                "Please upgrade with: pip install --upgrade 'dspy-ai>=2.7'"
            ) from e

        try:
            import optuna  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "Optuna is required for BootstrapFewShotWithOptuna optimizer. "
                "Install it with: pip install optuna"
            ) from e

        max_demos = min(
            self.config.optimization.max_bootstrapped_demos, num_train_examples
        )
        max_labeled = min(
            self.config.optimization.max_labeled_demos, num_train_examples
        )

        max_errors = self.config.optimization.max_errors
        if max_errors is not None:
            self._log_progress(f"Setting max_errors={max_errors} for fault tolerance")

        self._log_progress(
            f"Configuring BootstrapFewShotWithOptuna with "
            f"max_bootstrapped_demos={max_demos}, "
            f"max_labeled_demos={max_labeled}, "
            f"num_trials={self.config.optimization.num_trials}..."
        )

        try:
            return BootstrapFewShotWithOptuna(  # type: ignore[call-arg]
                metric=metric,
                max_bootstrapped_demos=max_demos,
                max_labeled_demos=max_labeled,
                num_trials=self.config.optimization.num_trials,
                max_errors=max_errors,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize BootstrapFewShotWithOptuna optimizer: {e}."
            ) from e

    def _run_compilation(self, optimizer, program, train_examples, val_examples):
        """Run BootstrapFewShotWithOptuna compilation."""
        self._log_progress(
            "Running BootstrapFewShotWithOptuna (Bayesian optimization with Optuna)..."
        )
        try:
            return optimizer.compile(  # type: ignore[call-arg]
                program,
                trainset=train_examples,
                valset=val_examples,
            )
        except Exception as e:
            raise RuntimeError(
                f"Optimization failed during BootstrapFewShotWithOptuna compilation: {e}"
            ) from e

    def _build_result(self, optimizer_instance, **kwargs) -> OptimizationResult:
        """Build result with Optuna-specific metadata."""
        result = super()._build_result(optimizer_instance=optimizer_instance, **kwargs)
        result.metadata["num_demos"] = len(kwargs.get("optimized_demos", []))

        # Extract optimized hyperparameters
        if hasattr(optimizer_instance, "best_params"):
            try:
                result.optimized_hyperparameters = optimizer_instance.best_params  # type: ignore[attr-defined]
                self._log_progress(
                    f"Optuna found best hyperparameters: {result.optimized_hyperparameters}"
                )
            except Exception as e:
                logger.warning(f"Failed to extract optimized hyperparameters: {e}")

        # Use initial prompt since optuna doesn't optimize instructions
        initial_prompt = kwargs.get("initial_prompt")
        if initial_prompt:
            result.optimized_prompt = initial_prompt

        return result
