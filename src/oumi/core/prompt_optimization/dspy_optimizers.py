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

from typing import Any, Callable, Optional

from oumi.core.configs.prompt_config import PromptOptimizationConfig
from oumi.core.prompt_optimization.base import BaseOptimizer, OptimizationResult
from oumi.utils.logging import logger


class DSPyOptimizer(BaseOptimizer):
    """Base class for DSPy-based optimizers."""

    def __init__(
        self,
        config: PromptOptimizationConfig,
        metric_fn: Optional[Callable[[list[str], list[str]], float]] = None,
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

    def _extract_demos_from_program(self, optimized_program) -> list[dict[str, Any]]:
        """Extract demonstrations from an optimized DSPy program.

        This handles ChainOfThought's nested structure and extracts all fields
        dynamically.

        Args:
            optimized_program: The optimized DSPy program.

        Returns:
            List of demo dictionaries with all fields extracted.
        """
        optimized_demos = []
        try:
            # Handle ChainOfThought which wraps a Predict module
            predictor = optimized_program.predictor
            if hasattr(predictor, "predict"):
                # ChainOfThought has .predict attribute containing the actual
                # Predict module
                predictor = predictor.predict

            # Extract demos if they exist
            if hasattr(predictor, "demos") and predictor.demos:
                for demo in predictor.demos:
                    demo_dict = {}
                    # Extract all fields dynamically instead of hardcoding field names
                    # Skip private attributes (starting with _)
                    for field_name in dir(demo):
                        if not field_name.startswith("_"):
                            try:
                                value = getattr(demo, field_name, None)
                                # Only include simple types (str, int, float,
                                # bool, etc.). Skip methods and complex objects
                                if value is not None and not callable(value):
                                    if isinstance(
                                        value, (str, int, float, bool, list, dict)
                                    ):
                                        demo_dict[field_name] = value
                            except Exception:
                                # Skip attributes that raise errors on access
                                pass

                    # Only add if we got some data
                    if demo_dict:
                        optimized_demos.append(demo_dict)

                self._log_progress(f"Extracted {len(optimized_demos)} demonstrations")
            else:
                self._log_progress("No demonstrations found in optimized program")
        except Exception as e:
            logger.warning(f"Failed to extract demonstrations: {e}")

        return optimized_demos

    def _extract_prompt_from_program(
        self, optimized_program, initial_prompt: Optional[str] = None
    ) -> str:
        """Extract the optimized prompt/instructions from a DSPy program.

        This handles ChainOfThought's nested structure and properly extracts
        instructions.

        Args:
            optimized_program: The optimized DSPy program.
            initial_prompt: The initial prompt to use as fallback.

        Returns:
            The optimized prompt string.
        """
        optimized_prompt = initial_prompt or ""

        # Log the initial prompt
        self._log_progress("=" * 80)
        self._log_progress("PROMPT EXTRACTION DEBUG")
        self._log_progress("=" * 80)
        self._log_progress(f"Initial prompt: {repr(initial_prompt)}")

        try:
            # Debug: Show program structure
            self._log_progress(f"Optimized program type: {type(optimized_program)}")
            self._log_progress(
                f"Has 'predictor' attr: {hasattr(optimized_program, 'predictor')}"
            )

            # Handle ChainOfThought which wraps a Predict module
            predictor = optimized_program.predictor
            self._log_progress(f"Predictor type: {type(predictor)}")
            self._log_progress(
                f"Predictor has 'predict' attr: {hasattr(predictor, 'predict')}"
            )

            if hasattr(predictor, "predict"):
                # ChainOfThought has .predict attribute containing the actual
                # Predict module
                self._log_progress("ChainOfThought detected, unwrapping to .predict")
                predictor = predictor.predict
                self._log_progress(f"Unwrapped predictor type: {type(predictor)}")

            # Now extract the instructions from the signature
            if hasattr(predictor, "signature"):
                signature = predictor.signature
                self._log_progress(f"Signature type: {type(signature)}")
                has_instr = hasattr(signature, "instructions")
                self._log_progress(f"Signature has 'instructions' attr: {has_instr}")

                if hasattr(signature, "instructions"):
                    extracted = signature.instructions
                    self._log_progress(
                        f"Extracted instructions type: {type(extracted)}"
                    )
                    instr_len = len(extracted) if extracted else 0
                    self._log_progress(f"Extracted instructions length: {instr_len}")

                    if extracted and len(extracted.strip()) > 0:
                        optimized_prompt = extracted
                        self._log_progress(
                            "✓ Successfully extracted optimized instructions"
                        )
                        self._log_progress(
                            "Optimized prompt preview (first 200 chars):"
                        )
                        self._log_progress(f"  {optimized_prompt[:200]}")
                    else:
                        logger.warning(
                            "Extracted instructions are empty, keeping initial prompt"
                        )
                else:
                    logger.warning("Signature has no 'instructions' attribute")
                    # Try alternative extraction methods
                    self._log_progress(f"Signature attributes: {dir(signature)}")
                    if hasattr(signature, "__doc__"):
                        self._log_progress(
                            f"Signature __doc__: {repr(signature.__doc__)}"
                        )
                        if signature.__doc__:
                            optimized_prompt = signature.__doc__
            else:
                logger.warning("Predictor has no 'signature' attribute")
                self._log_progress(f"Predictor attributes: {dir(predictor)}")

        except Exception as e:
            logger.error(f"Failed to extract optimized prompt: {e}", exc_info=True)
            self._log_progress(f"Exception details: {e}")

        self._log_progress("=" * 80)
        self._log_progress(f"FINAL OPTIMIZED PROMPT ({len(optimized_prompt)} chars):")
        self._log_progress(optimized_prompt if optimized_prompt else "(empty)")
        self._log_progress("=" * 80)

        # If we still have an empty prompt, use a sensible default
        if not optimized_prompt or len(optimized_prompt.strip()) == 0:
            default_prompt = (
                "Given the task inputs, produce accurate and helpful outputs."
            )
            logger.warning(
                f"Optimized prompt is empty. Using default: {default_prompt}. "
                "Consider providing an initial_prompt in your config for "
                "better results."
            )
            optimized_prompt = default_prompt

        return optimized_prompt


class MiproOptimizer(DSPyOptimizer):
    """MIPRO (Multi-prompt Instruction PRoposal Optimizer) wrapper."""

    def get_optimizer_name(self) -> str:
        """Get the optimizer name."""
        return "MIPROv2"

    def optimize(
        self,
        train_data: list[dict[str, Any]],
        val_data: list[dict[str, Any]],
        initial_prompt: Optional[str] = None,
    ) -> OptimizationResult:
        """Optimize prompts using MIPROv2.

        Args:
            train_data: Training dataset.
            val_data: Validation dataset.
            initial_prompt: Optional initial prompt.

        Returns:
            OptimizationResult with optimized prompts and metadata.

        Raises:
            ImportError: If DSPy is not installed.
            RuntimeError: If optimization fails.
        """
        import dspy
        from dspy.teleprompt import MIPROv2
        from oumi.core.prompt_optimization.dspy_bridge import OumiDSPyBridge

        self._log_progress("Starting MIPROv2 optimization...")
        self._log_progress(
            f"Training examples: {len(train_data)}, "
            f"Validation examples: {len(val_data)}"
        )

        # Create bridge
        bridge = OumiDSPyBridge(self.config, self.metric_fn)

        # Setup DSPy
        self._log_progress("Initializing DSPy with Oumi inference engine...")
        bridge.setup_dspy()

        # Convert datasets
        self._log_progress("Converting datasets to DSPy format...")
        train_examples = bridge.create_dspy_dataset(train_data)
        val_examples = bridge.create_dspy_dataset(val_data)

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

        # Create program
        self._log_progress("Creating DSPy program...")
        program = bridge.create_simple_program("question -> answer")

        # Create metric
        metric = bridge.create_metric(self.metric_fn)  # type: ignore[arg-type]

        # Configure MIPRO
        self._log_progress(
            f"Configuring MIPROv2 with {self.config.optimization.num_trials} trials..."
        )

        # Set max_errors for fault tolerance
        max_errors = self.config.optimization.max_errors
        if max_errors is not None:
            self._log_progress(f"Setting max_errors={max_errors} for fault tolerance")

        try:
            # MIPROv2 API changed in dspy 3.1.0+
            # New API requires num_candidates when auto=None
            mipro = MIPROv2(  # type: ignore[call-arg]
                metric=metric,
                auto=None,  # Disable auto mode to allow manual trial configuration
                num_candidates=self.config.optimization.num_trials,
                init_temperature=0.7,
                max_errors=max_errors,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize MIPROv2 optimizer: {e}. "
                f"This may be due to an incompatible DSPy version. "
                f"Please ensure you have dspy-ai>=2.7 installed."
            ) from e

        # Run optimization
        self._log_progress("Running optimization (this may take a while)...")
        try:
            # Calculate appropriate minibatch size based on validation set size
            # DSPy internally may split the dataset, so use a conservative size
            val_size = len(val_examples)
            minibatch_size = min(25, max(4, val_size // 2))

            optimized_program = mipro.compile(
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

        # Evaluate on validation set (optional - can be skipped to save cost)
        if not self.config.optimization.skip_final_eval:
            self._log_progress("Evaluating optimized program on validation set...")
            final_score, val_scores, failed_examples, eval_stats = (
                self._evaluate_program_on_dataset(
                    optimized_program,
                    val_examples,
                    metric,
                    "Evaluating on validation set",
                )
            )
        else:
            # Skip final evaluation - DSPy already evaluated during optimization
            self._log_progress(
                "Skipping final validation evaluation to save time/cost. "
                "DSPy's internal evaluation was used during optimization."
            )
            # Use a conservative estimate since we don't have the actual score
            estimated_score = 0.5
            final_score, val_scores, failed_examples, eval_stats = (
                self._evaluate_program_on_dataset(
                    optimized_program,
                    val_examples,
                    metric,
                    "Evaluating on validation set",
                    skip=True,
                    estimated_score=estimated_score,
                )
            )

        # Extract optimized prompt using helper method
        optimized_prompt = self._extract_prompt_from_program(
            optimized_program, initial_prompt
        )

        self._log_progress(
            f"Optimization complete! Final score: {final_score:.4f} "
            f"(evaluated on {len(val_scores)} examples in "
            f"{eval_stats.get_elapsed_time():.1f}s)"
        )

        # Extract candidate programs if available
        candidate_programs = None
        if hasattr(mipro, "candidate_programs"):
            try:
                candidates = []
                for idx, (prog, score) in enumerate(
                    mipro.candidate_programs[:20]  # type: ignore[attr-defined]
                ):  # Limit to top 20
                    # Extract program signature/prompt
                    prog_str = str(getattr(prog.predictor, "signature", str(prog)))
                    candidates.append(
                        {
                            "index": idx,
                            "program": prog_str[:500],  # Limit length
                            "score": float(score) if score is not None else 0.0,
                        }
                    )
                candidate_programs = candidates
                self._log_progress(
                    f"MIPROv2 explored {len(mipro.candidate_programs)} "  # type: ignore[attr-defined]
                    f"candidate programs. Returning top {len(candidates)}."
                )
            except Exception as e:
                logger.warning(f"Failed to extract candidate programs: {e}")

        return OptimizationResult(
            optimized_prompt=optimized_prompt,
            optimized_demos=[],
            optimized_hyperparameters={},
            final_score=final_score,
            training_history=[],
            num_trials=self.config.optimization.num_trials,
            metadata={
                "optimizer": "mipro",
                "status": "completed",
                "dspy_version": dspy.__version__,
                "validation_examples_evaluated": len(val_scores),
                "failed_examples": failed_examples,
                "num_candidates_explored": len(mipro.candidate_programs)  # type: ignore[attr-defined]
                if hasattr(mipro, "candidate_programs")
                else None,
            },
            optimization_stats=eval_stats,
            candidate_programs=candidate_programs,
        )


class GepaOptimizer(DSPyOptimizer):
    """GEPA (Genetic-Pareto) optimizer wrapper."""

    def get_optimizer_name(self) -> str:
        """Get the optimizer name."""
        return "GEPA"

    def optimize(
        self,
        train_data: list[dict[str, Any]],
        val_data: list[dict[str, Any]],
        initial_prompt: Optional[str] = None,
    ) -> OptimizationResult:
        """Optimize prompts using GEPA.

        Args:
            train_data: Training dataset.
            val_data: Validation dataset.
            initial_prompt: Optional initial prompt.

        Returns:
            OptimizationResult with optimized prompts and metadata.

        Raises:
            ImportError: If GEPA is not available in DSPy.
            RuntimeError: If optimization fails.
        """
        import dspy
        from oumi.core.prompt_optimization.dspy_bridge import OumiDSPyBridge

        self._log_progress("Starting GEPA optimization...")
        self._log_progress(
            f"Training examples: {len(train_data)}, "
            f"Validation examples: {len(val_data)}"
        )

        # Check if GEPA is available
        try:
            from dspy.teleprompt import GEPA
        except ImportError as e:
            raise ImportError(
                "GEPA optimizer is not available in your DSPy version. "
                "GEPA requires a newer version of DSPy. "
                "Please upgrade with: pip install --upgrade 'dspy-ai>=2.7' "
                "or use a different optimizer like 'mipro' or 'bootstrap'."
            ) from e

        # Create bridge
        bridge = OumiDSPyBridge(self.config, self.metric_fn)

        # Setup DSPy
        self._log_progress("Initializing DSPy with Oumi inference engine...")
        bridge.setup_dspy()

        # Convert datasets
        self._log_progress("Converting datasets to DSPy format...")
        train_examples = bridge.create_dspy_dataset(train_data)
        val_examples = bridge.create_dspy_dataset(val_data)

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

        # Create program
        self._log_progress("Creating DSPy program...")
        program = bridge.create_simple_program("question -> answer")

        # Create metric with GEPA feedback support
        if self.metric_fn is None:
            raise ValueError("metric_fn is required for GEPA optimization")
        metric = bridge.create_metric(self.metric_fn, support_gepa_feedback=True)

        # Configure GEPA
        self._log_progress(
            f"Configuring GEPA with "
            f"breadth={self.config.optimization.num_trials}, depth=3..."
        )

        # Set max_errors for fault tolerance
        max_errors = self.config.optimization.max_errors
        if max_errors is not None:
            self._log_progress(f"Setting max_errors={max_errors} for fault tolerance")

        try:
            gepa = GEPA(  # type: ignore[call-arg]
                metric=metric,  # type: ignore[call-arg]
                breadth=self.config.optimization.num_trials,  # type: ignore[call-arg]
                depth=3,  # type: ignore[call-arg]
                max_errors=max_errors,  # type: ignore[call-arg]
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize GEPA optimizer: {e}. "
                f"This may be due to an incompatible DSPy version."
            ) from e

        # Run optimization
        self._log_progress("Running GEPA optimization (this may take a while)...")
        try:
            optimized_program = gepa.compile(
                program,
                trainset=train_examples,
            )
        except Exception as e:
            raise RuntimeError(
                f"Optimization failed during GEPA compilation: {e}"
            ) from e

        # Evaluate on validation set (optional - can be skipped to save cost)
        if not self.config.optimization.skip_final_eval:
            self._log_progress("Evaluating optimized program on validation set...")
            final_score, val_scores, failed_examples, eval_stats = (
                self._evaluate_program_on_dataset(
                    optimized_program,
                    val_examples,
                    metric,
                    "Evaluating on validation set",
                )
            )
        else:
            # Skip final evaluation - DSPy already evaluated during optimization
            self._log_progress(
                "Skipping final validation evaluation to save time/cost. "
                "DSPy's internal evaluation was used during optimization."
            )
            # Use a conservative estimate since we don't have the actual score
            estimated_score = 0.5
            final_score, val_scores, failed_examples, eval_stats = (
                self._evaluate_program_on_dataset(
                    optimized_program,
                    val_examples,
                    metric,
                    "Evaluating on validation set",
                    skip=True,
                    estimated_score=estimated_score,
                )
            )

        # Extract optimized prompt using helper method
        optimized_prompt = self._extract_prompt_from_program(
            optimized_program, initial_prompt
        )

        self._log_progress(
            f"Optimization complete! Final score: {final_score:.4f} "
            f"(evaluated on {len(val_scores)} examples in "
            f"{eval_stats.get_elapsed_time():.1f}s)"
        )

        # Extract detailed GEPA results if available
        detailed_results = None
        if hasattr(optimized_program, "detailed_results"):
            try:
                gepa_results = optimized_program.detailed_results
                if hasattr(gepa_results, "to_dict"):
                    detailed_results = gepa_results.to_dict()
                    num_candidates = len(detailed_results.get("candidates", []))
                    best_idx = detailed_results.get("best_idx", "N/A")
                    self._log_progress(
                        f"GEPA explored {num_candidates} candidates. "
                        f"Best candidate index: {best_idx}"
                    )
                else:
                    detailed_results = {"raw_results": str(gepa_results)}
            except Exception as e:
                logger.warning(f"Failed to extract GEPA detailed results: {e}")

        return OptimizationResult(
            optimized_prompt=optimized_prompt,
            optimized_demos=[],
            optimized_hyperparameters={},
            final_score=final_score,
            training_history=[],
            num_trials=self.config.optimization.num_trials,
            metadata={
                "optimizer": "gepa",
                "status": "completed",
                "dspy_version": dspy.__version__,
                "validation_examples_evaluated": len(val_scores),
                "failed_examples": failed_examples,
            },
            optimization_stats=eval_stats,
            detailed_results=detailed_results,
        )


class BootstrapFewShotOptimizer(DSPyOptimizer):
    """BootstrapFewShot optimizer wrapper."""

    def get_optimizer_name(self) -> str:
        """Get the optimizer name."""
        return "BootstrapFewShot"

    def optimize(
        self,
        train_data: list[dict[str, Any]],
        val_data: list[dict[str, Any]],
        initial_prompt: Optional[str] = None,
    ) -> OptimizationResult:
        """Optimize few-shot examples using Bootstrap.

        Args:
            train_data: Training dataset.
            val_data: Validation dataset.
            initial_prompt: Optional initial prompt.

        Returns:
            OptimizationResult with optimized few-shot examples.

        Raises:
            ImportError: If DSPy is not installed.
            RuntimeError: If optimization fails.
        """
        import dspy
        from dspy.teleprompt import BootstrapFewShot
        from oumi.core.prompt_optimization.dspy_bridge import OumiDSPyBridge

        self._log_progress("Starting BootstrapFewShot optimization...")
        self._log_progress(
            f"Selecting few-shot examples from {len(train_data)} training examples"
        )

        # Create bridge
        bridge = OumiDSPyBridge(self.config, self.metric_fn)

        # Setup DSPy
        self._log_progress("Initializing DSPy with Oumi inference engine...")
        bridge.setup_dspy()

        # Convert datasets
        self._log_progress("Converting datasets to DSPy format...")
        train_examples = bridge.create_dspy_dataset(train_data)
        val_examples = bridge.create_dspy_dataset(val_data)

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

        # Create program
        self._log_progress("Creating DSPy program...")
        program = bridge.create_simple_program("question -> answer")

        # Create metric
        metric = bridge.create_metric(self.metric_fn)  # type: ignore[arg-type]

        # Configure BootstrapFewShot
        max_demos = min(
            self.config.optimization.max_bootstrapped_demos, len(train_examples)
        )
        max_labeled = min(
            self.config.optimization.max_labeled_demos, len(train_examples)
        )

        self._log_progress(
            f"Configuring BootstrapFewShot with max_bootstrapped_demos={max_demos}, "
            f"max_labeled_demos={max_labeled}..."
        )

        # Set max_errors for fault tolerance
        max_errors = self.config.optimization.max_errors
        if max_errors is not None:
            self._log_progress(f"Setting max_errors={max_errors} for fault tolerance")

        try:
            bootstrap = BootstrapFewShot(
                metric=metric,
                max_bootstrapped_demos=max_demos,
                max_labeled_demos=max_labeled,
                max_errors=max_errors,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize BootstrapFewShot optimizer: {e}. "
                f"This may be due to an incompatible DSPy version."
            ) from e

        # Run optimization
        self._log_progress("Running BootstrapFewShot (selecting best examples)...")
        try:
            optimized_program = bootstrap.compile(
                program,
                trainset=train_examples,
            )
        except Exception as e:
            raise RuntimeError(
                f"Optimization failed during BootstrapFewShot compilation: {e}"
            ) from e

        # Evaluate on validation set (optional - can be skipped to save cost)
        if not self.config.optimization.skip_final_eval:
            self._log_progress("Evaluating optimized program on validation set...")
            final_score, val_scores, failed_examples, eval_stats = (
                self._evaluate_program_on_dataset(
                    optimized_program,
                    val_examples,
                    metric,
                    "Evaluating on validation set",
                )
            )
        else:
            # Skip final evaluation - DSPy already evaluated during optimization
            self._log_progress(
                "Skipping final validation evaluation to save time/cost. "
                "DSPy's internal evaluation was used during optimization."
            )
            # Use a conservative estimate since we don't have the actual score
            estimated_score = 0.5
            final_score, val_scores, failed_examples, eval_stats = (
                self._evaluate_program_on_dataset(
                    optimized_program,
                    val_examples,
                    metric,
                    "Evaluating on validation set",
                    skip=True,
                    estimated_score=estimated_score,
                )
            )

        # Extract optimized demos using the common helper method
        optimized_demos = self._extract_demos_from_program(optimized_program)

        self._log_progress(
            f"Optimization complete! Selected {len(optimized_demos)} examples. "
            f"Final score: {final_score:.4f} (evaluated on {len(val_scores)} "
            f"examples in {eval_stats.get_elapsed_time():.1f}s)"
        )

        return OptimizationResult(
            optimized_prompt=initial_prompt or "",
            optimized_demos=optimized_demos,
            optimized_hyperparameters={},
            final_score=final_score,
            training_history=[],
            num_trials=len(train_examples),
            metadata={
                "optimizer": "bootstrap",
                "status": "completed",
                "num_demos": len(optimized_demos),
                "dspy_version": dspy.__version__,
                "validation_examples_evaluated": len(val_scores),
                "failed_examples": failed_examples,
            },
            optimization_stats=eval_stats,
        )


class BootstrapFewShotWithOptunaOptimizer(DSPyOptimizer):
    """BootstrapFewShot with Bayesian optimization (Optuna) wrapper."""

    def get_optimizer_name(self) -> str:
        """Get the optimizer name."""
        return "BootstrapFewShotWithOptuna"

    def optimize(
        self,
        train_data: list[dict[str, Any]],
        val_data: list[dict[str, Any]],
        initial_prompt: Optional[str] = None,
    ) -> OptimizationResult:
        """Optimize few-shot examples and hyperparameters using Bootstrap with Optuna.

        This optimizer combines BootstrapFewShot's example selection with Optuna's
        Bayesian optimization for hyperparameter tuning, providing better optimization
        than random or grid search.

        Args:
            train_data: Training dataset.
            val_data: Validation dataset.
            initial_prompt: Optional initial prompt.

        Returns:
            OptimizationResult with optimized few-shot examples and hyperparameters.

        Raises:
            ImportError: If DSPy or Optuna is not installed.
            RuntimeError: If optimization fails.
        """
        import dspy
        from oumi.core.prompt_optimization.dspy_bridge import OumiDSPyBridge

        self._log_progress("Starting BootstrapFewShotWithOptuna optimization...")
        self._log_progress(
            f"Training examples: {len(train_data)}, "
            f"Validation examples: {len(val_data)}"
        )

        # Check if BootstrapFewShotWithOptuna is available
        try:
            from dspy.teleprompt import BootstrapFewShotWithOptuna
        except ImportError as e:
            raise ImportError(
                "BootstrapFewShotWithOptuna optimizer is not available in "
                "your DSPy version. "
                "Please upgrade with: pip install --upgrade 'dspy-ai>=2.7' "
                "or use a different optimizer like 'mipro' or 'bootstrap'."
            ) from e

        # Check if Optuna is installed
        try:
            import optuna  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "Optuna is required for BootstrapFewShotWithOptuna optimizer. "
                "Install it with: pip install optuna"
            ) from e

        # Create bridge
        bridge = OumiDSPyBridge(self.config, self.metric_fn)

        # Setup DSPy
        self._log_progress("Initializing DSPy with Oumi inference engine...")
        bridge.setup_dspy()

        # Convert datasets
        self._log_progress("Converting datasets to DSPy format...")
        train_examples = bridge.create_dspy_dataset(train_data)
        val_examples = bridge.create_dspy_dataset(val_data)

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

        # Create program
        self._log_progress("Creating DSPy program...")
        program = bridge.create_simple_program("question -> answer")

        # Create metric
        if self.metric_fn is None:
            raise ValueError("metric_fn is required for BootstrapFewShotWithOptuna")
        metric = bridge.create_metric(self.metric_fn)

        # Configure BootstrapFewShotWithOptuna
        max_demos = min(
            self.config.optimization.max_bootstrapped_demos, len(train_examples)
        )
        max_labeled = min(
            self.config.optimization.max_labeled_demos, len(train_examples)
        )

        # Set max_errors for fault tolerance
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
            bootstrap_optuna = BootstrapFewShotWithOptuna(  # type: ignore[call-arg]
                metric=metric,
                max_bootstrapped_demos=max_demos,
                max_labeled_demos=max_labeled,
                num_trials=self.config.optimization.num_trials,  # type: ignore[call-arg]
                max_errors=max_errors,  # type: ignore[call-arg]
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize BootstrapFewShotWithOptuna optimizer: {e}. "
                f"This may be due to an incompatible DSPy version."
            ) from e

        # Run optimization
        self._log_progress(
            "Running BootstrapFewShotWithOptuna (Bayesian optimization with Optuna)..."
        )
        try:
            optimized_program = bootstrap_optuna.compile(  # type: ignore[call-arg]
                program,
                trainset=train_examples,
                valset=val_examples,
            )
        except Exception as e:
            raise RuntimeError(
                f"Optimization failed during BootstrapFewShotWithOptuna "
                f"compilation: {e}"
            ) from e

        # Evaluate on validation set (optional - can be skipped to save cost)
        if not self.config.optimization.skip_final_eval:
            self._log_progress("Evaluating optimized program on validation set...")
            final_score, val_scores, failed_examples, eval_stats = (
                self._evaluate_program_on_dataset(
                    optimized_program,
                    val_examples,
                    metric,
                    "Evaluating on validation set",
                )
            )
        else:
            # Skip final evaluation - DSPy already evaluated during optimization
            self._log_progress(
                "Skipping final validation evaluation to save time/cost. "
                "DSPy's internal evaluation was used during optimization."
            )
            # Use a conservative estimate since we don't have the actual score
            estimated_score = 0.5
            final_score, val_scores, failed_examples, eval_stats = (
                self._evaluate_program_on_dataset(
                    optimized_program,
                    val_examples,
                    metric,
                    "Evaluating on validation set",
                    skip=True,
                    estimated_score=estimated_score,
                )
            )

        # Extract optimized demos using the common helper method
        optimized_demos = self._extract_demos_from_program(optimized_program)

        # Extract optimized hyperparameters if available
        optimized_hyperparameters = {}
        if hasattr(bootstrap_optuna, "best_params"):
            try:
                optimized_hyperparameters = bootstrap_optuna.best_params  # type: ignore[attr-defined]
                self._log_progress(
                    f"Optuna found best hyperparameters: {optimized_hyperparameters}"
                )
            except Exception as e:
                logger.warning(f"Failed to extract optimized hyperparameters: {e}")

        self._log_progress(
            f"Optimization complete! Selected {len(optimized_demos)} examples. "
            f"Final score: {final_score:.4f} (evaluated on {len(val_scores)} "
            f"examples in {eval_stats.get_elapsed_time():.1f}s)"
        )

        return OptimizationResult(
            optimized_prompt=initial_prompt or "",
            optimized_demos=optimized_demos,
            optimized_hyperparameters=optimized_hyperparameters,
            final_score=final_score,
            training_history=[],
            num_trials=self.config.optimization.num_trials,
            metadata={
                "optimizer": "optuna",
                "status": "completed",
                "num_demos": len(optimized_demos),
                "dspy_version": dspy.__version__,
                "validation_examples_evaluated": len(val_scores),
                "failed_examples": failed_examples,
            },
            optimization_stats=eval_stats,
        )
