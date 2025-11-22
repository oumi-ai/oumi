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

"""Bridge between Oumi inference engines and DSPy optimization framework."""

import asyncio
from contextlib import contextmanager
from typing import Any, Callable, Optional

from oumi.core.configs.prompt_config import PromptOptimizationConfig
from oumi.core.types.conversation import Conversation, Message, Role
from oumi.utils.logging import logger


class OumiDSPyBridge:
    """Bridge to use Oumi inference engines with DSPy.

    This class wraps Oumi's inference engines to make them compatible
    with DSPy's optimization framework.
    """

    def __init__(
        self,
        config: PromptOptimizationConfig,
        metric_fn: Optional[Callable[[list[str], list[str]], float]] = None,
        callbacks: Optional[list] = None,
    ):
        """Initialize the bridge.

        Args:
            config: Prompt optimization configuration.
            metric_fn: Custom metric function for evaluation.
            callbacks: Optional list of DSPy callbacks for instrumentation.
        """
        self.config = config
        self.metric_fn = metric_fn
        self.callbacks = callbacks or []
        self._inference_engine = None
        self._tokenizer = None

    def _get_inference_engine(self):
        """Lazy load inference engine."""
        if self._inference_engine is None:
            from oumi.builders.inference_engines import build_inference_engine

            self._inference_engine = build_inference_engine(
                engine_type=self.config.engine,  # type: ignore[arg-type]
                model_params=self.config.model,
                generation_params=self.config.generation,
            )
        return self._inference_engine

    def _get_tokenizer(self):
        """Lazy load tokenizer for token counting."""
        if self._tokenizer is None:
            model_name = self.config.model.model_name.lower()

            # Try tiktoken for OpenAI models first
            if any(
                prefix in model_name
                for prefix in [
                    "gpt-",
                    "o1-",
                    "text-",
                    "davinci",
                    "curie",
                    "babbage",
                    "ada",
                ]
            ):
                try:
                    import tiktoken

                    # Map model names to tiktoken encodings
                    if "gpt-4o" in model_name:
                        # GPT-4o and GPT-4o-mini use o200k_base encoding
                        self._tokenizer = tiktoken.get_encoding("o200k_base")
                    elif "gpt-4" in model_name or "gpt-3.5-turbo" in model_name:
                        # GPT-4 and GPT-3.5-turbo use cl100k_base encoding
                        self._tokenizer = tiktoken.get_encoding("cl100k_base")
                    else:
                        # Fallback to cl100k_base for other OpenAI models
                        self._tokenizer = tiktoken.get_encoding("cl100k_base")
                    logger.debug(f"Using tiktoken for {self.config.model.model_name}")
                    return self._tokenizer
                except ImportError:
                    logger.info(
                        "tiktoken not installed. Install with: pip install tiktoken. "
                        "Token counts will be estimated."
                    )
                except Exception as e:
                    logger.debug(
                        f"Failed to load tiktoken: {e}. Token counts will be estimated."
                    )

            # Try HuggingFace tokenizer for local/HF models
            # Skip for known remote API providers
            remote_providers = ["claude", "gemini", "llama", "mistral", "anthropic"]
            is_likely_remote = any(
                provider in model_name for provider in remote_providers
            )

            if not is_likely_remote:
                try:
                    from transformers import AutoTokenizer

                    self._tokenizer = AutoTokenizer.from_pretrained(
                        self.config.model.model_name,
                        trust_remote_code=True,
                    )
                    logger.debug(
                        f"Using HuggingFace tokenizer for "
                        f"{self.config.model.model_name}"
                    )
                    return self._tokenizer
                except Exception as e:
                    logger.debug(
                        f"Could not load HuggingFace tokenizer for "
                        f"{self.config.model.model_name}: {e}. "
                        f"Token counts will be estimated."
                    )

            # If we get here, no tokenizer was loaded
            self._tokenizer = None

        return self._tokenizer

    def _estimate_token_count(self, text: str) -> int:
        """Estimate token count for a given text.

        Args:
            text: Input text to count tokens for.

        Returns:
            Estimated token count.
        """
        if not text:
            return 0

        tokenizer = self._get_tokenizer()
        if tokenizer is not None:
            try:
                # Check if it's a tiktoken encoding (has encode method but
                # not add_special_tokens param)
                if hasattr(tokenizer, "encode") and not hasattr(
                    tokenizer, "add_special_tokens"
                ):
                    # tiktoken encoding
                    return len(tokenizer.encode(text))
                else:
                    # HuggingFace tokenizer
                    return len(tokenizer.encode(text, add_special_tokens=True))  # type: ignore[call-arg]
            except Exception as e:
                logger.debug(
                    f"Tokenizer encode failed: {e}. Using fallback estimation."
                )

        # Fallback: improved estimation based on word count
        # Average ~1.3 tokens per word in English for modern tokenizers
        words = text.split()
        return max(1, int(len(words) * 1.3))

    @contextmanager
    def _temporary_generation_params(self, **kwargs):
        """Temporarily override generation parameters.

        This context manager safely overrides generation parameters and
        guarantees they are restored even if an error occurs.

        Args:
            **kwargs: Generation parameters to temporarily override.

        Yields:
            None
        """
        original_params = {}
        param_mapping = {
            "temperature": "temperature",
            "max_tokens": "max_new_tokens",
            "max_new_tokens": "max_new_tokens",
            "top_p": "top_p",
            "n": "num_return_sequences",
        }

        # Save original values and set new ones
        for dspy_param, oumi_param in param_mapping.items():
            if dspy_param in kwargs and hasattr(self.config.generation, oumi_param):
                original_params[oumi_param] = getattr(
                    self.config.generation, oumi_param
                )
                setattr(self.config.generation, oumi_param, kwargs[dspy_param])

        try:
            yield
        finally:
            # Restore original values
            for param_name, param_value in original_params.items():
                setattr(self.config.generation, param_name, param_value)

    def create_dspy_lm(self):
        """Create a DSPy language model wrapper.

        Returns:
            A DSPy-compatible LM object that uses Oumi's inference engine.
        """
        try:
            import dspy
        except ImportError:
            raise ImportError(
                "DSPy is required for this optimizer. "
                "Install it with: pip install 'oumi[prompt-optimization]'"
            )

        class OumiDSPyLM(dspy.LM):  # type: ignore[misc]
            """DSPy LM wrapper for Oumi inference engines."""

            def __init__(self, bridge: "OumiDSPyBridge"):
                # Initialize parent dspy.LM class with callbacks
                super().__init__(
                    model=bridge.config.model.model_name, callbacks=bridge.callbacks
                )
                self.bridge = bridge
                self.history = []  # type: ignore[misc]
                self.num_calls = 0
                self.failed_calls = 0

            def forward(
                self,
                prompt: Optional[str] = None,
                messages: Optional[list[dict[str, Any]]] = None,
                **kwargs,
            ) -> list[dict[str, Any]]:
                """Generate text from prompt(s).

                Args:
                    prompt: Input prompt string (DSPy uses this for simple prompts).
                    messages: Alternative message format (not currently used by Oumi).
                    **kwargs: Additional generation parameters (e.g., temperature,
                        max_tokens).

                Returns:
                    List of response dictionaries in DSPy format.

                Raises:
                    RuntimeError: If inference fails.
                """
                # DSPy uses either prompt or messages
                # If messages is provided, extract the last user message as the prompt
                if prompt is None and messages:
                    # Extract text from messages format
                    for msg in reversed(messages):
                        if msg.get("role") == "user":
                            prompt = msg.get("content", "")
                            break

                if prompt is None:
                    raise ValueError("Either prompt or messages must be provided")

                engine = self.bridge._get_inference_engine()

                # DSPy typically sends single prompts, not batches at this level
                prompts = [prompt]

                # Create conversations from prompts
                conversations = []
                for p in prompts:
                    conversations.append(
                        Conversation(messages=[Message(role=Role.USER, content=p)])
                    )

                # Log if verbose and overriding params
                if kwargs and self.bridge.config.optimization.verbose:
                    logger.debug(
                        f"Temporarily overriding generation params: "
                        f"{list(kwargs.keys())}"
                    )

                # Run inference with error handling and temporary param override
                try:
                    self.num_calls += 1
                    # Use context manager for safe parameter override
                    with self.bridge._temporary_generation_params(**kwargs):
                        results = engine.infer(conversations)
                except Exception as e:
                    self.failed_calls += 1
                    logger.error(
                        f"Inference failed (call {self.num_calls}, "
                        f"{self.failed_calls} total failures). "
                        f"Prompt: '{prompt[:100]}...'"
                    )
                    raise RuntimeError(
                        f"Inference failed after {self.num_calls} calls "
                        f"({self.failed_calls} failures): {e}"
                    ) from e

                # Extract outputs
                outputs = []
                for i, result in enumerate(results):
                    if result and len(result.messages) > 0:
                        # Get the last assistant message
                        assistant_msg = None
                        for msg in reversed(result.messages):
                            if msg.role == Role.ASSISTANT:
                                assistant_msg = msg.content
                                break

                        if assistant_msg:
                            outputs.append(assistant_msg)
                        else:
                            # No assistant message found
                            logger.warning(
                                f"No assistant message found in inference result "
                                f"{i + 1}. Result has {len(result.messages)} messages."
                            )
                            outputs.append("")
                            self.failed_calls += 1
                    else:
                        # Empty result
                        logger.warning(
                            f"Empty inference result received for prompt {i + 1}."
                        )
                        outputs.append("")
                        self.failed_calls += 1

                # Calculate token counts for usage tracking
                prompt_tokens = sum(
                    self.bridge._estimate_token_count(p) for p in prompts
                )
                completion_tokens = sum(
                    self.bridge._estimate_token_count(o) for o in outputs
                )
                total_tokens = prompt_tokens + completion_tokens

                # Log to history
                for p, o in zip(prompts, outputs):
                    self.history.append(
                        {
                            "prompt": p,
                            "response": o,
                            "kwargs": kwargs,
                            "call_num": self.num_calls,
                            "prompt_tokens": prompt_tokens // len(prompts),
                            "completion_tokens": completion_tokens // len(outputs)
                            if outputs
                            else 0,
                        }
                    )

                # Return in the format DSPy expects
                # DSPy's _process_lm_response expects an object with
                # 'choices', 'usage', and 'model'
                from dataclasses import dataclass

                # Use dataclass for response structure
                @dataclass
                class Response:
                    choices: list
                    usage: dict
                    model: str

                response = Response(
                    choices=[{"text": output} for output in outputs],
                    usage={
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": total_tokens,
                    },
                    model=self.bridge.config.model.model_name,
                )

                # Add DSPy-expected attributes for cache tracking and cost monitoring
                # cache_hit: Used by DSPy to detect when a response came from cache
                response.cache_hit = False  # type: ignore[attr-defined]  # Oumi doesn't currently expose cache info

                # _hidden_params: Used by DSPy for cost tracking and additional
                # metadata
                response._hidden_params = {  # type: ignore[attr-defined]
                    # Could be populated if cost tracking is available
                    "response_cost": None,
                }

                return response  # type: ignore[return-value]

            def inspect_history(self, n: int = 1):
                """Inspect recent history."""
                return self.history[-n:]

            def get_stats(self) -> dict[str, Any]:
                """Get usage statistics including token counts.

                Returns:
                    Dict with call counts, failure rate, and token usage.
                """
                # Calculate total token usage from history
                total_prompt_tokens = sum(
                    entry.get("prompt_tokens", 0) for entry in self.history
                )
                total_completion_tokens = sum(
                    entry.get("completion_tokens", 0) for entry in self.history
                )
                total_tokens = total_prompt_tokens + total_completion_tokens

                return {
                    "total_calls": self.num_calls,
                    "failed_calls": self.failed_calls,
                    "success_rate": (
                        (self.num_calls - self.failed_calls) / self.num_calls
                        if self.num_calls > 0
                        else 0.0
                    ),
                    "total_history_entries": len(self.history),
                    "total_prompt_tokens": total_prompt_tokens,
                    "total_completion_tokens": total_completion_tokens,
                    "total_tokens": total_tokens,
                }

            async def aforward(
                self,
                prompt: Optional[str] = None,
                messages: Optional[list[dict[str, Any]]] = None,
                **kwargs,
            ) -> list[dict[str, Any]]:
                """Async generate text from prompt(s).

                This method provides async support by running the synchronous
                inference in a thread pool. This is useful for async optimization
                workflows and improves performance for I/O-bound operations.

                Args:
                    prompt: Input prompt string (DSPy uses this for simple prompts).
                    messages: Alternative message format (not currently used by Oumi).
                    **kwargs: Additional generation parameters (e.g., temperature,
                        max_tokens).

                Returns:
                    List of response dictionaries in DSPy format.

                Raises:
                    RuntimeError: If inference fails.
                """
                # Run the synchronous forward() method in a thread pool
                # This is the recommended approach when the underlying engine
                # doesn't have native async support
                return await asyncio.to_thread(
                    self.forward, prompt=prompt, messages=messages, **kwargs
                )

            def dump_state(self) -> dict[str, Any]:
                """Dump LM state for serialization.

                Returns a dictionary containing all the state needed to reconstruct
                this LM instance. This enables checkpointing and saving optimized
                programs with DSPy's built-in serialization.

                Returns:
                    Dict containing model name, config, history, and statistics.
                """
                return {
                    "model": self.bridge.config.model.model_name,
                    "model_config": {
                        "model_name": self.bridge.config.model.model_name,
                        # Add other relevant model config fields
                    },
                    "generation_config": {
                        "temperature": self.bridge.config.generation.temperature,
                        "max_new_tokens": self.bridge.config.generation.max_new_tokens,
                        "top_p": self.bridge.config.generation.top_p,
                    },
                    "num_calls": self.num_calls,
                    "failed_calls": self.failed_calls,
                    "history_length": len(self.history),
                    # Don't serialize full history as it can be very large
                    # Just save summary statistics
                    "stats": self.get_stats(),
                }

            def load_state(self, state: dict[str, Any]) -> "OumiDSPyLM":
                """Load LM state from serialization.

                Restores the state of this LM instance from a previously saved state.
                Note: The bridge configuration is not modified, only the LM's
                internal state (call counts, statistics, etc.).

                Args:
                    state: Dictionary containing saved state from dump_state().

                Returns:
                    Self to allow method chaining.

                Raises:
                    ValueError: If state is incompatible with current configuration.
                """
                # Verify model compatibility
                if state.get("model") != self.bridge.config.model.model_name:
                    logger.warning(
                        f"Loading state for model '{state.get('model')}' "
                        f"but current model is "
                        f"'{self.bridge.config.model.model_name}'. "
                        f"This may cause issues."
                    )

                # Restore call counts
                self.num_calls = state.get("num_calls", 0)
                self.failed_calls = state.get("failed_calls", 0)

                # Note: We don't restore history as it can be very large
                # and is typically not needed after optimization
                logger.info(
                    f"Loaded LM state: {self.num_calls} calls, "
                    f"{self.failed_calls} failures"
                )

                return self

        return OumiDSPyLM(self)

    def create_dspy_dataset(
        self, data: list[dict[str, Any]], input_key: str = "input"
    ) -> list:
        """Convert Oumi dataset format to DSPy examples.

        Args:
            data: List of dataset examples with 'input' and 'output' fields.
            input_key: Key for input field (default: 'input').

        Returns:
            List of DSPy Example objects.

        Raises:
            ImportError: If DSPy is not installed.
            ValueError: If examples cannot be converted.
        """
        try:
            import dspy
        except ImportError:
            raise ImportError(
                "DSPy is required. "
                "Install with: pip install 'oumi[prompt-optimization]'"
            )

        examples = []
        skipped = 0

        for i, item in enumerate(data):
            # Support both simple dict format and Conversation format
            if "input" in item and "output" in item:
                try:
                    examples.append(
                        dspy.Example(
                            question=item["input"], answer=item["output"]
                        ).with_inputs("question")
                    )
                except Exception as e:
                    skipped += 1
                    logger.warning(
                        f"Failed to convert example {i + 1} to DSPy format: {e}"
                    )

            elif "messages" in item:
                # Handle Conversation format
                try:
                    conv = Conversation(**item)
                    question = ""
                    answer = ""
                    for msg in conv.messages:
                        if msg.role == Role.USER:
                            question = msg.content
                        elif msg.role == Role.ASSISTANT:
                            answer = msg.content

                    if question and answer:
                        examples.append(
                            dspy.Example(question=question, answer=answer).with_inputs(
                                "question"
                            )
                        )
                    else:
                        skipped += 1
                        logger.warning(
                            f"Example {i + 1} has no valid question-answer pair"
                        )
                except Exception as e:
                    skipped += 1
                    logger.warning(
                        f"Failed to parse Conversation format for example {i + 1}: {e}"
                    )
            else:
                skipped += 1
                logger.warning(
                    f"Example {i + 1} has neither 'input'/'output' nor "
                    f"'messages' fields"
                )

        if skipped > 0:
            logger.warning(
                f"Skipped {skipped}/{len(data)} examples during DSPy "
                f"conversion. Converted {len(examples)} examples successfully."
            )

        return examples

    def create_metric(
        self,
        metric_fn: Callable[[list[str], list[str]], float],
        support_gepa_feedback: bool = True,
    ):
        """Create a DSPy-compatible metric function.

        Args:
            metric_fn: Oumi metric function.
            support_gepa_feedback: Whether to support GEPA's extended metric signature
                with pred_name and pred_trace parameters for fine-grained feedback.

        Returns:
            DSPy-compatible metric function.
        """
        import inspect

        # Check if metric_fn supports GEPA's extended signature
        sig = inspect.signature(metric_fn)
        params = list(sig.parameters.keys())
        supports_gepa = (
            support_gepa_feedback
            and len(params) >= 5
            and "pred_name" in params
            and "pred_trace" in params
        )

        if supports_gepa:
            logger.info(
                "Detected GEPA-compatible metric with feedback support. "
                "Using enhanced metric wrapper."
            )

        def dspy_metric(
            example, prediction, trace=None, pred_name=None, pred_trace=None
        ):
            """DSPy metric wrapper with optional GEPA feedback support.

            Args:
                example: DSPy Example object.
                prediction: DSPy Prediction object.
                trace: Optional trace information (standard DSPy).
                pred_name: Optional predictor name (GEPA-specific).
                pred_trace: Optional predictor trace (GEPA-specific).

            Returns:
                Score between 0 and 1, or dict with 'score' and 'feedback' for GEPA.
            """
            # Extract ground truth and prediction
            if hasattr(example, "answer"):
                ground_truth = [example.answer]
            else:
                ground_truth = [""]

            if hasattr(prediction, "answer"):
                pred = [prediction.answer]
            elif hasattr(prediction, "rationale"):
                pred = [prediction.rationale]
            else:
                pred = [""]

            # Call the metric function
            try:
                if supports_gepa and pred_name is not None:
                    # Call with GEPA's extended signature
                    # type: ignore is needed as base metric_fn doesn't have these params
                    result = metric_fn(  # type: ignore[call-arg]
                        pred,
                        ground_truth,
                        trace=trace,  # type: ignore[call-arg]
                        pred_name=pred_name,  # type: ignore[call-arg]
                        pred_trace=pred_trace,  # type: ignore[call-arg]
                    )
                    # Result can be float or dict with 'score' and 'feedback'
                    return result
                else:
                    # Standard metric call
                    score = metric_fn(pred, ground_truth)
                    return score
            except Exception as e:
                pred_str = pred[0][:50] if pred and pred[0] else ""
                gt_str = (
                    ground_truth[0][:50] if ground_truth and ground_truth[0] else ""
                )
                logger.warning(
                    f"Metric evaluation failed for prediction='{pred_str}...' "
                    f"and ground_truth='{gt_str}...': {e}. "
                    f"Returning score of 0.0"
                )
                return 0.0

        return dspy_metric

    def create_simple_program(self, signature: str = "question -> answer"):
        """Create a simple DSPy program for optimization.

        Args:
            signature: DSPy signature string.

        Returns:
            DSPy program module.
        """
        try:
            import dspy
        except ImportError:
            raise ImportError(
                "DSPy is required. "
                "Install with: pip install 'oumi[prompt-optimization]'"
            )

        class SimpleQA(dspy.Module):
            """Simple Q&A program for prompt optimization."""

            def __init__(self, signature):
                super().__init__()
                self.predictor = dspy.ChainOfThought(signature)

            def forward(self, question):
                """Forward pass."""
                return self.predictor(question=question)

        return SimpleQA(signature)

    def setup_dspy(self):
        """Setup DSPy with Oumi LM."""
        try:
            import dspy
        except ImportError:
            raise ImportError(
                "DSPy is required. "
                "Install with: pip install 'oumi[prompt-optimization]'"
            )

        # Create and configure DSPy LM
        lm = self.create_dspy_lm()
        dspy.settings.configure(lm=lm)  # type: ignore[attr-defined]

        return lm
