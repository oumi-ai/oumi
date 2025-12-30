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

"""DSPy integration for Oumi inference engines.

This module provides the bridge between Oumi's inference engines and DSPy's
optimization framework, including the LM wrapper and dataset conversion.
"""

import asyncio
import re
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from oumi.core.configs.prompt_config import PromptOptimizationConfig
from oumi.core.types.conversation import Conversation, Message, Role
from oumi.utils.logging import logger

# =============================================================================
# Constants
# =============================================================================

_DSPY_INSTALL_ERROR = (
    "DSPy is required. Install with: pip install 'oumi[prompt-optimization]'"
)

_OPENAI_MODEL_PREFIXES = ("gpt-", "o1-", "text-", "davinci", "curie", "babbage", "ada")
_REMOTE_PROVIDERS = ("claude", "gemini", "llama", "mistral", "anthropic")

_TOKENS_PER_WORD_ESTIMATE = 1.3


# =============================================================================
# LM Response Dataclass
# =============================================================================


@dataclass
class LMResponse:
    """Response structure for DSPy compatibility."""

    choices: list[dict[str, str]]
    usage: dict[str, int]
    model: str
    cache_hit: bool = False
    _hidden_params: dict[str, Any] | None = None


# =============================================================================
# Main Bridge Class
# =============================================================================


class OumiDSPyBridge:
    """Bridge to use Oumi inference engines with DSPy.

    This class wraps Oumi's inference engines to make them compatible
    with DSPy's optimization framework.
    """

    def __init__(
        self,
        config: PromptOptimizationConfig,
        metric_fn: Callable[[list[str], list[str]], float] | None = None,
        callbacks: list | None = None,
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

    # -------------------------------------------------------------------------
    # Inference Engine
    # -------------------------------------------------------------------------

    def _get_inference_engine(self):
        """Lazy load inference engine."""
        if self._inference_engine is None:
            from oumi.builders.inference_engines import build_inference_engine

            if self.config.engine is None:
                raise ValueError(
                    "Prompt optimization requires an inference engine to be specified."
                )

            self._inference_engine = build_inference_engine(
                engine_type=self.config.engine,
                model_params=self.config.model,
                remote_params=self.config.remote_params,
                generation_params=self.config.generation,
            )
        return self._inference_engine

    # -------------------------------------------------------------------------
    # Tokenizer & Token Counting
    # -------------------------------------------------------------------------

    def _get_tokenizer(self):
        """Lazy load tokenizer for token counting."""
        if self._tokenizer is not None:
            return self._tokenizer

        model_name = self.config.model.model_name.lower()

        # Try tiktoken for OpenAI models
        if any(prefix in model_name for prefix in _OPENAI_MODEL_PREFIXES):
            self._tokenizer = self._try_tiktoken(model_name)
            if self._tokenizer:
                return self._tokenizer

        # Try HuggingFace for non-remote models
        if not any(provider in model_name for provider in _REMOTE_PROVIDERS):
            self._tokenizer = self._try_huggingface()

        return self._tokenizer

    def _try_tiktoken(self, model_name: str):
        """Try to load tiktoken tokenizer for OpenAI models."""
        try:
            import tiktoken

            encoding = "o200k_base" if "gpt-4o" in model_name else "cl100k_base"
            tokenizer = tiktoken.get_encoding(encoding)
            logger.debug(
                f"Using tiktoken ({encoding}) for {self.config.model.model_name}"
            )
            return tokenizer
        except ImportError:
            logger.debug("tiktoken not installed, using fallback token estimation")
        except Exception as e:
            logger.debug(f"Failed to load tiktoken: {e}")
        return None

    def _try_huggingface(self):
        """Try to load HuggingFace tokenizer."""
        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                self.config.model.model_name,
                trust_remote_code=True,
            )
            logger.debug(
                f"Using HuggingFace tokenizer for {self.config.model.model_name}"
            )
            return tokenizer
        except Exception as e:
            logger.debug(f"Could not load HuggingFace tokenizer: {e}")
        return None

    def _estimate_token_count(self, text: str) -> int:
        """Estimate token count for a given text."""
        if not text:
            return 0

        tokenizer = self._get_tokenizer()
        if tokenizer is not None:
            try:
                if hasattr(tokenizer, "encode") and not hasattr(
                    tokenizer, "add_special_tokens"
                ):
                    return len(tokenizer.encode(text))
                else:
                    return len(
                        tokenizer.encode(text, add_special_tokens=True)  # type: ignore[call-arg]
                    )
            except Exception as e:
                logger.debug(f"Tokenizer encode failed: {e}. Using fallback.")

        return max(1, int(len(text.split()) * _TOKENS_PER_WORD_ESTIMATE))

    # -------------------------------------------------------------------------
    # Generation Parameter Override
    # -------------------------------------------------------------------------

    @contextmanager
    def _temporary_generation_params(self, **kwargs):
        """Temporarily override generation parameters."""
        param_mapping = {
            "temperature": "temperature",
            "max_tokens": "max_new_tokens",
            "max_new_tokens": "max_new_tokens",
            "top_p": "top_p",
            "n": "num_return_sequences",
        }

        original_params = {}
        for dspy_param, oumi_param in param_mapping.items():
            if dspy_param in kwargs and hasattr(self.config.generation, oumi_param):
                original_params[oumi_param] = getattr(
                    self.config.generation, oumi_param
                )
                setattr(self.config.generation, oumi_param, kwargs[dspy_param])

        try:
            yield
        finally:
            for param_name, param_value in original_params.items():
                setattr(self.config.generation, param_name, param_value)

    # -------------------------------------------------------------------------
    # DSPy LM Creation
    # -------------------------------------------------------------------------

    def create_dspy_lm(self):
        """Create a DSPy language model wrapper.

        Returns:
            A DSPy-compatible LM object that uses Oumi's inference engine.
        """
        try:
            import dspy
        except ImportError:
            raise ImportError(_DSPY_INSTALL_ERROR)

        from dspy.utils.callback import with_callbacks

        bridge = self

        class OumiDSPyLM(dspy.LM):
            """DSPy LM wrapper for Oumi inference engines."""

            def __init__(self):
                super().__init__(
                    model=bridge.config.model.model_name,
                    callbacks=bridge.callbacks,
                )
                self.bridge = bridge
                self.history: list[dict[str, Any]] = []
                self.num_calls = 0
                self.failed_calls = 0

            @with_callbacks
            def __call__(
                self,
                prompt: str | None = None,
                messages: list[dict[str, Any]] | None = None,
                **kwargs,
            ) -> list[str]:
                """DSPy entrypoint that routes through the Oumi engine."""
                prompt_text = self._extract_prompt(prompt, messages)
                response = self.forward(prompt=prompt, messages=messages, **kwargs)
                return [
                    self._ensure_dspy_output_format(
                        choice.get("text", ""),
                        prompt_text,
                    )
                    for choice in response.choices
                ]

            def forward(
                self,
                prompt: str | None = None,
                messages: list[dict[str, Any]] | None = None,
                **kwargs,
            ) -> LMResponse:
                """Generate text from prompt(s)."""
                prompt = self._extract_prompt(prompt, messages)
                engine = self.bridge._get_inference_engine()

                conversations = [
                    Conversation(messages=[Message(role=Role.USER, content=prompt)])
                ]

                if kwargs and self.bridge.config.optimization.verbose:
                    logger.debug(f"Overriding generation params: {list(kwargs.keys())}")

                try:
                    self.num_calls += 1
                    with self.bridge._temporary_generation_params(**kwargs):
                        results = engine.infer(conversations)
                except Exception as e:
                    self.failed_calls += 1
                    logger.error(
                        f"Inference failed (call {self.num_calls}, "
                        f"{self.failed_calls} failures). Prompt: '{prompt[:100]}...'"
                    )
                    raise RuntimeError(
                        f"Inference failed after {self.num_calls} calls "
                        f"({self.failed_calls} failures): {e}"
                    ) from e

                outputs = self._extract_outputs(results)
                return self._build_response(prompt, outputs, kwargs)

            def _ensure_dspy_output_format(self, output: str, prompt_text: str) -> str:
                """Ensure output includes DSPy field markers for parsing."""
                markers_all = re.findall(r"\[\[ ## (\w+) ## \]\]", prompt_text)
                output_fields = re.findall(r"`\[\[ ## (\w+) ## \]\]`", prompt_text)
                markers = output_fields or markers_all
                if not markers:
                    return output

                if output_fields and all(
                    f"[[ ## {field} ## ]]" in output for field in output_fields
                ):
                    return output

                if "completed" in markers_all and "completed" not in markers:
                    markers.append("completed")

                target = "answer" if "answer" in markers else None
                if target is None:
                    for name in reversed(markers):
                        if name != "completed":
                            target = name
                            break

                if target is None:
                    return output

                parts = []
                for name in markers:
                    if name == "completed":
                        content = ""
                    elif name == target:
                        content = output.strip()
                    else:
                        content = ""
                    parts.append(f"[[ ## {name} ## ]]\n{content}".rstrip())
                return "\n\n".join(parts).strip()

            def _extract_prompt(
                self,
                prompt: str | None,
                messages: list[dict[str, Any]] | None,
            ) -> str:
                """Extract prompt from either prompt string or messages."""
                if prompt is None and messages:
                    for msg in reversed(messages):
                        if msg.get("role") == "user":
                            prompt = msg.get("content", "")
                            break

                if prompt is None:
                    raise ValueError("Either prompt or messages must be provided")
                return prompt

            def _extract_outputs(self, results: list) -> list[str]:
                """Extract output strings from inference results."""
                outputs = []
                for i, result in enumerate(results):
                    if result and len(result.messages) > 0:
                        assistant_msg = None
                        for msg in reversed(result.messages):
                            if msg.role == Role.ASSISTANT:
                                assistant_msg = msg.content
                                break

                        if assistant_msg:
                            outputs.append(assistant_msg)
                        else:
                            logger.warning(f"No assistant message in result {i + 1}.")
                            outputs.append("")
                            self.failed_calls += 1
                    else:
                        logger.warning(f"Empty inference result for prompt {i + 1}.")
                        outputs.append("")
                        self.failed_calls += 1
                return outputs

            def _build_response(
                self,
                prompt: str,
                outputs: list[str],
                kwargs: dict[str, Any],
            ) -> LMResponse:
                """Build the LMResponse object."""
                prompt_tokens = self.bridge._estimate_token_count(prompt)
                completion_tokens = sum(
                    self.bridge._estimate_token_count(o) for o in outputs
                )

                for output in outputs:
                    self.history.append(
                        {
                            "prompt": prompt,
                            "response": output,
                            "kwargs": kwargs,
                            "call_num": self.num_calls,
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens // len(outputs)
                            if outputs
                            else 0,
                        }
                    )

                return LMResponse(
                    choices=[{"text": output} for output in outputs],
                    usage={
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                    },
                    model=self.bridge.config.model.model_name,
                    cache_hit=False,
                    _hidden_params={"response_cost": None},
                )

            async def aforward(
                self,
                prompt: str | None = None,
                messages: list[dict[str, Any]] | None = None,
                **kwargs,
            ) -> LMResponse:
                """Async generate text from prompt(s)."""
                return await asyncio.to_thread(
                    self.forward, prompt=prompt, messages=messages, **kwargs
                )

            def get_stats(self) -> dict[str, Any]:
                """Get usage statistics."""
                total_prompt_tokens = sum(
                    entry.get("prompt_tokens", 0) for entry in self.history
                )
                total_completion_tokens = sum(
                    entry.get("completion_tokens", 0) for entry in self.history
                )

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
                    "total_tokens": total_prompt_tokens + total_completion_tokens,
                }

            def dump_state(self) -> dict[str, Any]:
                """Dump LM state for serialization."""
                return {
                    "model": self.bridge.config.model.model_name,
                    "generation_config": {
                        "temperature": self.bridge.config.generation.temperature,
                        "max_new_tokens": self.bridge.config.generation.max_new_tokens,
                        "top_p": self.bridge.config.generation.top_p,
                    },
                    "num_calls": self.num_calls,
                    "failed_calls": self.failed_calls,
                    "history_length": len(self.history),
                    "stats": self.get_stats(),
                }

            def load_state(self, state: dict[str, Any]) -> "OumiDSPyLM":
                """Load LM state from serialization."""
                if state.get("model") != self.bridge.config.model.model_name:
                    logger.warning(
                        f"Loading state for model '{state.get('model')}' "
                        f"but current model is '{self.bridge.config.model.model_name}'."
                    )

                self.num_calls = state.get("num_calls", 0)
                self.failed_calls = state.get("failed_calls", 0)
                logger.info(
                    f"Loaded LM state: {self.num_calls} calls, "
                    f"{self.failed_calls} failures"
                )
                return self

        return OumiDSPyLM()

    # -------------------------------------------------------------------------
    # Dataset Conversion
    # -------------------------------------------------------------------------

    def create_dspy_dataset(
        self, data: list[dict[str, Any]], input_key: str = "input"
    ) -> list:
        """Convert Oumi dataset format to DSPy examples.

        Args:
            data: List of dataset examples with 'input' and 'output' fields.
            input_key: Key for input field (default: 'input').

        Returns:
            List of DSPy Example objects.
        """
        try:
            import dspy
        except ImportError:
            raise ImportError(_DSPY_INSTALL_ERROR)

        examples = []
        skipped = 0

        for i, item in enumerate(data):
            example = self._convert_item_to_example(dspy, item, i)
            if example:
                examples.append(example)
            else:
                skipped += 1

        if skipped > 0:
            logger.warning(
                f"Skipped {skipped}/{len(data)} examples during DSPy conversion. "
                f"Converted {len(examples)} examples successfully."
            )

        return examples

    def _convert_item_to_example(self, dspy, item: dict, index: int):
        """Convert a single item to a DSPy Example."""
        if "input" in item and "output" in item:
            try:
                return dspy.Example(
                    question=item["input"], answer=item["output"]
                ).with_inputs("question")
            except Exception as e:
                logger.warning(f"Failed to convert example {index + 1}: {e}")
                return None

        if "messages" in item:
            try:
                conv = Conversation(**item)
                question, answer = "", ""
                for msg in conv.messages:
                    if msg.role == Role.USER:
                        question = msg.content
                    elif msg.role == Role.ASSISTANT:
                        answer = msg.content

                if question and answer:
                    return dspy.Example(question=question, answer=answer).with_inputs(
                        "question"
                    )
                else:
                    logger.warning(f"Example {index + 1} has no valid Q&A pair")
            except Exception as e:
                logger.warning(
                    f"Failed to parse Conversation for example {index + 1}: {e}"
                )
            return None

        logger.warning(
            f"Example {index + 1} has neither 'input'/'output' nor 'messages' fields"
        )
        return None

    # -------------------------------------------------------------------------
    # Metric Creation (Simplified - no signature inspection)
    # -------------------------------------------------------------------------

    def create_metric(
        self,
        metric_fn: Callable[[list[str], list[str]], float],
        support_gepa_feedback: bool = False,
    ):
        """Create a DSPy-compatible metric function.

        Args:
            metric_fn: Oumi metric function (predictions, references) -> score.
            support_gepa_feedback: If True, passes GEPA feedback params to metric.
                The metric_fn must accept extra params: trace, pred_name, pred_trace.

        Returns:
            DSPy-compatible metric function.
        """
        if support_gepa_feedback:
            logger.debug("Creating GEPA-compatible metric with feedback support.")

        def dspy_metric(
            example, prediction, trace=None, pred_name=None, pred_trace=None
        ):
            """DSPy metric wrapper."""
            ground_truth = [getattr(example, "answer", "")]
            pred = [
                getattr(prediction, "answer", None)
                or getattr(prediction, "rationale", "")
            ]

            try:
                if support_gepa_feedback and pred_name is not None:
                    # GEPA passes feedback parameters - forward them if supported
                    # pyright: ignore[reportCallIssue]
                    return metric_fn(
                        pred,
                        ground_truth,
                        trace=trace,  # type: ignore[call-arg]
                        pred_name=pred_name,  # type: ignore[call-arg]
                        pred_trace=pred_trace,  # type: ignore[call-arg]
                    )
                return metric_fn(pred, ground_truth)
            except TypeError:
                # Metric doesn't support extra params, call without them
                return metric_fn(pred, ground_truth)
            except Exception as e:
                logger.warning(f"Metric evaluation failed: {e}. Returning 0.0")
                return 0.0

        return dspy_metric

    # -------------------------------------------------------------------------
    # Program Creation
    # -------------------------------------------------------------------------

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
            raise ImportError(_DSPY_INSTALL_ERROR)

        class SimpleQA(dspy.Module):
            """Simple Q&A program for prompt optimization."""

            def __init__(self, sig):
                super().__init__()
                self.predictor = dspy.ChainOfThought(sig)

            def forward(self, question):
                """Forward pass."""
                return self.predictor(question=question)

        return SimpleQA(signature)

    # -------------------------------------------------------------------------
    # DSPy Setup
    # -------------------------------------------------------------------------

    def setup_dspy(self):
        """Setup DSPy with Oumi LM."""
        try:
            import dspy
        except ImportError:
            raise ImportError(_DSPY_INSTALL_ERROR)

        lm = self.create_dspy_lm()
        dspy.settings.configure(lm=lm)
        return lm


# =============================================================================
# Backwards Compatibility - Factory Function
# =============================================================================


def create_oumi_dspy_lm(bridge: OumiDSPyBridge):
    """Create a DSPy LM that wraps Oumi inference engines.

    This function is kept for backwards compatibility.
    Prefer using bridge.create_dspy_lm() directly.

    Args:
        bridge: The OumiDSPyBridge instance to use for inference.

    Returns:
        An instance of OumiDSPyLM configured with the bridge.
    """
    return bridge.create_dspy_lm()
