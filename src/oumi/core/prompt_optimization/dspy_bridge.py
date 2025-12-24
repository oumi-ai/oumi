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

from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

from oumi.core.configs.prompt_config import PromptOptimizationConfig
from oumi.core.types.conversation import Conversation, Role
from oumi.utils.logging import logger

# Tokenizer strategy registry for cleaner tokenizer loading
OPENAI_MODEL_PREFIXES = ("gpt-", "o1-", "text-", "davinci", "curie", "babbage", "ada")
REMOTE_PROVIDERS = ("claude", "gemini", "llama", "mistral", "anthropic")


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
        if self._tokenizer is not None:
            return self._tokenizer

        model_name = self.config.model.model_name.lower()

        # Try tiktoken for OpenAI models
        if any(prefix in model_name for prefix in OPENAI_MODEL_PREFIXES):
            self._tokenizer = self._try_tiktoken(model_name)
            if self._tokenizer:
                return self._tokenizer

        # Try HuggingFace for non-remote models
        if not any(provider in model_name for provider in REMOTE_PROVIDERS):
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
            logger.info("tiktoken not installed. Token counts will be estimated.")
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
                # tiktoken has encode but not add_special_tokens
                if hasattr(tokenizer, "encode") and not hasattr(
                    tokenizer, "add_special_tokens"
                ):
                    return len(tokenizer.encode(text))
                else:
                    return len(tokenizer.encode(text, add_special_tokens=True))  # type: ignore[call-arg]
            except Exception as e:
                logger.debug(f"Tokenizer encode failed: {e}. Using fallback.")

        # Fallback: ~1.3 tokens per word for modern tokenizers
        return max(1, int(len(text.split()) * 1.3))

    @contextmanager
    def _temporary_generation_params(self, **kwargs):
        """Temporarily override generation parameters.

        Args:
            **kwargs: Generation parameters to temporarily override.

        Yields:
            None
        """
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

    def create_dspy_lm(self):
        """Create a DSPy language model wrapper.

        Returns:
            A DSPy-compatible LM object that uses Oumi's inference engine.
        """
        from oumi.core.prompt_optimization.oumi_dspy_lm import create_oumi_dspy_lm

        return create_oumi_dspy_lm(self)

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
        """Convert a single item to a DSPy Example.

        Args:
            dspy: The dspy module.
            item: Dictionary with input/output or messages.
            index: Item index for logging.

        Returns:
            DSPy Example or None if conversion fails.
        """
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

    def create_metric(
        self,
        metric_fn: Callable[[list[str], list[str]], float],
        support_gepa_feedback: bool = True,
    ):
        """Create a DSPy-compatible metric function.

        Args:
            metric_fn: Oumi metric function.
            support_gepa_feedback: Whether to support GEPA's extended signature.

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
            logger.info("Using GEPA-compatible metric with feedback support.")

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
                if supports_gepa and pred_name is not None:
                    return metric_fn(  # type: ignore[call-arg]
                        pred,
                        ground_truth,
                        trace=trace,  # type: ignore[call-arg]
                        pred_name=pred_name,  # type: ignore[call-arg]
                        pred_trace=pred_trace,  # type: ignore[call-arg]
                    )
                return metric_fn(pred, ground_truth)
            except Exception as e:
                logger.warning(f"Metric evaluation failed: {e}. Returning score of 0.0")
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

        lm = self.create_dspy_lm()
        dspy.settings.configure(lm=lm)  # type: ignore[attr-defined]
        return lm
