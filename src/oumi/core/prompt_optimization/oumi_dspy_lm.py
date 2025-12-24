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

"""DSPy LM wrapper for Oumi inference engines."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from oumi.core.types.conversation import Conversation, Message, Role
from oumi.utils.logging import logger

if TYPE_CHECKING:
    from oumi.core.prompt_optimization.dspy_bridge import OumiDSPyBridge


@dataclass
class LMResponse:
    """Response structure for DSPy compatibility."""

    choices: list[dict[str, str]]
    usage: dict[str, int]
    model: str
    cache_hit: bool = False
    _hidden_params: dict[str, Any] | None = None


def create_oumi_dspy_lm(bridge: "OumiDSPyBridge"):
    """Create a DSPy LM class that wraps Oumi inference engines.

    Args:
        bridge: The OumiDSPyBridge instance to use for inference.

    Returns:
        An instance of OumiDSPyLM configured with the bridge.
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
            """Initialize the LM wrapper.

            Args:
                bridge: OumiDSPyBridge instance for inference.
            """
            super().__init__(
                model=bridge.config.model.model_name, callbacks=bridge.callbacks
            )
            self.bridge = bridge
            self.history: list[dict[str, Any]] = []
            self.num_calls = 0
            self.failed_calls = 0

        def forward(
            self,
            prompt: str | None = None,
            messages: list[dict[str, Any]] | None = None,
            **kwargs,
        ) -> LMResponse:
            """Generate text from prompt(s).

            Args:
                prompt: Input prompt string.
                messages: Alternative message format.
                **kwargs: Additional generation parameters.

            Returns:
                LMResponse object in DSPy-compatible format.

            Raises:
                ValueError: If neither prompt nor messages is provided.
                RuntimeError: If inference fails.
            """
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
                    f"{self.failed_calls} total failures). "
                    f"Prompt: '{prompt[:100]}...'"
                )
                raise RuntimeError(
                    f"Inference failed after {self.num_calls} calls "
                    f"({self.failed_calls} failures): {e}"
                ) from e

            outputs = self._extract_outputs(results)
            response = self._build_response(prompt, outputs, kwargs)
            return response

        def _extract_prompt(
            self,
            prompt: str | None,
            messages: list[dict[str, Any]] | None,
        ) -> str:
            """Extract prompt from either prompt string or messages.

            Args:
                prompt: Direct prompt string.
                messages: Message format with role/content dicts.

            Returns:
                The extracted prompt string.

            Raises:
                ValueError: If no prompt can be extracted.
            """
            if prompt is None and messages:
                for msg in reversed(messages):
                    if msg.get("role") == "user":
                        prompt = msg.get("content", "")
                        break

            if prompt is None:
                raise ValueError("Either prompt or messages must be provided")
            return prompt

        def _extract_outputs(self, results: list) -> list[str]:
            """Extract output strings from inference results.

            Args:
                results: List of Conversation results from inference.

            Returns:
                List of output strings.
            """
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
                        logger.warning(
                            f"No assistant message found in result {i + 1}."
                        )
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
            """Build the LMResponse object.

            Args:
                prompt: The input prompt.
                outputs: List of generated outputs.
                kwargs: Generation parameters used.

            Returns:
                LMResponse object.
            """
            prompt_tokens = self.bridge._estimate_token_count(prompt)
            completion_tokens = sum(
                self.bridge._estimate_token_count(o) for o in outputs
            )
            total_tokens = prompt_tokens + completion_tokens

            # Log to history
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

            response = LMResponse(
                choices=[{"text": output} for output in outputs],
                usage={
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                },
                model=self.bridge.config.model.model_name,
                cache_hit=False,
                _hidden_params={"response_cost": None},
            )
            return response

        async def aforward(
            self,
            prompt: str | None = None,
            messages: list[dict[str, Any]] | None = None,
            **kwargs,
        ) -> LMResponse:
            """Async generate text from prompt(s).

            Args:
                prompt: Input prompt string.
                messages: Alternative message format.
                **kwargs: Additional generation parameters.

            Returns:
                LMResponse object in DSPy-compatible format.
            """
            return await asyncio.to_thread(
                self.forward, prompt=prompt, messages=messages, **kwargs
            )

        def get_stats(self) -> dict[str, Any]:
            """Get usage statistics including token counts.

            Returns:
                Dict with call counts, failure rate, and token usage.
            """
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
            """Dump LM state for serialization.

            Returns:
                Dict containing model name, config, and statistics.
            """
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
            """Load LM state from serialization.

            Args:
                state: Dictionary containing saved state.

            Returns:
                Self for method chaining.
            """
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

    return OumiDSPyLM(bridge)
