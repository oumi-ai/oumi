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

import warnings
from typing import Any

from transformers.data.data_collator import DataCollatorForLanguageModeling


class ToolAwareCompletionsCollator(DataCollatorForLanguageModeling):
    r"""Completion-only collator that correctly masks tool results and tool calls.

    The standard ``DataCollatorForCompletionOnlyLM`` uses template matching to
    decide which tokens to train on.  It works well for simple user/assistant
    conversations, but breaks when a third role (``tool``) sits between two
    assistant turns: the tool-result span ends up unmasked because the collator
    only knows about the user-role marker as an "instruction" boundary.

    This collator takes a different approach:

    1. Start with **all labels masked** (-100).
    2. Find every assistant response span by locating ``response_template``
       tokens and scanning forward to the next ``end_of_turn_template``.
    3. **Unmask** those spans so the model trains on them.
    4. Optionally **re-mask** spans that contain tool-call content (controlled
       by ``mask_tool_calls``), so the model only trains on plain-text replies.

    Because the algorithm never relies on user/instruction markers, it handles
    any number of tool turns, parallel tool calls, and multi-turn conversations
    correctly.

    Args:
        response_template: String or token-ID list that marks the *start* of an
            assistant response (e.g. ``"<|im_start|>assistant\n"`` for SmolLM2
            or ``"[/INST]"`` for Llama-2).
        end_of_turn_template: String or token-ID list that marks the *end* of a
            turn (e.g. ``"<|im_end|>"`` for SmolLM2 or ``"</s>"`` for Llama).
        mask_tool_calls: When ``True``, assistant spans that contain
            ``tool_call_start_template`` are re-masked.  Set this to ``True``
            if you only want to train on plain-text final answers.  Defaults to
            ``False`` (train on all assistant output including tool calls).
        tool_call_start_template: String or token-ID list that marks the start
            of a tool-call block inside an assistant turn (e.g.
            ``"<tool_call>"``).  Required when ``mask_tool_calls=True``.
        ignore_index: Value used for masked labels.  Must match the
            ``ignore_index`` of the loss function (default: -100).
    """

    def __init__(
        self,
        response_template: str | list[int],
        end_of_turn_template: str | list[int],
        *args,
        mask_tool_calls: bool = False,
        tool_call_start_template: str | list[int] | None = None,
        ignore_index: int = -100,
        mlm: bool = False,
        **kwargs,
    ):
        """Initializes ToolAwareCompletionsCollator."""
        super().__init__(*args, mlm=mlm, **kwargs)
        self.ignore_index = ignore_index

        if isinstance(response_template, str):
            self.response_token_ids: list[int] = self.tokenizer.encode(
                response_template, add_special_tokens=False
            )
        else:
            self.response_token_ids = list(response_template)

        if isinstance(end_of_turn_template, str):
            self.end_of_turn_token_ids: list[int] = self.tokenizer.encode(
                end_of_turn_template, add_special_tokens=False
            )
        else:
            self.end_of_turn_token_ids = list(end_of_turn_template)

        self.mask_tool_calls = mask_tool_calls
        self.tool_call_start_token_ids: list[int] | None = None
        if mask_tool_calls:
            if tool_call_start_template is None:
                raise ValueError(
                    "tool_call_start_template must be provided "
                    "when mask_tool_calls=True"
                )
            if isinstance(tool_call_start_template, str):
                self.tool_call_start_token_ids = self.tokenizer.encode(
                    tool_call_start_template, add_special_tokens=False
                )
            else:
                self.tool_call_start_token_ids = list(tool_call_start_template)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_pattern(self, seq: list[int], pattern: list[int]) -> list[int]:
        """Return all start positions where *pattern* appears in *seq*."""
        plen = len(pattern)
        if plen == 0:
            return []
        first = pattern[0]
        positions = []
        for i in range(len(seq) - plen + 1):
            if seq[i] == first and seq[i : i + plen] == pattern:
                positions.append(i)
        return positions

    def _span_contains(
        self, seq: list[int], span_start: int, span_end: int, pattern: list[int]
    ) -> bool:
        """Return True if *pattern* appears anywhere in seq[span_start:span_end]."""
        plen = len(pattern)
        if plen == 0:
            return False
        first = pattern[0]
        for i in range(span_start, span_end - plen + 1):
            if seq[i] == first and seq[i : i + plen] == pattern:
                return True
        return False

    # ------------------------------------------------------------------
    # Main collation
    # ------------------------------------------------------------------

    def torch_call(
        self, examples: list[list[int] | Any | dict[str, Any]]
    ) -> dict[str, Any]:
        """Collates examples and applies tool-aware label masking."""
        # Let the base class handle padding and create labels = input_ids.
        batch = super().torch_call(examples)

        resp_len = len(self.response_token_ids)
        pad_token_id = self.tokenizer.pad_token_id

        for i in range(len(examples)):
            # Step 1: mask everything.
            batch["labels"][i, :] = self.ignore_index

            seq: list[int] = batch["input_ids"][i].tolist()

            # Compute the effective sequence length excluding trailing padding.
            # This prevents false matches when end_of_turn_token_ids overlaps
            # with the pad token (common: e.g. <|im_end|> = eos = pad).
            if pad_token_id is not None:
                n = len(seq)
                while n > 0 and seq[n - 1] == pad_token_id:
                    n -= 1
            else:
                n = len(seq)

            # Step 2: find every assistant response start position
            # (within the non-padded region only).
            resp_positions = self._find_pattern(seq[:n], self.response_token_ids)

            if len(resp_positions) == 0:
                warnings.warn(
                    f"Could not find response template in the following instance: "
                    f"{self.tokenizer.decode(batch['input_ids'][i])}. "
                    "This instance will be ignored in loss calculation.",
                    UserWarning,
                )
                continue

            for resp_pos in resp_positions:
                # Content starts right after the response_template tokens.
                content_start = resp_pos + resp_len

                # Step 3: find the next end_of_turn after content_start
                # (within the non-padded region only).
                eot_positions = self._find_pattern(
                    seq[content_start:n], self.end_of_turn_token_ids
                )
                if eot_positions:
                    content_end = content_start + eot_positions[0]
                else:
                    # No closing marker found — unmask to end of real content.
                    content_end = n

                if content_start >= content_end:
                    continue

                # Step 4: optionally skip tool-call spans.
                if self.mask_tool_calls and self.tool_call_start_token_ids is not None:
                    if self._span_contains(
                        seq,
                        content_start,
                        content_end,
                        self.tool_call_start_token_ids,
                    ):
                        # Leave this span masked.
                        continue

                # Step 5: unmask this assistant response span.
                batch["labels"][i, content_start:content_end] = batch["input_ids"][
                    i, content_start:content_end
                ]

        return batch
