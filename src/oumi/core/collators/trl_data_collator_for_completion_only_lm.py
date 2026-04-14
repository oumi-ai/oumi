# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
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

import numpy as np
import torch
from transformers.data.data_collator import DataCollatorForLanguageModeling


class DataCollatorForCompletionOnlyLM(DataCollatorForLanguageModeling):
    """Data collator for completion-only training.

    Masks input labels so that the loss is only computed on specific
    tokens (typically assistant responses), while ignoring other tokens
    (system prompts, user messages, padding).

    The ``masking_method`` parameter selects the masking strategy:

    **``assistant_turn``**:
        Span-based masking for multi-turn and tool-calling conversations.
        Masks everything, then unmarks each assistant response span bounded
        by ``response_template`` .. ``end_of_turn_template`` (inclusive of EOT).
        Correctly handles interleaved tool results and parallel tool calls.

    **``assistant_turn_no_tools``**:
        Same as ``assistant_turn``, but additionally re-masks assistant
        turns that contain tool-call content. Requires
        ``tool_call_start_template``. Only natural-language responses
        contribute to the loss.

    **``final_assistant_turn``**:
        Masks all tokens before the *last* ``response_template`` occurrence.
        Only the final assistant response is trained on. Suitable for
        single-turn completions.

    Args:
        response_template: String or token IDs marking the start of an
            assistant response. Required for all modes.
        instruction_template: String or token IDs marking the start of a
            user instruction. Legacy — only used with the instruction+response
            fallback path.
        masking_method: One of ``"assistant_turn"``,
            ``"assistant_turn_no_tools"``, ``"final_assistant_turn"``.
            When None, inferred from template presence for backward compat.
        end_of_turn_template: String or token IDs marking the end of a
            conversational turn. Required for ``assistant_turn`` and
            ``assistant_turn_no_tools`` modes.
        tool_call_start_template: String or token IDs marking the start
            of a tool-call block. Required for ``assistant_turn_no_tools``.
        mlm: Whether to use masked language modeling. Default False.
        ignore_index: Label value for masked tokens. Default -100.
        padding_free: Remove padding and add position_ids. Default False.
    """

    def __init__(
        self,
        response_template: str | list[int],
        instruction_template: str | list[int] | None = None,
        *args,
        masking_method: str | None = None,
        end_of_turn_template: str | list[int] | None = None,
        tool_call_start_template: str | list[int] | None = None,
        mlm: bool = False,
        ignore_index: int = -100,
        padding_free: bool = False,
        **kwargs,
    ):
        """Initializes the DataCollatorForCompletionOnlyLM."""
        super().__init__(*args, mlm=mlm, **kwargs)

        self.instruction_template = instruction_template
        if isinstance(instruction_template, str):
            # The user provides a string, must tokenize
            self.instruction_token_ids = self.tokenizer.encode(
                self.instruction_template,  # type: ignore
                add_special_tokens=False,
            )
        else:
            # The user already provides the token ids
            self.instruction_token_ids = instruction_template

        self.response_template = response_template
        if isinstance(response_template, str):
            # The user provides a string, must tokenize
            self.response_token_ids = self.tokenizer.encode(
                self.response_template, add_special_tokens=False
            )
        else:
            # The user already provides the token ids
            self.response_token_ids = response_template

        # Tool-aware span-based masking parameters
        self.end_of_turn_template = end_of_turn_template
        if isinstance(end_of_turn_template, str):
            self.end_of_turn_token_ids: list[int] | None = self.tokenizer.encode(
                end_of_turn_template, add_special_tokens=False
            )
        elif end_of_turn_template is not None:
            self.end_of_turn_token_ids = list(end_of_turn_template)
        else:
            self.end_of_turn_token_ids = None

        _KNOWN_MASKING_METHODS = {
            "assistant_turn",
            "assistant_turn_no_tools",
            "final_assistant_turn",
            "_legacy_instruction_response",
        }

        # Infer masking_method from template presence for backward compatibility.
        if masking_method is not None:
            if masking_method not in _KNOWN_MASKING_METHODS:
                valid_methods = sorted(
                    _KNOWN_MASKING_METHODS - {"_legacy_instruction_response"}
                )
                raise ValueError(
                    f"Unknown masking_method='{masking_method}'. "
                    f"Must be one of: {valid_methods}"
                )
            self.masking_method = masking_method
        elif end_of_turn_template is not None:
            self.masking_method = "assistant_turn"
        elif instruction_template is None:
            self.masking_method = "final_assistant_turn"
        else:
            self.masking_method = "_legacy_instruction_response"

        if self.masking_method in ("assistant_turn", "assistant_turn_no_tools"):
            if end_of_turn_template is None:
                raise ValueError(
                    "end_of_turn_template must be provided "
                    f"when masking_method='{self.masking_method}'"
                )

        self.mask_tool_calls = self.masking_method == "assistant_turn_no_tools"
        self.tool_call_start_token_ids: list[int] | None = None
        if self.mask_tool_calls:
            if tool_call_start_template is None:
                raise ValueError(
                    "tool_call_start_template must be provided "
                    "when masking_method='assistant_turn_no_tools'"
                )
            if isinstance(tool_call_start_template, str):
                self.tool_call_start_token_ids = self.tokenizer.encode(
                    tool_call_start_template, add_special_tokens=False
                )
            else:
                self.tool_call_start_token_ids = list(tool_call_start_template)

        if (
            not self.mlm
            and self.instruction_template
            and self.tokenizer.pad_token_id == self.tokenizer.eos_token_id
        ):
            warnings.warn(
                "The pad_token_id and eos_token_id values of this tokenizer "
                "are identical. "
                "If you are planning for multi-turn training, "
                "it can result in the model continuously generating questions and "
                "answers without eos token. "
                "To avoid this, set the pad_token_id to a different value.",
                UserWarning,
            )

        self.ignore_index = ignore_index
        self.padding_free = padding_free

    @staticmethod
    def _find_pattern(seq: list[int], pattern: list[int]) -> list[int]:
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

    @staticmethod
    def _span_contains(
        seq: list[int], span_start: int, span_end: int, pattern: list[int]
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

    def _apply_span_masking(
        self, batch: dict[str, Any], examples: list[list[int] | Any | dict[str, Any]]
    ) -> None:
        """Apply span-based masking for tool-aware conversations.

        Masks all labels, then unmarks assistant response spans bounded by
        response_template and end_of_turn_template (inclusive — the EOT token
        is unmasked so the model learns to produce it).  Optionally re-masks
        spans that contain tool-call content.
        """
        resp_ids = self.response_token_ids
        eot_ids = self.end_of_turn_token_ids
        assert eot_ids is not None  # Caller checks end_of_turn_template is not None
        resp_len = len(resp_ids)
        pad_token_id = self.tokenizer.pad_token_id

        for i in range(len(examples)):
            # Step 1: mask everything.
            batch["labels"][i, :] = self.ignore_index

            seq: list[int] = batch["input_ids"][i].tolist()

            # Compute effective sequence length excluding trailing padding.
            # Prevents false matches when end_of_turn_token_ids overlaps
            # with the pad token (common: e.g. <|im_end|> = eos = pad).
            if pad_token_id is not None:
                n = len(seq)
                while n > 0 and seq[n - 1] == pad_token_id:
                    n -= 1
            else:
                n = len(seq)

            # Step 2: find every assistant response start position.
            resp_positions = self._find_pattern(seq[:n], resp_ids)

            if len(resp_positions) == 0:
                warnings.warn(
                    f"Could not find response template in the following instance: "
                    f"{self.tokenizer.decode(batch['input_ids'][i])}. "
                    "This instance will be ignored in loss calculation.",
                    UserWarning,
                )
                continue

            for resp_pos in resp_positions:
                content_start = resp_pos + resp_len

                # Step 3: find the next end_of_turn after content_start.
                eot_positions = self._find_pattern(seq[content_start:n], eot_ids)
                if eot_positions:
                    content_end = content_start + eot_positions[0]
                else:
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
                        continue

                # Step 5: unmask this assistant response span, including the
                # end-of-turn token so the model learns when to stop.
                if eot_positions:
                    eot_len = len(self.end_of_turn_token_ids)  # type: ignore
                    unmask_end = content_end + eot_len
                else:
                    # No EOT found — content_end == n (end of real content).
                    # Do NOT extend past n or we'd unmask into padding.
                    unmask_end = content_end
                batch["labels"][i, content_start:unmask_end] = batch["input_ids"][
                    i, content_start:unmask_end
                ]

    # ------------------------------------------------------------------
    # Main collation
    # ------------------------------------------------------------------

    def torch_call(
        self, examples: list[list[int] | Any | dict[str, Any]]
    ) -> dict[str, Any]:
        """Collates a list of examples into a batch."""
        batch = super().torch_call(examples)

        if self.masking_method in ("assistant_turn", "assistant_turn_no_tools"):
            self._apply_span_masking(batch, examples)
        elif self.masking_method == "final_assistant_turn":
            # Response-only: unmask only the final assistant response.
            for i in range(len(examples)):
                response_token_ids_start_idx = None

                for idx in np.where(batch["labels"][i] == self.response_token_ids[0])[
                    0
                ]:
                    # `response_token_ids` is `'### Response:\n'`,
                    # here we are just making sure that the token IDs match
                    if (
                        self.response_token_ids
                        == batch["labels"][i][
                            idx : idx + len(self.response_token_ids)
                        ].tolist()
                    ):
                        response_token_ids_start_idx = idx

                if response_token_ids_start_idx is None:
                    warnings.warn(
                        f"Could not find response key `{self.response_template}` "
                        "in the following instance: "
                        f"{self.tokenizer.decode(batch['input_ids'][i])}. "
                        "This instance will be ignored in loss "
                        "calculation. Note, if this happens often, consider "
                        "increasing the `max_length`.",
                        UserWarning,
                    )
                    batch["labels"][i, :] = self.ignore_index
                else:
                    response_token_ids_end_idx = response_token_ids_start_idx + len(
                        self.response_token_ids
                    )

                    # Make pytorch loss function ignore all tokens up through the end
                    # of the response key
                    batch["labels"][i, :response_token_ids_end_idx] = self.ignore_index

        else:
            for i in range(len(examples)):
                response_token_ids_idxs = []
                human_token_ids_idxs = []

                for assistant_idx in np.where(
                    batch["labels"][i] == self.response_token_ids[0]
                )[0]:
                    # find the indexes of the start of a response.
                    if (
                        self.response_token_ids
                        == batch["labels"][i][
                            assistant_idx : assistant_idx + len(self.response_token_ids)
                        ].tolist()
                    ):
                        response_token_ids_idxs.append(
                            assistant_idx + len(self.response_token_ids)
                        )

                if len(response_token_ids_idxs) == 0:
                    warnings.warn(
                        f"Could not find response key `{self.response_template}` "
                        "in the following instance: "
                        f"{self.tokenizer.decode(batch['input_ids'][i])}. "
                        "This instance will be ignored in loss "
                        "calculation. Note, if this happens often, consider "
                        "increasing the `max_length`.",
                        UserWarning,
                    )
                    batch["labels"][i, :] = self.ignore_index

                human_token_ids = self.instruction_token_ids
                for human_idx in np.where(batch["labels"][i] == human_token_ids[0])[0]:  # type: ignore
                    # find the indexes of the start of a human answer.
                    if (
                        human_token_ids
                        == batch["labels"][i][
                            human_idx : human_idx + len(human_token_ids)
                        ].tolist()
                    ):
                        human_token_ids_idxs.append(human_idx)

                if len(human_token_ids_idxs) == 0:
                    warnings.warn(
                        f"Could not find instruction key `{self.instruction_template}` "
                        "in the following instance: "
                        f"{self.tokenizer.decode(batch['input_ids'][i])}."
                        " This instance will be ignored in loss "
                        "calculation. Note, if this happens often, "
                        "consider increasing the `max_length`.",
                        UserWarning,
                    )
                    batch["labels"][i, :] = self.ignore_index

                if (
                    len(human_token_ids_idxs) > 0
                    and len(response_token_ids_idxs) > 0
                    and human_token_ids_idxs[0] > response_token_ids_idxs[0]
                ):
                    human_token_ids_idxs = [0] + human_token_ids_idxs

                for idx, (start, end) in enumerate(
                    zip(human_token_ids_idxs, response_token_ids_idxs)
                ):
                    # Make pytorch loss function ignore all non response tokens
                    if idx != 0:
                        batch["labels"][i, start:end] = self.ignore_index
                    else:
                        batch["labels"][i, :end] = self.ignore_index

                if len(response_token_ids_idxs) < len(human_token_ids_idxs):
                    batch["labels"][i, human_token_ids_idxs[-1] :] = self.ignore_index

        if self.padding_free:
            # remove padding, `attention_mask` and add `position_ids`
            attn_mask = batch.pop("attention_mask")
            batch["input_ids"] = batch["input_ids"][attn_mask.bool()].unsqueeze(0)
            batch["position_ids"] = (
                attn_mask.cumsum(1)[attn_mask.bool()].unsqueeze(0) - 1
            )
            batch["labels"] = batch["labels"][attn_mask.bool()].unsqueeze(0)
            batch["labels"][batch["position_ids"] == 0] = self.ignore_index

            # Calculate cumulative sequence lengths for queries and keys to prevent
            #  graph breaks during further computations.
            flattened_position_ids = batch["position_ids"].flatten()
            indices_q = torch.arange(
                flattened_position_ids.size(0),
                device=flattened_position_ids.device,
                dtype=torch.int32,
            )
            batch["cu_seq_lens_q"] = torch.cat(
                (
                    indices_q[flattened_position_ids == 0],
                    torch.tensor(
                        flattened_position_ids.size(),
                        device=flattened_position_ids.device,
                        dtype=torch.int32,
                    ),
                )
            ).unsqueeze(0)
            batch["cu_seq_lens_k"] = batch["cu_seq_lens_q"]

            # Determine maximum sequence lengths to prevent graph breaks during
            #  further computations.
            batch["max_length_k"] = torch.tensor(
                [flattened_position_ids.max().item() + 1]
            )
            batch["max_length_q"] = batch["max_length_k"]

        return batch
