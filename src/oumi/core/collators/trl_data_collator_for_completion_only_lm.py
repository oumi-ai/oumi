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

from oumi.core.configs.params.data_params import TrainTarget


class DataCollatorForCompletionOnlyLM(DataCollatorForLanguageModeling):
    """Data collator for completion-only training.

    Masks input labels so that the loss is only computed on specific
    tokens (typically assistant responses), while ignoring other tokens
    (system prompts, user messages, padding).

    The ``train_target`` parameter selects the training target:

    **``all_assistant_turns``**:
        Span-based masking for multi-turn and tool-calling conversations.
        Masks everything, then unmarks each assistant response span bounded
        by ``response_template`` .. ``end_of_turn_template`` (inclusive of EOT).
        Correctly handles interleaved tool results and parallel tool calls.

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
        train_target: One of ``"all_assistant_turns"``,
            ``"final_assistant_turn"``, ``"_legacy_instruction_response"``.
            Resolved by the builder before construction.
        end_of_turn_template: String or token IDs marking the end of a
            conversational turn. Required for ``all_assistant_turns`` mode.
        tool_response_template: String or token IDs opening a tool result that the
            chat template renders inside the assistant turn (e.g. gemma-4's
            ``<|tool_response>``). Resolved by the builder.
        end_of_tool_response_template: String or token IDs closing such a tool
            result. Both are needed to exclude tool results from the loss.
        mlm: Whether to use masked language modeling. Default False.
        ignore_index: Label value for masked tokens. Default -100.
        padding_free: Remove padding and add position_ids. Default False.
        *args: Forwarded unchanged to ``DataCollatorForLanguageModeling``.
        **kwargs: Forwarded unchanged to ``DataCollatorForLanguageModeling`` — this is
            how ``tokenizer``, ``pad_to_multiple_of`` and the rest of the base
            collator's settings are passed. Nothing here is read by this subclass.
    """

    _VALID_TRAIN_TARGETS = {t.value for t in TrainTarget} | {
        "_legacy_instruction_response",
    }

    def _tokenize_template(self, template: str | list[int] | None) -> list[int] | None:
        """Encode a template string into token IDs, or pass through if already IDs.

        Args:
            template: Boundary marker to tokenize. A string is encoded without special
                tokens; a list of token IDs is copied through unchanged, which is how
                the builder passes markers it resolved itself; None passes through.

        Returns:
            The token IDs, or None when `template` is None.
        """
        if template is None:
            return None
        if isinstance(template, str):
            return self.tokenizer.encode(template, add_special_tokens=False)
        return list(template)

    def __init__(
        self,
        response_template: str | list[int],
        instruction_template: str | list[int] | None = None,
        *args,
        train_target: str,
        end_of_turn_template: str | list[int] | None = None,
        tool_response_template: str | list[int] | None = None,
        end_of_tool_response_template: str | list[int] | None = None,
        mlm: bool = False,
        ignore_index: int = -100,
        padding_free: bool = False,
        **kwargs,
    ):
        """Initializes the DataCollatorForCompletionOnlyLM."""
        super().__init__(*args, mlm=mlm, **kwargs)

        # Tokenize templates.
        self.instruction_template = instruction_template
        self.instruction_token_ids = self._tokenize_template(instruction_template)
        self.response_template = response_template
        self.response_token_ids: list[int] = self._tokenize_template(response_template)  # type: ignore[assignment]
        self.end_of_turn_template = end_of_turn_template
        self.end_of_turn_token_ids = self._tokenize_template(end_of_turn_template)
        self.tool_response_token_ids = self._tokenize_template(tool_response_template)
        self.end_of_tool_response_token_ids = self._tokenize_template(
            end_of_tool_response_template
        )

        if train_target not in self._VALID_TRAIN_TARGETS:
            valid = sorted(self._VALID_TRAIN_TARGETS - {"_legacy_instruction_response"})
            raise ValueError(
                f"Unknown train_target='{train_target}'. Must be one of: {valid}"
            )
        self.train_target = train_target

        if self.train_target == "all_assistant_turns":
            if end_of_turn_template is None:
                raise ValueError(
                    "end_of_turn_template must be provided "
                    f"when train_target='{self.train_target}'"
                )
        if self.train_target == "_legacy_instruction_response":
            if instruction_template is None:
                raise ValueError(
                    "instruction_template must be provided "
                    f"when train_target='{self.train_target}'"
                )

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
        """Return all start positions where *pattern* appears in *seq*.

        Args:
            seq: Token IDs to search.
            pattern: Token IDs to find. An empty pattern matches nothing.

        Returns:
            Every index in `seq` where `pattern` starts, ascending. Overlapping
            occurrences are all reported -- searching ``[1, 1, 1]`` for ``[1, 1]``
            returns ``[0, 1]``, not just ``[0]``, because the scan does not skip
            past a match it has already found. Callers pass boundary markers that
            cannot overlap a copy of themselves, so this never comes up in practice.
        """
        plen = len(pattern)
        if plen == 0:
            return []
        first = pattern[0]
        positions = []
        for i in range(len(seq) - plen + 1):
            if seq[i] == first and seq[i : i + plen] == pattern:
                positions.append(i)
        return positions

    def _mask_tool_response_spans(
        self,
        batch: dict[str, Any],
        row_idx: int,
        row_token_ids: list[int],
        start_idx: int,
        end_idx: int,
    ) -> None:
        """Re-masks tool results that the chat template nested in an unmasked span.

        Some templates (e.g. gemma-4) render tool results inside the assistant turn, so
        the span between response_template and end_of_turn_template contains
        environment output the model never generates.

        An opener with no closer before `end_idx` masks through `end_idx`: truncation
        can cut a block in half, and over-masking is the safe direction.

        Args:
            batch: Collated batch. Modified in place: the labels of every bracketed
                tool result found in the span are set to ``ignore_index``.
            row_idx: Which example in the batch to mask.
            row_token_ids: That row's token IDs, as a Python list. Passed in rather
                than read off `batch` because the caller already has it decoded.
            start_idx: First position of the span to scan, inclusive. Callers pass the
                start of a just-unmasked assistant span.
            end_idx: Position to stop scanning at, exclusive.

        Returns:
            None. The masking is applied to `batch` in place.

        Examples:
            A gemma-4 style turn, where the model calls a tool, the environment
            answers inside the same turn, and the model then answers the user.
            Span masking has already unmasked the whole turn; this walks it and
            takes the tool result back out, leaving the call and the answer::

                <resp> call <open> result <close> answer <eot>
                       ^^^^ ^^^^^^^^^^^^^^^^^^^^ ^^^^^^
                       kept       masked          kept

            Parallel tool calls give two blocks in one turn. The loop resumes at
            `cursor = block_end`, so the second is found on the next pass::

                <resp> <open> a <close> <open> b <close> answer <eot>
                       ^^^^^^^^^^^^^^^^ ^^^^^^^^^^^^^^^^
                          masked            masked

            An opener whose closer was cut off by truncation masks everything to
            `end_idx`. Over-masking costs a little signal; under-masking would
            train on environment output, so the bias goes this way on purpose::

                <resp> call <open> result-cut-off-here|
                       ^^^^ ^^^^^^^^^^^^^^^^^^^^^^^^^^
                       kept        masked to end_idx
        """
        open_ids = self.tool_response_token_ids
        close_ids = self.end_of_tool_response_token_ids
        if not open_ids or not close_ids:
            return

        cursor = start_idx
        while cursor < end_idx:
            opens = self._find_pattern(row_token_ids[cursor:end_idx], open_ids)
            if not opens:
                return
            block_start = cursor + opens[0]
            closes = self._find_pattern(row_token_ids[block_start:end_idx], close_ids)
            block_end = block_start + closes[0] + len(close_ids) if closes else end_idx
            batch["labels"][row_idx, block_start:block_end] = self.ignore_index
            cursor = block_end

    def _apply_span_masking(
        self, batch: dict[str, Any], examples: list[list[int] | Any | dict[str, Any]]
    ) -> None:
        """Apply span-based masking for multi-turn conversations.

        Masks all labels, then unmarks assistant response spans bounded by
        response_template and end_of_turn_template (inclusive — the EOT token
        is unmasked so the model learns to produce it).

        Args:
            batch: Collated batch. Its ``labels`` are rewritten in place.
            examples: The pre-collation examples, used only for their count — the
                token IDs are read back off `batch` so that padding and truncation
                already applied by the base collator are accounted for.

        Returns:
            None. The masking is applied to `batch` in place.
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

                # Step 4: unmask this assistant response span, including the
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
                self._mask_tool_response_spans(batch, i, seq, content_start, unmask_end)

    # ------------------------------------------------------------------
    # Main collation
    # ------------------------------------------------------------------

    def torch_call(
        self, examples: list[list[int] | Any | dict[str, Any]]
    ) -> dict[str, Any]:
        """Collates a list of examples into a batch.

        Args:
            examples: Examples to collate, each a list of token IDs or a mapping
                holding at least ``input_ids``. Passed through to the base collator,
                which handles padding.

        Returns:
            The collated batch, with ``labels`` masked according to ``train_target``.
        """
        batch = super().torch_call(examples)

        if self.train_target == "all_assistant_turns":
            self._apply_span_masking(batch, examples)
        elif self.train_target == "final_assistant_turn":
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

                    # The unmasked tail is one assistant turn, so it carries the same
                    # nested-tool-result exposure as span masking.
                    self._mask_tool_response_spans(
                        batch,
                        i,
                        batch["input_ids"][i].tolist(),
                        response_token_ids_end_idx,
                        batch["input_ids"].shape[1],
                    )

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
