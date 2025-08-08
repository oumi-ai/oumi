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
from typing import Any, Optional, Union

import numpy as np
import torch
from transformers.data.data_collator import DataCollatorForLanguageModeling


class DataCollatorForCompletionOnlyLM(DataCollatorForLanguageModeling):
    """Data collator used for completion tasks.

    Copied from `trl`'s `DataCollatorForCompletionOnlyLM` class.
    """

    def __init__(
        self,
        response_template: Union[str, list[int]],
        instruction_template: Optional[Union[str, list[int]]] = None,
        *args,
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

    def torch_call(
        self, examples: list[Union[list[int], Any, dict[str, Any]]]
    ) -> dict[str, Any]:
        """Collates a list of examples into a batch."""
        batch = super().torch_call(examples)

        if self.instruction_template is None:
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
