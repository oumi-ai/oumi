#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.generation.utils import GenerateOutput
from transformers.modeling_outputs import CausalLMOutputWithPast

from oumi.models.experimental.cambrian.utils import IS_XLA_AVAILABLE

from ..cambrian_arch import CambrianMetaForCausalLM, CambrianMetaModel
from .phi3 import Phi3Config, Phi3ForCausalLM, Phi3Model


class CambrianConfig(Phi3Config):
    model_type = "cambrian_phi3"

    debug = "debug"


class CambrianPhi3Model(CambrianMetaModel, Phi3Model):
    config_class = CambrianConfig

    def __init__(self, config: Phi3Config):
        super().__init__(config)


class CambrianPhi3ForCausalLM(Phi3ForCausalLM, CambrianMetaForCausalLM):
    config_class = CambrianConfig

    def __init__(self, config):
        super(Phi3ForCausalLM, self).__init__(config)

        self.model = CambrianPhi3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_aux_attention_masks_list: Optional[List[torch.Tensor]] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
                vision_tower_aux_feature_list,
                vision_tower_aux_attention_masks_list,
                final_vision_feature_size,
                global_context_feature,
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_aux_attention_masks_list,
                image_sizes,
            )

        if IS_XLA_AVAILABLE:
            # Very Important for TorchXLA
            # self.model.gradient_checkpointing = False

            from torch_xla.utils.checkpoint import checkpoint

            self.model._gradient_checkpointing_func = checkpoint

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if IS_XLA_AVAILABLE:
            output = super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                vision_tower_aux_feature_list=vision_tower_aux_feature_list,
                vision_tower_aux_attention_masks_list=vision_tower_aux_attention_masks_list,
                final_vision_feature_size=final_vision_feature_size,
                global_context_feature=global_context_feature,
            )

        else:
            output = super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                vision_tower_aux_feature_list=vision_tower_aux_feature_list
                if inputs_embeds is None
                else self.vision_tower_aux_feature_list,
                vision_tower_aux_attention_masks_list=vision_tower_aux_attention_masks_list
                if inputs_embeds is None
                else self.vision_tower_aux_attention_masks_list,
                final_vision_feature_size=final_vision_feature_size
                if inputs_embeds is None
                else self.final_vision_feature_size,
                global_context_feature=global_context_feature
                if inputs_embeds is None
                else self.global_context_feature,
            )

        return output

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _,
                vision_tower_aux_feature_list,
                vision_tower_aux_attention_masks_list,
                final_vision_feature_size,
                global_context_feature,
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes,
            )
            self.vision_tower_aux_feature_list = vision_tower_aux_feature_list
            self.vision_tower_aux_attention_masks_list = (
                vision_tower_aux_attention_masks_list
            )
            self.final_vision_feature_size = final_vision_feature_size
            self.global_context_feature = global_context_feature
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs
    ):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        if images is not None:
            inputs["images"] = images
        if image_sizes is not None:
            inputs["image_sizes"] = image_sizes
        return inputs


AutoConfig.register("cambrian_phi3", CambrianConfig)
AutoModelForCausalLM.register(CambrianConfig, CambrianPhi3ForCausalLM)

# Used in the following published models:
# nyu-visionx/cambrian-phi3-3b
