import torch
import torch.utils.checkpoint
from torch import nn
import math
import numpy as np


# https://github.com/facebookresearch/mae/blob/efb2a8062c206524e35e47d04501ed4f544c0ae8/util/pos_embed.py#L20
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])

    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class CrossAttention(nn.Module):

    def __init__(self, q_dim, kv_dim, hidden_dim, num_heads, attention_bias=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = self.hidden_dim // self.num_heads

        if (self.head_dim * self.num_heads) != self.hidden_dim:
            raise ValueError(
                f"hidden_dim must be divisible by num_heads (got `hidden_dim`: {self.hidden_dim}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = nn.Sequential(nn.LayerNorm(q_dim), nn.Linear(q_dim, self.num_heads * self.head_dim, bias=attention_bias))
        self.k_proj = nn.Sequential(nn.LayerNorm(kv_dim), nn.Linear(kv_dim, self.num_heads * self.head_dim, bias=attention_bias))
        self.v_proj = nn.Sequential(nn.LayerNorm(kv_dim), nn.Linear(kv_dim, self.num_heads * self.head_dim, bias=attention_bias))
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, q_dim, bias=attention_bias)

    def forward(
        self,
        vision_latents, queries, attention_mask
    ):
        
        bsz, q_len, _ = queries.size()
        bsz, v_len, _ = vision_latents.size()

        query_states = self.q_proj(queries)
        key_states = self.k_proj(vision_latents)
        value_states = self.v_proj(vision_latents)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, v_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, v_len, self.num_heads, self.head_dim).transpose(1, 2)


        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, v_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, v_len)}, but is {attention_mask.size()}"
                )

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_dim)

        attn_output = self.o_proj(attn_output)

        return attn_output
    

class AggregationBlock(nn.Module):
    def __init__(self, attention, q_dim, kv_dim, hidden_dim, num_heads, attention_bias=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = self.hidden_dim // self.num_heads

        if (self.head_dim * self.num_heads) != self.hidden_dim:
            raise ValueError(
                f"hidden_dim must be divisible by num_heads (got `hidden_dim`: {self.hidden_dim}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.attention = attention
        if attention:
            self.attention_layer = CrossAttention(q_dim, kv_dim, hidden_dim, num_heads, attention_bias)
        else:
            self.attention_layer = MLP(kv_dim, q_dim, q_dim)        

    def forward(
        self,
        vision_latents, queries, attention_mask
    ):
        if self.attention:
            queries = self.attention_layer(vision_latents, queries, attention_mask)
        else:
            queries = self.attention_layer(vision_latents)

        return queries


class MultiKVCrossAttention(nn.Module):

    def __init__(self, q_dim, kv_dim_list, hidden_dim, num_heads, attention_bias=False):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = self.hidden_dim // self.num_heads

        if (self.head_dim * self.num_heads) != self.hidden_dim:
            raise ValueError(
                f"hidden_dim must be divisible by num_heads (got `hidden_dim`: {self.hidden_dim}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = nn.Sequential(nn.LayerNorm(q_dim), nn.Linear(q_dim, self.num_heads * self.head_dim, bias=attention_bias))
        self.num_of_kvs = len(kv_dim_list)
        for i, kv_dim in enumerate(kv_dim_list):
            setattr(self, 'k_proj_{}'.format(i), nn.Sequential(nn.LayerNorm(kv_dim), nn.Linear(kv_dim, self.num_heads * self.head_dim, bias=attention_bias)))
            setattr(self, 'v_proj_{}'.format(i), nn.Sequential(nn.LayerNorm(kv_dim), nn.Linear(kv_dim, self.num_heads * self.head_dim, bias=attention_bias)))
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, q_dim, bias=attention_bias)

    def forward(
        self,
        queries, *vision_latents_attention_mask_list,
    ):
        
        vision_latents_list = vision_latents_attention_mask_list[:self.num_of_kvs]
        attention_mask_list = vision_latents_attention_mask_list[self.num_of_kvs:]
        
        bsz, q_len, _ = queries.size()

        query_states = self.q_proj(queries)
        key_states = torch.cat([getattr(self, 'k_proj_{}'.format(i))(vision_latents_list[i]) for i in range(self.num_of_kvs)], dim=1)
        value_states = torch.cat([getattr(self, 'v_proj_{}'.format(i))(vision_latents_list[i]) for i in range(self.num_of_kvs)], dim=1)

        v_len = key_states.shape[1]

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, v_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, v_len, self.num_heads, self.head_dim).transpose(1, 2)

        # if kv_weight is not None:
        #     kv_weight = kv_weight.unsqueeze(1).expand(-1, self.num_heads, -1, -1)

        attention_mask = torch.cat(attention_mask_list, dim=-1)

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, v_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, v_len)}, but is {attention_mask.size()}"
                )

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
        )
        # attn_output = spda(
        #     query_states,
        #     key_states,
        #     value_states,
        #     attn_mask=attention_mask,
        #     additional_score=kv_weight
        # )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_dim)

        attn_output = self.o_proj(attn_output)

        return attn_output


class MLP(nn.Module):
    def __init__(self, d_in, d_hidden, d_out):
        super().__init__() 
        self.linear_1 = nn.Linear(d_in, d_hidden, bias=False)
        self.act = nn.GELU()
        self.linear_2 = nn.Linear(d_hidden, d_out, bias=False)

    def forward(self, x):
        return self.linear_2(self.act(self.linear_1(x)))


class VisionCrossAttentionLayer(nn.Module):
    def __init__(self, q_dim, context_dim, kv_dim_list, kv_size_list, hidden_dim = 1024, layer_idx=0):
        super().__init__()
        num_heads = 16
        self.num_of_kvs = len(kv_dim_list)

        self.proj_context = nn.Linear(context_dim, hidden_dim, bias=False)
        self.proj_in = nn.Linear(q_dim+hidden_dim, hidden_dim, bias=False)
        # if self.num_of_kvs > 1:
        #     self.weight_mlp = MLP(q_dim+hidden_dim, hidden_dim, self.num_of_kvs)
        #     self.tower_weight = nn.Parameter(torch.zeros((self.num_of_kvs)))
        self.proj_out = MLP(hidden_dim, hidden_dim, q_dim)

        self.norm = nn.LayerNorm(hidden_dim)

        self.cross_attn = MultiKVCrossAttention(hidden_dim, kv_dim_list, hidden_dim, num_heads)
        self.kv_size_list = kv_size_list
        for i, kv_size in enumerate(kv_size_list):
            if kv_size > 1:
                setattr(self, "pos_embed_{}".format(i), nn.Parameter(torch.randn(kv_size**2, hidden_dim)))
                # self.register_buffer("pos_embed_{}".format(i), torch.from_numpy(get_2d_sincos_pos_embed(hidden_dim, kv_size)).float(), persistent=False)

    def forward(
        self,
        queries,
        context_feature,
        *vision_latents_attention_mask_list,
    ) -> torch.FloatTensor:

        residual = queries
        # queries = self.proj_in(queries)
        context_feature = self.proj_context(context_feature)
        # queries = queries + context_feature
        queries = torch.cat([queries, context_feature], -1)

        # if self.num_of_kvs > 1:
        #     kv_weight = self.weight_mlp(queries) # B * 1 * num_tower
        #     kv_weight = kv_weight + self.tower_weight.view(1, 1, -1)
        #     kv_weight = kv_weight.softmax(-1)
        #     kv_number_list = [size**2 for size in self.kv_size_list]
        #     kv_weight = torch.repeat_interleave(kv_weight, torch.tensor(kv_number_list).to(kv_weight.device), dim=-1)
        # else:
        #     kv_weight = None

        queries = self.proj_in(queries)

        vision_latents_list = vision_latents_attention_mask_list[:self.num_of_kvs]
        attention_mask_list = vision_latents_attention_mask_list[self.num_of_kvs:]

        attention_mask_list_reshaped = []
        if attention_mask_list is not None:
            for attention_mask in attention_mask_list:
                attention_mask = attention_mask.view(attention_mask.shape[0], 1, 1, -1)
                attention_mask = attention_mask.expand(-1, -1, queries.shape[1], -1)
                attention_mask_list_reshaped.append(attention_mask)

        vision_latents_pos_list = []
        for i, vision_latents in enumerate(vision_latents_list):
            if vision_latents.shape[1] > 1:
                vision_latents_pos_list.append(vision_latents + getattr(self, "pos_embed_{}".format(i))[None, :, :].to(vision_latents.dtype))
            else:
                vision_latents_pos_list.append(vision_latents)

        # Cross Attention
        attention_output = self.cross_attn(
            queries,
            *vision_latents_pos_list,
            *attention_mask_list_reshaped
        )

        # attention_output = (attention_output * combination_weight).sum(2)
        queries = queries + attention_output

        queries = self.norm(queries)

        queries = self.proj_out(queries)

        queries = queries + residual

        return queries


class VisionAggregationLayer(nn.Module):
    def __init__(self, q_dim, context_dim, kv_dim_list, kv_size_list, hidden_dim = 1024, layer_idx=0):
        super().__init__()
        num_heads = 16
        self.num_of_kvs = len(kv_dim_list)

        self.proj_context = nn.Linear(context_dim, hidden_dim, bias=False)
        self.proj_in = nn.Linear(q_dim+hidden_dim, hidden_dim, bias=False)

        self.proj_out = MLP(hidden_dim, hidden_dim, q_dim)

        self.norm = nn.LayerNorm(hidden_dim)

        if self.num_of_kvs > 1:
            self.weight_mlp = MLP(q_dim+hidden_dim, hidden_dim, self.num_of_kvs)

        for i, kv_size in enumerate(kv_size_list):
            if kv_size > 1:
                setattr(self, "pos_embed_{}".format(i), nn.Parameter(torch.randn(kv_size**2, hidden_dim)))
                setattr(self, "aggregate_{}".format(i), AggregationBlock(True, hidden_dim, kv_dim_list[i], hidden_dim, num_heads))
            else:
                setattr(self, "aggregate_{}".format(i), AggregationBlock(False, hidden_dim, kv_dim_list[i], hidden_dim, num_heads))

    def forward(
        self,
        queries,
        context_feature,
        *vision_latents_attention_mask_list,
    ) -> torch.FloatTensor:

        residual = queries
        # queries = self.proj_in(queries)
        context_feature = self.proj_context(context_feature)
        # queries = queries + context_feature
        queries = torch.cat([queries, context_feature], -1)

        if self.num_of_kvs > 1:
            combination_weight = self.weight_mlp(queries).softmax(-1) # B * 1 * num_tower
            combination_weight = combination_weight.unsqueeze(-1)
        else:
            combination_weight = 1

        queries = self.proj_in(queries)

        vision_latents_list = vision_latents_attention_mask_list[:self.num_of_kvs]
        attention_mask_list = vision_latents_attention_mask_list[self.num_of_kvs:]

        attention_mask_list_reshaped = []
        if attention_mask_list is not None:
            for attention_mask in attention_mask_list:
                attention_mask = attention_mask.view(attention_mask.shape[0], 1, 1, -1)
                attention_mask = attention_mask.expand(-1, -1, queries.shape[1], -1)
                attention_mask_list_reshaped.append(attention_mask)

        vision_latents_pos_list = []
        for i, vision_latents in enumerate(vision_latents_list):
            if vision_latents.shape[1] > 1:
                vision_latents_pos_list.append(vision_latents + getattr(self, "pos_embed_{}".format(i))[None, :, :].to(vision_latents.dtype))
            else:
                vision_latents_pos_list.append(vision_latents)

        aggregated_vision_latents_list = []
        for i, (vision_latents, attention_mask) in enumerate(zip(vision_latents_pos_list,attention_mask_list_reshaped)):
            aggregated_vision_latents_list.append(getattr(self, "aggregate_{}".format(i))(vision_latents, queries, attention_mask))

        aggregated_vision_latents = torch.stack(aggregated_vision_latents_list, 2)

        queries = queries + (aggregated_vision_latents * combination_weight).sum(2)

        queries = self.norm(queries)

        queries = self.proj_out(queries)

        queries = queries + residual

        return queries

class VisionTokenSampler(nn.Module):
    def __init__(self, q_dim, context_dim, kv_dim_list, kv_size_list, vision_hidden_size, num_of_layers=1, layer_type="joint"):
        super().__init__()
        assert layer_type in ['joint', 'sep']
        if layer_type == 'joint':
            self.layers = nn.ModuleList([VisionCrossAttentionLayer(q_dim, context_dim, kv_dim_list, kv_size_list, vision_hidden_size, idx) for idx in range(num_of_layers)])
        else:
            self.layers = nn.ModuleList([VisionAggregationLayer(q_dim, context_dim, kv_dim_list, kv_size_list, vision_hidden_size, idx) for idx in range(num_of_layers)])

    def forward(self, queries, context_feature, *vision_latents_attention_mask_list):
        for layer in self.layers:
            queries = layer(queries, context_feature, *vision_latents_attention_mask_list)
        return queries
