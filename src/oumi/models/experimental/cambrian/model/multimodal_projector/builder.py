import re

import torch
import torch.nn as nn

from .projectors import CAbstractor


class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": "identity"}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels), nn.GELU(), nn.Linear(channels, channels)
        )

    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)


class SEMLP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.se = nn.Sequential(
            nn.Linear(in_channels, in_channels, bias=False),
            nn.GELU(),
            nn.Linear(in_channels, in_channels, bias=False),
            nn.Sigmoid(),
        )

        self.proj = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.GELU(),
            nn.Linear(out_channels, out_channels),
        )

    def forward(self, x):
        global_x = torch.mean(x, 1, keepdim=True)
        weight = self.se(global_x)
        x = x * weight + x
        return self.proj(x)


def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, "mm_projector_type", "linear")

    if projector_type == "linear":
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    mlp_gelu_match = re.match(r"^mlp(\d+)x_gelu$", projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    if projector_type == "identity":
        return IdentityMap()

    if projector_type == "se_mlp":
        return SEMLP(config.mm_hidden_size, config.hidden_size)

    if projector_type == "CAbstractor":
        return CAbstractor(config.mm_hidden_size, config.hidden_size)

    raise ValueError(f"Unknown projector type: {projector_type}")
