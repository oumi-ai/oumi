import torch
import torch.nn as nn
from einops import rearrange

# from timm.layers import LayerNorm, LayerNorm2d
# from timm.models.regnet import RegStage


def build_pos_embeds(num_input_tokens: int, vision_hidden_size: int):
    # pos emb
    pos_emb = torch.nn.Parameter(torch.zeros(1, num_input_tokens, vision_hidden_size))
    nn.init.trunc_normal_(pos_emb, mean=0.0, std=0.02)

    return pos_emb


def build_mlp(depth, hidden_size, output_hidden_size):
    layers = [nn.Linear(hidden_size, output_hidden_size)]
    for _ in range(1, depth):
        layers.append(nn.SiLU())
        layers.append(nn.Linear(output_hidden_size, output_hidden_size))
    return nn.Sequential(*layers)


class Projector(nn.Module):
    """Base projector class"""

    def __init__(
        self,
        encoder_hidden_size: int,
        num_input_tokens: int,
        num_queries: int,
        output_hidden_size: int,
    ):
        super().__init__()
        self.encoder_hidden_size = encoder_hidden_size
        self.num_input_tokens = num_input_tokens
        self.num_queries = num_queries
        self.hidden_size = 1024
        self.output_hidden_size = output_hidden_size

        # pos emb
        self.pos_emb = build_pos_embeds(num_input_tokens, encoder_hidden_size)
        # self.pos_emb = None

        self.build_net()

    def build_net(self):
        raise NotImplementedError

    def _forward(self, x):
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
        x: (B, L, encoder_hidden_size) tensor from the visual backbone (CLIP visual encoder), including cls token.
        """
        if self.pos_emb is not None:
            x = x + self.pos_emb

        dtype = x.dtype
        # x = self._forward(x.to(torch.float32))  # (B, L, output_hidden_size)
        x = self._forward(x)

        return x.to(dtype)


class ConvProjector(Projector):
    def _forward(self, x):
        hw = int(x.size(1) ** 0.5)
        x = rearrange(x, "b (h w) d -> b d h w", h=hw, w=hw)
        x = self.net(x)
        x = rearrange(x, "b d h w -> b (h w) d")
        x = self.readout(x)
        return x


class CAbstractor(nn.Module):
    """Base projector class"""

    def __init__(
        self,
        encoder_hidden_size: int,
        output_hidden_size: int,
    ):
        super().__init__()
        self.encoder_hidden_size = encoder_hidden_size
        self.hidden_size = 1024
        self.output_hidden_size = output_hidden_size

        # pos emb
        # self.pos_emb = build_pos_embeds(num_input_tokens, encoder_hidden_size)
        self.pos_emb = None

        self.downsamples = nn.Conv2d(
            encoder_hidden_size,
            self.hidden_size,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )

        self.readout = nn.Sequential(
            nn.Linear(self.hidden_size, output_hidden_size),
            nn.GELU(),
            nn.Linear(output_hidden_size, output_hidden_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.to(torch.float32)

        hw = int(x.size(1) ** 0.5)
        x = rearrange(x, "b (h w) d -> b d h w", h=hw, w=hw)
        x = self.downsamples(x)
        x = rearrange(x, "b d h w -> b (h w) d")
        x = self.readout(x)

        return x.to(dtype)


# class CAbstractor(ConvProjector):
#     """C-Abstractor"""
#     def build_net(self):
#         encoder_hidden_size = self.encoder_hidden_size
#         hidden_size = self.hidden_size
#         output_hidden_size = self.output_hidden_size
#         depth = 3
#         mlp_depth = 2

#         n_queries = self.num_queries
#         assert (n_queries ** 0.5).is_integer(), "n_queries must be square number"
#         hw = int(n_queries ** 0.5)

#         # RegBlock = ResBlock + SE
#         RegBlock = partial(
#             RegStage,
#             stride=1,
#             dilation=1,
#             act_layer=nn.SiLU,
#             norm_layer=LayerNorm2d,
#         )

#         s1 = RegBlock(
#             depth,
#             encoder_hidden_size,
#             hidden_size,
#         )
#         sampler = nn.AdaptiveAvgPool2d((hw, hw))
#         s2 = RegBlock(
#             depth,
#             hidden_size,
#             hidden_size,
#         )

#         self.net = nn.Sequential(s1, sampler, s2)
#         self.readout = build_mlp(mlp_depth, hidden_size, output_hidden_size)
