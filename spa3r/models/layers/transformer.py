from functools import partial
from typing import Callable

import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_

from . import Block, Mlp, RotaryPositionEmbedding2D


def named_apply(fn: Callable,
                module: nn.Module,
                name="",
                depth_first=True,
                include_root=False) -> nn.Module:
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = ".".join((name, child_name)) if name else child_name
        named_apply(
            fn=fn,
            module=child_module,
            name=child_name,
            depth_first=depth_first,
            include_root=True,
        )
    if depth_first and include_root:
        fn(module=module, name=name)
    return module


def init_weights_vit(module: nn.Module, name: str = ""):
    """ViT weight initialization, original timm impl (for reproducibility)"""
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class Transformer(nn.Module):

    def __init__(
        self,
        embed_dim=768,
        depth=12,
        num_heads=12,
        ffn_ratio=4,
        qkv_bias=True,
        qk_norm=False,
        ffn_bias=True,
        proj_bias=True,
        drop_path_rate=0.,
        act_layer=nn.GELU,
        block_fn=Block,
        ffn_layer=Mlp,
        rope_freq=100,
        num_register_tokens=0,
    ):
        super().__init__()
        self.n_blocks = depth
        self.num_heads = num_heads
        self.num_register_tokens = num_register_tokens

        assert num_register_tokens >= 0
        self.register_tokens = nn.Parameter(
            torch.zeros(1, num_register_tokens, embed_dim)
        ) if num_register_tokens else None

        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.norm = norm_layer(embed_dim)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)
               ]  # stochastic depth decay rule

        rope = RotaryPositionEmbedding2D(frequency=rope_freq) if rope_freq > 0 else None

        blocks_list = [
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                ffn_ratio=ffn_ratio,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop_path=dpr[i],
                act_layer=act_layer,
                norm_layer=norm_layer,
                ffn_layer=ffn_layer,
                rope=rope,
            ) for i in range(depth)
        ]
        self.blocks = nn.ModuleList(blocks_list)

        self.init_weights()

    def init_weights(self):
        if self.register_tokens is not None:
            nn.init.normal_(self.register_tokens, std=1e-6)
        named_apply(init_weights_vit, self)

    def forward(self, x, pos=None, **kwargs):
        if self.register_tokens is not None:
            x = torch.cat((self.register_tokens.expand(x.size(0), -1, -1), x), dim=1)
            if pos is not None:
                pos = pos + 1
                pos_special = torch.zeros(
                    x.size(0), self.register_tokens.size(1), 2
                ).to(pos)
                pos = torch.cat([pos_special, pos], dim=1)
        for blk in self.blocks:
            x = blk(x, pos=pos, **kwargs)
        x_norm = self.norm(x)
        return x_norm[:, self.num_register_tokens:]
