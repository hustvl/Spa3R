from functools import partial

import torch.nn as nn
from einops import repeat
from torch.utils.checkpoint import checkpoint
from vggt.models.vggt import VGGT, Aggregator

from .layers.block import Block


class MaskedAggregator(Aggregator):

    def __init__(
        self,
        embed_dim=1024,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        qk_norm=True,
        init_values=0.01,
        *args,
        **kwargs,
    ):
        super().__init__(
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            ffn_bias=ffn_bias,
            qk_norm=qk_norm,
            init_values=init_values,
            *args,
            **kwargs,
        )

        self.global_blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                ffn_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                init_values=init_values,
                qk_norm=qk_norm,
                rope=self.rope,
            ) for _ in range(self.depth)
        ])

    def forward(self, images, view_mask=None):
        if view_mask is None:
            return super().forward(images)

        original_fn = self._process_global_attention
        patched_fn = partial(self._process_global_attention, view_mask=view_mask)
        self._process_global_attention = patched_fn
        outputs = super().forward(images)
        self._process_global_attention = original_fn
        return outputs

    def _process_global_attention(
        self, tokens, B, S, P, C, global_idx, pos=None, view_mask=None
    ):
        if tokens.shape != (B, S * P, C):
            tokens = tokens.reshape(B, S, P, C).reshape(B, S * P, C)

        if pos is not None and pos.shape != (B, S * P, 2):
            pos = pos.reshape(B, S, P, 2).reshape(B, S * P, 2)

        attn_mask = None
        if view_mask is not None:
            attn_mask = self._generate_attention_mask(view_mask, P).unsqueeze(1)

        intermediates = []

        # by default, self.aa_block_size=1, which processes one block at a time
        for _ in range(self.aa_block_size):
            if self.training:
                tokens = checkpoint(
                    self.global_blocks[global_idx],
                    tokens,
                    pos=pos,
                    attn_mask=attn_mask,
                    use_reentrant=self.use_reentrant,
                )
            else:
                tokens = self.global_blocks[global_idx](
                    tokens, pos=pos, attn_mask=attn_mask)
            global_idx += 1
            intermediates.append(tokens.reshape(B, S, P, C))

        return tokens, global_idx, intermediates

    def _generate_attention_mask(self, view_mask, seq_len):
        query_unmasked = view_mask.unsqueeze(2)  # (bs, v, 1)
        key_masked = ~view_mask.unsqueeze(1)  # (bs, 1, v)
        attn_mask = query_unmasked | key_masked  # (bs, v, v)
        attn_mask = repeat(attn_mask,
                           'b v1 v2 -> b (v1 n1) (v2 n2)',
                           n1=seq_len,
                           n2=seq_len)
        return attn_mask


class MaskedVGGT(VGGT):

    def __init__(self, img_size=518, patch_size=14, embed_dim=1024, *args, **kwargs):
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            *args,
            **kwargs,
        )
        self.aggregator = MaskedAggregator(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
        )
