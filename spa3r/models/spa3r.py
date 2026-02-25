import logging
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torchvision.transforms.v2 import functional as TF
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

from .layers import Block, Mlp, PositionGetter, PRoPEAttention, Transformer, init_weights_vit
from .masked_vggt import MaskedVGGT
from .utils import get_camera_rays

logger = logging.getLogger(__name__)


class Spa3R(nn.Module):

    def __init__(
        self,
        embed_dim,
        mask_ratio=0.5,
        asymmetric_masking=True,
        num_queries=256,
        num_encoder_layers=6,
        num_decoder_layers=6,
        num_heads=12,
        qk_norm=True,
        num_register_tokens=4,
    ):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.asymmetric_masking = asymmetric_masking
        self.num_queries = num_queries

        vggt = MaskedVGGT.from_pretrained('facebook/VGGT-1B')
        for param in vggt.parameters():
            param.requires_grad = False
        logger.info('Loaded VGGT model with parameters frozen')
        aggr_dim = vggt.aggregator.patch_embed.embed_dim * 2  # noqa: for VGGT

        self.projection = nn.Linear(aggr_dim, embed_dim)
        self.query_embed = nn.Embedding(num_queries, embed_dim)
        self.pose_tokenizer = nn.Linear(3, embed_dim)
        self.position_getter = PositionGetter()

        self.encoder = Transformer(
            embed_dim,
            depth=num_encoder_layers,
            num_heads=num_heads,
            qk_norm=qk_norm,
            num_register_tokens=num_register_tokens,
        )
        self.decoder = Transformer(
            embed_dim,
            depth=num_decoder_layers,
            num_heads=num_heads,
            qk_norm=qk_norm,
            num_register_tokens=num_register_tokens,
            block_fn=partial(Block, attn_layer=PRoPEAttention),
            rope_freq=0,
        )

        self.heads = nn.ModuleDict()
        self.heads['geo'] = Mlp(embed_dim,
                                hidden_features=embed_dim * 4,
                                out_features=aggr_dim)

        dino = torch.hub.load('facebookresearch/dinov3', 'dinov3_vitl16')
        for param in dino.parameters():
            param.requires_grad = False
        logger.info('Loaded DINOv3 model with parameters frozen')
        self.heads['sem'] = Mlp(embed_dim,
                                hidden_features=embed_dim * 4,
                                out_features=dino.embed_dim)

        self.depth_prob = Mlp(embed_dim, hidden_features=embed_dim * 4, out_features=1)

        self._init_weights()

        self.aggregator = vggt.aggregator
        self.patch_size = self.aggregator.patch_size
        self.camera_head = vggt.camera_head
        self.point_head = vggt.point_head
        self.depth_head = vggt.depth_head
        self.dino = dino

    def _init_weights(self):
        nn.init.trunc_normal_(self.query_embed.weight, std=0.02)
        self.apply(init_weights_vit)

    def _random_masking(self, shape, mask_ratio):
        bs, n = shape
        len_keep = int(n * (1 - mask_ratio)) - 1

        noise = torch.rand(bs, n - 1)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        mask = torch.ones((bs, n - 1))
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        mask = F.pad(mask, (1, 0), value=0)
        return mask.bool()

    def _extract_dino_features(self, x, target_shape):
        dino_input_shape = (224, 224)
        bs, v = x.shape[:2]
        x = F.interpolate(x.flatten(0, 1), dino_input_shape, mode='bilinear')
        x = TF.normalize(x, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        feats = self.dino(x, is_training=True)['x_norm_patchtokens']
        feats = rearrange(feats,
                          'bv (hp wp) c -> bv c hp wp',
                          hp=dino_input_shape[0] // self.dino.patch_size,
                          wp=dino_input_shape[1] // self.dino.patch_size)
        feats = F.interpolate(
            feats,
            (target_shape[0] // self.patch_size, target_shape[1] // self.patch_size),
            mode='bilinear')
        feats = rearrange(feats, '(b v) c hp wp -> b v (hp wp) c', b=bs, v=v)
        return feats

    def forward(self, inputs_dict, mode=None):
        assert mode in ('loss', 'predict')
        images = inputs_dict['images']
        bs, v, _, h, w = images.shape
        with torch.no_grad():
            if mode == 'loss':
                view_mask = self._random_masking(
                    (bs, v), self.mask_ratio).to(images.device)
                if self.asymmetric_masking:
                    aggr_tokens, patch_start_idx = self.aggregator(
                        images, view_mask=view_mask)
                    f_tgt = aggr_tokens[-1][:, :, patch_start_idx:]
                    f_ctx = batched_mask_indexing(f_tgt, ~view_mask)
                else:
                    aggr_tokens, patch_start_idx = self.aggregator(
                        batched_mask_indexing(images, ~view_mask))
                    f_ctx = aggr_tokens[-1][:, :, patch_start_idx:]
                    aggr_tokens, patch_start_idx = self.aggregator(images)
                    f_tgt = aggr_tokens[-1][:, :, patch_start_idx:]

                targets = {'geo': f_tgt}
                pose_enc = self.camera_head(aggr_tokens)[-1]

                if hasattr(self, 'depth_prob'):
                    depth_gt, _ = self.depth_head(
                        aggr_tokens, images=images, patch_start_idx=patch_start_idx)

                if hasattr(self, 'dino'):
                    targets['sem'] = self._extract_dino_features(images, (h, w))
            else:
                aggr_tokens, patch_start_idx = self.aggregator(images)
                f_ctx = aggr_tokens[-1][:, :, patch_start_idx:]

        x = self.projection(f_ctx)
        bs, v_ctx, n, c = x.shape
        pos = self.position_getter(
            bs, h // self.patch_size, w // self.patch_size, device=x.device)
        pos = pos.reshape(bs, 1, n, 2)

        query = self.query_embed.weight[None].expand(bs, -1, -1)
        query_pos = torch.zeros_like(query[..., :2]).to(pos) - 1  # will be added 1 in the Transformer
        encoder_input = torch.cat([query, x.flatten(1, 2)], dim=1)
        encoder_pos = torch.cat(
            [query_pos, pos.expand(-1, v_ctx, -1, -1).flatten(1, 2)], dim=1)
        enc_out = self.encoder(encoder_input, pos=encoder_pos)
        latents = enc_out[:, :self.num_queries]
        if mode == 'predict':
            return latents

        extrinsics, intrinsics = pose_encoding_to_extri_intri(pose_enc, (h, w))
        extrinsics = F.pad(extrinsics, (0, 0, 0, 1), value=0.0)
        extrinsics[..., 3, 3] = 1
        intrinsics_ = intrinsics.clone()
        intrinsics_[..., :2, :] /= self.patch_size
        pose_cond = get_camera_rays(intrinsics_, (h // self.patch_size, w // self.patch_size))
        pose_cond = rearrange(pose_cond, 'b v c h w -> (b v) (h w) c')
        pose_cond = self.pose_tokenizer(pose_cond)

        decoder_input = torch.cat(
            [repeat(latents, 'b n c -> (b v) n c', v=v), pose_cond], dim=1)
        decoder_pos = torch.cat([
            repeat(query_pos, 'b n c -> (b v) n c', v=v),
            pos.expand(-1, v, -1, -1).flatten(0, 1)
        ], dim=1)
        spatial_mask = torch.zeros(decoder_input.size(1) + self.decoder.num_register_tokens).bool().to(decoder_input.device)
        spatial_mask[-pose_cond.size(1):] = True
        dec_out = self.decoder(
            decoder_input,
            pos=decoder_pos,
            viewmats=extrinsics.flatten(0, 1).unsqueeze(1),
            Ks=intrinsics.flatten(0, 1).unsqueeze(1), 
            spatial_mask=spatial_mask,
            patches_x = w // self.patch_size,
            patches_y = h // self.patch_size,
            image_width = w,
            image_height = h,
        )
        dec_out = dec_out[:, self.num_queries:].reshape(bs, v, n, c)

        preds = []
        losses = {}
        for k, head in self.heads.items():
            pred = head(dec_out)
            preds.append(pred)
            losses[f'{k}/l1'] = F.smooth_l1_loss(pred, targets[k])
            pred = F.normalize(pred, p=2, dim=-1)
            tgt = F.normalize(targets[k], p=2, dim=-1)
            losses[f'{k}/cos'] = (1 - F.cosine_similarity(pred, tgt, dim=-1)).mean()

        if hasattr(self, 'depth_prob'):
            depth_hat = self.depth_prob(dec_out.detach())
            depth_gt = F.interpolate(
                rearrange(depth_gt, 'b v h w c -> (b v) c h w'),
                (h // self.patch_size, w // self.patch_size))
            depth_gt = rearrange(depth_gt, '(b v) c h w -> b v (h w) c', v=v)
            losses['depth_prob'] = F.smooth_l1_loss(depth_hat, depth_gt)

        # self.visualize(
        #     images,
        #     aggr_tokens,
        #     patch_start_idx,
        #     dec_out,
        #     *list(itertools.chain.from_iterable(zip(targets.values(), preds))),
        #     depth_gt,
        #     depth_hat,
        #     view_mask=view_mask.cpu().numpy(),
        #     scene_id=inputs_dict['scene_id'],
        # )
        return losses

    def visualize(
        self,
        images,
        aggr_tokens,
        patch_start_idx,
        *args,
        scene_id,
        view_mask=None,
        pcd_color_idx=None,
    ):
        """
        Args:
            *args: Feature maps to visualize (B, V, N, C)
        """
        import matplotlib.pyplot as plt
        from .utils import visualize, visualize_points

        if pcd_color_idx is not None:
            with torch.no_grad():
                pts3d, _ = self.point_head(
                    aggr_tokens, images=images, patch_start_idx=patch_start_idx)

        bs, v, _, h, w = images.shape
        for i in range(bs):
            vis_list = []
            vis_list.append(visualize(images[i]))
            for featmap in args:
                featmap = rearrange(featmap[i],
                                    'v (hp wp) c -> v c hp wp',
                                    v=v,
                                    hp=h // self.patch_size,
                                    wp=w // self.patch_size)
                featmap = visualize(featmap)
                if featmap.shape[-2:] != (h, w):
                    featmap = F.interpolate(
                        torch.from_numpy(featmap).permute(2, 0, 1)[None],
                        scale_factor=self.patch_size,
                    ).squeeze(0).permute(1, 2, 0).numpy()
                vis_list.append(featmap)

            vis = np.concatenate(vis_list, axis=0)
            if view_mask is not None:
                vis = vis.reshape(vis.shape[0], v, w, 3)
                sorted_indices = np.argsort(view_mask[i], kind='stable')
                vis = vis[:, sorted_indices, :, :].reshape(vis.shape[0], v * w, 3)
            plt.imsave(f'outputs/spa3r/{scene_id[i]}.png', vis)

            if pcd_color_idx is None:
                continue
            visualize_points(
                rearrange(vis_list[0], 'h (b v w) c -> b v h w c', b=bs, v=v),
                pts3d[i],
                save=f'{scene_id[i]}_rgb',
            )
            visualize_points(
                rearrange(vis_list[pcd_color_idx + 1],
                          'h (b v w) c -> b v h w c',
                          b=bs,
                          v=v),
                pts3d[i],
                save=f'{scene_id[i]}_feat',
            )


def batched_mask_indexing(x, mask):
    """
    Args:
        x: (B, V, ...)
        mask: (B, V)
    Return:
        x_: (B, V', ...)
    """
    bs, v = x.shape[:2]
    assert (mask.sum(1) == mask.sum(1)[0]).all()
    return x[mask].reshape(bs, -1, *x.shape[2:])
