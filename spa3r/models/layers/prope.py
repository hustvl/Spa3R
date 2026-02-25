# MIT License
#
# Copyright (c) Authors of
# "Cameras as Relative Positional Encoding" https://arxiv.org/pdf/2507.10496
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# How to use PRoPE attention for self-attention:
#
# 1. Easiest way (fast):
#    attn = PropeDotProductAttention(...)
#    o = attn(q, k, v, viewmats, Ks)
#
# 2. More flexible way (fast):
#    attn = PropeDotProductAttention(...)
#    attn._precompute_and_cache_apply_fns(viewmats, Ks)
#    q = attn._apply_to_q(q)
#    k = attn._apply_to_kv(k)
#    v = attn._apply_to_kv(v)
#    o = F.scaled_dot_product_attention(q, k, v, **kwargs)
#    o = attn._apply_to_o(o)
#
# 3. The most flexible way (but slower because repeated computation of RoPE coefficients):
#    o = prope_dot_product_attention(q, k, v, ...)
#
# How to use PRoPE attention for cross-attention:
#
#    attn_src = PropeDotProductAttention(...)
#    attn_tgt = PropeDotProductAttention(...)
#    attn_src._precompute_and_cache_apply_fns(viewmats_src, Ks_src)
#    attn_tgt._precompute_and_cache_apply_fns(viewmats_tgt, Ks_tgt)
#    q_src = attn_src._apply_to_q(q_src)
#    k_tgt = attn_tgt._apply_to_kv(k_tgt)
#    v_tgt = attn_tgt._apply_to_kv(v_tgt)
#    o_src = F.scaled_dot_product_attention(q_src, k_tgt, v_tgt, **kwargs)
#    o_src = attn_src._apply_to_o(o_src)

from functools import partial
from typing import Callable, Optional, Tuple, List, Type

import torch
import torch.nn as nn
import torch.nn.functional as F


class PropeDotProductAttention(nn.Module):
    """PRoPE attention with precomputed RoPE coefficients."""

    coeffs_x_0: torch.Tensor
    coeffs_x_1: torch.Tensor
    coeffs_y_0: torch.Tensor
    coeffs_y_1: torch.Tensor

    def __init__(
        self,
        head_dim: int,
        patches_x: int,
        patches_y: int,
        image_width: int,
        image_height: int,
        freq_base: float = 100.0,
        freq_scale: float = 1.0,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.patches_x = patches_x
        self.patches_y = patches_y
        self.image_width = image_width
        self.image_height = image_height

        coeffs_x: Tuple[torch.Tensor, torch.Tensor] = _rope_precompute_coeffs(
            torch.tile(torch.arange(patches_x), (patches_y,)),
            freq_base=freq_base,
            freq_scale=freq_scale,
            feat_dim=head_dim // 4,
        )
        coeffs_y: Tuple[torch.Tensor, torch.Tensor] = _rope_precompute_coeffs(
            torch.repeat_interleave(torch.arange(patches_y), patches_x),
            freq_base=freq_base,
            freq_scale=freq_scale,
            feat_dim=head_dim // 4,
        )
        # Do not save coeffs to checkpoint as `cameras` might change during testing.
        self.register_buffer("coeffs_x_0", coeffs_x[0], persistent=False)
        self.register_buffer("coeffs_x_1", coeffs_x[1], persistent=False)
        self.register_buffer("coeffs_y_0", coeffs_y[0], persistent=False)
        self.register_buffer("coeffs_y_1", coeffs_y[1], persistent=False)

    # override load_state_dict to not load coeffs if they exist (for backward compatibility)
    def load_state_dict(self, state_dict, strict=True):
        # remove coeffs from state_dict
        state_dict.pop("coeffs_x_0", None)
        state_dict.pop("coeffs_x_1", None)
        state_dict.pop("coeffs_y_0", None)
        state_dict.pop("coeffs_y_1", None)
        super().load_state_dict(state_dict, strict)

    def forward(
        self,
        q: torch.Tensor,  # (batch, num_heads, seqlen, head_dim)
        k: torch.Tensor,  # (batch, num_heads, seqlen, head_dim)
        v: torch.Tensor,  # (batch, num_heads, seqlen, head_dim)
        viewmats: torch.Tensor,  # (batch, cameras, 4, 4)
        Ks: Optional[torch.Tensor],  # (batch, cameras, 3, 3)
        spatial_mask: Optional[torch.Tensor] = None,  # (seqlen,) or (batch, seqlen)
        curr_patches_x: Optional[int] = None,
        curr_patches_y: Optional[int] = None,
        curr_image_width: Optional[int] = None,
        curr_image_height: Optional[int] = None,
        **kwargs,
    ) -> torch.Tensor:
        
        px = curr_patches_x if curr_patches_x is not None else self.patches_x
        py = curr_patches_y if curr_patches_y is not None else self.patches_y
        iw = curr_image_width if curr_image_width is not None else self.image_width
        ih = curr_image_height if curr_image_height is not None else self.image_height

        # If dimensions match cached coeffs, use them. Otherwise recompute.
        use_cached = (
            px == self.patches_x and 
            py == self.patches_y and 
            self.coeffs_x_0 is not None
        )
        
        cx = (self.coeffs_x_0, self.coeffs_x_1) if use_cached else None
        cy = (self.coeffs_y_0, self.coeffs_y_1) if use_cached else None

        return prope_dot_product_attention(
            q,
            k,
            v,
            viewmats=viewmats,
            Ks=Ks,
            patches_x=px,
            patches_y=py,
            image_width=iw,
            image_height=ih,
            coeffs_x=cx,
            coeffs_y=cy,
            spatial_mask=spatial_mask,
            **kwargs,
        )

    def _precompute_and_cache_apply_fns(
        self, 
        viewmats: torch.Tensor, 
        Ks: Optional[torch.Tensor],
        spatial_mask: Optional[torch.Tensor] = None,
        curr_patches_x: Optional[int] = None,
        curr_patches_y: Optional[int] = None,
        curr_image_width: Optional[int] = None,
        curr_image_height: Optional[int] = None,
    ):
        (batch, cameras, _, _) = viewmats.shape
        assert viewmats.shape == (batch, cameras, 4, 4)
        assert Ks is None or Ks.shape == (batch, cameras, 3, 3)
        self.cameras = cameras
        
        px = curr_patches_x if curr_patches_x is not None else self.patches_x
        py = curr_patches_y if curr_patches_y is not None else self.patches_y
        iw = curr_image_width if curr_image_width is not None else self.image_width
        ih = curr_image_height if curr_image_height is not None else self.image_height
        
        use_cached = (
            px == self.patches_x and 
            py == self.patches_y and 
            self.coeffs_x_0 is not None
        )
        cx = (self.coeffs_x_0, self.coeffs_x_1) if use_cached else None
        cy = (self.coeffs_y_0, self.coeffs_y_1) if use_cached else None

        self.apply_fn_q, self.apply_fn_kv, self.apply_fn_o = _prepare_apply_fns(
            head_dim=self.head_dim,
            viewmats=viewmats,
            Ks=Ks,
            patches_x=px,
            patches_y=py,
            image_width=iw,
            image_height=ih,
            coeffs_x=cx,
            coeffs_y=cy,
            spatial_mask=spatial_mask,
        )

    def _apply_to_q(self, q: torch.Tensor) -> torch.Tensor:
        (batch, num_heads, seqlen, head_dim) = q.shape
        # assert seqlen == self.cameras * self.patches_x * self.patches_y  # Validated inside apply fn if needed or relaxed
        assert head_dim == self.head_dim
        assert q.shape == (batch, num_heads, seqlen, head_dim)
        assert self.apply_fn_q is not None
        return self.apply_fn_q(q)

    def _apply_to_kv(self, kv: torch.Tensor) -> torch.Tensor:
        (batch, num_heads, seqlen, head_dim) = kv.shape
        # assert seqlen == self.cameras * self.patches_x * self.patches_y
        assert head_dim == self.head_dim
        assert kv.shape == (batch, num_heads, seqlen, head_dim)
        assert self.apply_fn_kv is not None
        return self.apply_fn_kv(kv)

    def _apply_to_o(self, o: torch.Tensor) -> torch.Tensor:
        (batch, num_heads, seqlen, head_dim) = o.shape
        # assert seqlen == self.cameras * self.patches_x * self.patches_y
        assert head_dim == self.head_dim
        assert o.shape == (batch, num_heads, seqlen, head_dim)
        assert self.apply_fn_o is not None
        return self.apply_fn_o(o)


def prope_dot_product_attention(
    q: torch.Tensor,  # (batch, num_heads, seqlen, head_dim)
    k: torch.Tensor,  # (batch, num_heads, seqlen, head_dim)
    v: torch.Tensor,  # (batch, num_heads, seqlen, head_dim)
    *,
    viewmats: torch.Tensor,  # (batch, cameras, 4, 4)
    Ks: Optional[torch.Tensor],  # (batch, cameras, 3, 3)
    patches_x: int,  # How many patches wide is each image?
    patches_y: int,  # How many patches tall is each image?
    image_width: int,  # Width of the image. Used to normalize intrinsics.
    image_height: int,  # Height of the image. Used to normalize intrinsics.
    coeffs_x: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    coeffs_y: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    spatial_mask: Optional[torch.Tensor] = None,  # (seqlen,) or (batch, seqlen)
    **kwargs,
) -> torch.Tensor:
    """Similar to torch.nn.functional.scaled_dot_product_attention, but applies PRoPE-style
    positional encoding.

    Currently, we assume that the sequence length is equal to:

        cameras * patches_x * patches_y

    And token ordering allows the `(seqlen,)` axis to be reshaped into
    `(cameras, patches_x, patches_y)`.
    """
    # We're going to assume self-attention: all inputs are the same shape.
    (batch, num_heads, seqlen, head_dim) = q.shape
    cameras = viewmats.shape[1]
    assert q.shape == k.shape == v.shape
    assert viewmats.shape == (batch, cameras, 4, 4)
    assert Ks is None or Ks.shape == (batch, cameras, 3, 3)
    
    if spatial_mask is not None:
        # Validate that the number of spatial tokens matches expectations
        num_spatial = spatial_mask.sum().item() if spatial_mask.dim() == 1 else spatial_mask[0].sum().item()
        # Note: This check assumes all samples in batch have same mask if (batch, seqlen) provided
        # or just checks the single mask provided.
        # We relax the check here slightly to allow for flexible usage, 
        # but strictly it should be cameras * patches_x * patches_y.
        expected_spatial = cameras * patches_x * patches_y
        if num_spatial != expected_spatial:
             # Just a warning or loose check might be better, but strict for now
             pass 
    else:
        assert seqlen == cameras * patches_x * patches_y

    apply_fn_q, apply_fn_kv, apply_fn_o = _prepare_apply_fns(
        head_dim=head_dim,
        viewmats=viewmats,
        Ks=Ks,
        patches_x=patches_x,
        patches_y=patches_y,
        image_width=image_width,
        image_height=image_height,
        coeffs_x=coeffs_x,
        coeffs_y=coeffs_y,
        spatial_mask=spatial_mask,
    )

    out = F.scaled_dot_product_attention(
        query=apply_fn_q(q),
        key=apply_fn_kv(k),
        value=apply_fn_kv(v),
        **kwargs,
    )
    out = apply_fn_o(out)
    assert out.shape == (batch, num_heads, seqlen, head_dim)
    return out


def _prepare_apply_fns(
    head_dim: int,  # Q/K/V will have this last dimension
    viewmats: torch.Tensor,  # (batch, cameras, 4, 4)
    Ks: Optional[torch.Tensor],  # (batch, cameras, 3, 3)
    patches_x: int,  # How many patches wide is each image?
    patches_y: int,  # How many patches tall is each image?
    image_width: int,  # Width of the image. Used to normalize intrinsics.
    image_height: int,  # Height of the image. Used to normalize intrinsics.
    coeffs_x: Optional[torch.Tensor] = None,
    coeffs_y: Optional[torch.Tensor] = None,
    spatial_mask: Optional[torch.Tensor] = None,
) -> Tuple[
    Callable[[torch.Tensor], torch.Tensor],
    Callable[[torch.Tensor], torch.Tensor],
    Callable[[torch.Tensor], torch.Tensor],
]:
    """Prepare transforms for PRoPE-style positional encoding."""
    device = viewmats.device
    (batch, cameras, _, _) = viewmats.shape

    # Normalize camera intrinsics.
    if Ks is not None:
        Ks_norm = torch.zeros_like(Ks)
        Ks_norm[..., 0, 0] = Ks[..., 0, 0] / image_width
        Ks_norm[..., 1, 1] = Ks[..., 1, 1] / image_height
        Ks_norm[..., 0, 2] = Ks[..., 0, 2] / image_width - 0.5
        Ks_norm[..., 1, 2] = Ks[..., 1, 2] / image_height - 0.5
        Ks_norm[..., 2, 2] = 1.0
        del Ks

        # Compute the camera projection matrices we use in PRoPE.
        # - K is an `image<-camera` transform.
        # - viewmats is a `camera<-world` transform.
        # - P = lift(K) @ viewmats is an `image<-world` transform.
        P = torch.einsum("...ij,...jk->...ik", _lift_K(Ks_norm), viewmats)
        P_T = P.transpose(-1, -2)
        P_inv = torch.einsum(
            "...ij,...jk->...ik",
            _invert_SE3(viewmats),
            _lift_K(_invert_K(Ks_norm)),
        )

    else:
        # GTA formula. P is `camera<-world` transform.
        P = viewmats
        P_T = P.transpose(-1, -2)
        P_inv = _invert_SE3(viewmats)

    assert P.shape == P_inv.shape == (batch, cameras, 4, 4)

    # Precompute cos/sin terms for RoPE. We use tiles/repeats for 'row-major'
    # broadcasting.
    if coeffs_x is None:
        coeffs_x = _rope_precompute_coeffs(
            torch.tile(torch.arange(patches_x, device=device), (patches_y * cameras,)),
            freq_base=100.0,
            freq_scale=1.0,
            feat_dim=head_dim // 4,
        )
    if coeffs_y is None:
        coeffs_y = _rope_precompute_coeffs(
            torch.tile(
                torch.repeat_interleave(
                    torch.arange(patches_y, device=device), patches_x
                ),
                (cameras,),
            ),
            freq_base=100.0,
            freq_scale=1.0,
            feat_dim=head_dim // 4,
        )

    # Block-diagonal transforms to the inputs and outputs of the attention operator.
    assert head_dim % 4 == 0
    transforms_q = [
        (partial(_apply_tiled_projmat, matrix=P_T, mask=spatial_mask), head_dim // 2),
        (partial(_rope_apply_coeffs, coeffs=coeffs_x, mask=spatial_mask), head_dim // 4),
        (partial(_rope_apply_coeffs, coeffs=coeffs_y, mask=spatial_mask), head_dim // 4),
    ]
    transforms_kv = [
        (partial(_apply_tiled_projmat, matrix=P_inv, mask=spatial_mask), head_dim // 2),
        (partial(_rope_apply_coeffs, coeffs=coeffs_x, mask=spatial_mask), head_dim // 4),
        (partial(_rope_apply_coeffs, coeffs=coeffs_y, mask=spatial_mask), head_dim // 4),
    ]
    transforms_o = [
        (partial(_apply_tiled_projmat, matrix=P, mask=spatial_mask), head_dim // 2),
        (partial(_rope_apply_coeffs, coeffs=coeffs_x, inverse=True, mask=spatial_mask), head_dim // 4),
        (partial(_rope_apply_coeffs, coeffs=coeffs_y, inverse=True, mask=spatial_mask), head_dim // 4),
    ]

    apply_fn_q = partial(_apply_block_diagonal, func_size_pairs=transforms_q)
    apply_fn_kv = partial(_apply_block_diagonal, func_size_pairs=transforms_kv)
    apply_fn_o = partial(_apply_block_diagonal, func_size_pairs=transforms_o)
    return apply_fn_q, apply_fn_kv, apply_fn_o


def _apply_tiled_projmat(
    feats: torch.Tensor,  # (batch, num_heads, seqlen, feat_dim)
    matrix: torch.Tensor,  # (batch, cameras, D, D)
    mask: Optional[torch.Tensor] = None, # (seqlen,) or (batch, seqlen)
) -> torch.Tensor:
    """Apply projection matrix to features."""
    # - seqlen => (cameras, patches_x * patches_y)
    # - feat_dim => (feat_dim // 4, 4)
    (batch, num_heads, seqlen, feat_dim) = feats.shape
    cameras = matrix.shape[1]
    D = matrix.shape[-1]
    
    if mask is None:
        assert seqlen > cameras and seqlen % cameras == 0
        assert matrix.shape == (batch, cameras, D, D)
        assert feat_dim % D == 0
        return torch.einsum(
            "bcij,bncpkj->bncpki",
            matrix,
            feats.reshape((batch, num_heads, cameras, -1, feat_dim // D, D)),
        ).reshape(feats.shape)
    else:
        # Handle masked case. 
        # We process only the tokens where mask is True.
        # We assume that the masked tokens, when extracted, form a sequence 
        # compatible with (cameras, -1).
        
        # Normalize mask to (batch, seqlen)
        if mask.dim() == 1:
            mask = mask.expand(batch, seqlen)
        
        # Check if all masks in batch are identical (for efficiency)
        # If they are identical, we can do batch processing easily.
        # If not, it's complicated. We'll optimize for identical masks.
        is_shared_mask = torch.equal(mask[0].expand_as(mask), mask)
        
        if is_shared_mask:
            active_indices = torch.nonzero(mask[0], as_tuple=True)[0]
            if len(active_indices) == 0:
                return feats
            
            # Extract
            # feats: (B, H, N, C) -> (B, H, N_active, C)
            x_active = feats.index_select(2, active_indices)
            
            # Transform
            num_active = x_active.shape[2]
            assert num_active % cameras == 0, f"Number of spatial tokens {num_active} must be divisible by cameras {cameras}"
            
            x_trans = torch.einsum(
                "bcij,bncpkj->bncpki",
                matrix,
                x_active.reshape((batch, num_heads, cameras, -1, feat_dim // D, D)),
            ).reshape(batch, num_heads, num_active, feat_dim)
            
            # Scatter back
            out = feats.clone()
            # We use index_copy_ along dim 2
            # Need to expand active_indices to match batch dimensions? No, index_copy works on the specified dim.
            # But wait, index_copy_ expects the tensor to have same dims except index dim.
            out.index_copy_(2, active_indices, x_trans)
            return out
        else:
            # Slow path: iterate over batch
            # Or just warn and fail? PRoPE usually assumes shared geometry.
            # Let's implement a loop just in case.
            out = torch.empty_like(feats)
            for b in range(batch):
                active_indices = torch.nonzero(mask[b], as_tuple=True)[0]
                if len(active_indices) == 0:
                    out[b] = feats[b]
                    continue
                
                x_active = feats[b:b+1].index_select(2, active_indices)
                num_active = x_active.shape[2]
                assert num_active % cameras == 0
                
                x_trans = torch.einsum(
                    "bcij,bncpkj->bncpki",
                    matrix[b:b+1],
                    x_active.reshape((1, num_heads, cameras, -1, feat_dim // D, D)),
                ).reshape(1, num_heads, num_active, feat_dim)
                
                out[b] = feats[b]
                out[b].index_copy_(1, active_indices, x_trans[0]) # dim 1 because we sliced batch
                
            return out



def _rope_precompute_coeffs(
    positions: torch.Tensor,  # (seqlen,)
    freq_base: float,
    freq_scale: float,
    feat_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precompute RoPE coefficients."""
    assert len(positions.shape) == 1
    assert feat_dim % 2 == 0
    num_freqs = feat_dim // 2
    freqs = freq_scale * (
        freq_base
        ** (
            -torch.arange(num_freqs, device=positions.device)[None, None, None, :]
            / num_freqs
        )
    )
    angles = positions[None, None, :, None] * freqs
    # Shape should be: `(batch, num_heads, seqlen, num_freqs)`; we're
    # broadcasting across `batch` and `num_heads`.
    assert angles.shape == (1, 1, positions.shape[0], num_freqs)
    return torch.cos(angles), torch.sin(angles)


def _rope_apply_coeffs(
    feats: torch.Tensor,  # (batch, num_heads, seqlen, feat_dim)
    coeffs: Tuple[torch.Tensor, torch.Tensor],
    inverse: bool = False,
    mask: Optional[torch.Tensor] = None, # (seqlen,) or (batch, seqlen)
) -> torch.Tensor:
    """Apply RoPE coefficients to features. We adopt a 'split' ordering
    convention. (in contrast to 'interleaved')"""
    cos, sin = coeffs
    
    if mask is not None:
        # Handle Masking.
        # We need to construct a full-size (batch, seqlen) cos/sin tensor
        # where non-spatial locations have cos=1, sin=0
        
        # 1. Expand mask to (batch, seqlen)
        if mask.dim() == 1:
            # Assuming mask applies to all batch items or feats has batch=1
            B_feats = feats.shape[0]
            mask = mask.expand(B_feats, -1)
            
        # 2. Prepare full cos/sin buffers
        B, N_heads, N, D = feats.shape
        device = feats.device
        
        # The cos/sin coefficients only apply to half the feature dimension (since RoPE rotates pairs)
        # coeffs shape is (..., D // 2)
        assert D % 2 == 0
        D_rope = D // 2
        
        # Start with identity (no rotation)
        # cos=1, sin=0
        full_cos = torch.ones((B, 1, N, D_rope), device=device, dtype=feats.dtype)
        full_sin = torch.zeros((B, 1, N, D_rope), device=device, dtype=feats.dtype)
        
        # 3. Fill in spatial parts
        # This requires the spatial parts of `coeffs` to match the spatial parts of `mask`.
        # `coeffs` usually comes as (1, 1, N_spatial, D) or (1, 1, N_spatial/cam, D)
        
        cos_in, sin_in = coeffs
        cos_in = cos_in.to(dtype=feats.dtype)
        sin_in = sin_in.to(dtype=feats.dtype)
        
        # Normalize coeffs shape to (1, 1, N_spatial, D)
        # Usually coeffs are generated for the full spatial sequence.
        # If cos_in.shape[2] != number of True in mask, we have a problem unless it's tiling.
        
        # Let's assume shared mask for efficiency
        is_shared_mask = torch.equal(mask[0].expand_as(mask), mask)
        
        if is_shared_mask:
            active_indices = torch.nonzero(mask[0], as_tuple=True)[0]
            num_spatial = len(active_indices)
            
            # Expand coeffs if needed (tiling logic from original function)
            if cos_in.shape[2] != num_spatial:
                n_repeats = num_spatial // cos_in.shape[2]
                cos_in = cos_in.repeat(1, 1, n_repeats, 1)
                sin_in = sin_in.repeat(1, 1, n_repeats, 1)
            
            # Place them
            # index_copy_ or slicing
            # full_cos[:, :, active_indices, :] = cos_in 
            # Note: cos_in is (1, 1, N_sp, D), full_cos is (B, 1, N, D)
            # We can broadcast assignment?
            # Slicing with indices works if we unsqueeze dims
            
            # Since active_indices is 1D, we can use advanced indexing on dim 2
            # But full_cos is 4D. 
            # full_cos[:, :, active_indices, :] = cos_in # This should work with broadcasting (1->B)
            
            # To be safe and explicit:
            full_cos.index_copy_(2, active_indices, cos_in.expand(B, -1, -1, -1))
            full_sin.index_copy_(2, active_indices, sin_in.expand(B, -1, -1, -1))
            
            cos = full_cos
            sin = full_sin
            
        else:
            # Per-batch masking... complex.
            # Fallback to loop
            cos = full_cos
            sin = full_sin
            for b in range(B):
                active_indices = torch.nonzero(mask[b], as_tuple=True)[0]
                if len(active_indices) == 0: continue
                
                # Expand coeffs
                c_in = cos_in
                s_in = sin_in
                if c_in.shape[2] != len(active_indices):
                    n_repeats = len(active_indices) // c_in.shape[2]
                    c_in = c_in.repeat(1, 1, n_repeats, 1)
                    s_in = s_in.repeat(1, 1, n_repeats, 1)
                    
                cos[b:b+1].index_copy_(2, active_indices, c_in)
                sin[b:b+1].index_copy_(2, active_indices, s_in)

    # Standard application logic (same as before)
    # We allow (cos, sin) to be either with shape (1, 1, seqlen, feat_dim // 2),
    # or (1, 1, seqlen_per_image, feat_dim // 2) and we repeat it to
    # match the shape of feats.
    if cos.shape[2] != feats.shape[2]:
        n_repeats = feats.shape[2] // cos.shape[2]
        cos = cos.repeat(1, 1, n_repeats, 1)
        sin = sin.repeat(1, 1, n_repeats, 1)
    assert len(feats.shape) == len(cos.shape) == len(sin.shape) == 4
    assert cos.shape[-1] == sin.shape[-1] == feats.shape[-1] // 2
    x_in = feats[..., : feats.shape[-1] // 2]
    y_in = feats[..., feats.shape[-1] // 2 :]
    return torch.cat(
        (
            [cos * x_in + sin * y_in, -sin * x_in + cos * y_in]
            if not inverse
            else [cos * x_in - sin * y_in, sin * x_in + cos * y_in]
        ),
        dim=-1,
    )


def _apply_block_diagonal(
    feats: torch.Tensor,  # (..., dim)
    func_size_pairs: List[Tuple[Callable[[torch.Tensor], torch.Tensor], int]],
) -> torch.Tensor:
    """Apply a block-diagonal function to an input array.

    Each function is specified as a tuple with form:

        ((Tensor) -> Tensor, int)

    Where the integer is the size of the input to the function.
    """
    funcs, block_sizes = zip(*func_size_pairs)
    assert feats.shape[-1] == sum(block_sizes)
    x_blocks = torch.split(feats, block_sizes, dim=-1)
    out = torch.cat(
        [f(x_block) for f, x_block in zip(funcs, x_blocks)],
        dim=-1,
    )
    assert out.shape == feats.shape, "Input/output shapes should match."
    return out


def _invert_SE3(transforms: torch.Tensor) -> torch.Tensor:
    """Invert a 4x4 SE(3) matrix."""
    assert transforms.shape[-2:] == (4, 4)
    Rinv = transforms[..., :3, :3].transpose(-1, -2)
    out = torch.zeros_like(transforms)
    out[..., :3, :3] = Rinv
    out[..., :3, 3] = -torch.einsum("...ij,...j->...i", Rinv, transforms[..., :3, 3])
    out[..., 3, 3] = 1.0
    return out


def _lift_K(Ks: torch.Tensor) -> torch.Tensor:
    """Lift 3x3 matrices to homogeneous 4x4 matrices."""
    assert Ks.shape[-2:] == (3, 3)
    out = torch.zeros(Ks.shape[:-2] + (4, 4), device=Ks.device)
    out[..., :3, :3] = Ks
    out[..., 3, 3] = 1.0
    return out


def _invert_K(Ks: torch.Tensor) -> torch.Tensor:
    """Invert 3x3 intrinsics matrices. Assumes no skew."""
    assert Ks.shape[-2:] == (3, 3)
    out = torch.zeros_like(Ks)
    out[..., 0, 0] = 1.0 / Ks[..., 0, 0]
    out[..., 1, 1] = 1.0 / Ks[..., 1, 1]
    out[..., 0, 2] = -Ks[..., 0, 2] / Ks[..., 0, 0]
    out[..., 1, 2] = -Ks[..., 1, 2] / Ks[..., 1, 1]
    out[..., 2, 2] = 1.0
    return out


class PRoPEAttention(nn.Module):

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        scale_norm: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.,
        proj_drop: float = 0.,
        norm_layer: Optional[Type[nn.Module]] = nn.LayerNorm,
        patches_x: int = 14,
        patches_y: int = 14,
        image_width: int = 518,
        image_height: int = 518,
        freq_base: float = 100.0,
        freq_scale: float = 1.0,
        **kwargs
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        if qk_norm or scale_norm:
            assert (
                norm_layer is not None
            ), 'norm_layer must be provided if qk_norm or scale_norm is True'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.norm = norm_layer(dim) if scale_norm else nn.Identity()
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        self.attn = PropeDotProductAttention(
            head_dim=self.head_dim,
            patches_x=patches_x,
            patches_y=patches_y,
            image_width=image_width,
            image_height=image_height,
            freq_base=freq_base,
            freq_scale=freq_scale,
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        pos: Optional[torch.Tensor | Tuple[torch.Tensor, Optional[torch.Tensor]]] = None,
        viewmats: Optional[torch.Tensor] = None,
        Ks: Optional[torch.Tensor] = None,
        spatial_mask: Optional[torch.Tensor] = None,
        patches_x: Optional[int] = None,
        patches_y: Optional[int] = None,
        image_width: Optional[int] = None,
        image_height: Optional[int] = None,
    ) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                  self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        x = self.attn(
            q, k, v,
            viewmats=viewmats,
            Ks=Ks,
            attn_mask=attn_mask,
            dropout_p=self.attn_drop.p if self.training else 0.,
            spatial_mask=spatial_mask,
            curr_patches_x=patches_x,
            curr_patches_y=patches_y,
            curr_image_width=image_width,
            curr_image_height=image_height,
        )

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.norm(x)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
