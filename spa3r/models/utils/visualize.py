import matplotlib.pyplot as plt
import numpy as np
import torch
from einops import rearrange
from matplotlib import cm

from .camera_utils import depth_to_points


def normalize(x, eps=1e-8):
    x = (x - x.min()) / (x.max() - x.min() + eps)
    return np.clip(x, 0, 1)


def visualize(x, cmap='Spectral_r', save=None):
    """
    Args:
        x: ((B, ...) , (1/3/C), H, W), assuming C should be greater than 64
        cmap: string or tensor of (N, 3)
    """
    if isinstance(x, list):
        x = torch.stack(x)
    # ((B), (C), H, W) -> (H, B * W, C)
    assert x.dim() >= 2
    if x.dim() == 2:
        x = x.unsqueeze(-1)
    if x.dim() == 3:
        if x.size(0) in (1, 3) or x.size(0) >= 64:
            x = x.permute(1, 2, 0)
        else:
            x = rearrange(x, 'b h w -> h (b w) c', c=1)
    else:
        if x.dim() >= 5:
            x = x.flatten(0, x.dim() - 4)
        x = rearrange(x, 'b c h w -> h (b w) c')

    if x.size(-1) > 3:
        with torch.amp.autocast('cuda', enabled=False):
            u, s, v = torch.pca_lowrank(x.flatten(0, -2).float(), q=3, niter=4)
        x = x @ v
    x = x.detach().cpu().numpy()
    if x.shape[-1] == 1:
        x = 1 / (x + 1e-6)
        v_min = np.nanpercentile(x, 2)
        v_max = np.nanpercentile(x, 98)
        x = np.clip((x - v_min) / (v_max - v_min), 0, 1)
        if isinstance(cmap, str):
            x = cm.get_cmap(cmap)(x)[..., 0, :3]
        else:
            x = cmap[x]
    elif x.shape[-1] == 3:
        x = normalize(x)

    if save is not None:
        plt.imsave(save, x)
    else:
        return x


def visualize_points(colors, points, extrinsics=None, intrinsics=None, save=None):
    import open3d as o3d

    if extrinsics is not None and intrinsics is not None:
        points = depth_to_points(points, extrinsics, intrinsics)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(
        points.flatten(0, -2).cpu().detach().numpy())
    pcd.colors = o3d.utility.Vector3dVector(colors.flatten(0, -2))
    if save is None:
        o3d.visualization.draw_geometries([pcd])
    else:
        o3d.io.write_point_cloud(f'{save}.ply', pcd)
