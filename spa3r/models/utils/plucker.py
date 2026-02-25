import torch
import torch.nn.functional as F


def quat_to_mat(quat):
    """
    Convert rotations given as quaternions to rotation matrices.
    Expects scalar-last (x, y, z, w) format.

    Args:
        quat: Quaternions of shape (..., 4).

    Returns:
        Tensor: Rotation matrices of shape (..., 3, 3).
    """
    quat /= torch.sqrt((quat**2).sum(dim=-1, keepdim=True))
    x, y, z, w = quat.unbind(-1)
    x2, y2, z2 = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    rots = torch.stack([
        1 - 2 * (y2 + z2), 2 * (xy - wz), 2 * (xz + wy),
        2 * (xy + wz), 1 - 2 * (x2 + z2), 2 * (yz - wx),
        2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (x2 + y2),
    ], dim=-1).reshape(*quat.shape[:-1], 3, 3)
    return rots


def pose_enc_to_c2w(pose_enc):
    """
    Args:
        pose_enc: (..., 7/8)
    """
    T = pose_enc[..., :3]
    quat = pose_enc[..., 3:7]
    R = quat_to_mat(quat)

    c2w = torch.cat([R, T.unsqueeze(-1)], dim=-1)  # (..., 3, 4)
    c2w = torch.cat([c2w, torch.zeros_like(c2w[..., :1, :])], dim=-2)
    c2w[..., 3, 3] = 1.0
    return c2w


def get_rays(c2w, intrinsics, image_size):
    """
    Args:
        c2w: Camera-to-world matrices of shape (..., 4, 4).
        intr: Camera intrinsics (fx, fy, cx, cy) of shape (..., 4), or (..., 3, 3).
        img_shape: (height, width).

    Returns:
        ray_o: (..., 3, h, w)
        ray_d: (..., 3, h, w)
    """
    bs = c2w.shape[:-2]  # (bs, (v))
    h, w = image_size
    y, x = torch.meshgrid(torch.arange(h).to(c2w),
                          torch.arange(w).to(c2w),
                          indexing='ij')
    pixel_coords = torch.stack([x, y], dim=-1) + 0.5  # (h, w, 2)

    if intrinsics.shape[-1] == 4:
        fx, fy, cx, cy = intrinsics.unbind(dim=-1)
    elif intrinsics.shape[-2:] == (3, 3):
        fx, fy = intrinsics[..., 0, 0], intrinsics[..., 1, 1]
        cx, cy = intrinsics[..., 0, 2], intrinsics[..., 1, 2]

    focal = torch.stack([fx, fy], dim=-1).reshape(*bs, 1, 1, 2)
    prin_pt = torch.stack([cx, cy], dim=-1).reshape(*bs, 1, 1, 2)

    c2w = c2w.reshape(*bs, 1, 1, 4, 4)
    ray_d = (pixel_coords - prin_pt) / focal
    ray_d = F.pad(ray_d, (0, 1), value=1.0)  # (..., h, w, 3)
    ray_d = torch.einsum('...ij,...j->...i', c2w[..., :3, :3], ray_d)

    ray_d = ray_d.to(torch.float32)
    ray_d = ray_d / torch.norm(ray_d, dim=-1, keepdim=True).clamp(min=1e-3)
    ray_d = ray_d.to(c2w.dtype)
    ray_o = c2w[..., :3, 3].expand_as(ray_d)

    ray_o = ray_o.mT.transpose(-2, -3)
    ray_d = ray_d.mT.transpose(-2, -3)
    return ray_o, ray_d


def compute_plucker_from_rays(ray_o, ray_d):
    o_cross_d = torch.cross(ray_o, ray_d, dim=2)
    return torch.cat([o_cross_d, ray_d], dim=2)


def compute_plucker_from_c2w(c2w, intrinsics, image_size):
    ray_o, ray_d = get_rays(c2w, intrinsics, image_size)
    return compute_plucker_from_rays(ray_o, ray_d)


def compute_plucker_from_7d_pose(pose_enc, intrinsics, image_size):
    """
    Args:
        pose_enc: (B, V, 7/8) absT_quaR
    """
    c2w = pose_enc_to_c2w(pose_enc)
    return compute_plucker_from_c2w(c2w, intrinsics, image_size)
