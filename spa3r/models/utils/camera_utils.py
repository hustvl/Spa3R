import torch
import torch.nn.functional as F


def extr_to_c2w(extr):
    c2w = torch.zeros_like(extr)
    c2w[..., :3, :3] = extr[..., :3, :3].mT
    c2w[..., :3, 3:4] = -c2w[..., :3, :3] @ extr[..., :3, 3:4]
    c2w[..., 3, 3] = 1
    return c2w


def depth_to_points(depth, extrinsics, intrinsics):
    """
    Args:
        depth: (bs, (v), 1, h, w)
        extrinsics: (bs, (v), 4, 4)
        intrinsics: (bs, (v), 3, 3)
    """
    h, w = depth.shape[-2:]
    c2w = extr_to_c2w(extrinsics)

    ys, xs = torch.meshgrid(torch.arange(h).to(depth),
                            torch.arange(w).to(depth),
                            indexing='ij')
    pixel_coords = torch.stack([xs, ys], dim=-1)  # (h, w, 2)
    pixel_coords = F.pad(pixel_coords, (0, 1), value=1.0)

    pts = pixel_coords * depth.transpose(-2, -3).transpose(-1, -2)  # bs, (v), h, w, 3
    pts = intrinsics.inverse().unsqueeze(-3).unsqueeze(-4) @ pts.unsqueeze(-1)
    pts = c2w.unsqueeze(-3).unsqueeze(-4) @ F.pad(pts, (0, 0, 0, 1), value=1.0)
    return pts.squeeze(-1)[..., :3]


def get_camera_rays(intrinsics, image_size):
    bs = intrinsics.shape[:-2] 
    h, w = image_size
    
    y, x = torch.meshgrid(torch.arange(h).to(intrinsics),
                          torch.arange(w).to(intrinsics),
                          indexing='ij')
    pixel_coords = torch.stack([x, y], dim=-1) + 0.5  # (h, w, 2)

    if intrinsics.shape[-1] == 4:
        fx, fy, cx, cy = intrinsics.unbind(dim=-1)
    elif intrinsics.shape[-2:] == (3, 3):
        fx, fy = intrinsics[..., 0, 0], intrinsics[..., 1, 1]
        cx, cy = intrinsics[..., 0, 2], intrinsics[..., 1, 2]

    focal = torch.stack([fx, fy], dim=-1).reshape(*bs, 1, 1, 2)
    prin_pt = torch.stack([cx, cy], dim=-1).reshape(*bs, 1, 1, 2)

    ray_d = (pixel_coords - prin_pt) / focal
    ray_d = F.pad(ray_d, (0, 1), value=1.0)  # (..., h, w, 3)

    ray_d = ray_d.to(torch.float32)
    ray_d = ray_d / torch.norm(ray_d, dim=-1, keepdim=True).clamp(min=1e-3)    
    ray_d = ray_d.mT.transpose(-2, -3)
    return ray_d
