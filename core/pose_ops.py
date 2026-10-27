"""Phép biến đổi pose SMPL dùng cho giả thuyết lật.

Silhouette là hình chiếu của cơ thể theo các tia nhìn từ camera. Soi gương cơ thể qua mặt phẳng đi
qua khung chậu và vuông góc với tia nhìn tới khung chậu (ma trận M_ray = I - 2 r rᵀ) cho ra gần
đúng silhouette đó. Cơ thể soi gương là "người quay ngược lại, trái <-> phải đổi chỗ".

Với pose SMPL: điểm thế giới p = R_gốc · b (b trong hệ cơ thể). Soi gương: M_ray · R_gốc · b
= (M_ray · R_gốc · M_x) · (M_x · b), mà M_x · b chính là cơ thể soi gương trong hệ cơ thể, tức
23 khớp đổi trái <-> phải và đổi dấu y, z của axis-angle (giống flip test của WHAM).
"""
import numpy as np
from scipy.spatial.transform import Rotation as R
from core.wham_io import SMPL_FLIP_PERM

M_X = np.diag([-1.0, 1.0, 1.0])

def mirror_body(body_pose):
    """Soi gương 23 khớp (N,69) qua mặt phẳng đối xứng của cơ thể."""
    n = len(body_pose)
    p = np.concatenate([np.zeros((n, 1, 3)), body_pose.reshape(n, 23, 3)], 1)[:, SMPL_FLIP_PERM]
    p[..., 1:] *= -1
    return p[:, 1:].reshape(n, 69)


def reflect_matrix(ray):
    """M = I - 2 r rᵀ cho mỗi tia (N,3) -> (N,3,3)."""
    r = ray / np.linalg.norm(ray, axis=-1, keepdims=True)
    return np.eye(3) - 2 * r[:, :, None] * r[:, None, :]


def turn_around(pose, pelvis_cam):
    """Giả thuyết "quay ngược": soi gương cơ thể qua mặt phẳng vuông góc tia nhìn tới khung chậu.

    pose: (N,72) gốc trong hệ camera. pelvis_cam: (N,3) vị trí khung chậu trong hệ camera.
    Khung chậu giữ nguyên vị trí; trên ảnh, khớp trái mới nằm gần chỗ khớp phải cũ.
    """
    M = reflect_matrix(pelvis_cam)
    root = R.from_rotvec(pose[:, :3]).as_matrix()
    out = np.empty_like(pose)
    out[:, :3] = R.from_matrix(M @ root @ M_X).as_rotvec()
    out[:, 3:] = mirror_body(pose[:, 3:])
    return out


def reflect_points(points, pelvis_cam):
    """Soi gương điểm (N,K,3) qua cùng mặt phẳng như turn_around (để kiểm tra)."""
    M = reflect_matrix(pelvis_cam)
    rel = points - pelvis_cam[:, None]
    return np.einsum('nij,nkj->nki', M, rel) + pelvis_cam[:, None]