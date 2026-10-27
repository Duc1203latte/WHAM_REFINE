"""Đọc output của WHAM (wham_output.pkl), camera và model SMPL.

Quy ước trong project:
- Hệ camera của WHAM: x sang phải ảnh, y xuống dưới ảnh, z hướng vào cảnh (OpenCV).
- `pose[:, :3]` là hướng gốc (khung chậu) trong hệ camera; `pose_world[:, :3]` là hướng gốc
  trong hệ world (Y hướng lên). 69 số còn lại (23 khớp) giống nhau ở hai nhánh.
- Dòng i của mọi mảng ứng với frame video `frame_ids[i]`.
"""

import os
from dataclasses import dataclass
import joblib
import numpy as np
from scipy.spatial.transform import Rotation as R

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SMPL_NPZ = os.path.join(ROOT, 'models', 'smpl', 'SMPL_NEUTRAL.npz')

SMPL_JOINT_NAMES = [
    'pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee', 'spine2',
    'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot', 'neck',
    'left_collar', 'right_collar', 'head', 'left_shoulder', 'right_shoulder',
    'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hand', 'right_hand']

SMPL_PARENTS = np.array([-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21])
# Khớp i của cơ thể soi gương là khớp SMPL_FLIP_PERM[i] của cơ thể gốc (trái <-> phải)
SMPL_FLIP_PERM = np.array([0, 2, 1, 3, 5, 4, 6, 8, 7, 9, 11, 10, 12, 14, 13, 15, 17, 16, 19, 18, 21, 20, 23, 22])

@dataclass
class WhamTrack:
    """Một track trong wham_output.pkl. N = số frame của track."""
    frame_ids: np.ndarray    # (N,)          frame trong video
    pose: np.ndarray         # (N, 72)       axis-angle (rad), gốc trong hệ camera
    pose_world: np.ndarray   # (N, 72)       như pose, gốc trong hệ world
    trans: np.ndarray        # (N, 3)        tịnh tiến SMPL trong hệ camera (m)
    trans_world: np.ndarray  # (N, 3)        tịnh tiến SMPL trong hệ world (m)
    betas: np.ndarray        # (N, 10)       hình dáng (thay đổi theo frame)
    verts: np.ndarray        # (N, 6890, 3)  mesh trong hệ camera, đã cộng tịnh tiến

    KEYS = ('frame_ids', 'pose', 'pose_world', 'trans', 'trans_world', 'betas', 'verts')

    @property
    def n(self):
        return len(self.frame_ids)

    @property
    def root_cam(self) -> R:
        return R.from_rotvec(self.pose[:, :3])

    @property
    def root_world(self) -> R:
        return R.from_rotvec(self.pose_world[:, :3])
    
    def to_dict(self):
        return {k: getattr(self, k) for k in self.KEYS} 

def load_wham(path, track_id = 0) -> WhamTrack:
    """Đọc một track của wham_output.pkl."""
    data = joblib.load(path)
    if track_id not in data:
        raise KeyError(f'Không có track {track_id}; các track có: {list(data.keys())}')
    d = data[track_id]
    return WhamTrack(**{k: np.asarray(d[k]) for k in WhamTrack.KEYS})


def save_wham(path, tracks):
    """Ghi pkl đúng format WHAM: {track_id: {khoá: mảng}}."""
    joblib.dump({tid: t.to_dict() for tid, t in tracks.items()}, path)


def load_calib(path):
    """calib.txt của WHAM (fx fy cx cy) -> ma trận nội K (3x3)."""
    fx, fy, cx, cy = np.loadtxt(path)[:4]
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])


def project(points, K):
    """Chiếu điểm hệ camera (..., 3) lên ảnh -> pixel (..., 2)."""
    p = points @ K.T
    return p[..., :2] / p[..., 2:3]


def load_smpl(batch_size=1, device='cpu'):
    """Model SMPL neutral từ npz (không cần chumpy)."""
    import smplx
    from smplx.utils import Struct
    return smplx.SMPL(os.path.dirname(SMPL_NPZ), data_struct=Struct(**np.load(SMPL_NPZ)),
                      batch_size=batch_size).to(device)


def smpl_forward(smpl, pose, betas):
    """Chạy SMPL cho N frame, không tịnh tiến. Trả (verts (N,6890,3), joints (N,24,3)).

    joints = J_regressor @ verts (24 khớp SMPL), giống cách đã kiểm chứng với WHAM.
    """
    import torch
    with torch.no_grad():
        p = torch.as_tensor(pose, dtype=torch.float32)
        b = torch.as_tensor(betas, dtype=torch.float32)
        out = smpl(global_orient=p[:, :3], body_pose=p[:, 3:], betas=b)
        verts = out.vertices
        joints = torch.einsum('jv,nvc->njc', smpl.J_regressor, verts)
    return verts.numpy(), joints.numpy()


def joints_from_verts(verts, smpl):
    """24 khớp SMPL từ mesh (N,6890,3), cùng hệ toạ độ với mesh."""
    return np.einsum('jv,nvc->njc', smpl.J_regressor.numpy(), verts)


def estimate_R_wc(track: WhamTrack, max_dev_deg=20.0, iters=5):
    """Phép quay camera -> world, coi là cố định (camera đứng yên).

    WHAM cho mỗi frame một R_wc = R(pose_world) R(pose)^-1, nhưng giá trị này trôi vì nhánh world
    tích phân riêng. Lấy trung bình rồi bỏ dần các frame lệch > max_dev_deg.
    Trả (R_wc, mask các frame được dùng).
    """
    rwc = track.root_world * track.root_cam.inv()
    keep = np.ones(track.n, bool)
    for _ in range(iters):
        mean = rwc[keep].mean()
        dev = np.degrees((rwc * mean.inv()).magnitude())
        new_keep = dev < max_dev_deg
        if new_keep.sum() < 3 or np.array_equal(new_keep, keep):
            break
        keep = new_keep
    return rwc[keep].mean(), keep