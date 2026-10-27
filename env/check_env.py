"""Kiểm tra env wham-refine chạy được đủ các phần project cần.

1. SMPL (.npz) + smplx dựng lại đúng verts của WHAM
2. COMETH: nạp mô hình BSM qua nimblephysics
3. cvxpy giải QP bằng OSQP
4. pytorch3d raster mesh trên GPU
5. imageio ghi video H.264
"""
import os
import tempfile

import joblib
import numpy as np
import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
WHAM_PKL = os.path.join(ROOT, 'data', 'toeloop9_cam4', 'wham_output.pkl')


def check_smpl():
    import smplx
    from smplx.utils import Struct
    smpl_dir = os.path.join(ROOT, 'models', 'smpl')
    m = smplx.SMPL(smpl_dir, data_struct=Struct(**np.load(os.path.join(smpl_dir, 'SMPL_NEUTRAL.npz'))))
    if not os.path.exists(WHAM_PKL):
        return f'nạp được, bỏ qua so sánh vì thiếu {WHAM_PKL}'
    d = joblib.load(WHAM_PKL)[0]
    p = torch.tensor(d['pose'][:5], dtype=torch.float32)
    b = torch.tensor(d['betas'][:5], dtype=torch.float32)
    v = m(global_orient=p[:, :3], body_pose=p[:, 3:], betas=b).vertices.detach().numpy()
    diff = d['verts'][:5] - v
    err = np.linalg.norm(diff - diff.mean(1, keepdims=True), axis=-1).mean() * 1000
    assert err < 2, f'sai {err:.2f} mm'
    return f'verts WHAM sai {err:.2f} mm'


def check_cometh():
    from COMETH.DynamicSkeleton import DynamicSkeleton
    n = DynamicSkeleton()._nimble.getNumDofs()
    assert n == 49
    return f'BSM {n} DOF'


def check_qp():
    import cvxpy as cp
    x = cp.Variable(3)
    cp.Problem(cp.Minimize(cp.sum_squares(x - 1)), [x <= 0.5]).solve(solver=cp.OSQP)
    assert np.allclose(x.value, 0.5, atol=1e-3)
    return 'OSQP OK'


def check_render():
    from pytorch3d.renderer import (FoVPerspectiveCameras, MeshRasterizer, RasterizationSettings,
                                    look_at_view_transform)
    from pytorch3d.utils import ico_sphere
    assert torch.cuda.is_available(), 'không thấy CUDA'
    R, T = look_at_view_transform(3, 0, 0)
    r = MeshRasterizer(cameras=FoVPerspectiveCameras(R=R, T=T, device='cuda'),
                       raster_settings=RasterizationSettings(image_size=64))
    n = int((r(ico_sphere(2, 'cuda')).pix_to_face >= 0).sum())
    assert n > 0
    return f'raster CUDA OK ({torch.cuda.get_device_name(0)})'


def check_video():
    import imageio
    path = os.path.join(tempfile.gettempdir(), 'wham_refine_check.mp4')
    w = imageio.get_writer(path, fps=30, codec='libx264')
    for i in range(5):
        w.append_data(np.full((64, 64, 3), i * 40, np.uint8))
    w.close()
    os.remove(path)
    return 'H.264 OK'


if __name__ == '__main__':
    ok = True
    for name, fn in [('SMPL', check_smpl), ('COMETH', check_cometh), ('QP', check_qp),
                     ('render', check_render), ('video', check_video)]:
        try:
            print(f'[OK]   {name}: {fn()}')
        except Exception as e:
            ok = False
            print(f'[LỖI]  {name}: {type(e).__name__}: {e}')
    print('Env sẵn sàng.' if ok else 'Env còn lỗi.')
    raise SystemExit(0 if ok else 1)
