"""Bước 1: phát hiện và sửa lật trước/sau (nhầm trái/phải) trên nhánh camera của WHAM.

Mỗi frame có 2 giả thuyết cho cùng một silhouette:
    0 = pose gốc của WHAM
    1 = quay ngược (soi gương qua mặt phẳng vuông góc tia nhìn tới khung chậu, đổi trái <-> phải)
Chọn chuỗi giả thuyết có chuyển động mượt nhất bằng Viterbi (bậc hai, để phạt cả đổi chiều xoay).

Output (thư mục --out):
    flips.csv          bảng theo frame
    flips.png          yaw và bước xoay trước / sau
    wham_step1.pkl     pkl format WHAM sau khi sửa
"""
import argparse
import os
import sys
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from core.pose_ops import turn_around                                    # noqa: E402
from core.wham_io import (WhamTrack, estimate_R_wc, joints_from_verts, load_smpl,  # noqa: E402
                          load_wham, save_wham, smpl_forward)

HYP_NAMES = ['wham', 'quay_nguoc']


def make_hypotheses(track: WhamTrack, smpl):
    """Pose (K,N,72), khớp hệ camera (K,N,24,3) và mesh (K,N,6890,3) của từng giả thuyết."""
    joints_wham = joints_from_verts(track.verts, smpl)
    pelvis = joints_wham[:, 0]
    poses = np.stack([track.pose, turn_around(track.pose, pelvis)])
    verts, joints = [], []
    for p in poses:
        v, j = smpl_forward(smpl, p, track.betas)
        off = pelvis - j[:, 0]                 # giữ khung chậu đúng chỗ của WHAM
        verts.append(v + off[:, None])
        joints.append(j + off[:, None])
    return poses, np.stack(joints), np.stack(verts)


def motion_terms(poses, joints):
    """Đại lượng chuyển động giữa các cặp giả thuyết ở hai frame liên tiếp.

    omega[t, a, b]: vận tốc góc khung chậu (rotvec, °/frame, hệ camera) từ giả thuyết a ở t-1 tới
                    b ở t.
    local[t, a, b]: dịch chuyển trung bình (m/frame) của 23 khớp trong hệ khung chậu (đo tư thế
                    tay chân, không tính hướng thân). Cao khi trái/phải bị đổi giữa hai frame.
    """
    K, N = poses.shape[:2]
    rots = [R.from_rotvec(p[:, :3]) for p in poses]
    # 23 khớp trong hệ khung chậu: R_gốcᵀ · (khớp - khung chậu)
    loc = np.stack([np.einsum('nji,nkj->nki', rots[k].as_matrix(), joints[k, :, 1:] - joints[k, :, :1])
                    for k in range(K)])
    omega = np.zeros((N, K, K, 3))
    local = np.zeros((N, K, K))
    for a in range(K):
        for b in range(K):
            omega[1:, a, b] = np.degrees((rots[b][1:] * rots[a][:-1].inv()).as_rotvec())
            local[1:, a, b] = np.linalg.norm(loc[b, 1:] - loc[a, :-1], axis=-1).mean(-1)
    return omega, local


def viterbi2(omega, local, sigma_rot, sigma_acc, sigma_local, switch_cost):
    """Viterbi bậc hai trên trạng thái cặp (s_{t-1}, s_t).

    Chi phí của bộ ba (a, b, c) ở frame t:
        (|ω_t|/σ_rot)²              xoay quá nhanh
      + (|ω_t - ω_{t-1}|/σ_acc)²     vận tốc góc đổi đột ngột (gồm cả đổi chiều xoay)
      + (local_t/σ_local)²           tay chân nhảy chỗ (đổi trái/phải)
      + switch_cost · [b ≠ c]        chi phí mỗi lần đổi giả thuyết (ưu tiên giữ WHAM khi ngang nhau)
    """
    N, K = local.shape[0], local.shape[1]
    c1 = (np.linalg.norm(omega, axis=-1) / sigma_rot) ** 2 + (local / sigma_local) ** 2 \
        + switch_cost * (1 - np.eye(K))[None]
    # D[b, c]: chi phí nhỏ nhất của đường đi kết thúc bằng (s_{t-1}=b, s_t=c)
    D = c1[1].copy()
    back = np.zeros((N, K, K), int)
    for t in range(2, N):
        acc = omega[t][None, :, :, :] - omega[t - 1][:, :, None, :]          # (a, b, c, 3)
        c2 = (np.linalg.norm(acc, axis=-1) / sigma_acc) ** 2
        tot = D[:, :, None] + c2                                            # (a, b, c)
        back[t] = tot.argmin(0)
        D = tot.min(0) + c1[t]
    b, c = np.unravel_index(D.argmin(), D.shape)
    states = np.zeros(N, int)
    states[-1], states[-2] = c, b
    for t in range(N - 1, 1, -1):
        a = back[t, states[t - 1], states[t]]
        states[t - 2] = a
    return states


def yaw_deg(root_cam: R, R_wc: R):
    """Yaw (°, đã unwrap, tính từ frame đầu) của hướng mặt khung chậu quanh trục Y-up world."""
    fwd = (R_wc * root_cam).apply([0, 0, 1])
    y = np.degrees(np.unwrap(np.arctan2(fwd[:, 0], fwd[:, 2])))
    return y - y[0]


def main():
    ap = argparse.ArgumentParser(description='Bước 1: phát hiện và sửa lật trước/sau trên nhánh camera của WHAM.')
    ap.add_argument('--wham', required=True, help='đường dẫn wham_output.pkl')
    ap.add_argument('--out', required=True, help='thư mục output')
    ap.add_argument('--track', type=int, default=0, help='track id trong pkl')
    ap.add_argument('--fps', type=float, default=60.0, help='fps của video (chỉ để đổi °/frame -> °/s)')
    ap.add_argument('--sigma-rot', type=float, default=30.0, help='thang vận tốc góc khung chậu (°/frame)')
    ap.add_argument('--sigma-acc', type=float, default=20.0, help='thang thay đổi vận tốc góc (°/frame²)')
    ap.add_argument('--sigma-local', type=float, default=0.05, help='thang dịch chuyển khớp trong hệ khung chậu (m/frame)')
    ap.add_argument('--switch-cost', type=float, default=2.0, help='chi phí mỗi lần đổi giả thuyết')
    ap.add_argument('--warn-deg', type=float, default=45.0, help='ngưỡng "đáng ngờ" của bước xoay (°/frame)')
    ap.add_argument('--flip-deg', type=float, default=90.0, help='ngưỡng "gần chắc lật" của bước xoay (°/frame)')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    track = load_wham(args.wham, args.track)
    smpl = load_smpl(batch_size=track.n)
    poses, joints, verts = make_hypotheses(track, smpl)
    omega, local = motion_terms(poses, joints)
    states = viterbi2(omega, local, args.sigma_rot, args.sigma_acc, args.sigma_local, args.switch_cost)

    idx = np.arange(track.n)
    prev = np.r_[0, states[:-1]]
    rot_wham = np.linalg.norm(omega[:, 0, 0], axis=-1)
    rot_fix = np.linalg.norm(omega[idx, prev, states], axis=-1)
    acc_wham = np.r_[0, np.linalg.norm(np.diff(omega[:, 0, 0], axis=0), axis=-1)]
    w_fix = omega[idx, prev, states]
    acc_fix = np.r_[0, np.linalg.norm(np.diff(w_fix, axis=0), axis=-1)]
    rot_wham[0] = rot_fix[0] = 0

    R_wc, used = estimate_R_wc(track)
    root_fix = R.from_rotvec(poses[states, idx, :3])
    flag = np.where(rot_wham > args.flip_deg, 'gan_chac_lat', np.where(rot_wham > args.warn_deg, 'dang_ngo', ''))

    df = pd.DataFrame({
        'frame': track.frame_ids,
        'gia_thuyet': [HYP_NAMES[s] for s in states],
        'doi_gia_thuyet': np.r_[False, states[1:] != states[:-1]],
        'canh_bao_wham': flag,
        'buoc_xoay_wham_deg': rot_wham.round(1),
        'buoc_xoay_sua_deg': rot_fix.round(1),
        'doi_van_toc_goc_wham_deg': acc_wham.round(1),
        'doi_van_toc_goc_sua_deg': acc_fix.round(1),
        'tay_chan_nhay_wham_mm': (1000 * local[:, 0, 0]).round(1),
        'tay_chan_nhay_sua_mm': (1000 * local[idx, prev, states]).round(1),
        'yaw_wham_cam_deg': yaw_deg(track.root_cam, R_wc).round(1),
        'yaw_wham_world_deg': yaw_deg(track.root_world, R.identity(track.n)).round(1),
        'yaw_sua_deg': yaw_deg(root_fix, R_wc).round(1),
        'R_wc_dung_frame': used,
    })
    df.to_csv(os.path.join(args.out, 'flips.csv'), index=False)

    # pkl format WHAM: gốc camera đã sửa, gốc world = R_wc cố định · gốc camera
    fixed = WhamTrack(**track.to_dict())
    fixed.pose = poses[states, idx].astype(np.float32)
    fixed.pose_world = fixed.pose.copy()
    fixed.pose_world[:, :3] = (R_wc * root_fix).as_rotvec()
    fixed.verts = verts[states, idx].astype(np.float32)
    save_wham(os.path.join(args.out, 'wham_step1.pkl'), {args.track: fixed})

    plot(df, args, os.path.join(args.out, 'flips.png'))

    seg = np.flatnonzero(np.diff(np.r_[0, states, 0]))
    print(f'R_wc cố định từ {used.sum()}/{track.n} frame')
    print('Đoạn quay ngược (frame):', [(int(track.frame_ids[a]), int(track.frame_ids[b - 1]))
                                       for a, b in zip(seg[::2], seg[1::2])])
    for name, col in [('WHAM', 'buoc_xoay_wham_deg'), ('sau sửa', 'buoc_xoay_sua_deg')]:
        v = df[col].to_numpy()
        print(f'{name:8s}: bước xoay max {v.max():6.1f}°/frame ({v.max() * args.fps:.0f}°/s), '
              f'> {args.warn_deg:.0f}°: {(v > args.warn_deg).sum()} frame, > {args.flip_deg:.0f}°: {(v > args.flip_deg).sum()} frame')


def plot(df, args, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    BLUE, ORANGE, AQUA, MUTED, GRID, BG = '#2a78d6', '#eb6834', '#1baf7a', '#52514e', '#e4e3df', '#fcfcfb'
    plt.rcParams.update({'font.size': 10, 'axes.edgecolor': GRID, 'axes.labelcolor': MUTED,
                         'xtick.color': MUTED, 'ytick.color': MUTED,
                         'axes.spines.top': False, 'axes.spines.right': False})
    f = df['frame'].to_numpy()
    flip = df['gia_thuyet'].to_numpy() != HYP_NAMES[0]
    fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True, facecolor=BG)
    for a in ax:
        a.set_facecolor(BG)
        a.grid(axis='y', color=GRID, lw=0.8)
        for x in f[flip]:
            a.axvspan(x - 0.5, x + 0.5, color='#f0efec', lw=0, zorder=0)
    ax[0].plot(f, df['yaw_wham_world_deg'], color=ORANGE, lw=2, label='WHAM world')
    ax[0].plot(f, df['yaw_wham_cam_deg'], color=BLUE, lw=2, label='WHAM camera')
    ax[0].plot(f, df['yaw_sua_deg'], color=AQUA, lw=2, label='sau bước 1')
    ax[0].set_ylabel('yaw tích luỹ (°)')
    ax[0].set_title('Bước 1: yaw khung chậu và bước xoay mỗi frame (nền xám = frame chọn "quay ngược")',
                    loc='left')
    ax[0].legend(frameon=False, loc='upper left')
    ax[1].plot(f, df['buoc_xoay_wham_deg'], color=BLUE, lw=2, label='WHAM camera')
    ax[1].plot(f, df['buoc_xoay_sua_deg'], color=AQUA, lw=2, label='sau bước 1')
    for th in (args.warn_deg, args.flip_deg):
        ax[1].axhline(th, color=MUTED, lw=1, ls='--')
        ax[1].text(f[-1], th + 2, f'{th:.0f}°', ha='right', color=MUTED)
    ax[1].set_ylabel('bước xoay khung chậu (°/frame)')
    ax[1].set_xlabel('frame')
    ax[1].legend(frameon=False, loc='upper center')
    fig.tight_layout()
    fig.savefig(path, dpi=110)


if __name__ == '__main__':
    main()
