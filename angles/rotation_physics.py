"""Bước 2a: báo cáo vật lý của chuyển động xoay quanh trục dọc (chỉ đo, chưa sửa).

Coi cơ thể là khối đặc đều giới hạn bởi mesh SMPL (betas cố định = trung vị). Trong hệ camera (camera
đứng yên nên là hệ quán tính), mỗi frame tính:
    I_v  quán tính quanh trục dọc qua khối tâm (kg·m²)
    L_v  mô-men động lượng quanh trục dọc qua khối tâm (kg·m²/s)
    ω    = L_v / I_v, tốc độ xoay "hiệu dụng" của toàn thân (°/s)
Lúc bay L_v phải không đổi (không có mô-men ngoài), nên ω không được đổi dấu.
Lúc trên băng, τ = dL_v/dt là mô-men ngoài mà băng phải tạo ra.

Pha bay suy từ video silhouette:
    1. đoạn dài nhất mà độ cao khối tâm khớp parabol rơi tự do (gia tốc -g);
    2. nới hai đầu khi điểm thấp nhất của silhouette còn cao hơn mức băng gần đó và khối tâm còn bám
       parabol đó.

Output (thư mục --out): physics.csv, physics.png.
"""
import argparse
import os
import sys
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from core.mass import angular_momentum, mass_properties                       # noqa: E402
from core.silhouette import lowest_points                                     # noqa: E402
from core.wham_io import estimate_R_wc, joints_from_verts, load_calib, load_smpl, load_wham, smpl_forward  # noqa: E402

G = 9.81


def fixed_shape_verts(track, smpl):
    """Mesh hệ camera với betas cố định (trung vị), khung chậu giữ đúng chỗ của WHAM."""
    betas = np.repeat(np.median(track.betas, 0, keepdims=True), track.n, 0)
    v, j = smpl_forward(smpl, track.pose, betas)
    pelvis = joints_from_verts(track.verts, smpl)[:, 0]
    return (v + (pelvis - j[:, 0])[:, None]).astype(np.float64)


def foot_height(track, low_uv, K, pelvis, up):
    """Độ cao (m, theo trục up, gốc tuỳ ý) của điểm thấp nhất silhouette.

    Lấy điểm trên tia nhìn qua pixel đó, ở cùng độ sâu với khung chậu (mặt phẳng qua khung chậu vuông
    góc tia nhìn). Phụ thuộc độ sâu của WHAM nên có thể trôi chậm theo thời gian.
    """
    Kinv = np.linalg.inv(K)
    uv = low_uv[track.frame_ids]
    rays = np.c_[uv, np.ones(len(uv))] @ Kinv.T
    depth = np.einsum('ni,ni->n', pelvis, rays) / np.einsum('ni,ni->n', rays, rays)
    return np.einsum('ni,i->n', rays * depth[:, None], up)


def detect_flight(com_h, foot_h, fps, rms_max=0.02, min_frames=12, clearance=0.15, ice_window=30, ext_tol=0.04):
    """Pha bay (mask bool) từ độ cao khối tâm và độ cao chân.

    Bước 1: tìm đoạn dài nhất mà com_h + g t²/2 tuyến tính (tức khối tâm rơi tự do), sai số RMS < rms_max.
    Bước 2: nới hai đầu chừng nào CẢ HAI còn đúng:
        - chân còn cao hơn mức băng (trung vị ice_window frame ngoài đoạn) + clearance;
        - khối tâm còn bám parabol của đoạn lõi (lệch < ext_tol), tức chưa có lực từ băng.
    """
    n = len(com_h)
    t = np.arange(n) / fps
    best = None
    for a in range(n):
        for b in range(a + min_frames, n):
            x = t[a:b + 1] - t[a]
            y = com_h[a:b + 1] + 0.5 * G * x ** 2
            A = np.c_[np.ones_like(x), x]
            r = A @ np.linalg.lstsq(A, y, rcond=None)[0] - y
            if np.sqrt(np.mean(r ** 2)) < rms_max and (best is None or b - a > best[1] - best[0]):
                best = (a, b)
    flight = np.zeros(n, bool)
    if best is None:
        return flight
    a, b = best
    # parabol của đoạn lõi, kéo dài ra cả clip
    x = t - t[a]
    A = np.c_[np.ones(b - a + 1), x[a:b + 1]]
    c = np.linalg.lstsq(A, com_h[a:b + 1] + 0.5 * G * x[a:b + 1] ** 2, rcond=None)[0]
    ballistic = np.abs(com_h - (c[0] + c[1] * x - 0.5 * G * x ** 2)) < ext_tol
    after = foot_h[b + 1:b + 1 + ice_window]
    before = foot_h[max(0, a - ice_window):a]
    ice = np.median(after if len(after) else before)
    airborne = (foot_h > ice + clearance) & ballistic
    while a > 0 and airborne[a - 1]:
        a -= 1
    while b < n - 1 and airborne[b + 1]:
        b += 1
    flight[a:b + 1] = True
    return flight


def main():
    ap = argparse.ArgumentParser(description='Bước 2a: báo cáo vật lý chuyển động xoay quanh trục dọc.')
    ap.add_argument('--wham', required=True, help='wham_output.pkl (hoặc wham_step1.pkl)')
    ap.add_argument('--silhouette', required=True, help='video silhouette đã đưa vào WHAM')
    ap.add_argument('--calib', required=True, help='calib.txt (fx fy cx cy)')
    ap.add_argument('--out', required=True, help='thư mục output')
    ap.add_argument('--track', type=int, default=0)
    ap.add_argument('--fps', type=float, default=60.0)
    ap.add_argument('--density', type=float, default=1000.0, help='khối lượng riêng cơ thể (kg/m³)')
    ap.add_argument('--flight', default=None, help='khai báo tay pha bay "đầu-cuối" (frame video), bỏ qua tự suy')
    ap.add_argument('--max-rate', type=float, default=2000.0, help='tốc độ xoay tối đa hợp lý (°/s)')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    track = load_wham(args.wham, args.track)
    smpl = load_smpl(batch_size=track.n)
    faces = smpl.faces.astype(np.int64)
    K = load_calib(args.calib)
    R_wc, _ = estimate_R_wc(track)
    up = R_wc.inv().apply([0.0, 1.0, 0.0])                 # trục thẳng đứng trong hệ camera

    verts = fixed_shape_verts(track, smpl)
    mass, com, inertia = mass_properties(verts, faces, args.density)
    L = angular_momentum(verts, faces, com, args.fps, args.density)
    I_v = np.einsum('i,nij,j->n', up, inertia, up)
    L_v = L @ up
    omega = np.degrees(L_v / I_v)                           # °/s
    tau = np.gradient(L_v, 1.0 / args.fps)                  # N·m, mô-men ngoài quanh trục dọc

    # Tốc độ yaw của khung chậu (để so với ω toàn thân)
    root = track.root_cam
    pelvis_rate = np.r_[0, np.degrees((root[1:] * root[:-1].inv()).as_rotvec()) @ up] * args.fps

    # Pha bay
    pelvis = joints_from_verts(track.verts, smpl)[:, 0]
    foot = foot_height(track, lowest_points(args.silhouette), K, pelvis, up)
    if args.flight:
        a, b = map(int, args.flight.split('-'))
        flight = (track.frame_ids >= a) & (track.frame_ids <= b)
    else:
        flight = detect_flight(com @ up, foot, args.fps)

    # Đánh giá lúc bay: chiều chính = dấu của trung vị L_v trong pha bay
    flags = np.full(track.n, '', dtype=object)
    if flight.any():
        L_ref = np.median(L_v[flight])
        sign = np.sign(L_ref)
        reversed_ = flight & (np.sign(omega) != sign)
        too_fast = np.abs(omega) > args.max_rate
        flags[reversed_] = 'nguoc_chieu'
        flags[too_fast & ~reversed_] = 'qua_nhanh'
        flags[too_fast & reversed_] = 'nguoc_chieu+qua_nhanh'

    df = pd.DataFrame({
        'frame': track.frame_ids,
        'bay': flight,
        'chan_cao_m': (foot - np.median(foot[~flight]) if (~flight).any() else foot).round(3),
        'khoi_tam_cao_m': (com @ up - (com @ up)[~flight].min() if (~flight).any() else com @ up).round(3),
        'I_v_kgm2': I_v.round(3),
        'L_v_kgm2s': L_v.round(2),
        'omega_deg_s': omega.round(0),
        'omega_khung_chau_deg_s': pelvis_rate.round(0),
        'tau_ngoai_Nm': tau.round(1),
        'canh_bao': flags,
    })
    df.to_csv(os.path.join(args.out, 'physics.csv'), index=False)
    plot(df, args, os.path.join(args.out, 'physics.png'))

    print(f'Khối lượng {mass[0]:.1f} kg (khối lượng riêng {args.density:.0f} kg/m³)')
    if flight.any():
        fb = track.frame_ids[flight]
        Lf = L_v[flight]
        print(f'Pha bay: frame {fb[0]}-{fb[-1]} ({flight.sum()} frame, {flight.sum() / args.fps:.2f} s)')
        print(f'  L_v trung vị {L_ref:.1f} kg·m²/s; dao động (độ lệch chuẩn / |trung vị|) {Lf.std() / abs(L_ref):.0%}')
        print(f'  ω ngược chiều chính: {(flags == "nguoc_chieu").sum() + (flags == "nguoc_chieu+qua_nhanh").sum()}'
              f'/{flight.sum()} frame; số lần đổi dấu ω: {(np.diff(np.sign(omega[flight])) != 0).sum()}')
        print(f'  |ω| max {np.abs(omega[flight]).max():.0f} °/s; > {args.max_rate:.0f} °/s: '
              f'{(np.abs(omega[flight]) > args.max_rate).sum()} frame')
    else:
        print('Không tìm thấy pha bay.')
    ice = ~flight
    print(f'Trên băng: |τ| trung vị {np.median(np.abs(tau[ice])):.0f} N·m, p95 {np.percentile(np.abs(tau[ice]), 95):.0f} N·m')


def plot(df, args, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    BLUE, ORANGE, AQUA, RED, MUTED, GRID, BG = '#2a78d6', '#eb6834', '#1baf7a', '#e34948', '#52514e', '#e4e3df', '#fcfcfb'
    plt.rcParams.update({'font.size': 10, 'axes.edgecolor': GRID, 'axes.labelcolor': MUTED,
                         'xtick.color': MUTED, 'ytick.color': MUTED,
                         'axes.spines.top': False, 'axes.spines.right': False})
    f = df['frame'].to_numpy()
    fl = df['bay'].to_numpy()
    fig, ax = plt.subplots(4, 1, figsize=(11, 11), sharex=True, facecolor=BG)
    for a in ax:
        a.set_facecolor(BG)
        a.grid(axis='y', color=GRID, lw=0.8)
        if fl.any():
            a.axvspan(f[fl][0] - 0.5, f[fl][-1] + 0.5, color='#f0efec', lw=0, zorder=0)
    ax[0].plot(f, df['khoi_tam_cao_m'], color=BLUE, lw=2, label='khối tâm')
    ax[0].plot(f, df['chan_cao_m'], color=ORANGE, lw=2, label='điểm thấp nhất silhouette')
    ax[0].set_ylabel('độ cao (m)')
    ax[0].set_title('Bước 2a: vật lý của chuyển động xoay quanh trục dọc (nền xám = pha bay suy từ silhouette)',
                    loc='left')
    ax[0].legend(frameon=False, loc='upper right')
    ax[1].plot(f, df['I_v_kgm2'], color=BLUE, lw=2)
    ax[1].set_ylabel('I quanh trục dọc (kg·m²)')
    ax[2].plot(f, df['L_v_kgm2s'], color=BLUE, lw=2)
    ax[2].axhline(0, color=MUTED, lw=1)
    ax[2].set_ylabel('L quanh trục dọc (kg·m²/s)\nlúc bay phải nằm ngang')
    ax[3].plot(f, df['omega_khung_chau_deg_s'], color=ORANGE, lw=1.5, label='khung chậu')
    ax[3].plot(f, df['omega_deg_s'], color=BLUE, lw=2, label='toàn thân ω = L/I')
    ax[3].axhline(0, color=MUTED, lw=1)
    for s in (1, -1):
        ax[3].axhline(s * args.max_rate, color=MUTED, lw=1, ls='--')
    bad = df['canh_bao'].astype(str).str.contains('nguoc_chieu').to_numpy()
    ax[3].scatter(f[bad], df['omega_deg_s'][bad], color=RED, s=18, zorder=3, label='ngược chiều lúc bay')
    ax[3].set_ylabel('tốc độ xoay quanh trục dọc (°/s)')
    ax[3].set_xlabel('frame')
    ax[3].legend(frameon=False, loc='lower right', ncol=3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)


if __name__ == '__main__':
    main()
