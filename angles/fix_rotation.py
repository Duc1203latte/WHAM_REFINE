"""Bước 2b: sửa hướng xoay quanh trục dọc trong pha bay theo bảo toàn mô-men động lượng.

Lúc bay không có mô-men ngoài, nên mô-men động lượng quanh trục dọc L không đổi và hướng thân θ(t) phải
theo đúng
    θ(t) = θ0 + L · ∫ dt / I(t)
với I(t) là quán tính quanh trục dọc (tính từ tư thế, xem core/mass.py). Cả đường θ(t) chỉ còn 2 ẩn
(θ0, L), luôn xoay một chiều và nhanh / chậm đúng theo tư thế (thu tay -> I giảm -> xoay nhanh).

Silhouette cho mỗi frame 2 hướng có thể: WHAM và soi gương (core/pose_ops.py: turn_around). Đường vật lý
chỉ cần gần MỘT trong hai. Dò lưới (θ0, L) với sai số cắt ngưỡng (frame lệch quá nhiều không kéo cả đường).

Silhouette không phân biệt được chiều xoay (soi gương cả đoạn bay cũng là một cú xoay hợp lệ theo chiều
kia), nên chiều xoay là tham số --chieu: ccw (mặc định, ngược chiều kim đồng hồ nhìn từ trên xuống, chiều
của đa số VĐV), cw, hoặc auto (chọn chiều có sai số nhỏ hơn; thường sát nhau, không đáng tin).

Trên băng (--ice, mặc định bật): L được phép đổi nhưng mô-men từ băng τ = dL/dt bị phạt. Quy hoạch động
trên lưới (hướng, tốc độ) neo vào hai đầu pha bay: sau tiếp đất đi xuôi thời gian, trước bật nhảy đi ngược.
Nhờ vậy chiều xoay không đổi qua lại khi chỉ có mô-men nhỏ (xem track_ice).

Mỗi frame: chọn giả thuyết gần hướng mục tiêu nhất, rồi xoay thêm quanh trục dọc (qua khung chậu) cho sát
hướng mục tiêu, tối đa --max-extra độ để silhouette không lệch nhiều. --no-ice: chỉ sửa pha bay.

Output (thư mục --out): wham_step2.pkl (format WHAM), rotation_fix.csv, rotation_fix.png.
"""
import argparse
import os
import sys
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, HERE)
from core.mass import angular_momentum, mass_properties                      # noqa: E402
from core.pose_ops import turn_around                                        # noqa: E402
from core.silhouette import lowest_points                                    # noqa: E402
from core.wham_io import (WhamTrack, estimate_R_wc, joints_from_verts, load_calib,  # noqa: E402
                          load_smpl, load_wham, save_wham, smpl_forward)
from rotation_physics import detect_flight, foot_height                      # noqa: E402

HYP_NAMES = ['wham', 'quay_nguoc']


def wrap(a):
    """Đưa góc (°) về [-180, 180)."""
    return (a + 180.0) % 360.0 - 180.0


def heading_basis(up):
    """Hai trục nằm ngang: e1 = hướng nhìn của camera chiếu lên mặt ngang, e2 = up × e1.
    Góc hướng tăng = xoay theo quy tắc bàn tay phải quanh trục up = ngược chiều kim đồng hồ nhìn từ trên.
    """
    e1 = np.array([0.0, 0.0, 1.0]) - up[2] * up
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(up, e1)


def heading_deg(rotvec, e1, e2):
    """Hướng mặt của khung chậu (trục z cơ thể SMPL) trên mặt ngang, °."""
    f = R.from_rotvec(rotvec).apply([0.0, 0.0, 1.0])
    return np.degrees(np.arctan2(f @ e2, f @ e1))


def fit_flight(obs, inv_I, fps, sign, L_max=60.0, L_step=0.25, cap=60.0):
    """Dò lưới (L, θ0) cho đường vật lý θ(t) = θ0 + L ∫dt/I.

    obs (2, M): hướng của 2 giả thuyết ở M frame bay (°). inv_I (M,): 1/I.
    sign: +1 chỉ thử L > 0, -1 chỉ L < 0.
    Sai số mỗi frame = khoảng cách góc tới giả thuyết gần hơn, cắt ở cap (°), rồi bình phương.
    Trả (L, θ0, đường θ (M,) theo °, tổng sai số).
    """
    # ∫ dt / I theo hình thang, đổi rad -> ° để nhân với L ra °
    cum = np.degrees(np.r_[0.0, np.cumsum((inv_I[1:] + inv_I[:-1]) / 2.0)] / fps)
    th0 = np.arange(0.0, 360.0, 1.0)
    best = None
    for L in sign * np.arange(L_step, L_max + 1e-9, L_step):
        curve = th0[:, None] + L * cum[None]                                 # (360, M)
        d = np.abs(wrap(curve[:, None, :] - obs[None])).min(1)               # (360, M)
        cost = (np.minimum(d, cap) ** 2).sum(1)
        k = int(cost.argmin())
        if best is None or cost[k] < best[3]:
            best = (L, th0[k], curve[k], cost[k])
    return best


def trim_and_fit(obs, I_v, idx, fps, sign, cap, trim_deg, max_trim):
    """Fit đường vật lý, rồi cắt dần hai đầu pha bay khi đầu / cuối lệch > trim_deg so với cả hai giả
    thuyết (không khớp mô hình L không đổi: còn đang đẩy khỏi băng hoặc đã chạm băng). "Lệch" ở mỗi đầu
    là trung vị của 3 frame sát đầu đó, để một frame tình cờ khớp không giữ lại cả đoạn sai.
    Mỗi đầu cắt tối đa max_trim frame. Trả (idx sau cắt, kết quả fit_flight).
    """
    a, b = 0, len(idx) - 1
    while True:
        sub = idx[a:b + 1]
        fit = fit_flight(obs[:, sub], 1.0 / I_v[sub], fps, sign, cap=cap)
        d = np.abs(wrap(fit[2][None] - obs[:, sub])).min(0)
        cut_a = np.median(d[:3]) > trim_deg and a < max_trim
        cut_b = np.median(d[-3:]) > trim_deg and len(idx) - 1 - b < max_trim
        if not (cut_a or cut_b) or b - a < 12:
            return sub, fit
        a += int(cut_a)
        b -= int(cut_b)


def track_ice(obs, I_v, frames, anchor, th0, w0, fps, sig_th, sig_tau, cap, w_max=40, a_max=8):
    """Hướng thân trên băng bằng quy hoạch động, bắt đầu từ trạng thái neo (th0 °, w0 °/frame).

    Trên băng L được phép đổi, nhưng mô-men từ băng τ = dL/dt bị phạt. Trạng thái là (hướng θ, tốc độ ω)
    trên lưới 1° x 1°/frame. Mỗi frame:
        ω' = ω + a,   |a| <= a_max          (đổi tốc độ có giới hạn)
        θ' = θ + ω'
        chi phí = (τ/σ_τ)² + (min(lệch tới giả thuyết gần nhất, cap)/σ_θ)²
        τ = (I' ω' - I ω) / dt              (N·m, ω đổi sang rad/s)
    frames: chỉ số frame theo thứ tự duyệt (đi ngược thời gian thì truyền chỉ số giảm dần và w0 đổi dấu).
    anchor: frame neo (frame bay sát frames[0]), nơi có trạng thái (th0, w0).
    Trả (θ (°, trong [0, 360)), ω (°/frame)) cho từng frame trong frames.
    """
    TH = np.arange(360)
    W = np.arange(-w_max, w_max + 1)
    D = np.full((360, len(W)), np.inf)
    D[int(round(th0)) % 360, int(np.argmin(np.abs(W - w0)))] = 0.0
    k_tau = np.pi / 180.0 * fps * fps                      # (kg·m² · °/frame) / frame -> N·m
    prev_I = I_v[anchor]
    backs = []
    for f in frames:
        dd = np.abs(wrap(TH[:, None] - obs[:, f][None])).min(1)
        data = (np.minimum(dd, cap) / sig_th) ** 2
        newD = np.full_like(D, np.inf)
        bk = np.zeros(D.shape, int)                        # chỉ số ω ở frame trước
        for j, wn in enumerate(W):
            best = np.full(360, np.inf)
            barg = np.zeros(360, int)
            for jp in range(max(0, j - a_max), min(len(W), j + a_max + 1)):
                tau = (I_v[f] * wn - prev_I * W[jp]) * k_tau
                c = np.roll(D[:, jp] + (tau / sig_tau) ** 2, int(wn))       # θ' = θ + ω'
                m = c < best
                best[m] = c[m]
                barg[m] = jp
            newD[:, j] = best + data
            bk[:, j] = barg
        backs.append(bk)
        D = newD
        prev_I = I_v[f]
    i, j = np.unravel_index(np.argmin(D), D.shape)
    th = np.zeros(len(frames))
    w = np.zeros(len(frames))
    for k in range(len(frames) - 1, -1, -1):
        th[k], w[k] = TH[i], W[j]
        jp = backs[k][i, j]
        i = (i - int(W[j])) % 360
        j = jp
    return th, w


def unwrap_deg(a):
    return np.degrees(np.unwrap(np.radians(a)))


def main():
    ap = argparse.ArgumentParser(description='Bước 2b: sửa hướng xoay quanh trục dọc trong pha bay (bảo toàn mô-men động lượng).')
    ap.add_argument('--wham', required=True, help='wham_output.pkl (hoặc wham_step1.pkl)')
    ap.add_argument('--silhouette', required=True, help='video silhouette đã đưa vào WHAM')
    ap.add_argument('--calib', required=True, help='calib.txt (fx fy cx cy)')
    ap.add_argument('--out', required=True, help='thư mục output')
    ap.add_argument('--track', type=int, default=0)
    ap.add_argument('--fps', type=float, default=60.0)
    ap.add_argument('--density', type=float, default=1000.0, help='khối lượng riêng cơ thể (kg/m³)')
    ap.add_argument('--flight', default=None, help='khai báo tay pha bay "đầu-cuối" (frame video), bỏ qua tự suy')
    ap.add_argument('--chieu', choices=['ccw', 'cw', 'auto'], default='ccw',
                    help='chiều xoay nhìn từ trên xuống: ccw ngược kim đồng hồ, cw thuận, auto theo sai số')
    ap.add_argument('--max-extra', type=float, default=20.0, help='pha bay: góc xoay thêm tối đa so với giả thuyết (°)')
    ap.add_argument('--max-extra-ice', type=float, default=30.0,
                    help='trên băng: góc xoay thêm tối đa so với giả thuyết (°)')
    ap.add_argument('--cap', type=float, default=60.0, help='ngưỡng cắt sai số mỗi frame khi dò lưới (°)')
    ap.add_argument('--trim-deg', type=float, default=30.0,
                    help='frame đầu / cuối pha bay lệch hơn mức này so với cả hai giả thuyết thì bị cắt khỏi pha bay (°)')
    ap.add_argument('--max-trim', type=int, default=8, help='số frame tối đa được cắt ở mỗi đầu pha bay')
    ap.add_argument('--ice', action=argparse.BooleanOptionalAction, default=True,
                    help='sửa cả phần trên băng (quy hoạch động neo vào pha bay); --no-ice: chỉ sửa pha bay')
    ap.add_argument('--sigma-theta', type=float, default=15.0,
                    help='trên băng: thang lệch cho phép tới giả thuyết gần nhất (°)')
    ap.add_argument('--sigma-tau', type=float, default=100.0,
                    help='trên băng: thang mô-men từ băng quanh trục dọc (N·m); lớn quá thì lại cho đổi chiều xoay')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    track = load_wham(args.wham, args.track)
    n = track.n
    smpl = load_smpl(batch_size=n)
    faces = smpl.faces.astype(np.int64)
    R_wc, _ = estimate_R_wc(track)
    up = R_wc.inv().apply([0.0, 1.0, 0.0])                 # trục thẳng đứng trong hệ camera
    e1, e2 = heading_basis(up)
    pelvis = joints_from_verts(track.verts, smpl)[:, 0]

    # Quán tính và mô-men động lượng (betas cố định), như bước 2a
    betas_med = np.repeat(np.median(track.betas, 0, keepdims=True), n, 0)

    def physics_of(pose):
        v, j = smpl_forward(smpl, pose, betas_med)
        v = (v + (pelvis - j[:, 0])[:, None]).astype(np.float64)
        _, com, inertia = mass_properties(v, faces, args.density)
        L = angular_momentum(v, faces, com, args.fps, args.density)
        return com, np.einsum('i,nij,j->n', up, inertia, up), L @ up

    com, I_v, L_wham = physics_of(track.pose)

    # Pha bay
    if args.flight:
        a, b = map(int, args.flight.split('-'))
        flight = (track.frame_ids >= a) & (track.frame_ids <= b)
    else:
        foot = foot_height(track, lowest_points(args.silhouette), load_calib(args.calib), pelvis, up)
        flight = detect_flight(com @ up, foot, args.fps)
    if not flight.any():
        sys.exit('Không tìm thấy pha bay; dùng --flight để khai báo tay.')
    idx = np.flatnonzero(flight)

    # Hai giả thuyết và hướng của chúng
    poses = np.stack([track.pose, turn_around(track.pose, pelvis)])
    obs = np.stack([heading_deg(p[:, :3], e1, e2) for p in poses])          # (2, N)

    # Cắt hai đầu pha bay theo mức khớp mô hình (dùng chiều --chieu; auto thì tạm dùng ccw, hai chiều
    # đối xứng gương nên cho cùng kết quả cắt)
    sign = {'ccw': 1, 'cw': -1}.get(args.chieu)
    if not args.flight:
        idx, _ = trim_and_fit(obs, I_v, idx, args.fps, sign or 1, args.cap, args.trim_deg, args.max_trim)
        flight[:] = False
        flight[idx] = True

    # Dò đường vật lý cho cả hai chiều, rồi chọn theo --chieu
    fits = {s: fit_flight(obs[:, idx], 1.0 / I_v[idx], args.fps, s, cap=args.cap) for s in (1, -1)}
    if sign is None:
        sign = min(fits, key=lambda s: fits[s][3])
    L_star, th0, curve, cost = fits[sign]

    # Hướng mục tiêu cho từng frame: pha bay = đường vật lý; trên băng = quy hoạch động neo vào hai đầu
    # pha bay (sau tiếp đất đi xuôi thời gian, trước bật nhảy đi ngược thời gian)
    target = np.full(n, np.nan)                                              # ° (liên tục trong pha bay)
    w_target = np.full(n, np.nan)                                            # °/frame
    target[idx] = curve
    w_target[idx] = np.degrees(L_star / I_v[idx]) / args.fps
    fixed_frames = idx.copy()
    if args.ice:
        after = np.arange(idx[-1] + 1, n)
        before = np.arange(idx[0] - 1, -1, -1)
        if len(after):
            th, w = track_ice(obs, I_v, after, idx[-1], curve[-1], curve[-1] - curve[-2], args.fps,
                              args.sigma_theta, args.sigma_tau, args.cap)
            target[after], w_target[after] = th, w
        if len(before):
            th, w = track_ice(obs, I_v, before, idx[0], curve[0], -(curve[1] - curve[0]), args.fps,
                              args.sigma_theta, args.sigma_tau, args.cap)
            target[before], w_target[before] = th, -w
        fixed_frames = np.arange(n)

    # Chọn giả thuyết + góc xoay thêm cho từng frame được sửa
    hyp = np.zeros(n, int)
    extra = np.zeros(n)
    gap = np.full(n, np.nan)                                                 # lệch còn lại tới hướng mục tiêu
    d = wrap(target[fixed_frames][None] - obs[:, fixed_frames])              # (2, M)
    hyp[fixed_frames] = np.abs(d).argmin(0)
    raw = d[hyp[fixed_frames], np.arange(len(fixed_frames))]
    limit = np.where(flight, args.max_extra, args.max_extra_ice)[fixed_frames]
    extra[fixed_frames] = np.clip(raw, -limit, limit)
    gap[fixed_frames] = raw - extra[fixed_frames]

    new_pose = poses[hyp, np.arange(n)].copy()
    turn = R.from_rotvec(np.radians(extra)[:, None] * up[None])             # xoay quanh trục dọc
    new_pose[:, :3] = (turn * R.from_rotvec(new_pose[:, :3])).as_rotvec()

    # pkl format WHAM (giống bước 1: gốc world = R_wc cố định · gốc camera)
    v, j = smpl_forward(smpl, new_pose, track.betas)
    fixed = WhamTrack(**track.to_dict())
    fixed.pose = new_pose.astype(np.float32)
    fixed.pose_world = fixed.pose.copy()
    fixed.pose_world[:, :3] = (R_wc * R.from_rotvec(new_pose[:, :3])).as_rotvec()
    fixed.verts = (v + (pelvis - j[:, 0])[:, None]).astype(np.float32)
    save_wham(os.path.join(args.out, 'wham_step2.pkl'), {args.track: fixed})

    # Đánh giá: L và tốc độ xoay sau sửa
    _, _, L_new = physics_of(new_pose)
    head_wham = unwrap_deg(obs[0])
    head_new = unwrap_deg(heading_deg(new_pose[:, :3], e1, e2))
    # Hướng mục tiêu (liên tục) dời cho cùng nhánh unwrap với head_new:
    # ở frame sửa đầu tiên, hướng sau sửa = hướng mục tiêu - lệch còn lại
    tgt = np.full(n, np.nan)
    k0 = fixed_frames[0]
    tgt[fixed_frames] = unwrap_deg(target[fixed_frames])
    tgt[fixed_frames] += head_new[k0] + gap[k0] - tgt[k0]
    rate = lambda h: np.r_[0.0, np.diff(h)] * args.fps                      # noqa: E731
    changed = np.zeros(n, bool)
    changed[fixed_frames] = (hyp[fixed_frames] == 1) | (np.abs(extra[fixed_frames]) > 0.5)

    df = pd.DataFrame({
        'frame': track.frame_ids,
        'bay': flight,
        'gia_thuyet': [HYP_NAMES[h] for h in hyp],
        'sua': changed,
        'goc_xoay_them_deg': extra.round(1),
        'lech_con_lai_deg': np.round(gap, 1),
        'huong_wham_deg': head_wham.round(1),
        'huong_muc_tieu_deg': np.round(tgt, 1),
        'huong_sau_sua_deg': head_new.round(1),
        'omega_wham_deg_s': rate(head_wham).round(0),
        'omega_sau_sua_deg_s': rate(head_new).round(0),
        'omega_muc_tieu_deg_s': (w_target * args.fps).round(0),
        'L_v_wham': L_wham.round(2),
        'L_v_sau_sua': L_new.round(2),
    })
    df.to_csv(os.path.join(args.out, 'rotation_fix.csv'), index=False)
    plot(df, L_star, os.path.join(args.out, 'rotation_fix.png'))

    fb = track.frame_ids[idx]
    print(f'Pha bay: frame {fb[0]}-{fb[-1]} ({len(idx)} frame)')
    for s, name in ((1, 'ccw'), (-1, 'cw')):
        Ls, _, cv, cs = fits[s]
        tot = cv[-1] - cv[0]
        print(f'  chiều {name}: L = {Ls:+.2f} kg·m²/s, xoay {tot:+.0f}° ({tot / 360:+.2f} vòng), sai số {cs:.0f}')
    print(f'Chọn chiều {"ccw" if sign > 0 else "cw"} (--chieu {args.chieu})')
    raw_all = np.full(n, np.nan)
    raw_all[fixed_frames] = raw
    ice = np.setdiff1d(fixed_frames, idx)
    for name, sel, lim in (('bay', idx, args.max_extra), ('trên băng', ice, args.max_extra_ice)):
        if len(sel) == 0:
            continue
        print(f'[{name}] {len(sel)} frame, chọn "quay ngược": {(hyp[sel] == 1).sum()}; xoay thêm |trung vị| '
              f'{np.median(np.abs(extra[sel])):.1f}°, chạm giới hạn {lim:.0f}°: '
              f'{(np.abs(raw_all[sel]) > lim).sum()} frame')
    for name, L in (('WHAM', L_wham), ('sau sửa', L_new)):
        Lf = L[idx]
        print(f'L lúc bay, {name:7s}: trung vị {np.median(Lf):+6.1f}, độ lệch chuẩn {Lf.std():5.1f} kg·m²/s, '
              f'số lần đổi dấu {(np.diff(np.sign(Lf)) != 0).sum()}')
    if len(ice):
        def flips(w):                                       # số lần đổi chiều, bỏ qua lúc gần đứng yên
            s = np.sign(w[np.abs(w) > 100])
            return int((np.diff(s) != 0).sum())
        landed = np.arange(idx[-1] + 1, n)
        print(f'Sau tiếp đất ({len(landed)} frame), số lần đổi chiều xoay (bỏ |ω| < 100°/s): '
              f'WHAM {flips(df["omega_wham_deg_s"].to_numpy()[landed])}, '
              f'mục tiêu {flips(df["omega_muc_tieu_deg_s"].to_numpy()[landed])}, '
              f'sau sửa {flips(df["omega_sau_sua_deg_s"].to_numpy()[landed])}')


def plot(df, L_star, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    BLUE, ORANGE, AQUA, MUTED, GRID, BG = '#2a78d6', '#eb6834', '#1baf7a', '#52514e', '#e4e3df', '#fcfcfb'
    plt.rcParams.update({'font.size': 10, 'axes.edgecolor': GRID, 'axes.labelcolor': MUTED,
                         'xtick.color': MUTED, 'ytick.color': MUTED,
                         'axes.spines.top': False, 'axes.spines.right': False})
    f = df['frame'].to_numpy()
    fl = df['bay'].to_numpy()
    fig, ax = plt.subplots(3, 1, figsize=(11, 9.5), sharex=True, facecolor=BG)
    for a in ax:
        a.set_facecolor(BG)
        a.grid(axis='y', color=GRID, lw=0.8)
        a.axvspan(f[fl][0] - 0.5, f[fl][-1] + 0.5, color='#f0efec', lw=0, zorder=0)
    ax[0].plot(f, df['huong_wham_deg'], color=ORANGE, lw=2, label='WHAM')
    ax[0].plot(f, df['huong_sau_sua_deg'], color=AQUA, lw=2, label='sau sửa')
    ax[0].plot(f, df['huong_muc_tieu_deg'], color=MUTED, lw=1.2, ls='--',
               label='mục tiêu (bay: L không đổi; băng: mô-men băng giới hạn)')
    ax[0].set_ylabel('hướng khung chậu (°)')
    ax[0].set_title(f'Bước 2b: sửa xoay quanh trục dọc (nền xám = pha bay), L* = {L_star:+.1f} kg·m²/s',
                    loc='left')
    ax[0].legend(frameon=False, loc='upper left')
    ax[1].plot(f, df['omega_wham_deg_s'], color=ORANGE, lw=1.5, label='WHAM')
    ax[1].plot(f, df['omega_sau_sua_deg_s'], color=AQUA, lw=2, label='sau sửa')
    ax[1].plot(f, df['omega_muc_tieu_deg_s'], color=MUTED, lw=1.2, ls='--', label='mục tiêu')
    ax[1].axhline(0, color=MUTED, lw=1)
    ax[1].set_ylabel('tốc độ xoay khung chậu (°/s)')
    ax[1].legend(frameon=False, loc='lower right', ncol=3)
    ax[2].plot(f, df['L_v_wham'], color=ORANGE, lw=1.5, label='WHAM')
    ax[2].plot(f, df['L_v_sau_sua'], color=AQUA, lw=2, label='sau sửa')
    ax[2].axhline(0, color=MUTED, lw=1)
    ax[2].axhline(L_star, color=MUTED, lw=1.2, ls='--', label='L*')
    ax[2].set_ylabel('L quanh trục dọc (kg·m²/s)')
    ax[2].set_xlabel('frame')
    ax[2].legend(frameon=False, loc='upper right', ncol=3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)


if __name__ == '__main__':
    main()
