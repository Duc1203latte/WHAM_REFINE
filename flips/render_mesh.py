"""Video so sánh mesh kiểu WHAM, hai khung cạnh nhau: trái = WHAM gốc, phải = bản đã sửa (bước 1 hoặc 2).

--view camera (mặc định): mesh phủ lên video nền, như nửa trái video demo của WHAM.
--view global:            góc nhìn toàn cục có mặt đất, như nửa phải video demo của WHAM.

Dùng renderer của repo WHAM (lib/vis/renderer.py) nên hình giống hệt video của WHAM. Mặt đất và
camera toàn cục tính từ WHAM gốc, dùng chung cho cả hai khung để so được trực tiếp.
Frame được đánh dấu (bước 1: chọn "quay ngược"; bước 2: frame bị sửa) có mesh màu cam và viền đỏ ở khung phải.

Video nền chỉ để hiển thị (mặc định nên dùng video silhouette đã đưa vào WHAM).
"""
import argparse
import os
import sys

import cv2
import imageio
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from core.wham_io import load_smpl, load_wham, smpl_forward  # noqa: E402

GRAY = [0.8, 0.8, 0.8]
ORANGE = [0.95, 0.55, 0.25]


def world_verts(track, smpl):
    """Mesh trong hệ world như run_vis của WHAM: SMPL(pose_world, betas) + trans_world."""
    v, _ = smpl_forward(smpl, track.pose_world, track.betas)
    return v + track.trans_world[:, None]


def label(img, text, y, scale=1.2):
    cv2.putText(img, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 6, cv2.LINE_AA)
    cv2.putText(img, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser(description='Video so sánh mesh kiểu WHAM: trái = WHAM gốc, phải = sau bước 1.')
    ap.add_argument('--wham', required=True, help='wham_output.pkl gốc')
    ap.add_argument('--step1', required=True, help='thư mục output của bước cần so (detect_flips.py hoặc fix_rotation.py)')
    ap.add_argument('--pkl-name', default='wham_step1.pkl', help='tên pkl đã sửa trong thư mục --step1')
    ap.add_argument('--csv-name', default='flips.csv',
                    help='bảng theo frame trong --step1; frame được đánh dấu = cột "sua" nếu có, '
                         'không thì gia_thuyet != wham')
    ap.add_argument('--name', default='sau buoc 1', help='nhãn của khung phải (không dấu)')
    ap.add_argument('--mark-label', default='QUAY NGUOC', help='chữ hiện ở frame được đánh dấu (không dấu)')
    ap.add_argument('--video', required=True, help='video nền (silhouette hoặc RGB), cùng frame với WHAM')
    ap.add_argument('--out', required=True, help='file mp4 output')
    ap.add_argument('--view', choices=['camera', 'global'], default='camera',
                    help='camera: mesh phủ lên video; global: góc nhìn toàn cục có mặt đất')
    ap.add_argument('--track', type=int, default=0)
    ap.add_argument('--wham-root', default='/home/Latte/projects/WHAM', help='repo WHAM (lấy renderer)')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    sys.path.insert(0, args.wham_root)
    from lib.vis.renderer import Renderer, get_global_cameras

    dev = torch.device(args.device)
    tracks = [load_wham(args.wham, args.track),
              load_wham(os.path.join(args.step1, args.pkl_name), args.track)]
    table = pd.read_csv(os.path.join(args.step1, args.csv_name))
    flipped = table['sua'].to_numpy(bool) if 'sua' in table else table['gia_thuyet'].to_numpy() != 'wham'

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    focal = (width ** 2 + height ** 2) ** 0.5        # như run_vis của WHAM (= fx trong calib.txt)
    smpl = load_smpl(batch_size=tracks[0].n)
    renderer = Renderer(width, height, focal, dev, smpl.faces)
    faces = renderer.faces.clone().squeeze(0)

    if args.view == 'global':
        # Mesh world của hai bản; nâng cùng một mức để WHAM gốc chạm mặt đất y = 0
        vg = [torch.from_numpy(world_verts(t, smpl)).float() for t in tracks]
        y0 = vg[0][..., 1].min()
        for v in vg:
            v[..., 1] -= y0
        # Mặt đất và camera toàn cục: tính từ WHAM gốc, giống hệt run_vis
        ref = vg[0]
        cx, cz = (ref.mean(1).max(0)[0] + ref.mean(1).min(0)[0])[[0, 2]] / 2.0
        sx, sz = (ref.mean(1).max(0)[0] - ref.mean(1).min(0)[0])[[0, 2]]
        renderer.set_ground(max(sx.item(), sz.item()) * 1.5, cx.item(), cz.item())
        global_R, global_T, global_lights = get_global_cameras(ref, dev)

    def render(k, i, img):
        """Một khung: mesh của bản k (0 = WHAM gốc, 1 = sau bước 1) ở frame i."""
        color = ORANGE if (k == 1 and flipped[i]) else GRAY
        if args.view == 'camera':
            renderer.create_camera(torch.eye(3), torch.zeros(3))
            return renderer.render_mesh(torch.from_numpy(tracks[k].verts[i]).float().to(dev), img, color)
        colors = torch.ones((1, 4), device=dev)
        # WHAM tô mesh toàn cục sáng hơn (0.9 thay vì 0.8); kẹp <= 1 để không tràn màu
        colors[..., :3] = (torch.tensor(color, device=dev) * (0.9 / 0.8)).clamp(max=1.0)
        cameras = renderer.create_camera(global_R[i], global_T[i])
        return renderer.render_with_ground(vg[k][[i]].to(dev), faces, colors, cameras, global_lights)

    row = {int(f): i for i, f in enumerate(tracks[0].frame_ids)}
    writer = imageio.get_writer(args.out, fps=fps, mode='I', format='FFMPEG', macro_block_size=1)
    fid = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        img = frame[..., ::-1].copy()                  # BGR -> RGB
        i = row.get(fid)
        panels = []
        for k, name in enumerate(['WHAM goc', args.name]):
            if i is None:                              # frame không có người trong track
                out = img.copy() if args.view == 'camera' else np.full_like(img, 255)
            else:
                out = np.ascontiguousarray(render(k, i, img).astype(np.uint8))
            label(out, f'{name}  frame {fid}', 50)
            if k == 1 and i is not None and flipped[i]:
                cv2.rectangle(out, (0, 0), (out.shape[1] - 1, out.shape[0] - 1), (255, 0, 0), 12)
                label(out, args.mark_label, 100)
            panels.append(out)
        writer.append_data(np.hstack(panels))
        fid += 1
        if fid % 50 == 0:
            print('  frame', fid)
    writer.close()
    print('Đã ghi', args.out)


if __name__ == '__main__':
    main()
