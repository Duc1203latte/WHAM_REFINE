"""Video kiểm chứng bước 1: skeleton SMPL chiếu lên video (RGB hoặc silhouette).

Trái = WHAM gốc, phải = sau bước 1. Tay chân bên trái màu xanh, bên phải màu cam, để thấy chỗ nhầm
trái/phải. Frame chọn giả thuyết "quay ngược" có viền đỏ ở khung bên phải.

Video RGB chỉ dùng để người xem kiểm chứng; thuật toán bước 1 không đọc video.
"""
import argparse
import os
import sys
import cv2
import imageio
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from core.wham_io import SMPL_JOINT_NAMES, SMPL_PARENTS, joints_from_verts, load_calib, load_smpl, load_wham, project  # noqa: E402

BGR = {'left': (214, 120, 42), 'right': (52, 104, 235), 'mid': (200, 200, 200)}   # xanh / cam / xám


def side(j):
    name = SMPL_JOINT_NAMES[j]
    return 'left' if name.startswith('left') else 'right' if name.startswith('right') else 'mid'


def draw_skeleton(img, uv, thick):
    for j, p in enumerate(SMPL_PARENTS):
        if p >= 0:
            cv2.line(img, tuple(np.int32(uv[p])), tuple(np.int32(uv[j])), BGR[side(j)], thick, cv2.LINE_AA)
    for j in range(len(uv)):
        cv2.circle(img, tuple(np.int32(uv[j])), thick + 1, BGR[side(j)], -1, cv2.LINE_AA)


def label(img, text, y, scale):
    cv2.putText(img, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(img, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser(description='Video kiểm chứng bước 1 (skeleton WHAM gốc | sau bước 1).')
    ap.add_argument('--wham', required=True, help='wham_output.pkl gốc')
    ap.add_argument('--step1', required=True, help='thư mục output của detect_flips.py')
    ap.add_argument('--video', required=True, help='video nền (RGB hoặc silhouette), cùng frame với WHAM')
    ap.add_argument('--calib', required=True, help='calib.txt (fx fy cx cy)')
    ap.add_argument('--out', required=True, help='file mp4 output')
    ap.add_argument('--track', type=int, default=0)
    ap.add_argument('--scale', type=float, default=0.5, help='thu nhỏ mỗi khung hình')
    args = ap.parse_args()

    wham = load_wham(args.wham, args.track)
    fixed = load_wham(os.path.join(args.step1, 'wham_step1.pkl'), args.track)
    df = pd.read_csv(os.path.join(args.step1, 'flips.csv'))
    K = load_calib(args.calib)
    smpl = load_smpl()
    uv = [project(joints_from_verts(t.verts, smpl), K) for t in (wham, fixed)]
    row = {f: i for i, f in enumerate(wham.frame_ids)}

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 60
    writer = imageio.get_writer(args.out, fps=fps, codec='libx264', quality=8, macro_block_size=8)
    fid = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        panels = []
        for k, name in enumerate(['WHAM goc', 'sau buoc 1']):
            img = frame.copy()
            i = row.get(fid)
            if i is not None:
                draw_skeleton(img, uv[k][i], 4)
                r = df.iloc[i]
                step = r['buoc_xoay_wham_deg'] if k == 0 else r['buoc_xoay_sua_deg']
                label(img, f'{name}  frame {fid}  xoay {step:.0f} deg/frame', 50, 1.3)
                if k == 1 and r['gia_thuyet'] != 'wham':
                    cv2.rectangle(img, (0, 0), (img.shape[1] - 1, img.shape[0] - 1), (0, 0, 255), 16)
                    label(img, 'QUAY NGUOC', 100, 1.3)
            panels.append(cv2.resize(img, None, fx=args.scale, fy=args.scale, interpolation=cv2.INTER_AREA))
        writer.append_data(cv2.cvtColor(np.hstack(panels), cv2.COLOR_BGR2RGB))
        fid += 1
    writer.close()
    print('Đã ghi', args.out)


if __name__ == '__main__':
    main()
