"""Đọc video silhouette (người màu đen trên nền trắng)."""
import cv2
import numpy as np


def lowest_points(video_path, thresh=128):
    """Điểm thấp nhất (u, v) pixel của bóng người trong từng frame.

    Lấy vùng đen liên thông lớn nhất. Frame không có người, hoặc gần như đen toàn bộ (đầu video
    silhouette có thể đen hết), trả NaN. Kết quả (số frame video, 2).
    """
    cap = cv2.VideoCapture(video_path)
    out = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        mask = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) < thresh
        n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
        if n < 2 or not (0 < mask.mean() < 0.5):
            out.append((np.nan, np.nan))
            continue
        k = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        ys, xs = np.nonzero(lab == k)
        v = ys.max()
        out.append((xs[ys >= v - 2].mean(), v))            # u = trung bình vài hàng dưới cùng
    return np.array(out, float)
