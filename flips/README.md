# Bước 1: phát hiện và sửa lật trước/sau trên nhánh camera của WHAM

Trạng thái: **tạm dừng, chưa giải quyết xong** (2026-10-27).
- Code đã có trong project; kết quả bên dưới chạy trên clip Toeloop_9 cam 4.
- Pipeline được phép bỏ qua bước này: bước 2 nhận `wham_output.pkl` gốc hoặc `wham_step1.pkl`.
- **Vấn đề còn lại:**
  - lật sai đoạn ~101–136;
  - đoạn cuối sau khi tiếp đất (người quay mặt ra camera) pose vẫn bị quay ngược.

## 1. Vấn đề

Video đầu vào là silhouette, người là bóng đen. Một bóng đen có hai cách hiểu:
- người quay mặt về camera;
- người quay lưng về camera, với tay chân trái và phải đổi chỗ cho nhau.

WHAM có lúc chọn nhầm, và có lúc nhảy qua lại giữa hai cách hiểu. Trên quỹ đạo, chỗ nhầm hiện ra thành:
- khung chậu xoay một bước rất lớn trong một frame (frame 38: 128°/frame, tương đương 7700°/s);
- tay chân đổi trái/phải giữa hai frame liên tiếp.

### Vì sao làm trên nhánh camera

WHAM cho hai hướng của khung chậu:

| | Khoá trong pkl | Từ đâu ra | Đặc điểm |
|---|---|---|---|
| Nhánh camera | `pose[:, :3]` | Dự đoán từ ảnh, từng frame | Gắn chặt với silhouette. Lật hiện ra thành bước nhảy rời rạc |
| Nhánh world | `pose_world[:, :3]` | Mạng riêng cộng dồn vận tốc góc | Mượt hơn, nhưng trôi. Cú lật bị làm mượt thành một đoạn quay ngược, không còn thấy bước nhảy |

Camera của clip này đứng yên (theo SLAM, vị trí chỉ dao động ~5 cm). Vì vậy hướng world có thể dựng lại bằng một phép quay cố định `R_wc` nhân với hướng trong hệ camera. Không cần tin nhánh world.

## 2. Hai giả thuyết cho mỗi frame

| Số | Tên | Pose |
|---|---|---|
| 0 | `wham` | Giữ nguyên pose của WHAM |
| 1 | `quay_nguoc` | Soi gương cơ thể qua mặt phẳng đi qua khung chậu, vuông góc với tia nhìn từ camera tới khung chậu |

### Công thức của giả thuyết 1 (`core/pose_ops.py: turn_around`)

- Đặt `r` = vector đơn vị từ camera tới khung chậu, và `M = I − 2 r rᵀ`.
- Hướng gốc mới: `R_gốc' = M · R_gốc · M_x`, với `M_x = diag(−1, 1, 1)`.
- 23 khớp: đổi chỗ khớp trái ↔ phải, rồi đổi dấu thành phần y và z của axis-angle. Đây là đúng phép flip mà WHAM dùng trong flip test.
- Khung chậu giữ nguyên vị trí.

### Kiểm chứng trên clip (186 frame)

- Khớp 3D lệch trung bình 7 mm so với ảnh soi gương lý tưởng, tối đa 21 mm. Phần lệch này do hình dáng SMPL không đối xứng tuyệt đối.
- Trên ảnh 2D, khớp mới lệch trung bình 3.1 px so với khớp đối xứng cũ (p95 10 px).
- Silhouette của hai giả thuyết trùng nhau, IoU ~0.9.

Nếu soi gương theo trục z của camera (thay vì theo tia nhìn) thì sai tới 89 px, vì người đứng lệch khỏi tâm ảnh.

### Tính chất quan trọng

Hai giả thuyết có cùng silhouette, cùng góc khớp (soi gương giữ nguyên độ gập của gối, khuỷu...) và cùng độ mượt bên trong một đoạn. Chúng chỉ khác nhau ở hai điểm:
- chiều xoay bị đảo;
- tay chân gần / xa camera bị đổi chỗ.

Khi người **nhìn ngang** camera, hai giả thuyết có gần như cùng hướng thân (chỉ lệch 12–19° ở frame 98–101). Vì thế người thật có thể "chuyển" từ cách hiểu này sang cách hiểu kia ở đúng lúc nhìn ngang mà không tạo ra bước nhảy nào.

## 3. Chọn chuỗi giả thuyết: Viterbi bậc hai

Trạng thái là cặp (giả thuyết frame t−1, giả thuyết frame t). Chi phí của bộ ba (a, b, c) ở frame t:

| Thành phần | Công thức | Mặc định | Ý nghĩa |
|---|---|---|---|
| Xoay nhanh | `(|ω_t| / σ_rot)²` | σ_rot = 30°/frame | ω = vận tốc góc khung chậu, tính trong hệ camera |
| Đổi vận tốc góc | `(|ω_t − ω_{t−1}| / σ_acc)²` | σ_acc = 20°/frame² | Phạt cả việc đổi chiều xoay |
| Tay chân nhảy chỗ | `(local_t / σ_local)²` | σ_local = 0.05 m/frame | local = dịch chuyển trung bình của 23 khớp trong hệ khung chậu |
| Đổi giả thuyết | `switch_cost · [b ≠ c]` | 2.0 | Khi hai bên ngang nhau thì ưu tiên giữ WHAM |

σ_rot = 30°/frame (1800°/s ở 60 fps) lấy theo tài liệu: tốc độ xoay của VĐV đỉnh cao khoảng 1800–2000°/s.

## 4. Output (`output/<clip>/step1/`)

| File | Nội dung |
|---|---|
| `flips.csv` | Bảng theo frame (cột bên dưới) |
| `flips.png` | Yaw tích luỹ (WHAM world / WHAM camera / sau sửa) và bước xoay mỗi frame |
| `wham_step1.pkl` | Format giống `wham_output.pkl`. `pose` đã sửa; `pose_world[:, :3] = R_wc · gốc camera`; `verts` dựng lại bằng SMPL. `trans`, `trans_world`, `betas` giữ nguyên |
| `check_rgb.mp4` | Video kiểm chứng: skeleton chiếu lên RGB. Trái = WHAM, phải = sau sửa. Tay chân trái màu xanh, phải màu cam. Frame "quay ngược" có viền đỏ |
| `mesh_camera.mp4` | `render_mesh.py`: mesh phủ lên video silhouette, kiểu WHAM. Trái = WHAM gốc, phải = sau bước 1. Frame "quay ngược" có mesh cam, viền đỏ |
| `mesh_global.mp4` | `render_mesh.py --view global`: góc nhìn toàn cục có mặt đất, cùng bố cục trái / phải |

**Cột của `flips.csv`:**
- `frame`: frame trong video;
- `gia_thuyet`: giả thuyết được chọn;
- `doi_gia_thuyet`: frame này có đổi giả thuyết so với frame trước không;
- `canh_bao_wham`: `dang_ngo` nếu bước xoay > 45°/frame, `gan_chac_lat` nếu > 90°/frame;
- `buoc_xoay_*_deg`, `doi_van_toc_goc_*_deg`, `tay_chan_nhay_*_mm`: bước xoay, mức đổi vận tốc góc, độ nhảy của tay chân, mỗi thứ có bản `wham` và bản `sua`;
- `yaw_wham_cam_deg`, `yaw_wham_world_deg`, `yaw_sua_deg`: yaw tích luỹ của từng nguồn;
- `R_wc_dung_frame`: frame có được dùng để ước `R_wc` không.

Video RGB **chỉ dùng để kiểm chứng**; thuật toán không đọc video.

## 5. Cách chạy (WSL, từ thư mục project)

```bash
PY=/home/Latte/miniconda3/envs/wham-refine/bin/python
D=data/toeloop9_cam4
O=output/toeloop9_cam4/step1
$PY flips/detect_flips.py --wham $D/wham_output.pkl --out $O
$PY flips/check.py --wham $D/wham_output.pkl --step1 $O \
    --video $D/rgb.mp4 --calib $D/calib.txt --out $O/check_rgb.mp4
$PY flips/render_mesh.py --wham $D/wham_output.pkl --step1 $O --video $D/input_silhouette.mp4 --out $O/mesh_camera.mp4
$PY flips/render_mesh.py --wham $D/wham_output.pkl --step1 $O --video $D/input_silhouette.mp4 --view global --out $O/mesh_global.mp4
```

`detect_flips.py` chạy ~3 s, `check.py` ~20 s, mỗi lần `render_mesh.py` ~35 s.

## 6. Kết quả bản thử trên Toeloop_9 cam 4

**Số liệu:**
- Viterbi chọn "quay ngược" ở frame 36–37, 69 và 71–178.
- Bước xoay lớn nhất giảm từ 128.5 xuống 74.7°/frame.
- Số frame > 45° giảm từ 6 xuống 4; số frame > 90° giảm từ 1 xuống 0.

**Kiểm bằng mắt trên RGB.** Cách đọc: người quay mặt về camera thì tay phải nằm bên trái ảnh.

| Frame | Người thật | WHAM gốc | Sau bước 1 |
|---|---|---|---|
| 80 | quay lưng | sai | **đúng** |
| 100 | nhìn ngang | khó nói (bỏ sót chân duỗi) | khó nói |
| 120, 130, 136 | quay mặt | đúng | **sai** |
| 138, 140 | quay mặt | sai (lật 2–3 frame) | đúng |
| 145 | quay mặt | đúng | **sai** |
| 178 | quay lưng | sai | **đúng** |

**Kết luận:**
- Viterbi tìm đúng **chỗ** WHAM nhảy (69–71, 137–141, 179).
- Nhưng nó không biết **bên nào** của bước nhảy là sai. Trong đoạn 101–178, hai giả thuyết mượt như nhau, nên Viterbi giữ "quay ngược" cả đoạn. Thực tế người đã quay từ lưng sang mặt lúc nhìn ngang (~frame 100), và WHAM đúng trong đoạn 101–136.
- Độ mượt một mình **không đủ** để xác định trước/sau.

## 7. Câu hỏi mở (cần bàn trước khi làm tiếp)

Cần thêm một bằng chứng nằm trong từng frame hoặc trong cả đoạn, để biết giả thuyết nào đúng. Các ứng viên (không dùng RGB):

1. **Chiều xoay nhất quán của VĐV.** Mỗi VĐV xoay một chiều cố định (đa số ngược chiều kim đồng hồ). Soi gương đảo chiều xoay, nên có thể chọn đoạn có chiều xoay khớp với chiều đa số trong cú nhảy. Đây là kiến thức về trượt băng nói chung, không phải về riêng clip này.
2. **Lưỡi giày trượt dọc theo hướng di chuyển.** Hướng bàn chân trên băng phải song song với vận tốc. Cần quỹ đạo world đáng tin, mà WHAM thì đang trôi.
3. **Hướng di chuyển so với hướng thân** (trượt tiến hay trượt lùi). Yếu, vì VĐV làm được cả hai.
4. **Dấu hiệu phối cảnh.** Phần gần camera trông to hơn, nên tay chân gần / xa có kích thước ảnh khác nhau. Silhouette có giữ thông tin này, nhưng tín hiệu nhỏ khi người ở xa (8–12 m).
5. **Chấp nhận giới hạn.** Bước 1 chỉ sửa những cú lật ngắn (vài frame, có bước nhảy ở cả hai đầu, như 36–37 và 138–140). Đoạn dài thì gắn cờ "không xác định" và để bước sau xử lý.
