# WHAM-REFINE

Tinh chỉnh pose / mesh 3D của [WHAM](https://github.com/yohanshin/WHAM) chạy trên **video silhouette** (người bị bôi đen trên nền trắng để bảo vệ riêng tư).

Vì chỉ thấy bóng đen, WHAM không phân biệt được người quay mặt hay quay lưng về camera. Kết quả là có những cú xoay vô lý: đang bay mà đổi chiều xoay, lật trước/sau, nhầm trái/phải. Project này phát hiện và sửa các lỗi đó, rồi xuất lại kết quả cùng format với output WHAM (`wham_output.pkl`), kèm bảng số liệu và video so sánh.

## Trạng thái

| Bước | Thư mục | Nội dung | Trạng thái |
|---|---|---|---|
| 1 | `flips/` | Phát hiện lật trước/sau bằng độ mượt chuyển động (Viterbi) | Tạm dừng; tìm được chỗ lật nhưng không biết bên nào sai. Có thể bỏ qua |
| 2 | `angles/` | Sửa chiều xoay bằng vật lý: bảo toàn mô-men động lượng lúc bay, mô-men giới hạn trên băng | **Đã chạy** |
| 3 | `report/` | Bảng đánh dấu các vị trí vô lý | Chưa làm |
| 4 | `fix/` | Sửa tổng hợp, xuất pkl / video / bảng | Chưa làm |

Chi tiết từng bước nằm trong README của thư mục tương ứng: [`flips/README.md`](flips/README.md), [`angles/README.md`](angles/README.md).

**Kết quả bước 2 trên clip thử (Toeloop_9, cam 4):**
- Sau tiếp đất, số lần đổi chiều xoay giảm từ 14 xuống 4.
- Lúc bay, mô-men động lượng đổi dấu 1 lần thay vì 3.
- Hướng thân đúng ở cả 9 frame đã đối chiếu với video RGB.

## Cấu trúc thư mục

```
WHAM-REFINE/
├── env/       cài đặt môi trường: setup_env.sh, check_env.py, prepare_smpl.py, conda_spec.txt, requirements.txt
├── core/      dùng chung
│   ├── wham_io.py      đọc / ghi pkl WHAM, nạp SMPL, camera, ước lượng phép quay camera → world
│   ├── pose_ops.py     giả thuyết "quay ngược" (soi gương) của một silhouette
│   ├── mass.py         khối lượng, khối tâm, quán tính, mô-men động lượng từ mesh SMPL
│   └── silhouette.py   đọc video silhouette (điểm thấp nhất của bóng người)
├── flips/     bước 1 + script render video dùng chung (render_mesh.py)
├── angles/    bước 2: rotation_physics.py (đo), fix_rotation.py (sửa)
├── report/    bước 3 (chưa có code)
├── fix/       bước 4 (chưa có code)
├── data/      dữ liệu đầu vào (không gồm video RGB)
├── models/    model SMPL          ┐ không đưa lên git
└── output/    kết quả chạy        ┘
```

## Cài đặt môi trường

Project chạy trong **WSL (Ubuntu)** với một env conda tên `wham-refine` (Python 3.9). Env này gom mọi thứ cần dùng:

| Thành phần | Phiên bản | Dùng cho |
|---|---|---|
| torch + CUDA | 1.11 + CUDA 11.3 | SMPL, render |
| pytorch3d | 0.7.2 | Render mesh kiểu WHAM |
| smplx | 0.1.28 | Dựng mesh SMPL |
| nimblephysics | 0.10.36 | Mô hình BSM (COMETH) |
| cvxpy (OSQP, Clarabel, SCS) | 1.6 | Tối ưu |
| numpy, scipy, pandas, matplotlib, opencv, imageio-ffmpeg | xem `env/requirements.txt` | Tính toán, biểu đồ, video H.264 |
| COMETH | cài editable | Mô hình BSM |

### Cần có trước

1. **WSL Ubuntu + GPU NVIDIA** với driver hỗ trợ CUDA 11.3.
2. **Miniconda** trong WSL.
3. **Repo WHAM đã cài theo hướng dẫn của WHAM,** gồm env conda `wham` có pytorch3d build sẵn và chumpy. Script lấy pytorch3d và chumpy từ env này, vì build pytorch3d từ source rất mất công. Renderer cũng lấy từ repo WHAM (`lib/vis/renderer.py`).
4. **Repo [COMETH](https://github.com/PARCO-LAB/Biomechanical-Model).**
5. **File zip SMPLify** (`mpips_smplify_public_v2.zip`, tải từ trang SMPLify sau khi đăng ký), chứa model SMPL neutral. Model SMPL có giấy phép riêng nên không có trong repo này.

### Dựng env

Mặc định script dùng đường dẫn của máy phát triển. Đặt biến môi trường cho đúng máy bạn trước khi chạy:

```bash
export CONDA=$HOME/miniconda3/bin/conda                 # conda dùng để tạo env
export WHAM_ENV=$HOME/miniconda3/envs/wham              # env của WHAM (lấy pytorch3d, chumpy)
export COMETH_DIR=/path/to/COMETH                       # repo COMETH
export SMPL_ZIP=/path/to/mpips_smplify_public_v2.zip    # zip SMPLify
export ENV_NAME=wham-refine                             # tên env sẽ tạo (tuỳ chọn)

bash env/setup_env.sh            # tạo env nếu chưa có, cài đặt, rồi kiểm tra
bash env/setup_env.sh --force    # xoá env cũ và tạo lại từ đầu
```

Từ Git Bash trên Windows, gọi qua WSL (cần `MSYS_NO_PATHCONV=1` để đường dẫn `/mnt/...` không bị đổi sai):

```bash
MSYS_NO_PATHCONV=1 wsl -e bash /mnt/d/WHAM-REFINE/env/setup_env.sh
```

**Script làm lần lượt:**
1. **Tạo env từ `env/conda_spec.txt`,** danh sách package conda (linux-64) giống env `wham`, trừ `cudatoolkit-dev`. Gói này chỉ cần khi build pytorch3d, và làm đầy `/tmp`.
2. **Cài package pip** từ `env/requirements.txt` (đã ghim phiên bản).
3. **Chép pytorch3d** từ env `wham` sang (cùng Python 3.9, torch 1.11 cu113 nên tương thích).
4. **Cài COMETH kiểu editable** (`pip install --no-deps -e $COMETH_DIR`).
5. **Chuẩn bị SMPL** bằng `env/prepare_smpl.py` (chạy bằng Python của env `wham`, vì cần chumpy):
   - giải nén `SMPL_NEUTRAL.pkl` vào `models/smpl/` và kiểm tra md5;
   - đổi sang `SMPL_NEUTRAL.npz` (mảng numpy thuần), nhờ vậy env `wham-refine` không cần chumpy.
6. **Kiểm tra** bằng `env/check_env.py`.

Không làm `conda create --clone wham` được, vì `lib64` của env `wham` có thể là symlink tạo tay và làm clone hỏng.

**Kết quả kiểm tra khi env đã đủ:**

```
[OK]   SMPL: verts WHAM sai 0.65 mm
[OK]   COMETH: BSM 49 DOF
[OK]   QP: OSQP OK
[OK]   render: raster CUDA OK (<tên GPU>)
[OK]   video: H.264 OK
Env sẵn sàng.
```

Chỉ kiểm tra lại env mà không cài: `$PY env/check_env.py`.

## Dữ liệu đầu vào

Mỗi clip đặt trong `data/<clip>/`. Ví dụ `data/toeloop9_cam4/`:

| File | Nội dung |
|---|---|
| `wham_output.pkl` | Output của WHAM (`demo.py`) |
| `calib.txt` | Thông số nội camera `fx fy cx cy` (WHAM ước lượng) |
| `input_silhouette.mp4` | Video silhouette đã đưa vào WHAM |
| `slam_results.pth`, `tracking_results.pth` | Kết quả trung gian của WHAM (tuỳ chọn) |
| `rgb.mp4` | Video RGB thật, cùng frame. **Chỉ dùng để kiểm chứng bằng mắt;** thuật toán không đọc |

Repo có sẵn dữ liệu clip `data/toeloop9_cam4/`, trừ `rgb.mp4` (hình người thật). `models/` (model SMPL có giấy phép riêng) và `output/` không đưa lên git.

## Chạy

Trong WSL, từ thư mục project:

```bash
PY=$HOME/miniconda3/envs/wham-refine/bin/python
D=data/toeloop9_cam4
O=output/toeloop9_cam4/step2

# Bước 2a: đo vi phạm vật lý của WHAM (~10 s)
$PY angles/rotation_physics.py --wham $D/wham_output.pkl --silhouette $D/input_silhouette.mp4 --calib $D/calib.txt --out $O

# Bước 2b: sửa chiều xoay (~13 s)
$PY angles/fix_rotation.py --wham $D/wham_output.pkl --silhouette $D/input_silhouette.mp4 --calib $D/calib.txt --out $O

# Video so sánh mesh: trái WHAM gốc, phải sau sửa (~35 s mỗi video)
for v in camera global; do
  $PY flips/render_mesh.py --wham $D/wham_output.pkl --step1 $O --pkl-name wham_step2.pkl --csv-name rotation_fix.csv \
      --name 'sau buoc 2' --mark-label 'DA SUA' --video $D/input_silhouette.mp4 --view $v --out $O/mesh_$v.mp4
done
```

Mọi script đều có `--help` mô tả đủ tham số. Các tham số hay dùng của `fix_rotation.py`:
- `--flight a-b`: khai báo tay pha bay, thay cho tự suy từ silhouette;
- `--chieu ccw|cw|auto`: chiều xoay nhìn từ trên xuống, mặc định ngược kim đồng hồ. Silhouette không tự quyết định được chiều này;
- `--no-ice`: chỉ sửa pha bay;
- `--sigma-tau`: mức mô-men cho phép từ mặt băng (mặc định 100 N·m).

**Output** trong `output/<clip>/step2/`:

| File | Nội dung |
|---|---|
| `physics.csv`, `physics.png` | Bước 2a: quán tính, mô-men động lượng, tốc độ xoay, pha bay, cảnh báo theo frame |
| `wham_step2.pkl` | Kết quả đã sửa, **cùng format `wham_output.pkl`** |
| `rotation_fix.csv`, `rotation_fix.png` | Bước 2b: giả thuyết được chọn, góc xoay thêm, hướng thân và tốc độ xoay trước / sau |
| `mesh_camera.mp4`, `mesh_global.mp4` | Video mesh kiểu WHAM: mesh trên silhouette, và góc nhìn toàn cục có mặt đất |

Bước 1 (tuỳ chọn, đang tạm dừng): xem lệnh chạy trong [`flips/README.md`](flips/README.md). Bước 2 nhận được cả `wham_output.pkl` gốc lẫn `wham_step1.pkl`.

## Ghi chú

- **Phạm vi:** mới kiểm trên một clip (Toeloop_9, cam 4), và một số tham số được chọn trên clip này. Cần kiểm thêm trên clip khác.
- **Giả định camera:** camera đứng yên. Mọi tính toán vật lý làm trong hệ camera, coi là hệ quán tính.
- **Ngôn ngữ:** tài liệu, comment và `--help` viết bằng tiếng Việt; tên biến và thuật ngữ giữ tiếng Anh.
