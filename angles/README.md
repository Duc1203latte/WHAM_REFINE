# Bước 2: sửa chuyển động xoay của WHAM bằng vật lý (inverse dynamics mức toàn thân)

Trạng thái (2026-10-27): **2a và 2b đã chạy**. 2b sửa cả pha bay lẫn phần trên băng (2c).

## Vấn đề

Silhouette của một người chỉ cho biết người đang ở **một trong hai** hướng: hướng của WHAM, hoặc hướng soi gương (quay ngược, trái ↔ phải). Xem `flips/README.md`.

Hai cách hiểu này **xoay ngược chiều nhau**. WHAM nhảy qua lại giữa chúng, nên trong video, đang xoay chiều này thì bỗng xoay ngược lại.

Bước 1 dùng độ mượt nên không bắt được lỗi này. Thứ còn thiếu là ràng buộc vật lý: khi bay, người xoay **một chiều**, với tốc độ do tư thế quyết định.

## Mô hình vật lý (`core/mass.py`)

- Coi cơ thể là **khối đặc, khối lượng riêng đều** (mặc định 1000 kg/m³), giới hạn bởi mesh SMPL. `betas` cố định bằng trung vị qua các frame, vì `betas` của WHAM đổi theo frame (khối lượng nhảy 51–61 kg).
- Chia khối thành các tứ diện (khối tâm + một mặt mesh), rồi dùng công thức đúng của từng tứ diện để tính khối lượng, khối tâm, tensor quán tính và mô-men động lượng.
- Hai công thức cùng dạng nên nhất quán: khi quay cứng thì `L = I·ω` đúng tuyệt đối (đã kiểm).
- Mọi thứ tính trong **hệ camera**. Camera đứng yên nên đây là hệ quán tính.
- Trục thẳng đứng `up = R_wc⁻¹·[0,1,0]`.

**Đại lượng quanh trục dọc qua khối tâm:**

| Ký hiệu | Ý nghĩa | Đơn vị |
|---|---|---|
| `I_v` | Quán tính | kg·m² |
| `L_v` | Mô-men động lượng | kg·m²/s |
| `ω = L_v / I_v` | Tốc độ xoay của toàn thân | °/s |
| `τ = dL_v/dt` | Mô-men ngoài mà băng phải tạo ra | N·m |

## Pha bay (`rotation_physics.py: detect_flight`)

1. **Đoạn lõi:** đoạn dài nhất mà độ cao khối tâm khớp parabol rơi tự do (gia tốc −g), sai số RMS < 2 cm. Trên băng thì sai 11–15 cm, nên tách rõ.
2. **Nới hai đầu** khi cả hai điều sau còn đúng:
   - điểm thấp nhất của **silhouette** còn cao hơn mức băng gần đó + 0.15 m (độ cao suy từ tia nhìn và độ sâu khung chậu);
   - khối tâm còn bám parabol (lệch < 4 cm).
3. Trong 2b còn có thêm bước **cắt hai đầu** theo mức khớp mô hình (xem dưới).

`--flight a-b` cho phép khai báo tay để bỏ qua phần tự suy.

## 2a: báo cáo (`rotation_physics.py`), chỉ đo, không sửa

Output trong `output/<clip>/step2/`:
- `physics.csv`: theo frame, gồm `bay`, độ cao chân / khối tâm, `I_v`, `L_v`, `ω`, ω riêng của khung chậu, `τ`, cảnh báo (`nguoc_chieu`, `qua_nhanh`);
- `physics.png`: biểu đồ 4 tầng.

**Kết quả Toeloop_9 cam 4:**
- Pha bay tự suy: frame 11–53.
- Lúc bay, `L_v` của WHAM dao động từ −25 đến +18 kg·m²/s và **đổi dấu 5 lần**. Đáng lẽ phải không đổi.
- `|ω|` lớn nhất 1876°/s, nằm trong ngưỡng 2000°/s. Vậy lỗi là ở **chiều** xoay, không phải độ lớn.
- `I_v` lúc bay giảm từ 1.9 xuống ~0.7 kg·m² (người thu tay chân), hợp lý.

## 2b: sửa xoay (`fix_rotation.py`): pha bay theo L không đổi, trên băng theo mô-men giới hạn

**Ý tưởng.** Lúc bay L không đổi, nên hướng thân phải là

```
θ(t) = θ0 + L · ∫ dt / I(t)
```

Cả đường chỉ còn **2 ẩn** (θ0, L). Nó luôn xoay một chiều, và nhanh / chậm đúng theo tư thế.

**Các bước:**
1. Mỗi frame có 2 hướng quan sát được (WHAM / soi gương). Hướng = trục z của khung chậu chiếu lên mặt ngang, tăng theo chiều ngược kim đồng hồ khi nhìn từ trên xuống.
2. **Dò lưới** (θ0 ∈ 0..359°, L ∈ ±0.25..60 kg·m²/s) để tìm đường θ(t) gần **một trong hai** hướng nhất. Sai số mỗi frame cắt ở 60°, để frame hỏng không kéo lệch cả đường.
3. **Cắt hai đầu pha bay:** nếu trung vị độ lệch của 3 frame ở một đầu > 30°, bỏ đầu đó (frame còn đang đẩy khỏi băng / đã chạm băng) rồi fit lại. Mỗi đầu cắt tối đa 8 frame.
4. **Chiều xoay:** silhouette không phân biệt được, vì soi gương cả đoạn bay cũng ra một cú xoay hợp lệ. Chọn bằng `--chieu`:
   - `ccw` (mặc định): ngược kim đồng hồ, chiều của đa số VĐV;
   - `cw`;
   - `auto`: theo sai số, nhưng hai chiều thường sát nhau nên không đáng tin.
5. **Trên băng** (`--ice`, mặc định bật; `--no-ice` để chỉ sửa pha bay). Xem mục "2c" bên dưới.
6. **Mỗi frame:** chọn giả thuyết gần hướng mục tiêu nhất, rồi xoay thêm quanh trục dọc (qua khung chậu) cho sát mục tiêu. Giới hạn góc xoay thêm để silhouette không lệch nhiều:
   - pha bay: `--max-extra` = 20°;
   - trên băng: `--max-extra-ice` = 30°.

### 2c: phần trên băng (`fix_rotation.py: track_ice`)

**Ý tưởng.**
- Trên băng, lưỡi giày tạo được mô-men nên L **được phép đổi**. Nhưng mô-men đó có giới hạn, nên chiều xoay không thể đổi qua lại liên tục như WHAM.
- Hướng thân lúc bay đã biết chắc, nên dùng làm **điểm neo** rồi đi tiếp ra hai phía:
  - sau tiếp đất: đi xuôi thời gian;
  - trước bật nhảy: đi ngược thời gian.

**Quy hoạch động** trên lưới trạng thái (hướng θ: bước 1°, tốc độ ω: bước 1°/frame, |ω| ≤ 40°/frame). Mỗi frame:

```
ω' = ω + a,   |a| ≤ 8 °/frame²            đổi tốc độ có giới hạn
θ' = θ + ω'
τ  = (I' · ω' − I · ω) / dt                mô-men từ băng (N·m), ω đổi sang rad/s
chi phí = (τ / σ_τ)²  +  (min(lệch tới giả thuyết gần nhất, 60°) / σ_θ)²
```

- σ_τ = `--sigma-tau` = 100 N·m. Để hãm cú xoay sau tiếp đất (L ≈ 23) về 0 trong ~0.3 s cần ~75 N·m.
- σ_θ = `--sigma-theta` = 15°.
- Trạng thái đầu là trạng thái (θ, ω) của đường vật lý ở frame bay sát đó. Cuối cùng truy vết ngược để lấy chuỗi tối ưu.

**Độ nhạy** (đã thử; chọn có nhìn RGB, nên cần kiểm thêm trên clip khác):
- σ_τ = 50 hoặc 100: đúng hết các frame kiểm RGB.
- σ_τ = 200: mô-men cho phép quá lớn, thuật toán lại cho đổi chiều xoay và sai ở frame 120–140.
- σ_θ = 10 / 15 / 25: đều đúng.

**Output** trong `output/<clip>/step2/`:

| File | Nội dung |
|---|---|
| `wham_step2.pkl` | Format WHAM. `pose` đã sửa; `pose_world[:, :3] = R_wc · gốc camera`; `verts` dựng lại bằng SMPL (betas gốc từng frame) |
| `rotation_fix.csv` | Theo frame: `bay`, `gia_thuyet`, `sua`, `goc_xoay_them_deg`, `lech_con_lai_deg` (lệch còn lại tới đường vật lý), hướng WHAM / vật lý / sau sửa, ω WHAM / sau sửa / vật lý, `L_v` WHAM / sau sửa |
| `rotation_fix.png` | Hướng, tốc độ xoay và L, trước / sau |
| `mesh_camera.mp4`, `mesh_global.mp4` | `flips/render_mesh.py`: trái WHAM, phải sau sửa. Frame bị sửa có mesh cam |

**Kết quả Toeloop_9 cam 4:**
- **Pha bay sau khi cắt:** frame 17–49 (RGB, chỉ để kiểm chứng: bay ~14–48).
- **Đường vật lý:** L* = +23.25 kg·m²/s, xoay 785° (~2.2 vòng), tốc độ ~1100 → 2000°/s.
  - Chiều `cw` cho sai số gần bằng (10582 so với 10557).
  - Theo RGB, chiều thật là ccw.
- **Xoay thêm:**
  - pha bay: trung vị 10.5°, 9/33 frame chạm giới hạn 20°;
  - trên băng: trung vị 7.0°, 21/153 frame chạm giới hạn 30°.
- **L lúc bay:**
  - WHAM: độ lệch chuẩn 13.1, đổi dấu 3 lần;
  - sau sửa: độ lệch chuẩn **5.4**, đổi dấu **1 lần**.
- **Sau tiếp đất,** số lần đổi chiều xoay (bỏ qua |ω| < 100°/s): WHAM **14**, hướng mục tiêu **0**, sau sửa **4**.
  - 4 lần còn lại nằm ở các frame mà mục tiêu cách cả hai giả thuyết > 30° (quanh frame 60–75, ngay sau tiếp đất).
- **Đối chiếu RGB** (chỉ để kiểm chứng): đúng **9/9** frame đã kiểm.

  | Frame | RGB | Chọn |
  |---|---|---|
  | 80 | quay lưng | quay ngược |
  | 120, 130, 136 | quay mặt | WHAM |
  | 138, 140 | quay mặt (WHAM lật) | quay ngược |
  | 145 | quay mặt | WHAM |
  | 178 | quay lưng (WHAM lật) | quay ngược |
  | 182 | quay lưng | WHAM |

- **Thử nới `--max-extra-ice`:** 45° không bớt đổi chiều hơn 30°. Còn nếu nới cả `--max-extra` của pha bay lên 30° thì pha bay xấu đi.

## Cách chạy (WSL, từ thư mục project)

```bash
PY=/home/Latte/miniconda3/envs/wham-refine/bin/python
D=data/toeloop9_cam4
O=output/toeloop9_cam4/step2
$PY angles/rotation_physics.py --wham $D/wham_output.pkl --silhouette $D/input_silhouette.mp4 --calib $D/calib.txt --out $O
$PY angles/fix_rotation.py     --wham $D/wham_output.pkl --silhouette $D/input_silhouette.mp4 --calib $D/calib.txt --out $O
for v in camera global; do
  $PY flips/render_mesh.py --wham $D/wham_output.pkl --step1 $O --pkl-name wham_step2.pkl --csv-name rotation_fix.csv \
      --name 'sau buoc 2' --mark-label 'DA SUA' --video $D/input_silhouette.mp4 --view $v --out $O/mesh_$v.mp4
done
```

Thời gian chạy: `rotation_physics.py` ~10 s, `fix_rotation.py` ~13 s, mỗi video ~35 s.

## Vấn đề còn lại / bước tiếp theo

1. **Ngay sau tiếp đất (frame ~60–75)** còn vài cú giật ngược chiều. Ở đó hướng mục tiêu cách cả hai giả thuyết > 30°, tức WHAM ở đoạn này sai nhiều hơn chuyện chọn nhầm giữa hai cách hiểu.
2. **σ_τ chọn có nhìn RGB của clip này.** Cần kiểm trên clip khác.
3. **Mesh nghiêng trong góc nhìn toàn cục.**
   - Hướng world dựng từ nhánh camera + `R_wc` cố định, nên mesh lúc bay nghiêng ~20–30°; mesh WHAM gốc thì thẳng.
   - Người có nghiêng thật trong ảnh, nhưng độ nghiêng theo chiều sâu không quan sát được.
   - Phương án khác: giữ độ nghiêng của nhánh world WHAM, chỉ thay hướng (yaw).
4. **Pha bay tự suy** lệch vài frame so với RGB (đầu muộn 3 frame).
