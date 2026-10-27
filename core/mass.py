"""Khối lượng, khối tâm, quán tính và mô-men động lượng của cơ thể từ mesh SMPL.

Coi cơ thể là khối đặc, khối lượng riêng đều, giới hạn bởi mesh SMPL (mesh SMPL kín). Chia khối thành
các tứ diện (một đỉnh ở khối tâm, ba đỉnh là một mặt tam giác của mesh) rồi cộng công thức đúng của
từng tứ diện. Thể tích có dấu, nên phần lõm tự trừ đi.

Với tứ diện có một đỉnh ở gốc toạ độ, ba đỉnh còn lại x1, x2, x3, khối lượng m:
    ∫ x xᵀ dm = m/20 · (Σ xᵢ xᵢᵀ + s sᵀ),            s = x1 + x2 + x3
Nếu vận tốc trong tứ diện nội suy tuyến tính từ vận tốc 4 đỉnh (đỉnh gốc có vận tốc 0):
    ∫ x × v dm = m/20 · (Σ xᵢ × uᵢ + s × Σ uᵢ)
Hai công thức cùng dạng, nên khi cơ thể quay cứng (u = ω × x) thì L = I ω đúng tuyệt đối.
"""
import numpy as np


def mass_properties(verts, faces, density=1000.0):
    """verts (N,V,3) m, faces (F,3). Trả (mass (N,), com (N,3), inertia quanh khối tâm (N,3,3))."""
    tri = verts[:, faces]                                  # (N,F,3,3): 3 đỉnh của mỗi mặt
    o = verts.mean(1)[:, None, None]                       # gốc tạm cho đỡ sai số làm tròn
    x = tri - o
    vol = np.einsum('nfi,nfi->nf', x[:, :, 0], np.cross(x[:, :, 1], x[:, :, 2])) / 6.0
    V = vol.sum(1)
    com = o[:, 0, 0] + np.einsum('nf,nfi->ni', vol, x.sum(2) / 4.0) / V[:, None]
    x = tri - com[:, None, None]                           # đổi gốc về khối tâm
    m = density * vol                                      # (N,F) khối lượng từng tứ diện (cùng thể tích)
    s = x.sum(2)
    C = np.einsum('nf,nfki,nfkj->nij', m, x, x) + np.einsum('nf,nfi,nfj->nij', m, s, s)
    C /= 20.0                                              # ∫ x xᵀ dm
    inertia = np.trace(C, axis1=1, axis2=2)[:, None, None] * np.eye(3) - C
    return density * V, com, inertia


def angular_momentum(verts, faces, com, fps, density=1000.0):
    """Mô-men động lượng quanh khối tâm (N,3), kg·m²/s. Vận tốc lấy sai phân trung tâm (2 đầu: một phía)."""
    vel = np.gradient(verts, 1.0 / fps, axis=0)            # (N,V,3)
    vcom = np.gradient(com, 1.0 / fps, axis=0)
    tri = verts[:, faces] - com[:, None, None]
    u = vel[:, faces] - vcom[:, None, None]                # vận tốc so với khối tâm
    vol = np.einsum('nfi,nfi->nf', tri[:, :, 0], np.cross(tri[:, :, 1], tri[:, :, 2])) / 6.0
    m = density * vol
    L = np.einsum('nf,nfi->ni', m, np.cross(tri, u).sum(2)) \
        + np.einsum('nf,nfi->ni', m, np.cross(tri.sum(2), u.sum(2)))
    return L / 20.0
