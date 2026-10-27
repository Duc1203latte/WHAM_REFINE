#!/usr/bin/env bash
# Dựng env conda wham-refine cho project (chạy trong WSL).
#
#   bash env/setup_env.sh            tạo env nếu chưa có, rồi cài / kiểm tra
#   bash env/setup_env.sh --force    xoá env cũ và tạo lại từ đầu
#
# Từ Git Bash trên Windows:
#   MSYS_NO_PATHCONV=1 wsl -e bash /mnt/d/WHAM-REFINE/env/setup_env.sh
#
# Cần có sẵn: miniconda, env conda `wham` (lấy pytorch3d đã build và chumpy),
# repo COMETH, file zip SMPLify chứa SMPL neutral.
set -euo pipefail

CONDA=${CONDA:-/home/Latte/miniconda3/bin/conda}
ENV_NAME=${ENV_NAME:-wham-refine}
WHAM_ENV=${WHAM_ENV:-/home/Latte/miniconda3/envs/wham}
COMETH_DIR=${COMETH_DIR:-/mnt/d/COMETH}
SMPL_ZIP=${SMPL_ZIP:-/mnt/d/mpips_smplify_public_v2.zip}

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ENV_DIR=$(dirname "$(dirname "$CONDA")")/envs/$ENV_NAME
PY=$ENV_DIR/bin/python
SITE=$ENV_DIR/lib/python3.9/site-packages

if [[ "${1:-}" == "--force" && -d "$ENV_DIR" ]]; then
    echo "== Xoá env cũ $ENV_DIR"
    "$CONDA" env remove -y -n "$ENV_NAME" || rm -rf "$ENV_DIR"
fi

# 1. Package conda: cùng bản với env wham (torch 1.11 + CUDA 11.3), bỏ cudatoolkit-dev.
#    Không dùng `conda create --clone wham`: lib64 của env wham là symlink tạo tay nên clone hỏng.
if [[ ! -x "$PY" ]]; then
    echo "== Tạo env $ENV_NAME từ conda_spec.txt"
    "$CONDA" create -y -n "$ENV_NAME" --file "$HERE/conda_spec.txt"
fi

# 2. Package pip (numpy, nimblephysics, cvxpy, smplx, ...).
echo "== Cài package pip"
"$PY" -m pip install -q -r "$HERE/requirements.txt"

# 3. pytorch3d: chép bản đã build từ source trong env wham (cùng Python 3.9, torch 1.11 cu113).
if ! "$PY" -c "import pytorch3d" 2>/dev/null; then
    echo "== Chép pytorch3d từ $WHAM_ENV"
    cp -a "$WHAM_ENV"/lib/python3.9/site-packages/pytorch3d \
          "$WHAM_ENV"/lib/python3.9/site-packages/pytorch3d-*.dist-info "$SITE"/
fi

# 4. COMETH kiểu editable, --no-deps để không đổi phiên bản package.
echo "== Cài COMETH (editable) từ $COMETH_DIR"
"$PY" -m pip install -q --no-deps -e "$COMETH_DIR"

# 5. SMPL: giải nén + chuyển sang npz bằng env wham (có chumpy).
if [[ ! -f "$HERE/../models/smpl/SMPL_NEUTRAL.npz" ]]; then
    echo "== Chuẩn bị SMPL từ $SMPL_ZIP"
    "$WHAM_ENV"/bin/python "$HERE/prepare_smpl.py" --zip "$SMPL_ZIP"
fi

# 6. Kiểm tra.
echo "== Kiểm tra env"
cd /tmp && "$PY" "$HERE/check_env.py"
