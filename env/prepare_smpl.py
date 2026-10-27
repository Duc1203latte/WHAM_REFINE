"""Giải nén SMPL neutral từ zip SMPLify và chuyển sang .npz (không cần chumpy).

Phải chạy bằng Python có chumpy (env conda `wham`), vì file pkl gốc chứa mảng chumpy.

Output trong models/smpl/:
    SMPL_NEUTRAL.pkl  bản gốc trong zip
    SMPL_NEUTRAL.npz  cùng nội dung, chỉ gồm mảng numpy
"""
import argparse
import hashlib
import os
import pickle
import zipfile

import numpy as np
import scipy.sparse as sp

ZIP_MEMBER = 'smplify_public/code/models/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl'
MD5 = '65dc7f162f3ef21a38637663c57e14a7'     # trùng file WHAM dùng


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description='Giải nén SMPL neutral và chuyển sang npz.')
    ap.add_argument('--zip', default='/mnt/d/mpips_smplify_public_v2.zip', help='file zip SMPLify')
    ap.add_argument('--out', default=os.path.join(here, '..', 'models', 'smpl'), help='thư mục output')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    pkl_path = os.path.join(args.out, 'SMPL_NEUTRAL.pkl')
    with zipfile.ZipFile(args.zip) as z, open(pkl_path, 'wb') as f:
        f.write(z.read(ZIP_MEMBER))
    md5 = hashlib.md5(open(pkl_path, 'rb').read()).hexdigest()
    print('SMPL_NEUTRAL.pkl md5', md5, 'OK' if md5 == MD5 else 'KHÁC file WHAM dùng!')

    d = pickle.load(open(pkl_path, 'rb'), encoding='latin1')
    out = {}
    for k, v in d.items():
        if sp.issparse(v):
            v = v.toarray()
        elif hasattr(v, 'r'):                      # mảng chumpy
            v = np.array(v.r)
        if isinstance(v, np.ndarray):
            out[k] = v
    np.savez(os.path.join(args.out, 'SMPL_NEUTRAL.npz'), **out)
    print('SMPL_NEUTRAL.npz:', ', '.join(f'{k}{v.shape}' for k, v in out.items()))


if __name__ == '__main__':
    main()
