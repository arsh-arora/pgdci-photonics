import numpy as np
import scipy.io as sio
import pandas as pd
from .config import cfg

def load_mat_as_nm(mat_path=None, key=None, px_to_nm=None, fps=None):
    mat_path = mat_path or cfg.paths.mat_path
    px_to_nm = px_to_nm or cfg.sampling.px_to_nm
    fps = fps or cfg.sampling.fps_in
    d = sio.loadmat(mat_path)
    if key is None:
        cands = [(k, np.array(v).squeeze()) for k,v in d.items() if not k.startswith("__")]
        cands = [(k,a) for k,a in cands if a.ndim==1 and 5000 < a.size < 200000]
        if not cands:
            raise ValueError("No 1D vector found in .mat")
        key, x = max(cands, key=lambda kv: kv[1].size)
    else:
        x = np.array(d[key]).squeeze()
    x_nm = x.astype(float) * px_to_nm
    t = np.arange(x_nm.size) / fps
    return t, x_nm, key

def save_csv(path, **cols):
    pd.DataFrame(cols).to_csv(path, index=False)
