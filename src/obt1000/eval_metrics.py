import numpy as np
from scipy.signal import welch

def rmse(a,b): return float(np.sqrt(np.mean((a-b)**2)))

def psd_distance(x, y, nperseg=4096):
    f, Px = welch(x, nperseg=min(nperseg, len(x)))
    _, Py = welch(y, nperseg=min(nperseg, len(y)))
    Px = np.log(Px+1e-12); Py = np.log(Py+1e-12)
    return float(np.mean(np.abs(Px-Py)))

def corner_frequency_match(x, fs, target_fc=None):
    # very rough: find -3dB corner vs. DC; return deviation if target_fc provided
    f, P = welch(x, fs=fs, nperseg=min(4096, len(x)))
    P = P / P.max()
    idx = np.where(P < 0.5)[0]
    fc = f[idx[0]] if len(idx) else np.nan
    if target_fc is None: return fc
    return abs(fc - target_fc)
