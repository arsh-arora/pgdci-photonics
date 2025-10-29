import numpy as np
import torch
from torch.utils.data import Dataset

def synth_ou(batch, T, dt, tau, sigma):
    a = np.exp(-dt/tau)
    q = sigma**2 * (1 - a*a)
    x = np.zeros((batch, T), dtype=np.float32)
    w = np.random.randn(batch, T).astype(np.float32)*np.sqrt(q)
    for t in range(1,T):
        x[:,t] = a*x[:,t-1] + w[:,t]
    return x

def color_psd(x, alpha=0.2):
    X = np.fft.rfft(x, axis=1)
    f = np.maximum(np.arange(1, X.shape[1]+1), 1.0)
    weight = (1.0/f)**alpha
    X *= weight[None,:]
    y = np.fft.irfft(X, n=x.shape[1], axis=1)
    return y.astype(np.float32)

class MaskedImputeDataset(Dataset):
    def __init__(self, n=1024, T=15000, dt=1/300):
        self.n, self.T, self.dt = n, T, dt
    def __len__(self): return self.n
    def __getitem__(self, i):
        tau = 10**np.random.uniform(-2.5, -1.5)   # ~3–30 ms
        sigma = 10**np.random.uniform(1.2, 2.1)   # ~16–125 nm
        x = synth_ou(1, self.T, self.dt, tau, sigma)[0]
        if np.random.rand()<0.3:
            x = color_psd(x[None,:], alpha=np.random.uniform(0.1,0.4))[0]
        stride = np.random.choice([3,4])
        mask = np.zeros_like(x); mask[::stride]=1.0
        y_obs = x*mask
        return torch.from_numpy(x).float(), torch.from_numpy(y_obs).float(), torch.from_numpy(mask).float()
