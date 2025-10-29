import torch, torch.nn.functional as F
import numpy as np
from scipy.signal import welch

def welch_psd_torch(x, nperseg=1024):
    # CPU SciPy for reliability; detaches graph by design (only used as regularizer target).
    x_np = x.detach().cpu().squeeze(1).numpy()
    logs = []
    for xi in x_np:
        f, P = welch(xi, nperseg=min(nperseg, len(xi)))
        logs.append(np.log(P+1e-12))
    return torch.tensor(np.stack(logs), dtype=torch.float32, device=x.device)

def loss_measurement(x0_hat, y_obs, mask):
    return F.mse_loss(x0_hat*mask, y_obs)

def loss_psd(x0_hat, prior):
    psd_t = welch_psd_torch(x0_hat)
    psd_p = welch_psd_torch(prior)
    return F.l1_loss(psd_t, psd_p)

def loss_energy(x0_hat):
    return (x0_hat**2).mean()
