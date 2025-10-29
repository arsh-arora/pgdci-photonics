import matplotlib.pyplot as plt
import numpy as np
import os

def plot_overlay(t_in, x_in, t_out, x_a, x_b, labels, path, tmax=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if tmax:
        mask_in = t_in <= tmax
        mask_out = t_out <= tmax
    else:
        mask_in = slice(None)
        mask_out = slice(None)
    plt.figure(figsize=(10,4))
    plt.plot(t_in[mask_in], x_in[mask_in], label="300 fps")
    plt.plot(t_out[mask_out], x_a[mask_out], label=labels[0])
    plt.plot(t_out[mask_out], x_b[mask_out], label=labels[1], alpha=0.8)
    plt.xlabel("time (s)"); plt.ylabel("position (nm)")
    plt.legend(); plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()

def plot_uncertainty(t, mean, var, path, tmax=None, sigmas=2.0):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if tmax:
        m = t <= tmax
    else:
        m = slice(None)
    std = np.sqrt(np.maximum(var, 0))
    plt.figure(figsize=(10,4))
    plt.plot(t[m], mean[m], label="mean")
    plt.fill_between(t[m], mean[m]-sigmas*std[m], mean[m]+sigmas*std[m], alpha=0.25, label=f"±{sigmas}σ")
    plt.xlabel("time (s)"); plt.ylabel("position (nm)")
    plt.legend(); plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
