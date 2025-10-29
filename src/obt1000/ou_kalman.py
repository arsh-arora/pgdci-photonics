import numpy as np
from .config import cfg

def estimate_ar1_params(x, dt, min_a=None, max_a=None):
    min_a = cfg.ou.min_a if min_a is None else min_a
    max_a = cfg.ou.max_a if max_a is None else max_a
    xz = x - x.mean()
    c0 = np.mean(xz*xz)
    c1 = np.mean(xz[:-1]*xz[1:])
    a = np.clip(c1/max(c0, 1e-12), min_a, max_a)
    tau = -dt/np.log(a)
    sigma2 = c0
    q = sigma2*(1-a*a)
    r = np.var(x[1:]-x[:-1]) * 0.25  # conservative camera noise guess
    return a, q, r, tau, sigma2

def kalman_rts(y, a, q, r):
    n = len(y)
    x_f = np.zeros(n); P_f = np.zeros(n)
    x_f[0]=y[0]; P_f[0]=q/(1-a*a)+1e-9
    for k in range(1,n):
        x_pred = a*x_f[k-1]; P_pred = a*a*P_f[k-1]+q
        K = P_pred/(P_pred+r)
        x_f[k] = x_pred + K*(y[k]-x_pred)
        P_f[k] = (1-K)*P_pred
    x_s = np.copy(x_f); P_s = np.copy(P_f)
    for k in range(n-2,-1,-1):
        A = a
        C = (A*P_f[k])/(A*A*P_f[k]+q)
        x_s[k] = x_f[k] + C*(x_s[k+1]-A*x_f[k])
        P_s[k] = P_f[k] + C*C*((A*A*P_f[k]+q)-P_f[k+1])
    return x_s, P_s

def ou_interpolate_to_1000fps(t_in, x_mean, tau, fps_out=None):
    fps_out = fps_out or cfg.sampling.fps_out
    dt_out = 1.0/fps_out
    t_out = np.arange(int(t_in[-1]*fps_out)+1)*dt_out
    x_out = np.zeros_like(t_out)
    j = 0; x_out[0]=x_mean[0]
    for i in range(1,len(t_out)):
        while j+1 < len(t_in) and t_in[j+1] <= t_out[i]:
            j += 1
        dt = t_out[i]-t_in[j]
        a = np.exp(-dt/tau)
        x_out[i] = a*x_mean[j]
    return t_out, x_out

def ou_fullgrid_1000fps(y_300, fps_in, fps_out=None, r_extra=None):
    """State-space model on the 1000 fps grid, with observations only at 300 fps times."""
    fps_out = fps_out or cfg.sampling.fps_out
    dt_in = 1.0/fps_in
    dt_out = 1.0/fps_out
    a300, q300, r, tau, _ = estimate_ar1_params(y_300, dt_in)
    if r_extra is not None: r = r_extra

    # derive per-1000fps AR params
    a1k = np.exp(-dt_out/tau)
    sigma2 = np.var(y_300 - y_300.mean())
    q1k = sigma2*(1 - a1k*a1k)

    # build observations mask at 1000fps grid
    T = int((len(y_300)-1)*fps_out/fps_in)+1
    t1k = np.arange(T)*dt_out
    # place 300fps obs onto 1000fps indices
    idx_obs = (np.arange(len(y_300))*fps_out/fps_in).round().astype(int)
    idx_obs = np.clip(idx_obs, 0, T-1)  # Ensure indices are within bounds
    mask = np.zeros(T, dtype=bool); mask[idx_obs]=True
    y1k = np.zeros(T); y1k[idx_obs] = y_300

    # Kalman on 1000fps grid with masked observations
    x_f = np.zeros(T); P_f = np.zeros(T)
    x_f[0]= y1k[0] if mask[0] else 0.0
    P_f[0]= q1k/(1-a1k*a1k)+1e-9
    for k in range(1,T):
        x_pred = a1k*x_f[k-1]; P_pred = a1k*a1k*P_f[k-1]+q1k
        if mask[k]:
            K = P_pred/(P_pred+r)
            x_f[k] = x_pred + K*(y1k[k]-x_pred)
            P_f[k] = (1-K)*P_pred
        else:
            x_f[k] = x_pred; P_f[k] = P_pred
    x_s = np.copy(x_f); P_s = np.copy(P_f)
    for k in range(T-2,-1,-1):
        C = (a1k*P_f[k])/(a1k*a1k*P_f[k]+q1k)
        x_s[k] = x_f[k] + C*(x_s[k+1]-a1k*x_f[k])
        P_s[k] = P_f[k] + C*C*((a1k*a1k*P_f[k]+q1k)-P_f[k+1])
    return t1k, x_s, P_s
