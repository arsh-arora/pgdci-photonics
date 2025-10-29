import numpy as np
from scipy import interpolate, signal
from .config import cfg

def upsample_cubic(t_in, x, fps_out=None):
    fps_out = fps_out or cfg.sampling.fps_out
    dt_out = 1.0/fps_out
    t_out = np.arange(int(t_in[-1]*fps_out)+1)*dt_out
    x_out = interpolate.CubicSpline(t_in, x)(t_out)
    return t_out, x_out

def upsample_fft_equal_span(x, num_out):
    # FFT-based resampling across the fixed span (0…T)
    return signal.resample(x, num_out)

def upsample_polyphase(t_in, x, fps_in=None, fps_out=None):
    fps_in = fps_in or cfg.sampling.fps_in
    fps_out = fps_out or cfg.sampling.fps_out
    # rational approximation for polyphase resample
    from math import gcd
    g = gcd(int(fps_out), int(fps_in))
    up = int(fps_out // g)
    down = int(fps_in // g)
    x_out = signal.resample_poly(x, up, down, window=("kaiser", 8.0))
    t_out = np.arange(x_out.size) / fps_out
    # ensure same total duration
    last_t = t_in[-1]
    valid = t_out <= last_t + 1e-12
    return t_out[valid], x_out[valid]
