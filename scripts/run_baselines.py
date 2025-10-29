import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.obt1000.config import cfg
from src.obt1000.io_prep import load_mat_as_nm, save_csv
from src.obt1000.baselines import upsample_cubic, upsample_polyphase, upsample_fft_equal_span
from src.obt1000.plotting import plot_overlay
from src.obt1000.utils import ensure_dir
import numpy as np

if __name__ == "__main__":
    ensure_dir(cfg.paths.out_dir)
    t, x_nm, key = load_mat_as_nm(cfg.paths.mat_path, cfg.paths.mat_key,
                                  cfg.sampling.px_to_nm, cfg.sampling.fps_in)

    # cubic
    t_c, x_c = upsample_cubic(t, x_nm, cfg.sampling.fps_out)
    save_csv(f"{cfg.paths.out_dir}/1000fps_cubic_nm.csv", t_s=t_c, x_nm=x_c)

    # polyphase
    t_p, x_p = upsample_polyphase(t, x_nm, cfg.sampling.fps_in, cfg.sampling.fps_out)
    save_csv(f"{cfg.paths.out_dir}/1000fps_polyphase_nm.csv", t_s=t_p, x_nm=x_p)

    # FFT (span-locked)
    num_out = int(t[-1]*cfg.sampling.fps_out)+1
    x_f = upsample_fft_equal_span(x_nm, num_out)
    t_f = np.linspace(0, t[-1], num_out)
    save_csv(f"{cfg.paths.out_dir}/1000fps_fft_nm.csv", t_s=t_f, x_nm=x_f)

    plot_overlay(t, x_nm, t_c, x_c, x_f,
                 labels=("cubic", "fft"),
                 path=f"{cfg.paths.out_dir}/overlay_cubic_fft.png", tmax=0.6)
    print("Baselines saved to", cfg.paths.out_dir)
