import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.obt1000.config import cfg
from src.obt1000.io_prep import load_mat_as_nm, save_csv
from src.obt1000.ou_kalman import estimate_ar1_params, kalman_rts, ou_interpolate_to_1000fps, ou_fullgrid_1000fps
from src.obt1000.plotting import plot_uncertainty
from src.obt1000.utils import ensure_dir
import numpy as np

if __name__=="__main__":
    ensure_dir(cfg.paths.out_dir)
    t, x_nm, key = load_mat_as_nm(cfg.paths.mat_path, cfg.paths.mat_key,
                                  cfg.sampling.px_to_nm, cfg.sampling.fps_in)
    dt = 1.0/cfg.sampling.fps_in
    a,q,r,tau,_ = estimate_ar1_params(x_nm, dt)
    x_sm, P_sm = kalman_rts(x_nm, a,q,r)

    # Continuous OU interpolation from smoothed 300 fps → 1000 fps
    t1k_cont, x1k_cont = ou_interpolate_to_1000fps(t, x_sm, tau, cfg.sampling.fps_out)
    save_csv(f"{cfg.paths.out_dir}/1000fps_ou_interp_nm.csv", t_s=t1k_cont, x_nm=x1k_cont)

    # Full 1000 fps state-space smoothing (masked obs)
    t1k_grid, x1k_grid, P1k_grid = ou_fullgrid_1000fps(x_nm, cfg.sampling.fps_in, cfg.sampling.fps_out)
    save_csv(f"{cfg.paths.out_dir}/1000fps_ou_fullgrid_nm.csv", t_s=t1k_grid, x_nm=x1k_grid, var_nm2=P1k_grid)

    plot_uncertainty(t1k_grid, x1k_grid, P1k_grid, f"{cfg.paths.out_dir}/ou_fullgrid_uncertainty.png", tmax=0.6)
    print(f"Estimated tau={tau:.4g}s  a={a:.6f}  r≈{r:.1f} nm^2")
    print("OU outputs saved to", cfg.paths.out_dir)
