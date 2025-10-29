import pandas as pd, numpy as np, json, os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.obt1000.config import cfg
from src.obt1000.io_prep import load_mat_as_nm
from src.obt1000.eval_metrics import rmse, psd_distance, corner_frequency_match

def load_series(path):
    df = pd.read_csv(path)
    return df["t_s"].values, df["x_nm"].values

if __name__=="__main__":
    t300, x300, key = load_mat_as_nm(cfg.paths.mat_path, cfg.paths.mat_key,
                                     cfg.sampling.px_to_nm, cfg.sampling.fps_in)

    results = {}
    series = {
        "cubic":      f"{cfg.paths.out_dir}/1000fps_cubic_nm.csv",
        "polyphase":  f"{cfg.paths.out_dir}/1000fps_polyphase_nm.csv",
        "fft":        f"{cfg.paths.out_dir}/1000fps_fft_nm.csv",
        "ou_interp":  f"{cfg.paths.out_dir}/1000fps_ou_interp_nm.csv",
        "ou_full":    f"{cfg.paths.out_dir}/1000fps_ou_fullgrid_nm.csv",
        "pgcdi":      f"{cfg.paths.out_dir}/1000fps_pgcdi_nm.csv",
    }
    # mask-based CV: keep every 3rd point as observed → compare recon at missing points
    idx_obs = np.arange(0, len(x300), 3)
    idx_mis = np.setdiff1d(np.arange(len(x300)), idx_obs)

    for name, path in series.items():
        if not os.path.exists(path): continue
        t1k, x1k = load_series(path)
        # downsample back: pick 1000fps indices corresponding to 300fps timestamps
        idx1k = (t300*cfg.sampling.fps_out).round().astype(int)
        idx1k = np.clip(idx1k, 0, len(x1k)-1)  # Ensure indices are within bounds
        x_back = x1k[idx1k]
        results[name] = {
            "rmse_nm": rmse(x_back[idx_mis], x300[idx_mis]),
            "psd_dist": psd_distance(x_back, x300),
            "fc_1k_Hz": corner_frequency_match(x_back, fs=cfg.sampling.fps_in)  # compare on 300 grid proxy
        }

    out_path = f"{cfg.paths.out_dir}/eval_summary.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print("Saved evaluation to", out_path)
    print(pd.DataFrame(results).T)
