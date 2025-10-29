import torch, numpy as np
from ..obt1000.config import cfg
from ..obt1000.io_prep import load_mat_as_nm, save_csv
from ..obt1000.ou_kalman import estimate_ar1_params, kalman_rts
from ..obt1000.baselines import upsample_cubic
from .model import UNet1D
from .scheduler import NoiseSched

@torch.no_grad()
def ddpm_sample(cond, steps=1000, device="cpu"):
    sched = NoiseSched(T=steps, device=device)
    B,_,T = cond.shape
    x = torch.randn(B,1,T, device=device)
    for t in reversed(range(steps)):
        # simplified sampling; sufficient for imputation
        beta_t = sched.beta[t]
        ab_t = sched.ab[t]
        model = ddpm_sample.model
        eps = model(cond, x)
        x = (x - (beta_t/torch.sqrt(1-ab_t))*eps) / torch.sqrt(1-beta_t)
        if t > 0:
            x = x + torch.sqrt(beta_t)*torch.randn_like(x)
    return x

def run_sample():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t, x_nm, key = load_mat_as_nm(cfg.paths.mat_path, cfg.paths.mat_key,
                                  cfg.sampling.px_to_nm, cfg.sampling.fps_in)
    # OU prior from 300 fps
    a,q,r,tau,_ = estimate_ar1_params(x_nm, 1.0/cfg.sampling.fps_in)
    x_sm,_ = kalman_rts(x_nm, a,q,r)

    # move all to 1000 fps via cubic for alignment
    t1k, y1k = upsample_cubic(t, x_nm, cfg.sampling.fps_out)
    _, ou1k   = upsample_cubic(t, x_sm, cfg.sampling.fps_out)

    mask1k = np.zeros_like(y1k); mask1k[(t*cfg.sampling.fps_out).astype(int)] = 1.0

    y = torch.from_numpy(y1k).float().unsqueeze(0).unsqueeze(0).to(device)
    m = torch.from_numpy(mask1k).float().unsqueeze(0).unsqueeze(0).to(device)
    o = torch.from_numpy(ou1k).float().unsqueeze(0).unsqueeze(0).to(device)
    cond = torch.cat([y,m,o], dim=1)

    model = UNet1D().to(device)
    model.load_state_dict(torch.load(cfg.train.save_path, map_location=device))
    ddpm_sample.model = model

    xhat = ddpm_sample(cond, steps=1000, device=device).cpu().squeeze().numpy()
    save_csv(f"{cfg.paths.out_dir}/1000fps_pgcdi_nm.csv", t_s=t1k, x_nm=xhat)
    print("Saved PG-CDI:", f"{cfg.paths.out_dir}/1000fps_pgcdi_nm.csv")
