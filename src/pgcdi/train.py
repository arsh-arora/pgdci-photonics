import torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from ..obt1000.config import cfg
from .data import MaskedImputeDataset
from .model import UNet1D
from .scheduler import NoiseSched
from .physics_losses import loss_measurement, loss_psd, loss_energy

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = MaskedImputeDataset(n=512, T=cfg.train.T, dt=1/cfg.sampling.fps_in)
    dl = DataLoader(ds, batch_size=cfg.train.batch_size, shuffle=True, drop_last=True)
    model = UNet1D().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr)
    sched = NoiseSched(T=1000, device=device)

    for ep in range(cfg.train.epochs):
        pbar = tqdm(dl, desc=f"epoch {ep+1}/{cfg.train.epochs}")
        for x, y_obs, mask in pbar:
            x = x.to(device).unsqueeze(1)         # [B,1,T]
            y_obs = y_obs.to(device).unsqueeze(1) # [B,1,T]
            mask = mask.to(device).unsqueeze(1)   # [B,1,T]

            # OU prior proxy: lowpass of y_obs (or later: feed a real OU smoother output)
            ou_prior = torch.nn.functional.avg_pool1d(y_obs, 9, stride=1, padding=4)

            t = torch.randint(0, sched.T, (x.size(0),), device=device)
            eps = torch.randn_like(x)
            ab = sched.ab[t].view(-1,1,1)
            x_t = torch.sqrt(ab)*x + torch.sqrt(1-ab)*eps

            cond = torch.cat([y_obs, mask, ou_prior], dim=1)
            eps_pred = model(cond, x_t)

            # denoise estimate
            x0_hat = (x_t - torch.sqrt(1-ab)*eps_pred)/torch.sqrt(ab)

            l_eps = F.mse_loss(eps_pred, eps)
            l_meas = loss_measurement(x0_hat, y_obs, mask)
            l_psd = loss_psd(x0_hat, ou_prior)
            l_eng = loss_energy(x0_hat)

            loss = l_eps + 2.0*l_meas + 0.2*l_psd + 1e-4*l_eng
            opt.zero_grad(); loss.backward(); opt.step()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    torch.save(model.state_dict(), cfg.train.save_path)
    print("Saved:", cfg.train.save_path)
