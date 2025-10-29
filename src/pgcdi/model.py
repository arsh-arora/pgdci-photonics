import torch, torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(c, c, 7, padding=3), nn.GELU(),
            nn.Conv1d(c, c, 7, padding=3), nn.GELU()
        )
    def forward(self, x): return self.net(x)

class UNet1D(nn.Module):
    def __init__(self, ch=64, cin=4):
        super().__init__()
        # cond has 3 channels [y_obs, mask, ou_prior]; x_t is appended → total 4
        self.inp = nn.Conv1d(cin, ch, 7, padding=3)
        self.b1 = ConvBlock(ch); self.d1 = nn.Conv1d(ch, ch, 4, stride=2, padding=1)
        self.b2 = ConvBlock(ch); self.d2 = nn.Conv1d(ch, ch, 4, stride=2, padding=1)
        self.mid= ConvBlock(ch)
        self.u2 = nn.ConvTranspose1d(ch, ch, 4, stride=2, padding=1)
        self.u1 = nn.ConvTranspose1d(ch, ch, 4, stride=2, padding=1)
        self.out = nn.Conv1d(ch, 1, 7, padding=3)
    def forward(self, cond, x_t):
        h = torch.cat([cond, x_t], dim=1)
        h = self.inp(h)
        h1 = self.b1(h);  d1 = self.d1(h1)
        h2 = self.b2(d1); d2 = self.d2(h2)
        m  = self.mid(d2)
        u2 = self.u2(m)
        u1 = self.u1(u2 + h2)
        out= self.out(u1 + h1)
        return out
