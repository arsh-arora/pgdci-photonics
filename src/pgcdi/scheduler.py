import torch

class NoiseSched:
    def __init__(self, T=1000, beta1=1e-4, beta2=2e-2, device="cpu"):
        self.T=T
        self.beta=torch.linspace(beta1,beta2,T, device=device)
        self.alph=1.0-self.beta
        self.ab  = torch.cumprod(self.alph, dim=0)

    def add_noise(self, x0, t, eps):
        abt = self.ab[t].view(-1,1,1)
        return torch.sqrt(abt)*x0 + torch.sqrt(1-abt)*eps
