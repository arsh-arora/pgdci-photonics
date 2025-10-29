# Configuration Examples

Complete examples of different use cases and configurations.

## Example 1: Basic 300→1000 fps (Default)

```python
# src/obt1000/config.py
from dataclasses import dataclass

@dataclass
class Paths:
    mat_path: str = "data/300fps_15k.mat"
    mat_key: str | None = None
    out_dir: str = "out"

@dataclass
class Sampling:
    fps_in: float = 300.0
    fps_out: float = 1000.0
    px_to_nm: float = 35.0
```

**Usage:**
```bash
make all
```

## Example 2: High Frame Rate (500→2000 fps)

```python
@dataclass
class Sampling:
    fps_in: float = 500.0
    fps_out: float = 2000.0
    px_to_nm: float = 35.0

@dataclass
class Train:
    epochs: int = 30           # More epochs for harder task
    batch_size: int = 2        # Smaller batches for longer sequences
    T: int = 20000             # Longer synthetic sequences
    lr: float = 1e-4           # Lower learning rate
```

## Example 3: Low Noise Data (High SNR)

```python
@dataclass
class OU:
    min_a: float = 1e-8        # Tighter constraints
    max_a: float = 0.9999

@dataclass
class Train:
    epochs: int = 15
    batch_size: int = 8
    T: int = 15000
    lr: float = 3e-4           # Faster learning for cleaner data
```

## Example 4: Noisy Data (Low SNR)

```python
@dataclass
class OU:
    min_a: float = 1e-4        # Allow more flexibility
    max_a: float = 0.99

@dataclass
class Train:
    epochs: int = 40           # More training
    batch_size: int = 4
    T: int = 15000
    lr: float = 1e-4           # Careful learning
```

## Example 5: Multiple Experiments Batch Processing

Create a script:

```python
# batch_process.py
from src.obt1000.config import cfg
from scripts.run_baselines import main as run_baselines
from scripts.run_ou import main as run_ou

experiments = [
    {"file": "exp1_300fps.mat", "px_nm": 35.0, "key": "x_position"},
    {"file": "exp2_300fps.mat", "px_nm": 37.5, "key": "trajectory"},
    {"file": "exp3_300fps.mat", "px_nm": 33.2, "key": None},
]

for exp in experiments:
    print(f"\nProcessing {exp['file']}...")

    # Update config
    cfg.paths.mat_path = f"data/{exp['file']}"
    cfg.paths.mat_key = exp['key']
    cfg.paths.out_dir = f"out/{exp['file'][:-4]}"
    cfg.sampling.px_to_nm = exp['px_nm']

    # Run pipeline
    run_baselines()
    run_ou()
    # ... etc
```

## Example 6: Custom Trap Parameters

If you know your trap parameters:

```python
# Add to config.py
@dataclass
class TrapPhysics:
    kappa: float = 1e-6        # Trap stiffness (N/m)
    gamma: float = 6e-8        # Friction coefficient (kg/s)
    T_kelvin: float = 293.0    # Temperature (K)

    @property
    def tau(self):
        """Relaxation time (s)"""
        return self.gamma / self.kappa

    @property
    def D(self):
        """Diffusion coefficient (m²/s)"""
        from scipy.constants import k as kB
        return kB * self.T_kelvin / self.gamma

@dataclass
class Config:
    paths: Paths = Paths()
    sampling: Sampling = Sampling()
    ou: OU = OU()
    train: Train = Train()
    trap: TrapPhysics = TrapPhysics()
```

## Example 7: GPU/CPU Control

```python
# In train.py or sample.py
import os

# Force CPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""
device = "cpu"

# Or specify GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use GPU 0
device = "cuda"

# Or multi-GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
device = "cuda:0"  # Primary GPU
```

## Example 8: Quick Test (Small Data)

```python
@dataclass
class Train:
    epochs: int = 5            # Fast iteration
    batch_size: int = 2
    T: int = 5000              # Shorter sequences
    lr: float = 2e-4
    save_path: str = "out/pgcdi_test.pt"

# In data.py, reduce dataset size:
ds = MaskedImputeDataset(n=128, T=5000, dt=1/300)  # Only 128 samples
```

## Example 9: Production Quality

```python
@dataclass
class Train:
    epochs: int = 100          # Thorough training
    batch_size: int = 8
    T: int = 20000             # Long sequences
    lr: float = 1e-4           # Conservative LR
    save_path: str = "out/pgcdi_production.pt"

# In train.py, add:
from torch.optim.lr_scheduler import ReduceLROnPlateau
scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5)
```

## Example 10: Different Output Directories per Method

```python
import os
from src.obt1000.config import cfg

# Baselines
cfg.paths.out_dir = "out/baselines"
os.makedirs(cfg.paths.out_dir, exist_ok=True)
# run_baselines()

# OU methods
cfg.paths.out_dir = "out/ou_kalman"
os.makedirs(cfg.paths.out_dir, exist_ok=True)
# run_ou()

# PG-CDI
cfg.paths.out_dir = "out/pgcdi"
cfg.train.save_path = "out/pgcdi/model.pt"
os.makedirs(cfg.paths.out_dir, exist_ok=True)
# run_train_pgcdi()
```

## Example 11: Different Diffusion Configurations

### Fast sampling (fewer steps)
```python
# In sample.py
xhat = ddpm_sample(cond, steps=200, device=device)
```

### Custom noise schedule
```python
# In scheduler.py
class NoiseSched:
    def __init__(self, T=1000, beta1=1e-5, beta2=1e-2, device="cpu"):
        # Cosine schedule
        s = 0.008
        t = torch.linspace(0, T, T, device=device)
        f = torch.cos(((t/T + s)/(1+s)) * np.pi/2)**2
        self.ab = f / f[0]
        self.alph = torch.cat([self.ab[:1], self.ab[1:]/self.ab[:-1]])
        self.beta = 1.0 - self.alph
```

## Example 12: Multiple Variables (X, Y, Z)

```python
# Process each axis separately
axes = ['x', 'y', 'z']
for axis in axes:
    cfg.paths.mat_key = f"position_{axis}"
    cfg.paths.out_dir = f"out/axis_{axis}"

    # Run pipeline for this axis
    # ... (baselines, ou, train, sample, eval)
```

## Example 13: Cross-Validation

```python
# In eval script, add k-fold CV
from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=True, random_state=42)
results = []

for fold, (train_idx, test_idx) in enumerate(kf.split(x300)):
    print(f"Fold {fold+1}/5")

    # Use only train_idx for OU estimation
    # Use test_idx for RMSE computation
    # ... store results

# Average across folds
```

## Example 14: Export for Analysis Software

```python
# After generating 1000 fps data
import scipy.io as sio
import pandas as pd

# Load CSV
df = pd.read_csv("out/1000fps_pgcdi_nm.csv")

# Export to .mat (MATLAB)
sio.savemat("out/1000fps_pgcdi.mat", {
    "t": df["t_s"].values,
    "x": df["x_nm"].values,
    "fps": 1000,
    "method": "PG-CDI"
})

# Export to HDF5 (larger datasets)
df.to_hdf("out/1000fps_pgcdi.h5", key="trajectory", mode="w")
```

## Example 15: Real-Time Monitoring During Training

```python
# In train.py, add logging
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(log_dir="runs/pgcdi")

# In training loop:
writer.add_scalar("Loss/total", loss.item(), global_step)
writer.add_scalar("Loss/eps", l_eps.item(), global_step)
writer.add_scalar("Loss/measurement", l_meas.item(), global_step)
# ...

# View with: tensorboard --logdir=runs
```

## Example 16: Ensemble Methods

```python
# Train multiple models with different seeds
seeds = [42, 123, 456, 789, 1000]
models = []

for seed in seeds:
    torch.manual_seed(seed)
    np.random.seed(seed)

    cfg.train.save_path = f"out/pgcdi_seed_{seed}.pt"
    # train()
    models.append(cfg.train.save_path)

# Average predictions
predictions = []
for model_path in models:
    model.load_state_dict(torch.load(model_path))
    pred = ddpm_sample(cond, steps=1000, device=device)
    predictions.append(pred.cpu().numpy())

ensemble_mean = np.mean(predictions, axis=0)
ensemble_std = np.std(predictions, axis=0)
```

## Running Configuration Examples

1. Copy desired configuration to `src/obt1000/config.py`
2. Verify: `python verify_setup.py`
3. Run: `make all` or individual scripts

For dynamic configs, create a custom script in `scripts/` that imports and modifies `cfg` before running the pipeline.
