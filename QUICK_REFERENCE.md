# Quick Reference Card

## Installation (One-time)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
python verify_setup.py
```

## Basic Workflow

```bash
# 1. Place data
cp your_data.mat data/

# 2. Run everything
make all

# Or run step-by-step:
make baselines   # Cubic, polyphase, FFT → ~5s
make ou          # OU-Kalman → ~10s
make train       # Train diffusion model → ~10min (CPU) / ~2min (GPU)
make sample      # Generate with PG-CDI → ~2min
make eval        # Evaluate all methods → ~5s
```

## Output Files

```
out/
├── 1000fps_cubic_nm.csv        # Cubic spline
├── 1000fps_polyphase_nm.csv    # Polyphase resampling
├── 1000fps_fft_nm.csv          # FFT resampling
├── 1000fps_ou_interp_nm.csv    # OU continuous interpolation
├── 1000fps_ou_fullgrid_nm.csv  # OU full state-space (best classical)
├── 1000fps_pgcdi_nm.csv        # PG-CDI (novel method)
├── overlay_cubic_fft.png       # Visual comparison
├── ou_fullgrid_uncertainty.png # Confidence bands
├── eval_summary.json           # Quantitative metrics
└── pgcdi.pt                    # Trained model
```

## Configuration

Edit `src/obt1000/config.py`:

```python
# Essential settings
mat_path = "data/your_file.mat"    # Input file
fps_in = 300.0                     # Input frame rate
fps_out = 1000.0                   # Target frame rate
px_to_nm = 35.0                    # Calibration factor

# Training (PG-CDI)
epochs = 20                        # Training epochs
batch_size = 4                     # Batch size
lr = 2e-4                          # Learning rate
```

## Methods Summary

| Method | Type | Speed | Uncertainty | Physics-aware |
|--------|------|-------|-------------|---------------|
| Cubic | Classical | ⚡⚡⚡ | ❌ | ❌ |
| Polyphase | Classical | ⚡⚡⚡ | ❌ | ❌ |
| FFT | Classical | ⚡⚡⚡ | ❌ | ❌ |
| OU Interp | Physics | ⚡⚡ | ❌ | ✅ |
| OU Fullgrid | Physics | ⚡⚡ | ✅ | ✅ |
| PG-CDI | ML+Physics | ⚡ | ❌ | ✅ |

## Commands

```bash
# Verify setup
python verify_setup.py

# Individual scripts
python scripts/run_baselines.py
python scripts/run_ou.py
python scripts/run_train_pgcdi.py
python scripts/run_sample_pgcdi.py
python scripts/run_eval_all.py

# Clean outputs
rm -rf out/*

# Re-train model
make train

# View results
cat out/eval_summary.json
open out/overlay_cubic_fft.png
open out/ou_fullgrid_uncertainty.png
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Import errors | `pip install -e .` |
| No .mat file | Check `data/` directory |
| Wrong variable | Set `mat_key` in config |
| Out of memory | Reduce `batch_size` or `T` |
| Slow training | Use GPU or reduce `epochs` |
| Bad results | Check `px_to_nm` calibration |

## Quick Tests

```bash
# Test imports
python -c "from src.obt1000.config import cfg; print(cfg)"

# Test data loading (requires .mat file)
python -c "from src.obt1000.io_prep import load_mat_as_nm; print(load_mat_as_nm())"

# Check GPU availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

## Typical Performance (15k points @ 300 fps → 50k @ 1000 fps)

| Step | CPU Time | GPU Time | Output |
|------|----------|----------|--------|
| Baselines | 5s | N/A | 3 CSV files |
| OU-Kalman | 10s | N/A | 2 CSV files |
| Training | 10-20min | 1-3min | 1 model file |
| Sampling | 2-5min | 30-60s | 1 CSV file |
| Evaluation | 5s | N/A | 1 JSON file |

## Key Equations

**OU Process:**
```
dx/dt = -(x/τ) + √(2D) * ξ(t)
AR(1): x[n+1] = a*x[n] + w[n], a = exp(-Δt/τ)
```

**Kalman Filter:**
```
Predict: x̂[k|k-1] = a*x̂[k-1]
Update:  x̂[k] = x̂[k|k-1] + K*(y[k] - x̂[k|k-1])
```

**Diffusion:**
```
Forward:  x_t = √(ᾱ_t)*x_0 + √(1-ᾱ_t)*ε
Reverse:  x_{t-1} = (x_t - β_t*ε_θ)/√(1-β_t)
```

## File Structure

```
.
├── README.md                    # Main readme
├── USAGE_GUIDE.md              # Detailed usage
├── CONFIGURATION_EXAMPLES.md    # Config examples
├── QUICK_REFERENCE.md          # This file
├── requirements.txt            # Dependencies
├── pyproject.toml              # Package config
├── Makefile                    # Build commands
├── verify_setup.py             # Verification script
├── data/                       # Input data (your .mat)
├── out/                        # Output results
├── scripts/                    # Runnable scripts
│   ├── run_baselines.py
│   ├── run_ou.py
│   ├── run_train_pgcdi.py
│   ├── run_sample_pgcdi.py
│   └── run_eval_all.py
└── src/
    ├── obt1000/               # Classical methods
    │   ├── config.py          # Configuration
    │   ├── io_prep.py         # Data I/O
    │   ├── baselines.py       # Interpolation
    │   ├── ou_kalman.py       # OU filtering
    │   ├── eval_metrics.py    # Evaluation
    │   ├── plotting.py        # Visualization
    │   └── utils.py           # Utilities
    └── pgcdi/                 # Novel ML method
        ├── data.py            # Synthetic data
        ├── model.py           # UNet architecture
        ├── physics_losses.py  # Physics constraints
        ├── scheduler.py       # Noise schedule
        ├── train.py           # Training loop
        └── sample.py          # Inference
```

## Citation

```bibtex
@software{optical_bead_1000fps,
  title = {Physics-Guided Upsampling of Optical Tweezer Trajectories},
  year = {2025},
  url = {https://github.com/yourusername/optical-bead-trajectory-1000fps}
}
```

## Support

1. ✅ Run `python verify_setup.py`
2. 📖 Read `USAGE_GUIDE.md`
3. 🔧 Check `CONFIGURATION_EXAMPLES.md`
4. 🐛 Open GitHub issue
