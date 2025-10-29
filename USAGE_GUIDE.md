# Usage Guide: Optical Bead Trajectory Upsampling

## Overview

This repository provides a complete pipeline for upsampling optical tweezer bead trajectories from 300 fps to 1000 fps using both classical and novel machine learning methods.

## Quick Start

### 1. Installation

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -U pip
pip install -e .

# Verify installation
python verify_setup.py
```

### 2. Prepare Your Data

Place your `.mat` file in the `data/` directory:
```bash
cp /path/to/your/300fps_15k.mat data/
```

### 3. Configure Parameters

Edit `src/obt1000/config.py` to match your data:

```python
@dataclass
class Paths:
    mat_path: str = "data/300fps_15k.mat"    # Your .mat file
    mat_key: str | None = None               # Auto-detect if None
    out_dir: str = "out"

@dataclass
class Sampling:
    fps_in: float = 300.0      # Input frame rate
    fps_out: float = 1000.0    # Target frame rate
    px_to_nm: float = 35.0     # Pixel to nanometer conversion
```

## Running the Pipeline

### Method 1: Using Makefile (Recommended)

```bash
# Run all steps at once
make all

# Or run individual steps:
make baselines   # Classical methods (cubic, polyphase, FFT)
make ou          # OU-Kalman smoother
make train       # Train PG-CDI model (novel method)
make sample      # Generate 1000 fps with PG-CDI
make eval        # Evaluate all methods
```

### Method 2: Running Scripts Directly

```bash
# Step 1: Classical baselines
python scripts/run_baselines.py
# Outputs: out/1000fps_cubic_nm.csv
#          out/1000fps_polyphase_nm.csv
#          out/1000fps_fft_nm.csv
#          out/overlay_cubic_fft.png

# Step 2: OU-Kalman smoother
python scripts/run_ou.py
# Outputs: out/1000fps_ou_interp_nm.csv
#          out/1000fps_ou_fullgrid_nm.csv
#          out/ou_fullgrid_uncertainty.png

# Step 3: Train PG-CDI (Physics-Guided Conditional Diffusion Imputer)
python scripts/run_train_pgcdi.py
# Outputs: out/pgcdi.pt (trained model)

# Step 4: Sample with PG-CDI
python scripts/run_sample_pgcdi.py
# Outputs: out/1000fps_pgcdi_nm.csv

# Step 5: Evaluate all methods
python scripts/run_eval_all.py
# Outputs: out/eval_summary.json
```

## Configuration Options

### Basic Configuration

```python
# src/obt1000/config.py

# Input/Output paths
cfg.paths.mat_path = "data/your_file.mat"
cfg.paths.out_dir = "out"

# Sampling parameters
cfg.sampling.fps_in = 300.0     # Input frame rate
cfg.sampling.fps_out = 1000.0   # Output frame rate
cfg.sampling.px_to_nm = 35.0    # Calibration factor

# OU process constraints
cfg.ou.min_a = 1e-6             # Minimum AR(1) coefficient
cfg.ou.max_a = 0.999999         # Maximum AR(1) coefficient

# Training parameters for PG-CDI
cfg.train.epochs = 20           # Number of training epochs
cfg.train.batch_size = 4        # Batch size
cfg.train.T = 15000             # Sequence length for synthetic data
cfg.train.lr = 2e-4             # Learning rate
cfg.train.save_path = "out/pgcdi.pt"  # Model save path
```

### Advanced: Custom Configurations

You can modify the config dynamically in scripts:

```python
from src.obt1000.config import cfg

# Change paths
cfg.paths.mat_path = "data/custom_experiment.mat"
cfg.paths.mat_key = "trajectory_x"  # Specific variable name

# Adjust sampling
cfg.sampling.fps_out = 2000.0  # Go to 2000 fps instead
cfg.sampling.px_to_nm = 40.5   # Different calibration

# Training adjustments
cfg.train.epochs = 50          # More epochs
cfg.train.batch_size = 8       # Larger batches
```

## Output Files

All outputs are saved to `out/` (configurable):

### CSV Files
- `1000fps_cubic_nm.csv` - Cubic spline interpolation
- `1000fps_polyphase_nm.csv` - Polyphase resampling
- `1000fps_fft_nm.csv` - FFT-based resampling
- `1000fps_ou_interp_nm.csv` - OU continuous interpolation
- `1000fps_ou_fullgrid_nm.csv` - OU full state-space (with variance)
- `1000fps_pgcdi_nm.csv` - Physics-Guided CDI output

Each CSV contains:
- `t_s`: Time in seconds
- `x_nm`: Position in nanometers
- `var_nm2`: Variance (OU fullgrid only)

### Plots
- `overlay_cubic_fft.png` - Comparison of methods (first 0.6s)
- `ou_fullgrid_uncertainty.png` - OU mean ± 2σ confidence bands

### Evaluation
- `eval_summary.json` - RMSE, PSD distance, corner frequency

### Model
- `pgcdi.pt` - Trained PG-CDI model weights

## Methods Explained

### Classical Methods

1. **Cubic Spline**: Smooth C² interpolation
2. **Polyphase**: Rational resampling with Kaiser window
3. **FFT**: Fourier-domain resampling (assumes periodic/zero-padded)

### OU-Kalman Methods

4. **OU Interpolation**: Continuous-time OU process interpolation
5. **OU Fullgrid**: Full 1000 fps state-space Kalman filter/smoother
   - Handles missing observations at intermediate times
   - Provides uncertainty quantification (variance)

### Novel Method

6. **PG-CDI (Physics-Guided Conditional Diffusion Imputer)**
   - Conditional 1D UNet diffusion model
   - Trained on synthetic OU/GLE processes
   - Guided by:
     * OU prior from Kalman smoother
     * Measurement loss (fidelity to observed points)
     * PSD matching (spectral consistency)
     * Energy regularization

## Evaluation Metrics

The `eval_summary.json` contains:

```json
{
  "method_name": {
    "rmse_nm": 12.34,        // Root mean square error (nm)
    "psd_dist": 0.567,       // Log-PSD distance
    "fc_1k_Hz": 1234.5       // Corner frequency (Hz)
  }
}
```

- **RMSE**: Computed on masked points (cross-validation)
- **PSD Distance**: Mean absolute difference of log-PSDs
- **Corner Frequency**: -3dB point relative to DC

## Troubleshooting

### Import Errors
```bash
python verify_setup.py
pip install -e .
```

### No .mat file found
- Ensure file is in `data/` directory
- Check `cfg.paths.mat_path` in config.py

### Wrong variable loaded
- Set `cfg.paths.mat_key = "your_variable_name"`

### GPU/CPU Selection
PG-CDI automatically uses CUDA if available:
```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```

### Memory Issues
Reduce batch size or sequence length:
```python
cfg.train.batch_size = 2
cfg.train.T = 10000
```

## Customization Examples

### 1. Different Frame Rates

```python
# 500 fps → 2000 fps
cfg.sampling.fps_in = 500.0
cfg.sampling.fps_out = 2000.0
```

### 2. Multiple Data Files

```python
files = ["exp1.mat", "exp2.mat", "exp3.mat"]
for f in files:
    cfg.paths.mat_path = f"data/{f}"
    cfg.paths.out_dir = f"out/{f[:-4]}"
    # Run pipeline...
```

### 3. Custom Training Data

Modify `src/pgcdi/data.py`:
```python
def __getitem__(self, i):
    tau = 10**np.random.uniform(-3.0, -1.0)      # Wider range
    sigma = 10**np.random.uniform(1.0, 2.5)
    # ... add GLE or other physics
```

### 4. Different Diffusion Steps

```python
# In sample.py
xhat = ddpm_sample(cond, steps=500, device=device)  # Faster
```

## Performance Tips

1. **Parallel Training**: Use larger batch size on GPU
2. **Faster Sampling**: Reduce DDPM steps (500 instead of 1000)
3. **Memory**: Process long sequences in chunks
4. **I/O**: Use HDF5 for very large datasets

## Citation

If you use this code, please cite:
```
@software{optical_bead_1000fps,
  title = {Physics-Guided Upsampling of Optical Tweezer Trajectories},
  year = {2025},
  url = {https://github.com/yourusername/optical-bead-trajectory-1000fps}
}
```

## Support

For issues or questions:
1. Check `verify_setup.py` output
2. Review this guide
3. Examine example outputs in `out/`
4. Open an issue on GitHub
