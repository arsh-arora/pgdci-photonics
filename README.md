# Optical Bead Trajectory Upsampling (300 → 1000 fps)

**Tracks**
- Classical: cubic spline, polyphase/FFT resample, OU–Kalman smoother (continuous interpolation + 1000 fps state-space).
- Novel: PG-CDI (Physics-Guided Conditional Diffusion Imputer) with OU prior + physics losses.

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate   # (Windows: .venv\Scripts\activate)
pip install -U pip && pip install -e .
# Place your .mat in data/ and set paths in src/obt1000/config.py if needed.

make baselines       # cubic + polyphase/FFT → out/
make ou              # OU–Kalman outputs (+bands) → out/
make train           # train PG-CDI on synthetic physics data
make sample          # sample PG-CDI on your trajectory → 1000 fps csv
make eval            # RMSE/PSD comparisons + plots
```

Outputs appear under `out/` as CSV/PNGs and a summary report.
# pgdci-photonics
