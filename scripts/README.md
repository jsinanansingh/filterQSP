# Plotting Scripts

Scripts for generating paper-quality figures of filter functions.

## Scripts

### `plot_instantaneous_vs_continuous.py`

Compares filter functions for analogous pulse sequences:
- **Continuous**: Rotation about y-axis with Rabi frequency Omega
- **Instantaneous**: R_x(pi) -> R_z(theta) free evolution -> R_x(-pi)

The conjugation relation R_x(pi) . R_z(theta) . R_x(-pi) = R_z(-theta) is verified.

**Key parameters** (edit at top of script):
- `THETA`: Rotation angle
- `DELTA`: Detuning
- `OMEGA_CONTINUOUS`: Rabi frequency
- `FREQ_MIN`, `FREQ_MAX`: Frequency range
- `USE_LOG_SCALE`: Log-log or linear plot

### `plot_cpmg_comparison.py`

Compares CPMG sequences with instantaneous vs continuous pi-pulses:
- Same color for each N value
- Solid lines: instantaneous pulses
- Dashed lines: continuous pulses

**Key parameters** (edit at top of script):
- `TAU_TOTAL`: Total free evolution time
- `N_PULSES_LIST`: List of N values to plot (default: [1, 4, 8])
- `OMEGA_PI`: Rabi frequency for pi-pulses
- `FREQ_MIN`, `FREQ_MAX`: Frequency range

## Usage

```bash
# From repo root
conda activate qsp
python scripts/plot_instantaneous_vs_continuous.py
python scripts/plot_cpmg_comparison.py
```

## Output

Figures are saved to `figures/` directory:
- `instantaneous_vs_continuous_filter.pdf`
- `cpmg_instant_vs_continuous.pdf`

## Customization

Edit the `# Plot Configuration` section at the top of each script to adjust:
- Sequence parameters (angles, times, frequencies)
- Plot styling (colors, line widths, font sizes)
- Output format (PNG, PDF, etc.)
