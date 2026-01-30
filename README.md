# Quantum Pulse Sequence Suite

A comprehensive Python toolkit for designing and analyzing composite pulse sequences using Quantum Signal Processing (QSP) for quantum applications, including atomic clocks and quantum algorithms.

## Features

### Core Capabilities
- **Hamiltonian to Unitary Evolution** with and without Noise 
- **Analytic Filter function calculations** for both instantaneous and continuous QSP laser pulses as well as for detuning and amplitude noise hamiltonians
- **Sensitivity and Fidelity Metrics** for filter functions to first order, second order, etc.
- **Noise generation** tool for colored noise types (white, 1/f, 1/f², Lorentzian) in freq and time domain
- **Analytical formula validation** for a veraging time-domain trajectories with noise to compare to perturbative analytical filter function calculations
- **Comprehensive visualization** tools for visualizing pulse sequences, filter functions, Bloch sphere trajectories and comparing pulse sequences

### Pulse Types Supported for Qubit
1. **Instantaneous Pulses**: Ideal SU(2) laser pulses characterized by:
   - Rotation axis: n̂ = (x, y, z) (automatically normalized)
   - Rotation angle: θ (radians)
   - Noise only during free evolution periods
       - characterized by detuning δ and time τ

2. **Continuous Pulses**: Finite-duration pulses characterized by:
   - Rabi frequency: Ω (rad/s)
   - Rotation axis: n̂ = (x, y, z) (automatically normalized)
   - Pulse duration: τ (s)
   - Detuning: δ (rad/s)
   - Noise affects both pulses and free evolution
   
### Noise Hamiltonians
- Detuning noise: \beta(t)*\sigma_z
- Amplitude noise: \beta(t)*H_{QSP}(t)

### Noise Models
- **White noise**: S(ω) = S₀
- **1/f noise**: S(ω) = S₀/ω
- **1/f² noise**: S(ω) = S₀/ω²
- **Lorentzian**: S(ω) = S₀γ/(γ² + ω²)
- Custom power spectral densities

## Installation

### Requirements
```bash
pip install numpy scipy matplotlib --break-system-packages
```

### Quick Start
```bash
cd quantum_pulse_suite
python examples.py  # Run all examples
```

## Usage

### 1. Programmatic Usage

#### Creating Pulse Sequences

```python
from core.pulse_sequence import (PulseSequence, InstantaneousPulse, 
                                 FreeEvolution, pi_pulse, pi_half_pulse)
import numpy as np

# Create a custom sequence
seq = PulseSequence(name="My Sequence")

# Add π/2 pulse around x-axis
seq.add_element(InstantaneousPulse(
    axis=[1, 0, 0],
    angle=np.pi/2
))

# Add free evolution
seq.add_element(FreeEvolution(duration=1e-6))  # 1 μs

# Add π pulse around y-axis
seq.add_element(InstantaneousPulse(
    axis=[0, 1, 0],
    angle=np.pi
))

# Add another free evolution
seq.add_element(FreeEvolution(duration=1e-6))

# Add final pulse around arbitrary axis
seq.add_element(InstantaneousPulse(
    axis=[1, 1, 0],  # Will be normalized
    angle=np.pi/2
))

# Print summary
print(seq.summary())
```

#### Using Pre-defined Sequences

```python
from core.pulse_sequence import ramsey_sequence, spin_echo_sequence, cpmg_sequence

# Ramsey sequence: π/2 - τ - π/2
ramsey = ramsey_sequence(tau=1e-6)

# Spin echo: π/2 - τ - π - τ - π/2
echo = spin_echo_sequence(tau=5e-7)

# CPMG with n refocusing pulses
cpmg = cpmg_sequence(tau=2.5e-7, n=4)
```

#### Computing Filter Functions

```python
from core.filter_functions import FilterFunctionNum, FilterFunctionAn, ColoredNoisePSD

# Create filter function calculator for given sequence, up to perturbative order n numerically
ff = FilterFunctionNum(sequence, order)

# Create filter function calculator for given sequence, up to perturbative order n using analytic QSP formulas
ffAn = FilterFunctionAn(sequence, order)

# Compute filter function at specific frequency
omega = 1e6  # rad/s
F_omega = ff.filter_function(omega)

# Compute for array of frequencies
omega_array = np.logspace(3, 9, 1000)
F_array = ff.filter_function_array(omega_array)

# Define noise PSD
white_noise = ColoredNoisePSD.white_noise(amplitude=1.0)
one_over_f = ColoredNoisePSD.one_over_f(amplitude=1.0)

# Compute noise susceptibility
chi = ff.noise_susceptibility(white_noise)
print(f"Susceptibility: {chi:.6e}")

# Compute fidelity (first order)
from core.filter_functions import compute_fidelity
fidelity = compute_fidelity(sequence, white_noise)
print(f"Fidelity: {fidelity:.6f}")
```

#### Visualization

```python
from core.visualization import (plot_pulse_sequence, plot_filter_function,
                                compare_filter_functions)
import matplotlib.pyplot as plt

# Plot single sequence as train of step or delta functions spaced with correct free evolutions
fig, ax = plt.subplots()
plot_pulse_sequence(sequence, ax=ax)
plt.show()

# Plot filter function
fig, ax = plt.subplots()
plot_filter_function(sequence, ax=ax, log_scale=True)
plt.show()

```

### General Sequences

#### Ramsey Sequence
- Structure: π/2 - τ - π/2
- Purpose: Measure dephasing and frequency
- Sensitivity: Maximum at ω ≈ 1/τ
- White noise: χ ≈ S₀τ²

#### Spin Echo (Hahn Echo)
- Structure: π/2 - τ - π - τ - π/2
- Purpose: Refocus inhomogeneous broadening
- Suppresses: DC and low-frequency noise
- White noise: χ ≈ 0 (first order)

#### CPMG (Carr-Purcell-Meiboom-Gill)
- Structure: π/2 - [τ - π - τ]ⁿ - π/2
- Purpose: Enhanced decoupling with multiple refocusing
- Suppresses: Low-frequency noise better than single echo
- Scaling: χ ∝ 1/n for n refocusing pulses

## File Structure

```
quantum_pulse_suite/
├── __init__.py              # Package initialization
├── core/                    # Core modules
│   ├── __init__.py
│   ├── pulse_sequence.py    # Pulse sequence data structures
│   ├── filter_functions.py  # Filter function calculations, numerical and analytical
│   ├── noise.py             # Noise generation with unit tests to ensure proper time-domain noise sequences
│   └── visualization.py     # Plotting and visualization
├── analysis/                # Analysis tools
│   └── batch_numerical.py         # Batch time-domain trajectories to average for comparison to frequency domain analytics
├── examples.py              # Example scripts
├── gui/                     # Graphical interface? maybe add
│   └── pulse_designer.py    # Main GUI application? maybe add
└── README.md               # This file
```

## Examples

The `examples.py` script demonstrates:
1. **Basic sequences** - Creating and visualizing Ramsey, Echo, CPMG
2. **Filter functions** - Computing and comparing filter functions
3. **Noise susceptibility** - Analyzing impact on sensitivity and fidelity for different noise types
4. **Analytical validation** - Verifying numerical results
5. **Custom sequences** - Building arbitrary pulse sequences

Run all examples:
```bash
python examples.py
```

## Extensions for Qudits (Future)

The current implementation focuses on qubits (spin-1/2), but the architecture is designed to be extended to qudits:

1. Generalize rotation matrices to higher dimensions
2. Implement spin-displacement pulses
3. Extend filter function formalism to higher-dimensional Hilbert spaces

## Troubleshooting

### Import Errors
Make sure you're running from the correct directory or add to Python path:
```python
import sys
sys.path.append('/path/to/quantum_pulse_suite')
```

### Numerical Integration Warnings
For very long sequences or extreme frequency ranges, adjust integration limits:
```python
chi = ff.noise_susceptibility(psd, omega_max=1e10)
```

## Contributing

This suite is designed to be extensible. Key areas for contribution:
- Higher-order perturbation theory
- Additional noise models (i.e. not wide-sense stationary noise)
- Pulse optimization algorithms

## References

### Key Papers on Filter Functions:
1. Biercuk et al., "Optimized dynamical decoupling in a model quantum memory" (2009)
2. Green et al., "Arbitrary quantum control of qubits in the presence of universal noise" (2013)
3. Ball et al., "Software tools for quantum control: Improving quantum computer performance through noise and error suppression" (2021)

### Dynamical Decoupling:
1. Carr & Purcell, "Effects of diffusion on free precession in NMR" (1954)
2. Meiboom & Gill, "Modified spin-echo method for measuring nuclear relaxation times" (1958)
3. Viola & Lloyd, "Dynamical suppression of decoherence in two-state quantum systems" (1998)
