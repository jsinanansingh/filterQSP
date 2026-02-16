# Code Plan: Qudit Irrep-Resolved Filter Functions

## What Already Exists (and can be reused)

| Component | File | Status |
|---|---|---|
| GGM matrices for any `d` | `core/operators.py` | Done |
| SU(d) structure constants | `core/operators.py` | Done |
| Matrix FFT filter function (any d) | `core/fft_filter_function.py` | Done |
| `noise_susceptibility_from_matrix` | `core/fft_filter_function.py` | Done |
| `bloch_components_from_matrix` | `core/fft_filter_function.py` | Done (needs GGM basis passed for d>2) |
| Multi-level pulse sequence | `core/multilevel.py` | Done (2-level subspace only) |
| Three-level clock filter Fe, Ff, Phi | `core/three_level_filter.py` | Done |

## What Needs to Be Built

### Phase 1: Spin-Displacement Pulse Infrastructure

**New file: `core/spin_displacement.py`**

```python
# Key functions:
def spin_j_operators(d):
    """Return Jx, Jy, Jz for spin j=(d-1)/2."""
    # Jz = diag(-j, -j+1, ..., j)
    # J+, J- with sqrt(j(j+1) - m(m±1)) matrix elements
    # Jx = (J+ + J-)/2, Jy = (J+ - J-)/(2i)

def spin_displacement_hamiltonian(d, phases, omega=1.0, detunings=None):
    """Build H_rot(phases) from Eq (eq:H_rot) of draft.

    When phases=zeros and detunings=zeros, returns omega * Jx.
    Rabi rates: Omega_k = omega * sqrt((k+1)(d-k-1))
    """

def spin_displacement_pulse(d, phases, theta, omega=1.0, detunings=None):
    """Compute D(phases, theta) = expm(-i * theta * H_rot / omega)."""

def snap_gate(d, phases):
    """Diagonal phase gate S(phases) = diag(exp(i*phi_0), ..., exp(i*phi_{d-1}))."""
```

**New file: `core/qudit_pulse_sequence.py`**

A new `QuditPulseSequence` class that holds a sequence of spin-displacement
pulses and is compatible with `fft_filter_function`:

```python
class SpinDisplacementElement(PulseElement):
    """Single spin-displacement pulse D(phases, theta)."""
    # .duration() = tau (finite) or 0 (instantaneous limit)
    # .hamiltonian() = H_rot(phases) from Eq (eq:H_rot)
    # .unitary() = expm(-i H_rot tau)

class QuditPulseSequence:
    """Sequence of spin-displacement pulses for d-level system.

    Compatible with fft_filter_function() via .elements and .total_duration().
    """

    @classmethod
    def from_mqs_prog(cls, d, params_dict):
        """Import pulse parameters from MQS Prog PyTorch output.

        params_dict contains:
        - 'phases': list of d vectors of d-1 phases each
        - 'thetas': list of d rotation angles
        - 'omega': Rabi frequency
        - 'tau': pulse duration (None = instantaneous)
        """

    @classmethod
    def grover_diffusion(cls, d, phases, thetas, omega=1.0, tau=None):
        """Create d-pulse Grover diffusion operator sequence."""
```

### Phase 2: Irrep Decomposition Tools

**New file: `core/irrep_decomposition.py`**

```python
def angular_momentum_irrep_projectors(d):
    """Compute projectors P_L for L=0,1,...,2j in Liouville space.

    Uses Clebsch-Gordan coefficients for j x j -> L.
    Returns dict {L: P_L} where P_L is (d^2 x d^2) projector
    acting on vectorized operators.
    """

def irrep_resolved_filter_function(F_matrix, d):
    """Decompose matrix filter function by angular momentum irreps.

    Parameters
    ----------
    F_matrix : (n_freq, d, d) array from fft_filter_function
    d : int, system dimension

    Returns
    -------
    dict {L: F_L(omega)} where F_L is (n_freq,) array
    """

def transition_resolved_filter_function(F_matrix, d):
    """Decompose by transition (j,k) pairs.

    Returns dict {(j,k): F_jk(omega)} for j<k
    plus {('diag',): F_diag(omega)}
    """

def ggm_component_filter_functions(F_matrix, d):
    """Return |R_a(omega)|^2 for each GGM component a=1..d^2-1.

    Wraps bloch_components_from_matrix with GGM basis.
    """
```

### Phase 3: Tests (FFT as Ground Truth)

**New file: `tests/test_qudit_filter_function.py`**

```python
class TestQuditFilterFunction(unittest.TestCase):

    def test_qudit_ramsey_d3(self):
        """Qutrit Ramsey: Jx rotation, free evolution, Jx rotation.
        FFT susceptibility vs known formula."""

    def test_qudit_ramsey_d4(self):
        """d=4 Ramsey with spin-displacement pulses."""

    def test_ggm_components_sum(self):
        """Sum of |R_a|^2 = noise_susceptibility for d=3,4,5."""

    def test_irrep_components_sum(self):
        """Sum over L of F_L(w) = F_fid(w) for d=3,4,5."""

    def test_qubit_limit(self):
        """d=2 spin-displacement reduces to standard Pauli rotation.
        Filter function matches existing qubit tests."""

    def test_three_level_clock_embedding(self):
        """d=3 QSP on {|g>,|e>} subspace: new qudit code
        matches existing three_level_filter.py results."""

    def test_grover_d3(self):
        """3-level Grover diffusion operator: FFT filter function."""

    def test_grover_d4(self):
        """4-level Grover diffusion operator: FFT filter function."""

    def test_jz_noise_only_L1_first_order(self):
        """For Jz noise at first order, only L=1 sector contributes."""
```

### Phase 4: MQS Prog Interface

**New file: `core/mqs_interface.py`**

```python
def load_mqs_prog_sequence(filepath, d):
    """Load pulse sequence from MQS Prog output file.

    Expected format (to be confirmed with MQS Prog):
    - JSON or pickle with keys 'phases', 'thetas', 'omega'
    - phases: (n_pulses, d-1) array
    - thetas: (n_pulses,) array
    """

def compare_sequences(sequences, noise_psd, d, omega_range):
    """Compare filter functions of multiple sequences.

    Returns DataFrame with columns:
    - sequence_id
    - total_infidelity (for given PSD)
    - peak_frequency
    - low_freq_suppression
    """
```

### Phase 5: Analysis Scripts

**New file: `scripts/plot_qudit_filter_functions.py`**

```python
# 1. Plot GGM component filter functions for d=3 Grover
# 2. Plot irrep-resolved F_L(w) for d=3,4,5 Grover
# 3. Compare different Grover sequences under various PSDs
# 4. Heatmap: F_{jk}(w) by transition for a Grover sequence
```

## Implementation Order

1. **`core/spin_displacement.py`** - Foundational: Jx, Jy, Jz operators and
   spin-displacement pulse construction. Test: `D(0, 2*pi)` = identity for
   all d. Test: `D(0, pi)` reproduces known spin-j rotation matrices.

2. **`core/qudit_pulse_sequence.py`** - Build sequences from spin-displacement
   pulses. Test: compatibility with `fft_filter_function()`.

3. **`core/irrep_decomposition.py`** - GGM and irrep projection. Test:
   projectors are orthogonal, sum to identity, and dimensions match.

4. **`tests/test_qudit_filter_function.py`** - Comprehensive tests using FFT
   as ground truth.

5. **`core/mqs_interface.py`** - Import from MQS Prog (once format is known).

6. **`scripts/plot_qudit_filter_functions.py`** - Analysis and visualization.

## Key Design Decisions

- **Reuse FFT infrastructure**: The existing `fft_filter_function()` already
  works for any d. We just need to feed it proper d-dimensional pulse
  sequences and Jz noise Hamiltonians.

- **No analytic formulas initially**: Unlike the qubit case where closed-form
  cj/sj integrals exist, the d-level spin-displacement case has no simple
  analytic Fourier integrals. We rely on FFT for the filter function and
  only decompose the result into GGM/irrep components.

- **Irrep decomposition is post-processing**: Compute the full matrix F(w)
  via FFT, then project onto irreps. This avoids needing per-irrep analytic
  formulas.

- **Instantaneous limit first**: Start with instantaneous spin-displacement
  pulses (like the existing InstantaneousPulse but in d dimensions), then
  extend to finite-duration pulses if needed.
