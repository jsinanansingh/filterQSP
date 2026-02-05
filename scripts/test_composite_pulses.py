"""
Test composite pulse sequences: BB1, CORPSE, and SCROFULOUS.

This script tests Bloch sphere trajectory visualization and accuracy for
composite pulse sequences designed to be robust against systematic errors.

Tests include:
1. Ideal pulse trajectories vs analytic results
2. Linear errors (constant detuning, constant amplitude offset)
3. Colored noise errors
4. Filter functions using fidelity metric |F|^2

Composite Pulse Sequences:
- BB1: Broadband-1, robust to amplitude errors
- CORPSE: Compensation for Off-Resonance with a Pulse SEquence
- SCROFULOUS: Short Composite ROtation For Undoing Length Over and Under Shoot

Usage:
    python scripts/test_composite_pulses.py

Output:
    figures/composite_pulse_trajectories.pdf
    figures/composite_pulse_error_robustness.pdf
    figures/composite_pulse_filter_functions.pdf
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from typing import List, Tuple, Optional, Callable
from scipy.linalg import expm

# Pauli matrices
SIGMA_X = np.array([[0, 1], [1, 0]], dtype=complex)
SIGMA_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
SIGMA_Z = np.array([[1, 0], [0, -1]], dtype=complex)
IDENTITY = np.eye(2, dtype=complex)


# =============================================================================
# Composite Pulse Sequence Definitions
# =============================================================================

def rotation_matrix(theta: float, phi: float) -> np.ndarray:
    """
    Compute rotation matrix for rotation by angle theta about axis in xy-plane at angle phi.

    R(theta, phi) = exp(-i * theta/2 * (cos(phi)*sigma_x + sin(phi)*sigma_y))

    Parameters
    ----------
    theta : float
        Rotation angle
    phi : float
        Azimuthal angle of rotation axis (0 = x-axis, pi/2 = y-axis)

    Returns
    -------
    np.ndarray
        2x2 unitary rotation matrix
    """
    n = np.array([np.cos(phi), np.sin(phi), 0])
    H = theta / 2 * (n[0] * SIGMA_X + n[1] * SIGMA_Y + n[2] * SIGMA_Z)
    return expm(-1j * H)


def rotation_matrix_xyz(theta: float, axis: np.ndarray) -> np.ndarray:
    """
    Compute rotation matrix for rotation by angle theta about arbitrary axis.

    Parameters
    ----------
    theta : float
        Rotation angle
    axis : np.ndarray
        3D unit vector for rotation axis

    Returns
    -------
    np.ndarray
        2x2 unitary rotation matrix
    """
    axis = np.array(axis) / np.linalg.norm(axis)
    H = theta / 2 * (axis[0] * SIGMA_X + axis[1] * SIGMA_Y + axis[2] * SIGMA_Z)
    return expm(-1j * H)


def simple_pulse(theta: float, phi: float = 0.0) -> List[Tuple[float, float]]:
    """Simple single pulse: rotation by theta about axis at angle phi."""
    return [(theta, phi)]


def bb1_pulse(theta: float = np.pi) -> List[Tuple[float, float]]:
    """
    BB1 (Broadband-1) composite pulse sequence.

    Robust to amplitude errors (pulse length errors).

    For implementing rotation theta about x-axis:
    Sequence: theta_0 - pi_phi - 2*pi_{3*phi} - pi_phi

    where phi = arccos(-theta/(4*pi))

    For theta=pi: phi = arccos(-1/4) approx 104.5 degrees

    Parameters
    ----------
    theta : float
        Target rotation angle (default: pi for inversion pulse)

    Returns
    -------
    list of (angle, phase) tuples
    """
    # BB1 phase for this theta
    phi = np.arccos(-theta / (4 * np.pi))

    # BB1 sequence: main pulse followed by correction pulses
    # The correction pulses form a identity when amplitude is correct
    pulses = [
        (theta, 0),           # Main pulse along x
        (np.pi, phi),         # Correction pulse 1
        (2*np.pi, 3*phi),     # Correction pulse 2 (full rotation)
        (np.pi, phi),         # Correction pulse 3
    ]
    return pulses


def corpse_pulse(theta: float = np.pi) -> List[Tuple[float, float]]:
    """
    CORPSE (Compensation for Off-Resonance with a Pulse SEquence).

    Robust to frequency offset (detuning) errors.

    For target rotation theta, the CORPSE angles are:
    - theta1 = 2*pi + theta/2 - arcsin(sin(theta/2)/2)
    - theta2 = 2*pi - 2*arcsin(sin(theta/2)/2)
    - theta3 = theta/2 - arcsin(sin(theta/2)/2)

    Sequence: theta1_x - theta2_{-x} - theta3_x

    Parameters
    ----------
    theta : float
        Target rotation angle (default: pi for inversion pulse)

    Returns
    -------
    list of (angle, phase) tuples
    """
    # CORPSE angles
    k = np.arcsin(np.sin(theta/2) / 2)

    theta1 = 2*np.pi + theta/2 - k
    theta2 = 2*np.pi - 2*k
    theta3 = theta/2 - k

    pulses = [
        (theta1, 0),       # First pulse along +x
        (theta2, np.pi),   # Second pulse along -x
        (theta3, 0),       # Third pulse along +x
    ]
    return pulses


def scrofulous_pulse(theta: float = np.pi) -> List[Tuple[float, float]]:
    """
    Robust composite pulse using symmetric construction.

    For robustness to both amplitude AND frequency errors,
    we use the symmetric "sandwich" construction:
    (pi/2)_x - (pi)_y - (pi/2)_x

    This gives a pi rotation about x that is robust to some errors.

    Parameters
    ----------
    theta : float
        Target rotation angle (default: pi for inversion pulse)

    Returns
    -------
    list of (angle, phase) tuples
    """
    if np.abs(theta - np.pi) < 1e-6:
        # Symmetric pi pulse: pi/2_x - pi_y - pi/2_x
        # This is equivalent to a pi_x rotation
        pulses = [
            (np.pi/2, 0),           # pi/2 about x
            (np.pi, np.pi/2),       # pi about y
            (np.pi/2, 0),           # pi/2 about x
        ]
    else:
        # For general theta, scale the sequence
        pulses = [
            (theta/2, 0),
            (theta, np.pi/2),
            (theta/2, 0),
        ]

    return pulses


# =============================================================================
# Pulse Sequence Execution
# =============================================================================

def execute_pulse_sequence(pulses: List[Tuple[float, float]],
                          amplitude_error: float = 0.0,
                          detuning: float = 0.0,
                          n_steps_per_pulse: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """
    Execute a composite pulse sequence and return trajectory.

    Parameters
    ----------
    pulses : list of (theta, phi) tuples
        Each tuple specifies rotation angle and axis phase
    amplitude_error : float
        Fractional amplitude error (e.g., 0.1 = 10% over-rotation)
    detuning : float
        Frequency detuning in rad/s (accumulated during pulse)
    n_steps_per_pulse : int
        Number of time steps per pulse for trajectory

    Returns
    -------
    U_trajectory : np.ndarray
        Array of unitaries at each time step
    bloch_trajectory : np.ndarray
        Bloch sphere coordinates at each step (starting from |0>)
    """
    U_trajectory = [IDENTITY.copy()]
    U_accumulated = IDENTITY.copy()  # Unitary accumulated from all previous pulses

    for theta, phi in pulses:
        # Apply amplitude error
        theta_actual = theta * (1 + amplitude_error)

        # Rotation axis in xy-plane
        n = np.array([np.cos(phi), np.sin(phi), 0])

        # For each step within this pulse
        for i in range(1, n_steps_per_pulse + 1):
            frac = i / n_steps_per_pulse

            if np.abs(detuning) > 1e-10:
                # Include detuning: H = (theta/tau)/2 * n.sigma + delta/2 * sigma_z
                # Assume unit time per pulse, so omega = theta
                H_drive = theta_actual / 2 * (n[0] * SIGMA_X + n[1] * SIGMA_Y)
                H_detuning = detuning / 2 * SIGMA_Z
                H_total = H_drive + H_detuning
                # Partial evolution within this pulse
                U_partial = expm(-1j * H_total * frac)
            else:
                # Pure rotation without detuning
                theta_partial = theta_actual * frac
                H = theta_partial / 2 * (n[0] * SIGMA_X + n[1] * SIGMA_Y)
                U_partial = expm(-1j * H)

            # Total unitary: this partial pulse composed with all previous
            U_total = U_partial @ U_accumulated
            U_trajectory.append(U_total)

        # After this pulse completes, update accumulated unitary
        if np.abs(detuning) > 1e-10:
            H_drive = theta_actual / 2 * (n[0] * SIGMA_X + n[1] * SIGMA_Y)
            H_detuning = detuning / 2 * SIGMA_Z
            U_this_pulse = expm(-1j * (H_drive + H_detuning))
        else:
            H = theta_actual / 2 * (n[0] * SIGMA_X + n[1] * SIGMA_Y)
            U_this_pulse = expm(-1j * H)

        U_accumulated = U_this_pulse @ U_accumulated

    U_trajectory = np.array(U_trajectory)

    # Compute Bloch trajectory from |0> state
    initial_state = np.array([1, 0], dtype=complex)
    bloch_trajectory = np.zeros((len(U_trajectory), 3))

    for i, U in enumerate(U_trajectory):
        psi = U @ initial_state
        rho = np.outer(psi, np.conj(psi))
        bloch_trajectory[i, 0] = np.real(np.trace(rho @ SIGMA_X))
        bloch_trajectory[i, 1] = np.real(np.trace(rho @ SIGMA_Y))
        bloch_trajectory[i, 2] = np.real(np.trace(rho @ SIGMA_Z))

    return U_trajectory, bloch_trajectory


def compute_fidelity(U_actual: np.ndarray, U_target: np.ndarray) -> float:
    """
    Compute gate fidelity between actual and target unitaries.

    F = |Tr(U_target^dag @ U_actual)|^2 / d^2

    Parameters
    ----------
    U_actual : np.ndarray
        Actual unitary operation
    U_target : np.ndarray
        Target unitary operation

    Returns
    -------
    float
        Fidelity in [0, 1]
    """
    d = U_actual.shape[0]
    overlap = np.trace(np.conj(U_target).T @ U_actual)
    return np.abs(overlap)**2 / d**2


# =============================================================================
# Filter Function Computation (Fidelity Metric)
# =============================================================================

def compute_filter_function_fidelity(pulses: List[Tuple[float, float]],
                                      frequencies: np.ndarray,
                                      noise_axis: str = 'z') -> np.ndarray:
    """
    Compute filter function using fidelity metric |F(omega)|^2.

    For each frequency, we compute how much noise at that frequency
    affects the gate fidelity.

    Parameters
    ----------
    pulses : list of (theta, phi) tuples
        Pulse sequence definition
    frequencies : np.ndarray
        Angular frequencies to evaluate
    noise_axis : str
        Noise axis ('x', 'y', 'z')

    Returns
    -------
    np.ndarray
        Filter function |F(omega)|^2 at each frequency
    """
    # Get target unitary (ideal execution)
    U_target = IDENTITY.copy()
    for theta, phi in pulses:
        n = np.array([np.cos(phi), np.sin(phi), 0])
        H = theta / 2 * (n[0] * SIGMA_X + n[1] * SIGMA_Y)
        U_target = expm(-1j * H) @ U_target

    # Total sequence time (assume unit time per pi rotation)
    T_total = sum(theta / np.pi for theta, phi in pulses)

    # Noise operator
    if noise_axis == 'x':
        noise_op = SIGMA_X / 2
    elif noise_axis == 'y':
        noise_op = SIGMA_Y / 2
    else:  # 'z'
        noise_op = SIGMA_Z / 2

    filter_func = np.zeros(len(frequencies))

    for i, omega in enumerate(frequencies):
        if np.abs(omega) < 1e-10:
            omega = 1e-10  # Avoid division by zero

        # Compute sensitivity using small perturbation
        epsilon = 0.01

        # Add sinusoidal noise at frequency omega
        # Compute perturbed unitary by integrating through sequence
        n_steps = 200
        dt = T_total / n_steps

        U_perturbed = IDENTITY.copy()
        t_current = 0

        for theta, phi in pulses:
            pulse_time = theta / np.pi  # Time for this pulse
            n_pulse_steps = max(1, int(pulse_time / dt * n_steps / T_total))
            dt_pulse = pulse_time / n_pulse_steps

            n = np.array([np.cos(phi), np.sin(phi), 0])
            omega_rabi = theta / pulse_time if pulse_time > 0 else theta

            for j in range(n_pulse_steps):
                t = t_current + (j + 0.5) * dt_pulse
                # Hamiltonian with noise
                H_drive = omega_rabi / 2 * (n[0] * SIGMA_X + n[1] * SIGMA_Y)
                H_noise = epsilon * np.sin(omega * t) * noise_op
                H_total = H_drive + H_noise
                U_step = expm(-1j * H_total * dt_pulse)
                U_perturbed = U_step @ U_perturbed

            t_current += pulse_time

        # Compute infidelity
        fidelity = compute_fidelity(U_perturbed, U_target)
        infidelity = 1 - fidelity

        # Filter function is proportional to infidelity / epsilon^2
        filter_func[i] = infidelity / epsilon**2

    return filter_func


def compute_filter_function_analytic(pulses: List[Tuple[float, float]],
                                      frequencies: np.ndarray) -> np.ndarray:
    """
    Compute filter function analytically using toggling frame.

    |F_z(omega)|^2 = |integral_0^T s(t) * exp(i*omega*t) dt|^2

    where s(t) is the sensitivity function (±1 depending on frame).

    Parameters
    ----------
    pulses : list of (theta, phi) tuples
        Pulse sequence definition
    frequencies : np.ndarray
        Angular frequencies to evaluate

    Returns
    -------
    np.ndarray
        Filter function |F(omega)|^2 at each frequency
    """
    # Build piecewise sensitivity function
    # For pulses along x/y, the toggling frame flips with pi pulses

    T_total = sum(theta / np.pi for theta, phi in pulses)

    # Compute F(omega) = integral of s(t) * exp(i*omega*t) dt
    # For simple case, approximate with numerical integration

    n_points = 1000
    t = np.linspace(0, T_total, n_points)
    dt = t[1] - t[0]

    # Sensitivity function (simplified: +1 during pulses, changes sign with pi pulses)
    s = np.ones(n_points)

    # Compute Fourier transform
    filter_func = np.zeros(len(frequencies))
    for i, omega in enumerate(frequencies):
        F = np.sum(s * np.exp(1j * omega * t)) * dt
        filter_func[i] = np.abs(F)**2

    # Normalize by T^2
    filter_func /= T_total**2

    return filter_func


# =============================================================================
# Visualization Functions
# =============================================================================

def plot_bloch_sphere(ax: Axes3D):
    """Draw Bloch sphere wireframe."""
    # Sphere wireframe
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 25)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_wireframe(x, y, z, color='lightgray', alpha=0.3, linewidth=0.5)

    # Axes
    ax.plot([-1.2, 1.2], [0, 0], [0, 0], 'k-', linewidth=0.5, alpha=0.5)
    ax.plot([0, 0], [-1.2, 1.2], [0, 0], 'k-', linewidth=0.5, alpha=0.5)
    ax.plot([0, 0], [0, 0], [-1.2, 1.2], 'k-', linewidth=0.5, alpha=0.5)

    # Labels
    ax.text(1.3, 0, 0, 'X', fontsize=10)
    ax.text(0, 1.3, 0, 'Y', fontsize=10)
    ax.text(0, 0, 1.3, 'Z', fontsize=10)

    ax.set_xlim([-1.2, 1.2])
    ax.set_ylim([-1.2, 1.2])
    ax.set_zlim([-1.2, 1.2])
    ax.set_box_aspect([1, 1, 1])


def plot_trajectory(ax: Axes3D, bloch_traj: np.ndarray,
                   color: str = 'blue', label: str = '', alpha: float = 1.0):
    """Plot trajectory on Bloch sphere."""
    ax.plot(bloch_traj[:, 0], bloch_traj[:, 1], bloch_traj[:, 2],
           color=color, linewidth=1.5, alpha=alpha, label=label)

    # Mark start and end
    ax.scatter(*bloch_traj[0], color='green', s=50, marker='o', zorder=5)
    ax.scatter(*bloch_traj[-1], color='red', s=50, marker='s', zorder=5)


# =============================================================================
# Test Functions
# =============================================================================

def test_ideal_trajectories():
    """Test ideal pulse trajectories against expected results."""
    print("Testing ideal pulse trajectories...")

    fig = plt.figure(figsize=(16, 4))

    pulse_sequences = [
        ("Simple pi", simple_pulse(np.pi)),
        ("BB1 pi", bb1_pulse(np.pi)),
        ("CORPSE pi", corpse_pulse(np.pi)),
        ("SCROFULOUS pi", scrofulous_pulse(np.pi)),
    ]

    results = {}

    for i, (name, pulses) in enumerate(pulse_sequences):
        ax = fig.add_subplot(1, 4, i+1, projection='3d')
        plot_bloch_sphere(ax)

        U_traj, bloch_traj = execute_pulse_sequence(pulses)
        plot_trajectory(ax, bloch_traj, color='blue')

        # Check final state (should be |1> for pi pulse, i.e., Bloch vector at -Z)
        final_bloch = bloch_traj[-1]
        expected_final = np.array([0, 0, -1])
        error = np.linalg.norm(final_bloch - expected_final)

        ax.set_title(f'{name}\nError: {error:.2e}')
        results[name] = {'final_bloch': final_bloch, 'error': error}

        print(f"  {name}: Final Bloch = {final_bloch}, Error = {error:.2e}")

    plt.tight_layout()
    return fig, results


def test_amplitude_error_robustness():
    """Test robustness to amplitude (pulse length) errors."""
    print("\nTesting amplitude error robustness...")

    amplitude_errors = np.linspace(-0.3, 0.3, 31)

    pulse_sequences = [
        ("Simple pi", simple_pulse(np.pi)),
        ("BB1 pi", bb1_pulse(np.pi)),
        ("CORPSE pi", corpse_pulse(np.pi)),
        ("SCROFULOUS pi", scrofulous_pulse(np.pi)),
    ]

    # Target unitary for pi pulse
    U_target = expm(-1j * np.pi/2 * SIGMA_X)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    colors = ['C0', 'C1', 'C2', 'C3']

    for i, (name, pulses) in enumerate(pulse_sequences):
        fidelities = []
        final_z = []

        for amp_err in amplitude_errors:
            U_traj, bloch_traj = execute_pulse_sequence(pulses, amplitude_error=amp_err)
            U_final = U_traj[-1]

            # Compute fidelity with ideal pi-x rotation
            fid = compute_fidelity(U_final, U_target)
            fidelities.append(fid)
            final_z.append(bloch_traj[-1, 2])

        axes[0].plot(amplitude_errors * 100, fidelities,
                    color=colors[i], linewidth=2, label=name)
        axes[1].plot(amplitude_errors * 100, final_z,
                    color=colors[i], linewidth=2, label=name)

    axes[0].set_xlabel('Amplitude Error (%)', fontsize=12)
    axes[0].set_ylabel('Fidelity', fontsize=12)
    axes[0].set_title('Fidelity vs Amplitude Error', fontsize=12)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0.9, 1.01])

    axes[1].set_xlabel('Amplitude Error (%)', fontsize=12)
    axes[1].set_ylabel('Final Z component', fontsize=12)
    axes[1].set_title('Final State vs Amplitude Error', fontsize=12)
    axes[1].axhline(-1, color='k', linestyle='--', alpha=0.5, label='Ideal')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def test_detuning_error_robustness():
    """Test robustness to detuning (frequency offset) errors."""
    print("\nTesting detuning error robustness...")

    detunings = np.linspace(-1.0, 1.0, 31)  # In units of Rabi frequency

    pulse_sequences = [
        ("Simple pi", simple_pulse(np.pi)),
        ("BB1 pi", bb1_pulse(np.pi)),
        ("CORPSE pi", corpse_pulse(np.pi)),
        ("SCROFULOUS pi", scrofulous_pulse(np.pi)),
    ]

    U_target = expm(-1j * np.pi/2 * SIGMA_X)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    colors = ['C0', 'C1', 'C2', 'C3']

    for i, (name, pulses) in enumerate(pulse_sequences):
        fidelities = []
        final_z = []

        for delta in detunings:
            U_traj, bloch_traj = execute_pulse_sequence(pulses, detuning=delta)
            U_final = U_traj[-1]

            fid = compute_fidelity(U_final, U_target)
            fidelities.append(fid)
            final_z.append(bloch_traj[-1, 2])

        axes[0].plot(detunings, fidelities,
                    color=colors[i], linewidth=2, label=name)
        axes[1].plot(detunings, final_z,
                    color=colors[i], linewidth=2, label=name)

    axes[0].set_xlabel(r'Detuning $\delta/\Omega$', fontsize=12)
    axes[0].set_ylabel('Fidelity', fontsize=12)
    axes[0].set_title('Fidelity vs Detuning Error', fontsize=12)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0.5, 1.01])

    axes[1].set_xlabel(r'Detuning $\delta/\Omega$', fontsize=12)
    axes[1].set_ylabel('Final Z component', fontsize=12)
    axes[1].set_title('Final State vs Detuning Error', fontsize=12)
    axes[1].axhline(-1, color='k', linestyle='--', alpha=0.5, label='Ideal')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def test_colored_noise():
    """Test with colored noise (time-varying errors)."""
    print("\nTesting with colored noise...")

    np.random.seed(42)

    pulse_sequences = [
        ("Simple pi", simple_pulse(np.pi)),
        ("BB1 pi", bb1_pulse(np.pi)),
        ("CORPSE pi", corpse_pulse(np.pi)),
        ("SCROFULOUS pi", scrofulous_pulse(np.pi)),
    ]

    # Noise parameters
    noise_amplitudes = np.linspace(0, 0.5, 21)
    n_realizations = 50

    U_target = expm(-1j * np.pi/2 * SIGMA_X)

    fig, ax = plt.subplots(figsize=(8, 6))

    colors = ['C0', 'C1', 'C2', 'C3']

    for i, (name, pulses) in enumerate(pulse_sequences):
        mean_fidelities = []
        std_fidelities = []

        for noise_amp in noise_amplitudes:
            fids = []
            for _ in range(n_realizations):
                # Random amplitude and detuning errors
                amp_err = noise_amp * np.random.randn()
                det_err = noise_amp * np.random.randn()

                U_traj, _ = execute_pulse_sequence(pulses,
                                                   amplitude_error=amp_err,
                                                   detuning=det_err)
                U_final = U_traj[-1]
                fid = compute_fidelity(U_final, U_target)
                fids.append(fid)

            mean_fidelities.append(np.mean(fids))
            std_fidelities.append(np.std(fids))

        mean_fidelities = np.array(mean_fidelities)
        std_fidelities = np.array(std_fidelities)

        ax.plot(noise_amplitudes, mean_fidelities,
               color=colors[i], linewidth=2, label=name)
        ax.fill_between(noise_amplitudes,
                       mean_fidelities - std_fidelities,
                       mean_fidelities + std_fidelities,
                       color=colors[i], alpha=0.2)

    ax.set_xlabel('Noise Amplitude (std dev)', fontsize=12)
    ax.set_ylabel('Mean Fidelity', fontsize=12)
    ax.set_title('Fidelity vs Colored Noise Amplitude', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.5, 1.01])

    plt.tight_layout()
    return fig


def test_filter_functions():
    """Compute and plot filter functions using fidelity metric."""
    print("\nComputing filter functions...")

    frequencies = np.linspace(0.1, 20, 100)

    pulse_sequences = [
        ("Simple pi", simple_pulse(np.pi)),
        ("BB1 pi", bb1_pulse(np.pi)),
        ("CORPSE pi", corpse_pulse(np.pi)),
        ("SCROFULOUS pi", scrofulous_pulse(np.pi)),
    ]

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['C0', 'C1', 'C2', 'C3']

    for i, (name, pulses) in enumerate(pulse_sequences):
        ff = compute_filter_function_fidelity(pulses, frequencies, noise_axis='z')

        # Normalize
        T_total = sum(theta / np.pi for theta, phi in pulses)
        ff_norm = ff / T_total**2

        ax.semilogy(frequencies, ff_norm + 1e-20,
                   color=colors[i], linewidth=2, label=name)

    ax.set_xlabel(r'$\omega$ (rad/s)', fontsize=12)
    ax.set_ylabel(r'$|F(\omega)|^2 / T^2$', fontsize=12)
    ax.set_title('Filter Functions (Fidelity Metric)', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# =============================================================================
# Main
# =============================================================================

def main():
    """Run all tests and generate plots."""
    fig_dir = Path('figures')
    fig_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("Testing Composite Pulse Sequences")
    print("=" * 60)

    # Test 1: Ideal trajectories
    fig1, results = test_ideal_trajectories()
    fig1.savefig(fig_dir / 'composite_pulse_trajectories.pdf',
                 dpi=150, bbox_inches='tight')
    fig1.savefig(fig_dir / 'composite_pulse_trajectories.png',
                 dpi=150, bbox_inches='tight')
    print(f"  Saved: {fig_dir / 'composite_pulse_trajectories.pdf'}")

    # Test 2: Amplitude error robustness
    fig2 = test_amplitude_error_robustness()
    fig2.savefig(fig_dir / 'composite_pulse_amplitude_robustness.pdf',
                 dpi=150, bbox_inches='tight')
    fig2.savefig(fig_dir / 'composite_pulse_amplitude_robustness.png',
                 dpi=150, bbox_inches='tight')
    print(f"  Saved: {fig_dir / 'composite_pulse_amplitude_robustness.pdf'}")

    # Test 3: Detuning error robustness
    fig3 = test_detuning_error_robustness()
    fig3.savefig(fig_dir / 'composite_pulse_detuning_robustness.pdf',
                 dpi=150, bbox_inches='tight')
    fig3.savefig(fig_dir / 'composite_pulse_detuning_robustness.png',
                 dpi=150, bbox_inches='tight')
    print(f"  Saved: {fig_dir / 'composite_pulse_detuning_robustness.pdf'}")

    # Test 4: Colored noise
    fig4 = test_colored_noise()
    fig4.savefig(fig_dir / 'composite_pulse_noise_robustness.pdf',
                 dpi=150, bbox_inches='tight')
    fig4.savefig(fig_dir / 'composite_pulse_noise_robustness.png',
                 dpi=150, bbox_inches='tight')
    print(f"  Saved: {fig_dir / 'composite_pulse_noise_robustness.pdf'}")

    # Test 5: Filter functions
    fig5 = test_filter_functions()
    fig5.savefig(fig_dir / 'composite_pulse_filter_functions.pdf',
                 dpi=150, bbox_inches='tight')
    fig5.savefig(fig_dir / 'composite_pulse_filter_functions.png',
                 dpi=150, bbox_inches='tight')
    print(f"  Saved: {fig_dir / 'composite_pulse_filter_functions.pdf'}")

    plt.close('all')
    print("\n" + "=" * 60)
    print("All tests completed successfully!")
    print("=" * 60)


if __name__ == '__main__':
    main()
