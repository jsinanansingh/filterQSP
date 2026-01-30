"""
Visualization utilities for quantum pulse sequences.

This module provides plotting functions for pulse sequences, filter functions,
and simulation results. (Placeholder for future implementation)
"""

from typing import Optional, Tuple, List
import numpy as np

# Optional matplotlib import
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def _check_matplotlib():
    """Check if matplotlib is available."""
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for visualization. "
                         "Install with: pip install matplotlib")


def plot_filter_function(
    frequencies: np.ndarray,
    filter_func: 'FilterFunction',
    ax: Optional['plt.Axes'] = None,
    components: bool = False,
    loglog: bool = True,
    label: Optional[str] = None
    ) -> 'plt.Axes':
    """
    Plot the filter function magnitude.

    Parameters
    ----------
    frequencies : np.ndarray
        Angular frequencies at which to evaluate
    filter_func : FilterFunction
        Filter function calculator instance
    ax : plt.Axes, optional
        Matplotlib axes to plot on. If None, creates new figure.
    components : bool
        If True, plot individual Fx, Fy, Fz components
    loglog : bool
        If True, use log-log scale
    label : str, optional
        Label for the plot

    Returns
    -------
    plt.Axes
        The matplotlib axes object
    """
    _check_matplotlib()

    if ax is None:
        fig, ax = plt.subplots()

    Fx, Fy, Fz = filter_func.filter_function(frequencies)
    total = np.abs(Fx)**2 + np.abs(Fy)**2 + np.abs(Fz)**2

    plot_func = ax.loglog if loglog else ax.plot

    if components:
        plot_func(frequencies, np.abs(Fx)**2, label=f'{label} Fx' if label else 'Fx', alpha=0.7)
        plot_func(frequencies, np.abs(Fy)**2, label=f'{label} Fy' if label else 'Fy', alpha=0.7)
        plot_func(frequencies, np.abs(Fz)**2, label=f'{label} Fz' if label else 'Fz', alpha=0.7)

    plot_func(frequencies, total, label=label or '|F(ω)|²', linewidth=2)

    ax.set_xlabel('Angular frequency ω')
    ax.set_ylabel('|F(ω)|²')
    ax.legend()
    ax.grid(True, alpha=0.3)

    return ax


def plot_unitary_trajectory(
    times: np.ndarray,
    U_trajectory: np.ndarray,
    ax: Optional['plt.Axes'] = None,
    components: str = 'populations'
) -> 'plt.Axes':
    """
    Plot the unitary evolution trajectory.

    Parameters
    ----------
    times : np.ndarray
        Time points
    U_trajectory : np.ndarray
        Array of shape (n_times, 2, 2) containing unitaries
    ax : plt.Axes, optional
        Matplotlib axes to plot on
    components : str
        What to plot: 'populations' for |U_ij|², 'real', 'imag', or 'both'

    Returns
    -------
    plt.Axes
        The matplotlib axes object
    """
    _check_matplotlib()

    if ax is None:
        fig, ax = plt.subplots()

    if components == 'populations':
        ax.plot(times, np.abs(U_trajectory[:, 0, 0])**2, label='|U₀₀|²')
        ax.plot(times, np.abs(U_trajectory[:, 0, 1])**2, label='|U₀₁|²')
        ax.plot(times, np.abs(U_trajectory[:, 1, 0])**2, label='|U₁₀|²')
        ax.plot(times, np.abs(U_trajectory[:, 1, 1])**2, label='|U₁₁|²')
        ax.set_ylabel('Population')
    elif components == 'real':
        ax.plot(times, np.real(U_trajectory[:, 0, 0]), label='Re(U₀₀)')
        ax.plot(times, np.real(U_trajectory[:, 0, 1]), label='Re(U₀₁)')
        ax.set_ylabel('Real part')
    elif components == 'imag':
        ax.plot(times, np.imag(U_trajectory[:, 0, 0]), label='Im(U₀₀)')
        ax.plot(times, np.imag(U_trajectory[:, 0, 1]), label='Im(U₀₁)')
        ax.set_ylabel('Imaginary part')

    ax.set_xlabel('Time')
    ax.legend()
    ax.grid(True, alpha=0.3)

    return ax


def plot_bloch_trajectory(
    U_trajectory: np.ndarray,
    ax: Optional['plt.Axes'] = None,
    initial_state: np.ndarray = None
) -> 'plt.Axes':
    """
    Plot the Bloch sphere trajectory.

    Parameters
    ----------
    U_trajectory : np.ndarray
        Array of shape (n_times, 2, 2) containing unitaries
    ax : plt.Axes, optional
        Matplotlib 3D axes to plot on
    initial_state : np.ndarray, optional
        Initial state vector, defaults to |0⟩

    Returns
    -------
    plt.Axes
        The matplotlib 3D axes object
    """
    _check_matplotlib()
    from mpl_toolkits.mplot3d import Axes3D

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    if initial_state is None:
        initial_state = np.array([1, 0], dtype=complex)

    # Compute Bloch vector trajectory
    n_times = len(U_trajectory)
    bloch_vectors = np.zeros((n_times, 3))

    for i, U in enumerate(U_trajectory):
        psi = U @ initial_state
        # Bloch vector: (⟨σx⟩, ⟨σy⟩, ⟨σz⟩)
        from .pulse_sequence import SIGMA_X, SIGMA_Y, SIGMA_Z
        bloch_vectors[i, 0] = np.real(np.conj(psi) @ SIGMA_X @ psi)
        bloch_vectors[i, 1] = np.real(np.conj(psi) @ SIGMA_Y @ psi)
        bloch_vectors[i, 2] = np.real(np.conj(psi) @ SIGMA_Z @ psi)

    # Plot trajectory
    ax.plot(bloch_vectors[:, 0], bloch_vectors[:, 1], bloch_vectors[:, 2],
            'b-', linewidth=1.5)
    ax.scatter(*bloch_vectors[0], color='green', s=50, label='Start')
    ax.scatter(*bloch_vectors[-1], color='red', s=50, label='End')

    # Draw Bloch sphere wireframe
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 25)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_wireframe(x, y, z, color='gray', alpha=0.1)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    ax.set_zlim([-1.1, 1.1])
    ax.legend()

    return ax


def compare_filter_functions(
    frequencies: np.ndarray,
    filter_funcs: List[Tuple['FilterFunction', str]],
    ax: Optional['plt.Axes'] = None,
    loglog: bool = True,
    reference_lines: Optional[List[Tuple[float, str]]] = None
) -> 'plt.Axes':
    """
    Compare multiple filter functions on the same plot.

    Parameters
    ----------
    frequencies : np.ndarray
        Angular frequencies
    filter_funcs : list
        List of (FilterFunction, label) tuples
    ax : plt.Axes, optional
        Matplotlib axes
    loglog : bool
        Use log-log scale
    reference_lines : list, optional
        List of (exponent, label) for 1/ω^n reference lines

    Returns
    -------
    plt.Axes
        The matplotlib axes object
    """
    _check_matplotlib()

    if ax is None:
        fig, ax = plt.subplots()

    plot_func = ax.loglog if loglog else ax.plot

    for ff, label in filter_funcs:
        susceptibility = ff.noise_susceptibility(frequencies)
        plot_func(frequencies, susceptibility, label=label, linewidth=2)

    if reference_lines:
        for exp, label in reference_lines:
            ref = frequencies[0]**(-exp) * frequencies**(-exp)
            plot_func(frequencies, ref, '--', alpha=0.5, label=label)

    ax.set_xlabel('Angular frequency ω')
    ax.set_ylabel('|F(ω)|²')
    ax.legend()
    ax.grid(True, alpha=0.3)

    return ax
