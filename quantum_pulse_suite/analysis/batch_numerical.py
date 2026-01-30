"""
Numerical evolution and trajectory analysis for quantum pulse sequences.

This module provides classes for numerical simulation of pulse sequences
with noise, including variance computation using the Kubo formula.
"""

from typing import List, Tuple, Callable, Optional
import numpy as np
from scipy.linalg import expm
from scipy.integrate import simpson

from ..core.pulse_sequence import SIGMA_X, SIGMA_Y, SIGMA_Z
from ..core.noise import generate_time_series, noise_interpolation


def bloch_vector_from_operator(op: np.ndarray) -> np.ndarray:
    """
    Extract Bloch vector representation from a 2x2 operator.

    Parameters
    ----------
    op : np.ndarray
        2x2 complex matrix

    Returns
    -------
    np.ndarray
        3D Bloch vector [x, y, z]
    """
    x = np.real(op[0, 1])
    y = np.imag(op[0, 1])
    z = np.real(op[0, 0])
    return np.array([x, y, z])


class NumericalEvolution:
    """
    Class for numerical piecewise-constant Hamiltonian evolution.

    This class creates time-dependent Hamiltonians from pulse parameters
    and computes the unitary evolution numerically.

    Parameters
    ----------
    dt : float
        Base time step for discretization
    """

    def __init__(self, dt: float = 0.001):
        self.dt = dt

    def create_H_list(
        self,
        params: np.ndarray,
        noise_type: int = -1,
        psd_amplitude: float = 1.0
    ) -> Tuple[List[Tuple[np.ndarray, float]], List[float]]:
        """
        Create a list of Hamiltonians for piecewise evolution.

        Parameters
        ----------
        params : np.ndarray
            Array of shape (n_pulses, 6) with columns:
            [omega, n_x, n_y, n_z, delta, tau]
            - omega: Rabi frequency
            - n_x, n_y, n_z: rotation axis components
            - delta: detuning
            - tau: duration
        noise_type : int
            Type of noise (-1 = none, 0 = white, 1 = 1/f, 2 = 1/f^2)
        psd_amplitude : float
            Amplitude scaling for noise PSD

        Returns
        -------
        tuple
            (H_list, times) where H_list is [(H_k, dt_k), ...] and
            times is the cumulative time array
        """
        params = np.atleast_2d(params)
        total_time = np.sum(params[:, 5])
        total_num_bins = int(total_time / self.dt) + len(params)

        # Generate noise if requested
        if noise_type >= 0:
            def psd(w):
                return psd_amplitude / (np.abs(w)**noise_type + 1e-14)
            betas = generate_time_series(psd, total_num_bins, self.dt)[0]
        else:
            betas = np.zeros(total_num_bins)

        H_list = []
        times = [0.0]
        curr_time = 0.0

        for row in params:
            omega = row[0]
            axis = [row[1], row[2], row[3]]
            delta = row[4]
            tau = row[5]

            # Base Hamiltonian for this segment
            H_base = (omega * (axis[0] * SIGMA_X + axis[1] * SIGMA_Y +
                              axis[2] * SIGMA_Z) / 2 +
                     delta * SIGMA_Z / 2)

            # Discretize this segment
            num_bins = int(tau / self.dt)
            final_dt = tau - num_bins * self.dt

            # Get noise values at discretized times
            segment_times = np.arange(num_bins + 1) * self.dt + curr_time
            if final_dt > 1e-12:
                segment_times = np.append(segment_times, segment_times[-1] + final_dt)

            if noise_type >= 0:
                noise_values = [noise_interpolation(betas, self.dt, t)
                               for t in segment_times]
            else:
                noise_values = [0.0] * len(segment_times)

            # Add Hamiltonian segments
            for k in range(num_bins):
                H = H_base + noise_values[k] * SIGMA_Z / 2
                H_list.append((H, self.dt))
                times.append(times[-1] + self.dt)

            if final_dt > 1e-12:
                H = H_base + noise_values[-1] * SIGMA_Z / 2
                H_list.append((H, final_dt))
                times.append(times[-1] + final_dt)

            curr_time += tau

        return H_list, times[:-1]  # Remove last duplicate time

    def unitary_evo_piecewise(
        self,
        H_list: List[Tuple[np.ndarray, float]]
    ) -> np.ndarray:
        """
        Compute unitary evolution from piecewise constant Hamiltonians.

        Parameters
        ----------
        H_list : list
            List of (H_k, dt_k) tuples

        Returns
        -------
        np.ndarray
            Array of shape (n_steps+1, 2, 2) containing the unitary
            at each time step
        """
        U = np.eye(2, dtype=complex)
        U_history = [U.copy()]

        for H, dt in H_list:
            U_step = expm(-1j * H * dt)
            U = U_step @ U
            U_history.append(U.copy())

        return np.array(U_history)

    def evolve_sequence(
        self,
        params: np.ndarray,
        noise_type: int = -1,
        psd_amplitude: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evolve a pulse sequence and return the unitary trajectory.

        Parameters
        ----------
        params : np.ndarray
            Pulse parameters array
        noise_type : int
            Noise type
        psd_amplitude : float
            Noise amplitude

        Returns
        -------
        tuple
            (U_trajectory, times) arrays
        """
        H_list, times = self.create_H_list(params, noise_type, psd_amplitude)
        U_trajectory = self.unitary_evo_piecewise(H_list)
        return U_trajectory, np.array(times + [times[-1] + H_list[-1][1]])


class TrajectoryAnalyzer:
    """
    Class for analyzing quantum trajectories with noise.

    Provides methods for computing variance of observables using
    the Kubo formula approach.

    Parameters
    ----------
    numerical_evo : NumericalEvolution
        Numerical evolution engine to use
    """

    def __init__(self, numerical_evo: Optional[NumericalEvolution] = None):
        if numerical_evo is None:
            numerical_evo = NumericalEvolution()
        self.numerical_evo = numerical_evo

    def var_delta_op(
        self,
        measurement_op: np.ndarray,
        U_trajectory: np.ndarray,
        noise_time_series: np.ndarray,
        times: np.ndarray,
        initial_state: np.ndarray = None
    ) -> float:
        """
        Calculate variance of measurement operator using Kubo formula.

        Computes the variance of ⟨B⟩ due to noise perturbations using
        linear response theory.

        Parameters
        ----------
        measurement_op : np.ndarray
            2x2 Hermitian measurement operator
        U_trajectory : np.ndarray
            Array of unitaries at each time point
        noise_time_series : np.ndarray
            Noise values at each time point
        times : np.ndarray
            Time points
        initial_state : np.ndarray, optional
            Initial state vector, defaults to |0⟩

        Returns
        -------
        float
            Variance of the measurement expectation value
        """
        if initial_state is None:
            initial_state = np.array([1, 0], dtype=complex)

        num_time_points = len(U_trajectory)

        # Transform measurement operator to interaction picture at final time
        U_final = U_trajectory[-1]
        B_I = U_final.conj().T @ measurement_op @ U_final

        # Compute commutator trajectory
        comm_x = np.zeros(num_time_points)
        comm_y = np.zeros(num_time_points)
        comm_z = np.zeros(num_time_points)

        for t in range(num_time_points):
            # Interaction picture noise Hamiltonian
            U_t = U_trajectory[t]
            H_tilde = noise_time_series[t] * U_t.conj().T @ (SIGMA_Z / 2) @ U_t

            # Commutator [B_I(T), H_tilde(t)]
            commutator = B_I @ H_tilde - H_tilde @ B_I
            bloch = bloch_vector_from_operator(commutator)

            comm_x[t] = bloch[0]
            comm_y[t] = bloch[1]
            comm_z[t] = np.real(bloch[2])

        # Integrate commutators using Simpson's rule
        int_x = simpson(comm_x, x=times)
        int_y = simpson(comm_y, x=times)
        int_z = simpson(comm_z, x=times)

        # Construct integrated operator
        integral = 2 * (int_x * SIGMA_X + int_y * SIGMA_Y + int_z * SIGMA_Z)
        int_squared = integral @ integral

        # Compute variance
        variance = np.abs(initial_state.conj() @ int_squared @ initial_state)
        return variance

    def avg_var_trajectories(
        self,
        params: np.ndarray,
        measurement_op: np.ndarray,
        n_trajectories: int = 10,
        noise_type: int = -1,
        psd_amplitude: float = 1.0,
        initial_state: np.ndarray = None
    ) -> float:
        """
        Compute average variance over multiple noise trajectories.

        Parameters
        ----------
        params : np.ndarray
            Pulse sequence parameters
        measurement_op : np.ndarray
            Measurement operator
        n_trajectories : int
            Number of noise realizations to average over
        noise_type : int
            Noise type
        psd_amplitude : float
            Noise PSD amplitude
        initial_state : np.ndarray, optional
            Initial state

        Returns
        -------
        float
            Mean variance over all trajectories
        """
        if initial_state is None:
            initial_state = np.array([1, 0], dtype=complex)

        # Get unitary trajectory (without noise for base evolution)
        H_list, times = self.numerical_evo.create_H_list(
            params, noise_type=-1, psd_amplitude=0
        )
        U_trajectory = self.numerical_evo.unitary_evo_piecewise(H_list)
        times = np.array(times + [times[-1] + self.numerical_evo.dt])

        # Ensure times and U_trajectory have compatible lengths
        min_len = min(len(times), len(U_trajectory))
        times = times[:min_len]
        U_trajectory = U_trajectory[:min_len]

        total_num_bins = len(times)
        variances = []

        for _ in range(n_trajectories):
            # Generate noise time series
            if noise_type >= 0:
                def psd(w):
                    return psd_amplitude / (np.abs(w)**noise_type + 1e-14)
                noise_series = generate_time_series(
                    psd, total_num_bins, self.numerical_evo.dt
                )[0]
            else:
                noise_series = np.zeros(total_num_bins)

            var = self.var_delta_op(
                measurement_op, U_trajectory, noise_series, times, initial_state
            )
            variances.append(var)

        return np.mean(variances)

    def compute_filter_function_numerically(
        self,
        params: np.ndarray,
        frequencies: np.ndarray,
        n_trajectories: int = 100,
        psd_amplitude: float = 1.0
    ) -> np.ndarray:
        """
        Estimate filter function numerically via noise averaging.

        Uses white noise simulations to estimate the filter function.

        Parameters
        ----------
        params : np.ndarray
            Pulse sequence parameters
        frequencies : np.ndarray
            Frequencies at which to evaluate
        n_trajectories : int
            Number of noise realizations
        psd_amplitude : float
            Noise amplitude

        Returns
        -------
        np.ndarray
            Estimated |F(w)|^2 at each frequency
        """
        # This is a placeholder for a more sophisticated implementation
        # that would use spectral analysis of the variance
        measurement_op = SIGMA_Z

        var = self.avg_var_trajectories(
            params, measurement_op,
            n_trajectories=n_trajectories,
            noise_type=0,  # White noise
            psd_amplitude=psd_amplitude
        )

        # For white noise, variance ~ integral of |F(w)|^2 * S(w)
        # This is a rough approximation
        return var * np.ones_like(frequencies)
