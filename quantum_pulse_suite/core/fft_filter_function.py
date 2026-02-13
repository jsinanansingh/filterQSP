"""
FFT-based filter function computation.

Computes the filter function by sampling the toggling-frame noise Hamiltonian
in the time domain and taking the FFT. This gives a matrix-valued filter
function that works for arbitrary Hilbert space dimension.

For qubits (d=2), the matrix filter function can be decomposed into Pauli
components that match the analytic Bloch-vector filter function in total
noise susceptibility.
"""

import numpy as np
from scipy.linalg import expm


def _unitary_at_time(elements, t, d):
    """Compute the forward propagator U(t) from time 0 to time t.

    Instantaneous pulses (duration=0) at positions <= t are applied.
    Finite-duration elements are partially evolved up to time t.
    """
    U = np.eye(d, dtype=complex)
    cumtime = 0.0

    for element in elements:
        dur = element.duration()
        if dur == 0:
            # Instantaneous pulse: apply if at or before target time
            if cumtime <= t:
                U = element.unitary() @ U
        else:
            if cumtime >= t:
                break
            elif cumtime + dur <= t:
                U = element.unitary() @ U
                cumtime += dur
            else:
                # Partial evolution within this element
                dt_in = t - cumtime
                H = element.hamiltonian()
                U = expm(-1j * H * dt_in) @ U
                cumtime = t
                break

    return U


def fft_filter_function(seq, noise_hamiltonian, n_samples=4096, pad_factor=4):
    """
    Compute the filter function via FFT of the toggling-frame noise Hamiltonian.

    The toggling-frame noise Hamiltonian is:
        h(t) = U(t)^dag @ H_noise @ U(t)

    where U(t) is the forward propagator from time 0 to time t.
    The FFT of h(t) gives the matrix-valued filter function F(w).

    Zero-padding is used to interpolate the spectrum between the natural
    DFT grid points (w_k = 2*pi*k/T). Without padding, the filter function
    evaluates to zero at these grid points for constant toggling-frame
    Hamiltonians (e.g., free evolution at zero detuning).

    Parameters
    ----------
    seq : PulseSequence
        The pulse sequence (polynomials need not be computed).
    noise_hamiltonian : np.ndarray, shape (d, d)
        The noise Hamiltonian in the lab frame (e.g., sigma_z / 2).
    n_samples : int
        Number of time samples (power of 2 recommended).
    pad_factor : int
        Zero-padding factor. The signal is padded to pad_factor * n_samples
        points before FFT, giving pad_factor times finer frequency resolution.
        Default is 4.

    Returns
    -------
    frequencies : np.ndarray, shape (n_freq,)
        Positive angular frequencies.
    F_matrix : np.ndarray, shape (n_freq, d, d)
        Matrix-valued filter function at each frequency.
    """
    T = seq.total_duration()
    d = noise_hamiltonian.shape[0]
    dt = T / n_samples
    elements = seq.elements

    # Precompute segment boundaries and cumulative unitaries for efficiency
    segments = []  # (start_time, end_time, eigvals, V, Vdag, U_before)
    U_cumulative = np.eye(d, dtype=complex)
    cumtime = 0.0

    for elem in elements:
        dur = elem.duration()
        if dur == 0:
            U_cumulative = elem.unitary() @ U_cumulative
        else:
            H_elem = elem.hamiltonian()
            eigvals, V = np.linalg.eigh(H_elem)
            segments.append((
                cumtime, cumtime + dur,
                eigvals, V, V.conj().T,
                U_cumulative.copy()
            ))
            U_cumulative = elem.unitary() @ U_cumulative
            cumtime += dur

    # Sample toggling-frame Hamiltonian at discrete times
    times = np.linspace(0, T, n_samples, endpoint=False)
    H_tf = np.zeros((n_samples, d, d), dtype=complex)

    for seg_start, seg_end, eigvals, V, Vdag, U_before in segments:
        mask = (times >= seg_start) & (times < seg_end)
        indices = np.where(mask)[0]
        t_local = times[indices] - seg_start

        for j, idx in enumerate(indices):
            exp_diag = np.exp(-1j * eigvals * t_local[j])
            U_partial = V @ (exp_diag[:, None] * Vdag)
            U_t = U_partial @ U_before
            H_tf[idx] = U_t.conj().T @ noise_hamiltonian @ U_t

    # Zero-pad for spectral interpolation
    n_padded = pad_factor * n_samples
    H_padded = np.zeros((n_padded, d, d), dtype=complex)
    H_padded[:n_samples] = H_tf

    # FFT along time axis
    # Convention: F(w) = integral h(t) exp(-iwt) dt ~ dt * sum h[n] exp(-iwt_n)
    # numpy.fft.fft computes X[k] = sum x[n] exp(-2pi*j*k*n/N)
    # with w_k = 2*pi*k/(N_padded*dt), so X[k]*dt approximates F(w_k)
    F_fft = np.fft.fft(H_padded, axis=0) * dt

    # Angular frequencies (based on padded length)
    freqs = 2 * np.pi * np.fft.fftfreq(n_padded, d=dt)

    # Return positive frequencies only
    pos_mask = freqs > 0
    return freqs[pos_mask], F_fft[pos_mask]


def noise_susceptibility_from_matrix(F_matrix):
    """
    Compute the total noise susceptibility from the matrix filter function.

    For a traceless filter function matrix F(w):
        |F|^2 = 2 * Tr(F(w) @ F(w)^dag)

    This equals sum_k |F_k(w)|^2 where F_k are the Bloch/GGM components.

    Parameters
    ----------
    F_matrix : np.ndarray, shape (n_freq, d, d)
        Matrix-valued filter function.

    Returns
    -------
    np.ndarray, shape (n_freq,)
        Total noise susceptibility at each frequency.
    """
    # Tr(F @ F^dag) for each frequency
    # = sum over matrix elements of |F_{ij}|^2
    trace_FFdag = np.real(
        np.einsum('...ij,...ij->...', F_matrix, np.conj(F_matrix))
    )
    return 2.0 * trace_FFdag


def bloch_components_from_matrix(F_matrix, basis_matrices=None):
    """
    Extract Bloch vector filter function components from matrix filter function.

    For d=2 (qubit), decomposes F(w) in the Pauli basis:
        F_k(w) = Tr(sigma_k @ F(w))   for k = x, y, z

    For general d, uses the provided basis matrices (e.g., GGM matrices).

    Parameters
    ----------
    F_matrix : np.ndarray, shape (n_freq, d, d)
        Matrix-valued filter function.
    basis_matrices : list of np.ndarray, optional
        Operator basis matrices. If None, uses Pauli matrices for d=2.

    Returns
    -------
    components : list of np.ndarray
        F_k(w) for each basis matrix, each shape (n_freq,).
    """
    d = F_matrix.shape[-1]

    if basis_matrices is None:
        if d != 2:
            raise ValueError(
                "basis_matrices required for d != 2. "
                "Use generalized_gell_mann_matrices() for higher dimensions."
            )
        # Pauli matrices
        sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
        sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
        basis_matrices = [sigma_x, sigma_y, sigma_z]

    components = []
    for sigma_k in basis_matrices:
        # F_k = Tr(sigma_k @ F) for each frequency
        F_k = np.einsum('ij,...ji->...', sigma_k, F_matrix)
        components.append(F_k)

    return components
