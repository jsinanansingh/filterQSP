"""
Three-level clock filter functions.

Computes the classical noise variance Var[<dM>] for a three-level Lambda clock
system where:
- QSP control acts on {|g>, |e>} subspace (probe transition)
- Initial state is (|g> + |m>)/sqrt(2)
- Measurement observable M = m . sigma_gm acts on {|g>, |m>} manifold
- Noise source: beta_e(t) on |e><e|, and separately beta_m(t) on |m><m|

The relevant noise variance for frequency estimation is the shot-to-shot
variance of the signal mean <M>, averaged over classical stochastic noise:

    Var[<dM>]_noise = int dw/2pi  S_be(w) F(w)

where the filter function F(w) is determined by the sensitivity trajectory

    r(t) = <psi_I(t)|[M_I(t), H_e]|psi_I(t)>  =  i|G(t)|^2

and F(w) = |FT{r(t)}(w)|^2 = |Chi(w)|^2  (Chi = FT[|G(t)|^2]).

This satisfies the DC consistency condition  F(0) = sens_sq  (the signal
slope squared equals the filter function value at zero frequency).

The figure of merit is the normalised frequency variance:

    sigma_nu = Var[<dM>]_noise / F(0) = int dw/2pi S(w) F(w) / F(0)

Two filter functions are implemented:
- F(w)  = m_y^2 * |Chi(w)|^2   where  Chi(w) = FT[|G(t)|^2]   (e-noise)
- Ff(w) = (1 - m_z^2) * (1 - cos(w*T)) / w^2                  (m-noise)
"""

import numpy as np
from scipy.integrate import simpson, quad


def default_omega_cutoff(T):
    """Default low-frequency cutoff in angular-frequency units: 2*pi/T."""
    return 2.0 * np.pi / T


def resolve_omega_cutoff(T, omega_cutoff=None):
    """
    Resolve the low-frequency integration cutoff.

    Parameters
    ----------
    T : float
        Total interrogation time.
    omega_cutoff : float or None
        Lower cutoff in angular-frequency units.  If None, use the Fourier
        limit 2*pi/T.  Set to 0.0 to integrate from DC.
    """
    if omega_cutoff is None:
        return default_omega_cutoff(T)
    return float(omega_cutoff)


def Ff_analytic(frequencies, T, m_z):
    """
    Compute the f-noise filter function Ff(w).

    Protocol-independent filter function for noise on the |f> state.
        Ff(w) = (1 - m_z^2) * (1 - cos(w*T)) / w^2

    Parameters
    ----------
    frequencies : array_like
        Angular frequencies.
    T : float
        Total sequence duration.
    m_z : float
        z-component of measurement direction in {|g>, |f>} Bloch sphere.

    Returns
    -------
    np.ndarray
        Ff at each frequency.
    """
    w = np.asarray(frequencies, dtype=float)
    result = np.zeros_like(w)

    # Handle w -> 0 limit: (1 - cos(wT))/w^2 -> T^2/2
    small = np.abs(w) < 1e-10
    large = ~small

    if np.any(small):
        result[small] = (1 - m_z**2) * T**2 / 2
    if np.any(large):
        result[large] = (1 - m_z**2) * (1 - np.cos(w[large] * T)) / w[large]**2

    return result


def _exp_integral(w, t_start, t_end):
    """Compute int_{t_start}^{t_end} e^{-iwt} dt for array of frequencies."""
    w = np.asarray(w, dtype=float)
    result = np.zeros(len(w), dtype=complex)

    small = np.abs(w) < 1e-12
    large = ~small

    if np.any(small):
        result[small] = t_end - t_start
    if np.any(large):
        wl = w[large]
        result[large] = (np.exp(-1j * wl * t_end) -
                         np.exp(-1j * wl * t_start)) / (-1j * wl)

    return result


def fft_three_level_filter(seq, n_samples=4096, pad_factor=4,
                           M=None, psi0=None, m_z=0.0,
                           m_x=0.0, m_y=0.0):
    """
    Compute the three-level clock filter function via direct matrix multiplication.

    This is the ground-truth validation function.  At each time point it forms
    the full propagator U(t), the toggled noise A_e(t) = U†(t)|e><e|U(t), and
    the interaction-frame observable M_I(T) = U†(T) M U(T), then computes the
    sensitivity trajectory

        r(t) = <psi0 | [M_I(T), A_e(t)] | psi0>

    by explicit d×d matrix multiplication.  No Cayley-Klein / phi-chi
    decomposition is assumed.  The filter function is

        Fe(w) = |FT[r(t)](w)|^2.

    Since [M_I(T), A_e(t)] is anti-Hermitian, r(t) is purely imaginary; the
    signal slope S = -i ∫ r(t) dt is therefore manifestly real.

    Parameters
    ----------
    seq : MultiLevelPulseSequence
    n_samples : int
    pad_factor : int
    M : ndarray (d,d), optional
        Measurement observable.  Default: sigma_y^{gm}.
    psi0 : ndarray (d,), optional
        Initial state.  Default: (|g>+|m>)/sqrt(2).
    m_z : float
        Used only for the clock-noise Ff (protocol-independent).
    m_x, m_y : float
        Accepted for API compatibility; ignored for Fe.

    Returns
    -------
    frequencies : np.ndarray  (positive angular frequencies)
    Fe : np.ndarray            filter function |FT[r(t)]|^2
    Ff : np.ndarray            clock-noise filter function (protocol-independent)
    r_fft : np.ndarray         FT[r(t)] at positive frequencies (complex, for diagnostics)
    """
    d = seq.dim
    subspace = seq.subspace
    i_g, i_e = subspace._levels
    i_m = next(i for i in range(d) if i not in (i_g, i_e))

    if M is None:
        M = np.zeros((d, d), dtype=complex)
        M[i_g, i_m] = -1j
        M[i_m, i_g] =  1j

    if psi0 is None:
        psi0 = np.zeros(d, dtype=complex)
        psi0[i_g] = 1.0 / np.sqrt(2)
        psi0[i_m] = 1.0 / np.sqrt(2)

    M    = np.asarray(M,    dtype=complex)
    psi0 = np.asarray(psi0, dtype=complex)

    # Noise operator |e><e| in full space
    H_noise = np.zeros((d, d), dtype=complex)
    H_noise[i_e, i_e] = 1.0

    T = seq.total_duration()
    dt = T / n_samples
    elements = seq.elements

    # Build segment list (non-zero duration elements) with cumulative U before each
    segments = []
    U_cumulative = np.eye(d, dtype=complex)
    cumtime = 0.0
    for elem in elements:
        dur = elem.duration()
        if dur == 0:
            U_cumulative = elem.unitary() @ U_cumulative
        else:
            H_elem = elem.hamiltonian()
            eigvals, V = np.linalg.eigh(H_elem)
            segments.append((cumtime, cumtime + dur, eigvals, V, V.conj().T,
                             U_cumulative.copy()))
            U_cumulative = elem.unitary() @ U_cumulative
            cumtime += dur

    # M_I(T) = U†(T) M U(T)
    M_I_T = U_cumulative.conj().T @ M @ U_cumulative

    # Sample r(t) at uniform time grid
    times = np.linspace(0, T, n_samples, endpoint=False)
    r_samples = np.zeros(n_samples, dtype=complex)

    for seg_start, seg_end, eigvals, V, Vdag, U_before in segments:
        mask = (times >= seg_start) & (times < seg_end)
        indices = np.where(mask)[0]
        t_local = times[indices] - seg_start

        for j, idx in enumerate(indices):
            exp_diag = np.exp(-1j * eigvals * t_local[j])
            U_partial = V @ (exp_diag[:, None] * Vdag)
            U_t = U_partial @ U_before

            # A_e(t) = U†(t) |e><e| U(t)
            A_e = U_t.conj().T @ H_noise @ U_t

            # [M_I(T), A_e(t)]
            comm = M_I_T @ A_e - A_e @ M_I_T

            # r(t) = <psi0|comm|psi0>  (purely imaginary)
            r_samples[idx] = psi0.conj() @ comm @ psi0

    # Zero-pad and FFT
    n_padded = pad_factor * n_samples
    r_padded = np.zeros(n_padded, dtype=complex)
    r_padded[:n_samples] = r_samples
    r_fft = np.fft.fft(r_padded) * dt
    freqs = 2 * np.pi * np.fft.fftfreq(n_padded, d=dt)

    pos_mask = freqs > 0
    freqs_pos = freqs[pos_mask]
    r_pos = r_fft[pos_mask]

    Fe = np.abs(r_pos)**2
    Ff = Ff_analytic(freqs_pos, T, m_z)

    return freqs_pos, Fe, Ff, r_pos


def direct_dft_filter(seq, omega_array, n_samples=4096, m_y=1.0):
    """
    Compute Fe(omega) at arbitrary frequencies via direct DFT summation.

    Unlike fft_three_level_filter (which returns uniformly spaced frequencies),
    this evaluates the Fourier transform at exactly the requested omega values.
    Use with log-spaced omega_array for smooth log-log filter function plots.

    The computation is O(n_samples * len(omega_array)) — fast for n_samples=4096
    and a few hundred omega points.

    Parameters
    ----------
    seq : MultiLevelPulseSequence
        Pulse sequence on the probe transition {|g>, |e>}.
    omega_array : array_like
        Angular frequencies at which to evaluate Fe (rad/s).  Can be any set
        of frequencies, e.g. np.logspace(-2, 2, 500).
    n_samples : int
        Number of uniform time samples for the path functions.
    m_y : float
        y-component of measurement direction (typically 1.0 for sigma_y^{gm}).

    Returns
    -------
    omega_array : np.ndarray
        Same as input, as a numpy array.
    Fe : np.ndarray
        e-noise filter function F(omega) = m_y^2 * |Chi|^2.
    """
    omega_array = np.asarray(omega_array, dtype=float)
    T = seq.total_duration()
    dt = T / n_samples
    subspace = seq.subspace
    elements = seq.elements
    d = seq.dim

    # Build segment list (same as fft_three_level_filter)
    segments = []
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

    # Sample phi(t) = F*(t)G(t) and chi(t) = |G(t)|^2 on uniform time grid
    times = np.linspace(0, T, n_samples, endpoint=False)
    phi_samples = np.zeros(n_samples, dtype=complex)
    chi_samples = np.zeros(n_samples, dtype=float)

    for seg_start, seg_end, eigvals, V, Vdag, U_before in segments:
        mask = (times >= seg_start) & (times < seg_end)
        indices = np.where(mask)[0]
        t_local = times[indices] - seg_start
        for j, idx in enumerate(indices):
            exp_diag = np.exp(-1j * eigvals * t_local[j])
            U_partial = V @ (exp_diag[:, None] * Vdag)
            U_t = U_partial @ U_before
            U_sub = subspace.project_operator(U_t)
            f_t = U_sub[0, 0]
            g_t = U_sub[0, 1] / 1j
            phi_samples[idx] = np.conj(f_t) * g_t
            chi_samples[idx] = np.abs(g_t)**2

    # Extract F(T), G(T) from the final propagator U_cumulative = U(T)
    U_T_sub = subspace.project_operator(U_cumulative)
    F_T = U_T_sub[0, 0]
    G_T = U_T_sub[0, 1] / 1j

    # Correct sensitivity kernel h(t) = Re[F(T)]*chi(t) + Re[G(T)*phi(t)]
    h_samples = np.real(F_T) * chi_samples + np.real(G_T * phi_samples)

    # Direct DFT at requested frequencies
    phases = np.exp(-1j * np.outer(omega_array, times))  # (n_omega, n_time)
    H = phases @ h_samples * dt

    Fe = np.abs(H)**2
    return omega_array, Fe


def fft_phase_filter(seq, n_samples=4096, pad_factor=4,
                     m_x=0.0, m_y=0.0, m_z=0.0):
    """
    Compute phase-derivative filter function via FFT.

    Samples phi(t) = F*(t) G(t) and chi(t) = |G(t)|^2, takes their numerical
    time derivatives, zero-pads, and FFTs to obtain Phi_dot(w) and Chi_dot(w).
    Then constructs:

        F_phase(w) = 1/2*(1 + 2*m_z*m_x)*|Phi_dot(w)|^2
                     + (m_x^2 + m_y^2)*|Chi_dot(w)|^2

    Since Phi_dot(w) = iw*Phi(w) and Chi_dot(w) = iw*Chi(w), this equals
    w^2 * Fe(w), emphasising sensitivity to frequency noise.  For instantaneous
    sequences phi(t) is piecewise constant so d phi/dt ≈ 0; the function is
    most meaningful for continuous protocols (GPS, Rabi).

    Parameters
    ----------
    seq : MultiLevelPulseSequence
        Pulse sequence on the probe transition {|g>, |e>}.
    n_samples : int
        Number of time samples (power of 2 recommended).
    pad_factor : int
        Zero-padding factor for spectral interpolation.
    m_x, m_y, m_z : float
        Components of measurement direction in {|g>, |f>} Bloch sphere.

    Returns
    -------
    frequencies : np.ndarray
        Positive angular frequencies.
    F_phase : np.ndarray
        Phase-derivative filter function at each frequency.
    Ff : np.ndarray
        f-noise filter function at each frequency (protocol-independent).
    Phi_dot : np.ndarray
        FFT of d phi/dt at each frequency.
    """
    T = seq.total_duration()
    dt = T / n_samples
    subspace = seq.subspace
    elements = seq.elements
    d = seq.dim

    # Precompute segment boundaries and cumulative unitaries
    segments = []
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

    # Sample phi(t) = F*(t) G(t) and chi(t) = |G(t)|^2 at discrete times
    times = np.linspace(0, T, n_samples, endpoint=False)
    phi_samples = np.zeros(n_samples, dtype=complex)
    chi_samples = np.zeros(n_samples, dtype=float)

    for seg_start, seg_end, eigvals, V, Vdag, U_before in segments:
        mask = (times >= seg_start) & (times < seg_end)
        indices = np.where(mask)[0]
        t_local = times[indices] - seg_start

        for j, idx in enumerate(indices):
            exp_diag = np.exp(-1j * eigvals * t_local[j])
            U_partial = V @ (exp_diag[:, None] * Vdag)
            U_t = U_partial @ U_before

            U_sub = subspace.project_operator(U_t)
            f_t = U_sub[0, 0]
            g_t = U_sub[0, 1] / 1j
            phi_samples[idx] = np.conj(f_t) * g_t
            chi_samples[idx] = np.abs(g_t)**2

    # Numerical time derivatives
    dphi_dt = np.gradient(phi_samples, dt)
    dchi_dt = np.gradient(chi_samples, dt)

    # Zero-pad and FFT the derivatives
    n_padded = pad_factor * n_samples
    dphi_padded = np.zeros(n_padded, dtype=complex)
    dphi_padded[:n_samples] = dphi_dt
    dchi_padded = np.zeros(n_padded, dtype=float)
    dchi_padded[:n_samples] = dchi_dt

    Phi_dot_fft = np.fft.fft(dphi_padded) * dt
    Chi_dot_fft = np.fft.fft(dchi_padded) * dt
    freqs = 2 * np.pi * np.fft.fftfreq(n_padded, d=dt)

    # Positive frequencies only
    pos_mask = freqs > 0
    freqs_pos = freqs[pos_mask]
    Phi_dot_pos = Phi_dot_fft[pos_mask]
    Chi_dot_pos = Chi_dot_fft[pos_mask]

    # F_phase(w) = 1/2*(1+2*m_z*m_x)*|Phi_dot|^2 + (m_x^2+m_y^2)*|Chi_dot|^2
    F_phase = (0.5 * (1 + 2 * m_z * m_x) * np.abs(Phi_dot_pos)**2
               + (m_x**2 + m_y**2) * np.abs(Chi_dot_pos)**2)

    Ff = Ff_analytic(freqs_pos, T, m_z)

    return freqs_pos, F_phase, Ff, Phi_dot_pos


def fft_phase_filter_2level(seq, n_samples=4096, pad_factor=4):
    """
    Phase-derivative filter function for a qubit (2-level) pulse sequence.

    Samples phi(t) = F*(t) G(t) from the qubit Cayley-Klein trajectory,
    takes its numerical time derivative d phi/dt, zero-pads, and FFTs.
    Returns |FFT[d phi/dt]|^2, which equals w^2 * |Phi(w)|^2.

    Parameters
    ----------
    seq : PulseSequence
        Qubit pulse sequence with compute_polynomials() already called (or
        will be called here). Must expose _polynomial_segments storing
        (F_func, G_func, t_start, t_end) tuples.
    n_samples : int
        Number of time samples (power of 2 recommended).
    pad_factor : int
        Zero-padding factor for spectral interpolation.

    Returns
    -------
    frequencies : np.ndarray
        Positive angular frequencies.
    F_phase : np.ndarray
        |FFT[d phi/dt]|^2 at each frequency.
    """
    if not seq._polynomials_computed:
        seq.compute_polynomials()

    T = seq.total_duration()
    dt = T / n_samples
    times = np.linspace(0, T, n_samples, endpoint=False)
    phi_samples = np.zeros(n_samples, dtype=complex)

    for F_func, G_func, t_start, t_end in seq._polynomial_segments:
        mask = (times >= t_start) & (times < t_end)
        for idx in np.where(mask)[0]:
            t = times[idx]
            phi_samples[idx] = np.conj(F_func(t)) * G_func(t)

    # Numerical time derivative
    dphi_dt = np.gradient(phi_samples, dt)

    # Zero-pad and FFT
    n_padded = pad_factor * n_samples
    dphi_padded = np.zeros(n_padded, dtype=complex)
    dphi_padded[:n_samples] = dphi_dt

    Phi_dot_fft = np.fft.fft(dphi_padded) * dt
    freqs = 2 * np.pi * np.fft.fftfreq(n_padded, d=dt)

    pos_mask = freqs > 0
    freqs_pos = freqs[pos_mask]
    F_phase = np.abs(Phi_dot_fft[pos_mask])**2

    return freqs_pos, F_phase


def kubo_filter_2level(qubit_seq, m_hat=(0., 1., 0.), r0=None,
                       n_samples=4096, pad_factor=4):
    """
    Kubo variance filter function for a qubit (2-level) pulse sequence.

    Computes the first-order sensitivity trajectory

        r(t) = (m̂ × R(t)) · r₀

    where R(t) is the Bloch vector of the toggling-frame noise operator
    U†(t)(σ_z/2)U(t), evaluated analytically from the QSP polynomial
    segments.  Returns the variance filter function |r(ω)|².

    Parameters
    ----------
    qubit_seq : PulseSequence
        Qubit pulse sequence.  compute_polynomials() is called if needed.
    m_hat : array_like, shape (3,)
        Bloch vector of measurement observable M.
        Default (0, 1, 0) = σ_y.
    r0 : array_like, shape (3,), optional
        Bloch vector of initial state ψ₀.
        Default (1, 0, 0) for equal superposition (|0⟩+|1⟩)/√2.
    n_samples : int
        Number of time samples (power of 2 recommended).
    pad_factor : int
        Zero-padding factor for spectral interpolation.

    Returns
    -------
    frequencies : np.ndarray
        Positive angular frequencies.
    F_kubo : np.ndarray
        Kubo variance filter function |r(ω)|² at each frequency.
    """
    if not qubit_seq._polynomials_computed:
        qubit_seq.compute_polynomials()

    m_hat = np.asarray(m_hat, dtype=float)
    if r0 is None:
        r0 = np.array([1., 0., 0.])   # (|0⟩+|1⟩)/√2 → Bloch x-axis
    else:
        r0 = np.asarray(r0, dtype=float)

    T = qubit_seq.total_duration()
    dt = T / n_samples
    times = np.linspace(0, T, n_samples, endpoint=False)
    r_samples = np.zeros(n_samples, dtype=complex)

    for F_func, G_func, t_start, t_end in qubit_seq._polynomial_segments:
        mask = (times >= t_start) & (times < t_end)
        for idx in np.where(mask)[0]:
            t = times[idx]
            f_t = F_func(t)
            g_t = G_func(t)

            # Bloch vector of U†(σ_z/2)U:
            #   H̃_{01} = i g f*  →  n_x = 2 Re(igf*), n_y = -2 Im(igf*)
            #   n_z = |f|² - |g|²
            igf_star = 1j * g_t * np.conj(f_t)
            R = np.array([2.0 * np.real(igf_star),
                          -2.0 * np.imag(igf_star),
                          np.abs(f_t)**2 - np.abs(g_t)**2])

            r_samples[idx] = np.dot(np.cross(m_hat, R), r0)

    # Zero-pad and FFT
    n_padded = pad_factor * n_samples
    r_padded = np.zeros(n_padded, dtype=complex)
    r_padded[:n_samples] = r_samples

    r_fft = np.fft.fft(r_padded) * dt
    freqs = 2 * np.pi * np.fft.fftfreq(n_padded, d=dt)

    pos_mask = freqs > 0
    return freqs[pos_mask], np.abs(r_fft[pos_mask])**2


def kubo_filter_3level(seq, M=None, psi0=None, n_samples=4096, pad_factor=4):
    """
    Kubo variance filter function for the three-level clock system.

    Computes the first-order sensitivity trajectory

        r(t) = ⟨ψ₀ | [M, H̃(t)] | ψ₀⟩

    where H̃(t) = U†(t)|e⟩⟨e|U(t) is the toggling-frame probe noise
    Hamiltonian constructed analytically from the QSP polynomial segments.
    Returns the variance filter function |r(ω)|².

    For the default choice M = σ_y^{gm} and ψ₀ = (|g⟩+|m⟩)/√2, this
    reduces to r(t) = i|g(t)|², so |r(ω)|² = |Chi(ω)|² — identical to
    the Chi term in Fe with m_y=1.

    Parameters
    ----------
    seq : MultiLevelPulseSequence
        Pulse sequence on the probe {|g⟩, |e⟩} subspace.
        compute_polynomials() is called if needed.
    M : np.ndarray, shape (d, d), optional
        Measurement observable.  Default: σ_y on the |g⟩-|m⟩ clock
        transition, i.e. -i|g⟩⟨m| + i|m⟩⟨g|.
    psi0 : np.ndarray, shape (d,), optional
        Initial state vector.  Default: (|g⟩+|m⟩)/√2.
    n_samples : int
        Number of time samples (power of 2 recommended).
    pad_factor : int
        Zero-padding factor for spectral interpolation.

    Returns
    -------
    frequencies : np.ndarray
        Positive angular frequencies.
    F_kubo : np.ndarray
        Kubo variance filter function |r(ω)|² at each frequency.
    """
    if not seq._polynomials_computed:
        seq.compute_polynomials()

    d = seq.dim
    i_g, i_e = seq.subspace._levels          # probe subspace indices
    i_m = next(i for i in range(d) if i not in (i_g, i_e))   # clock state

    if M is None:
        M = np.zeros((d, d), dtype=complex)
        M[i_g, i_m] = -1j   # σ_y^{gm} = -i|g⟩⟨m| + i|m⟩⟨g|
        M[i_m, i_g] =  1j

    if psi0 is None:
        psi0 = np.zeros(d, dtype=complex)
        psi0[i_g] = 1.0 / np.sqrt(2)
        psi0[i_m] = 1.0 / np.sqrt(2)

    M    = np.asarray(M,    dtype=complex)
    psi0 = np.asarray(psi0, dtype=complex)

    # Compute U(T) to form M_I(T) = U†(T) M U(T)
    # U(T) is the product of all element unitaries.
    U_T = np.eye(d, dtype=complex)
    for elem in seq.elements:
        U_T = elem.unitary() @ U_T
    M_I_T = U_T.conj().T @ M @ U_T

    # Noise: |e⟩⟨e| in probe subspace → [[0,0],[0,1]]
    H_noise_sub = np.array([[0, 0], [0, 1]], dtype=complex)

    T = seq.total_duration()
    dt = T / n_samples
    times = np.linspace(0, T, n_samples, endpoint=False)
    r_samples = np.zeros(n_samples, dtype=complex)

    for F_func, G_func, t_start, t_end in seq._polynomial_segments:
        mask = (times >= t_start) & (times < t_end)
        for idx in np.where(mask)[0]:
            t = times[idx]
            f_t = F_func(t)
            g_t = G_func(t)

            # Probe-subspace unitary U_sub(t) = [[f, ig],[ig*, f*]]
            U_sub = np.array([[f_t,            1j * g_t],
                              [1j * np.conj(g_t), np.conj(f_t)]])

            # H̃ in probe subspace: U_sub† H_noise_sub U_sub
            H_tilde_sub = U_sub.conj().T @ H_noise_sub @ U_sub

            # Embed H̃ into full d×d space at probe indices
            H_tilde = np.zeros((d, d), dtype=complex)
            for li, i in enumerate((i_g, i_e)):
                for lj, j in enumerate((i_g, i_e)):
                    H_tilde[i, j] = H_tilde_sub[li, lj]

            # r(t) = ⟨ψ₀|[M_I(T), H̃(t)]|ψ₀⟩  (purely imaginary)
            # Using M_I(T) = U†(T)MU(T) instead of bare M.
            MH = M_I_T @ H_tilde
            commutator = MH - MH.conj().T        # = [M_I(T), H̃]
            r_samples[idx] = psi0.conj() @ commutator @ psi0

    # Zero-pad and FFT
    n_padded = pad_factor * n_samples
    r_padded = np.zeros(n_padded, dtype=complex)
    r_padded[:n_samples] = r_samples

    r_fft = np.fft.fft(r_padded) * dt
    freqs = 2 * np.pi * np.fft.fftfreq(n_padded, d=dt)

    pos_mask = freqs > 0
    return freqs[pos_mask], np.abs(r_fft[pos_mask])**2


def kubo_filter_2level_analytic(qubit_seq, frequencies,
                                m_hat=(0., 1., 0.), r0=None):
    """
    Analytic (QSP-polynomial) Kubo variance filter function for a qubit.

    Evaluates r(ω) = ∫₀ᵀ r(t) e^{-iωt} dt analytically by decomposing

        r(t) = C_f2·|F(t)|² + C_g2·|G(t)|² + C_q·G(t)F*(t) + C_q*·G*(t)F(t)

    where the coefficients are determined by m̂ and r₀.  For free-evolution
    segments |F|², |G|², GF* are constant → exact via _exp_integral.
    For continuous-pulse segments: per-frequency numerical quadrature.

    Parameters
    ----------
    qubit_seq : PulseSequence
        Qubit pulse sequence with _polynomial_segments available.
    frequencies : array_like
        Angular frequencies at which to evaluate.
    m_hat : array_like, shape (3,)
        Bloch vector of measurement M.  Default (0, 1, 0) = σ_y.
    r0 : array_like, shape (3,), optional
        Bloch vector of initial state.  Default (1, 0, 0) for equal superposition.

    Returns
    -------
    frequencies : np.ndarray
        Input frequencies.
    F_kubo : np.ndarray
        |r(ω)|² at each frequency.
    """
    if not qubit_seq._polynomials_computed:
        qubit_seq.compute_polynomials()

    m_hat = np.asarray(m_hat, dtype=float)
    if r0 is None:
        r0 = np.array([1., 0., 0.])
    else:
        r0 = np.asarray(r0, dtype=float)

    frequencies = np.asarray(frequencies, dtype=float)

    # Precompute coefficients:
    # R(t) = (-2 Im(GF*), -2 Re(GF*), |F|²-|G|²)
    #      = i*(GF* - G*F) * (1,0,0) direction +
    #        -(GF* + G*F) * (0,1,0) direction +
    #        (|F|²-|G|²)  * (0,0,1) direction
    # Coefficients of GF*, G*F, |F|², |G|² in r = (m̂ × R)·r₀:
    C_q  = np.dot(np.cross(m_hat, np.array([1j,  -1., 0.])), r0)  # coeff of GF*
    C_qc = np.dot(np.cross(m_hat, np.array([-1j, -1., 0.])), r0)  # coeff of G*F (= C_q*)
    C_f2 = np.dot(np.cross(m_hat, np.array([0.,   0., 1.])), r0)  # coeff of |F|²
    C_g2 = -C_f2                                                    # coeff of |G|²

    r_omega = np.zeros(len(frequencies), dtype=complex)
    poly_list = qubit_seq._poly_list

    for seg_idx, (F_func, G_func, t_start, t_end) in enumerate(
            qubit_seq._polynomial_segments):
        tau = t_end - t_start
        if tau < 1e-15:
            continue

        poly_entry = poly_list[seg_idx]

        if len(poly_entry) == 5:
            # Free evolution: |F|², |G|², GF* all constant
            f0, g0 = poly_entry[0], poly_entry[1]
            q0 = g0 * np.conj(f0)
            r_const = (C_g2 * abs(g0)**2 + C_f2 * abs(f0)**2
                       + C_q * q0 + C_qc * np.conj(q0))
            r_omega += r_const * _exp_integral(frequencies, t_start, t_end)
        else:
            # Continuous pulse: numerical quadrature
            def _r(t):
                ft, gt = F_func(t), G_func(t)
                q = gt * np.conj(ft)
                return (C_g2 * abs(gt)**2 + C_f2 * abs(ft)**2
                        + C_q * q + C_qc * np.conj(q))

            for fi, w in enumerate(frequencies):
                def re_integrand(t, ww=w): return np.real(_r(t) * np.exp(-1j * ww * t))
                def im_integrand(t, ww=w): return np.imag(_r(t) * np.exp(-1j * ww * t))
                re_, _ = quad(re_integrand, t_start, t_end, limit=100)
                im_, _ = quad(im_integrand, t_start, t_end, limit=100)
                r_omega[fi] += re_ + 1j * im_

    return frequencies, np.abs(r_omega)**2


def kubo_filter_2level_full_analytic(seq, frequencies,
                                     m_hat=(0., 1., 0.), r0=None):
    """
    Simple and full Kubo variance filter functions for a qubit.

    Computes F̃(ω) = FT[R(t)](ω) for all three components of the
    toggling-frame Bloch vector R(t) (Bloch vector of U†σ_z U), then
    returns two filter functions derived from the sensitivity trajectory
    Â(t) = (m̂ × R(t))·σ:

        F_simple(ω) = |(m̂ × F̃(ω))·r₀|²
            First-order Kubo: expectation value r(t) = ⟨ψ₀|Â(t)|ψ₀⟩ is
            taken before the Fourier transform.  Equivalent to
            kubo_filter_2level_analytic for these parameters.

        F_full(ω) = |m̂ × F̃(ω)|² = |F̃(ω)|² - |m̂·F̃(ω)|²
            FT of the two-time quantum correlator
            ⟨ψ₀|Â(t)Â(t')|ψ₀⟩ (symmetric part), using
            (a·σ)(b·σ) = a·b + i(a×b)·σ with the antisymmetric imaginary
            term vanishing for symmetric noise PSDs.

    The gap F_full − F_simple = |(m̂×F̃) − ((m̂×F̃)·r̂₀)r̂₀|² is the
    component of m̂×F̃(ω) perpendicular to r₀, squared.

    For free-evolution segments R(t) is constant → exact via
    _exp_integral.  For continuous-pulse segments a dense time grid
    with rectangular quadrature is used (n_quad=1024 points/segment).

    Parameters
    ----------
    seq : PulseSequence
        Qubit pulse sequence with compute_polynomials() already called.
    frequencies : array_like
        Angular frequencies at which to evaluate.
    m_hat : array_like, shape (3,)
        Bloch vector of measurement M = m̂·σ.  Default (0,1,0) = σ_y.
    r0 : array_like, shape (3,), optional
        Bloch vector of initial state ψ₀.  Default (1,0,0) for equal
        superposition.  Pass (0,0,1) for the σ_z eigenstate |0⟩.

    Returns
    -------
    frequencies : np.ndarray
        Input frequencies.
    F_simple : np.ndarray
        |(m̂×F̃(ω))·r₀|² at each frequency.
    F_full : np.ndarray
        |m̂×F̃(ω)|² at each frequency.
    """
    if not seq._polynomials_computed:
        seq.compute_polynomials()

    m_hat = np.asarray(m_hat, dtype=float)
    if r0 is None:
        r0 = np.array([1., 0., 0.])
    else:
        r0 = np.asarray(r0, dtype=float)

    frequencies = np.asarray(frequencies, dtype=float)
    nf = len(frequencies)

    # Ftilde[k, fi] = int R_k(t) e^{-i w_fi t} dt, accumulated over segments
    Ftilde = np.zeros((3, nf), dtype=complex)
    poly_list = seq._poly_list

    for seg_idx, (F_func, G_func, t_start, t_end) in enumerate(
            seq._polynomial_segments):
        tau = t_end - t_start
        if tau < 1e-15:
            continue

        poly_entry = poly_list[seg_idx]

        if len(poly_entry) == 5:
            # Free evolution: R(t) is constant (detuning phases cancel in
            # f*g and |f|², |g|²)
            f0, g0 = poly_entry[0], poly_entry[1]
            fsg = np.conj(f0) * g0
            R_const = np.array([
                -2.0 * np.imag(fsg),       # R_x = -2 Im(f*g)
                -2.0 * np.real(fsg),       # R_y = -2 Re(f*g)
                abs(f0)**2 - abs(g0)**2,   # R_z = |f|² - |g|²
            ])
            exp_int = _exp_integral(frequencies, t_start, t_end)
            Ftilde += R_const[:, np.newaxis] * exp_int[np.newaxis, :]

        else:
            # Continuous pulse: sample R(t) on a dense grid, then DFT
            n_quad = 1024
            t_grid = np.linspace(t_start, t_end, n_quad, endpoint=False)
            dt = tau / n_quad

            R_grid = np.empty((3, n_quad))
            for i, t in enumerate(t_grid):
                ft, gt = F_func(t), G_func(t)
                fsg = np.conj(ft) * gt
                R_grid[0, i] = -2.0 * np.imag(fsg)
                R_grid[1, i] = -2.0 * np.real(fsg)
                R_grid[2, i] = abs(ft)**2 - abs(gt)**2

            # Vectorised DFT: (nf, n_quad) @ (n_quad, 3) → (nf, 3)
            exp_kernel = np.exp(
                -1j * np.outer(frequencies, t_grid))   # (nf, n_quad)
            Ftilde += dt * (exp_kernel @ R_grid.T).T   # (3, nf)

    # m̂ × F̃(ω): cross product of real m̂ with complex F̃ at each frequency
    # np.cross(m_hat, Ftilde.T) broadcasts to (nf, 3); transpose → (3, nf)
    mx_Ftilde = np.cross(m_hat, Ftilde.T).T    # (3, nf), complex

    # F_simple(ω) = |(m̂×F̃)·r₀|²
    F_simple = np.abs(r0 @ mx_Ftilde) ** 2     # (nf,)

    # F_full(ω) = |m̂×F̃|² = Σ_k |(m̂×F̃)_k|²
    F_full = np.sum(np.abs(mx_Ftilde) ** 2, axis=0)   # (nf,)

    return frequencies, F_simple, F_full


def kubo_filter_3level_analytic(seq, frequencies, m_y=1.0):
    """
    Analytic Kubo variance filter function for the 3-level clock system.

    For measurement M = m_y·σ_y^{gm} on the |g⟩-|m⟩ clock transition and
    initial state ψ₀ = (|g⟩+|m⟩)/√2, the sensitivity trajectory reduces to

        r(t) = i·m_y·|G(t)|²

    so that r(ω) = i·m_y·Chi(ω) and |r(ω)|² = m_y²·|Chi(ω)|², where

        Chi(ω) = ∫₀ᵀ |G(t)|² e^{-iωt} dt.

    For free-evolution segments |G(t)|² is constant → exact via
    _exp_integral.  For continuous-pulse segments: numerical quadrature.

    Parameters
    ----------
    seq : MultiLevelPulseSequence
        Probe pulse sequence with compute_polynomials() already called.
    frequencies : array_like
        Angular frequencies at which to evaluate.
    m_y : float
        y-component of measurement direction in {|g⟩, |m⟩} Bloch sphere.

    Returns
    -------
    frequencies : np.ndarray
        Input frequencies.
    F_kubo : np.ndarray
        m_y²·|Chi(ω)|² at each frequency.
    """
    if not seq._polynomials_computed:
        seq.compute_polynomials()

    frequencies = np.asarray(frequencies, dtype=float)
    Chi = np.zeros(len(frequencies), dtype=complex)
    poly_list = seq._poly_list

    for seg_idx, (F_func, G_func, t_start, t_end) in enumerate(seq._polynomial_segments):
        tau = t_end - t_start
        if tau < 1e-15:
            continue

        poly_entry = poly_list[seg_idx]

        if len(poly_entry) == 5:
            # Free evolution: |G(t)|² = |g0|² (constant)
            g0 = poly_entry[1]
            Chi += abs(g0)**2 * _exp_integral(frequencies, t_start, t_end)

        elif len(poly_entry) == 7:
            # Continuous pulse: numerical quadrature
            _add_continuous_segment_chi(Chi, G_func, t_start, t_end, frequencies)

    return frequencies, m_y**2 * np.abs(Chi)**2


def analytic_three_level_filter(seq, frequencies, m_x=0.0, m_y=0.0, m_z=0.0):
    """
    Compute three-level clock filter functions analytically.

    Uses the polynomial segments from compute_polynomials() to evaluate:
      Phi(w) = sum_j int F*_j(t) G_j(t) e^{-iwt} dt
      Chi(w) = sum_j int |G_j(t)|^2 e^{-iwt} dt

    For free evolution segments F*G and |G|^2 are both constant (phase factors
    cancel for |G|^2; F*G picks up constant phase), giving simple exponential
    integrals.  For continuous pulse segments, per-segment numerical quadrature
    is used.

    Parameters
    ----------
    seq : MultiLevelPulseSequence
        Pulse sequence with compute_polynomials() already called.
    frequencies : array_like
        Angular frequencies at which to evaluate.
    m_x, m_y, m_z : float
        Components of measurement direction in {|g>, |m>} Bloch sphere.
        Only m_y enters Fe; m_x is accepted for API compatibility.

    Returns
    -------
    frequencies : np.ndarray
        Angular frequencies (same as input).
    Fe : np.ndarray
        e-noise filter function at each frequency.
    Ff : np.ndarray
        f-noise filter function at each frequency.
    Chi : np.ndarray
        Core spectral quantity Chi(w) = FT[|G(t)|^2] at each frequency.
    """
    frequencies = np.asarray(frequencies, dtype=float)
    T = seq.total_duration()

    # Ensure polynomials are computed
    if not seq._polynomials_computed:
        seq.compute_polynomials()

    segments = seq._polynomial_segments
    poly_list = seq._poly_list

    Phi = np.zeros(len(frequencies), dtype=complex)
    Chi = np.zeros(len(frequencies), dtype=complex)

    for seg_idx, (F_func, G_func, t_start, t_end) in enumerate(segments):
        tau = t_end - t_start
        if tau < 1e-15:
            continue

        poly_entry = poly_list[seg_idx]

        if len(poly_entry) == 5:
            # Free evolution: F*G = conj(f_prev)*g_prev (constant)
            #                 |G|^2 = |g_prev|^2 (constant, phase cancels)
            f_prev, g_prev = poly_entry[0], poly_entry[1]
            fg_product = np.conj(f_prev) * g_prev
            g2 = abs(g_prev)**2

            exp_int = _exp_integral(frequencies, t_start, t_end)
            if abs(fg_product) >= 1e-15:
                Phi += fg_product * exp_int
            if g2 >= 1e-15:
                Chi += g2 * exp_int

        elif len(poly_entry) == 7:
            # Continuous pulse: numerical quadrature for both integrals
            _add_continuous_segment_phi(
                Phi, F_func, G_func, t_start, t_end, frequencies)
            _add_continuous_segment_chi(
                Chi, G_func, t_start, t_end, frequencies)

    # F(w) = m_y^2 * |Chi|^2  (classical noise filter function)
    Fe = m_y**2 * np.abs(Chi)**2

    # Construct Ff(w)
    Ff = Ff_analytic(frequencies, T, m_z)

    return frequencies, Fe, Ff, Chi


def _add_continuous_segment_phi(Phi, F_func, G_func, t_start, t_end,
                                frequencies):
    """Add contribution of a continuous pulse segment to Phi via quadrature."""
    for fi, w in enumerate(frequencies):
        def integrand_re(t, ww=w):
            fg = np.conj(F_func(t)) * G_func(t)
            return np.real(fg * np.exp(-1j * ww * t))

        def integrand_im(t, ww=w):
            fg = np.conj(F_func(t)) * G_func(t)
            return np.imag(fg * np.exp(-1j * ww * t))

        re_part, _ = quad(integrand_re, t_start, t_end, limit=100)
        im_part, _ = quad(integrand_im, t_start, t_end, limit=100)
        Phi[fi] += re_part + 1j * im_part


def _add_continuous_segment_chi(Chi, G_func, t_start, t_end, frequencies):
    """Add contribution of a continuous pulse segment to Chi = int |G|^2 e^{-iwt} dt."""
    for fi, w in enumerate(frequencies):
        def integrand_re(t, ww=w):
            return np.abs(G_func(t))**2 * np.cos(ww * t)

        def integrand_im(t, ww=w):
            return -np.abs(G_func(t))**2 * np.sin(ww * t)

        re_part, _ = quad(integrand_re, t_start, t_end, limit=100)
        im_part, _ = quad(integrand_im, t_start, t_end, limit=100)
        Chi[fi] += re_part + 1j * im_part


def _probe_ck_at_delta(seq, delta_extra):
    """
    Evaluate the probe-subspace Cayley-Klein parameters (f, g) with an
    additional global laser detuning delta_extra on top of each element's
    stored delta.

    The physical laser detuning Hamiltonian is
        H_delta = delta * |e><e|
    which equals the traceless SU(2) part  delta/2*(|e><e|-|g><g|)  plus a
    global probe-subspace phase  delta/2 * I.  The SU(2) update for free
    evolution of duration tau is therefore
        diag(exp(+i delta tau/2),  exp(-i delta tau/2))
    (same as the traceless convention), but additionally a global phase
        exp(-i delta tau/2)
    accumulates on the probe subspace relative to the untouched clock state
    |m>.  This function returns both the SU(2) parameters (f, g) and the
    accumulated probe global phase so that callers can apply the correct
    relative phase between probe and clock components.

    Continuous-pulse elements use the full off-resonant Rabi formula with
    delta_total = element.delta + delta_extra and also accumulate a global
    phase exp(-i delta_total tau/2).  Instantaneous pulses are
    delta-independent and contribute no global phase.

    Works for both MultiLevelPulseSequence (3-level) and PulseSequence (2-level).

    Parameters
    ----------
    seq : MultiLevelPulseSequence or ContinuousPulseSequence
        Pulse sequence whose elements have .delta, .tau, .omega, .axis attributes.
    delta_extra : float
        Extra detuning to add globally to every segment.

    Returns
    -------
    f, g : complex
        Final Cayley-Klein amplitudes of the probe subspace (SU(2) part).
    probe_global_phase : complex
        Accumulated global phase exp(-i * sum_j delta_j * tau_j / 2) from the
        delta * |e><e| Hamiltonian.  Multiply probe-subspace amplitudes by this
        factor when computing expectation values in the three-level system.
    """
    # Local imports to avoid circular dependencies
    from quantum_pulse_suite.core.multilevel import (
        MultiLevelFreeEvolution, MultiLevelInstantPulse, MultiLevelContinuousPulse,
    )
    from quantum_pulse_suite.core.pulse_sequence import (
        FreeEvolution, InstantaneousPulse, ContinuousPulse,
    )

    f, g = 1.0 + 0j, 0.0 + 0j
    probe_global_phase = 1.0 + 0j   # accumulates exp(-i delta tau/2) per segment

    for element in seq.elements:

        # ── Free evolution ───────────────────────────────────────────────────
        if isinstance(element, (MultiLevelFreeEvolution, FreeEvolution)):
            delta_total = element.delta + delta_extra
            tau = element.tau
            # SU(2) part of delta*|e><e|: diag(exp(+i d tau/2), exp(-i d tau/2))
            # Both f and g pick up the same phase factor:
            phase = np.exp(1j * delta_total * tau / 2)
            f, g = phase * f, phase * g
            # Global phase: delta*|e><e| = SU(2) part * exp(-i delta tau/2)
            probe_global_phase *= np.exp(-1j * delta_total * tau / 2)

        # ── Instantaneous pulse ──────────────────────────────────────────────
        elif isinstance(element, (MultiLevelInstantPulse, InstantaneousPulse)):
            axis = element.axis
            angle = element.angle
            ch = np.cos(angle / 2)
            sh = np.sin(angle / 2)
            if np.allclose(axis, [1, 0, 0]):
                f, g = (f * ch - np.conj(g) * sh,
                        g * ch + np.conj(f) * sh)
            elif np.allclose(axis, [0, 1, 0]):
                f, g = (f * ch - 1j * np.conj(g) * sh,
                        g * ch + 1j * np.conj(f) * sh)
            else:
                # General axis: use the stored 2×2 unitary
                U2 = (element.subspace_unitary()
                      if hasattr(element, 'subspace_unitary') else element.unitary())
                f, g = (U2[0, 0] * f + U2[0, 1] / 1j * np.conj(g),
                        U2[1, 0] / 1j * np.conj(f) + U2[1, 1] * g)

        # ── Continuous pulse ─────────────────────────────────────────────────
        elif isinstance(element, (MultiLevelContinuousPulse, ContinuousPulse)):
            omega = element.omega
            axis  = element.axis
            n_x, n_y, n_z = axis
            delta_total = element.delta + delta_extra
            tau  = element.tau
            # Effective Rabi frequency at total detuning
            rabi = np.sqrt(delta_total**2 + 2 * n_z * delta_total * omega + omega**2)
            if rabi < 1e-15:
                rabi = 1e-15
            ch = np.cos(rabi * tau / 2)
            sh = np.sin(rabi * tau / 2)
            # Common diagonal term of the SU(2) propagator
            diag = ch + 1j * (delta_total + n_z * omega) / rabi * sh
            off  = omega / rabi * sh
            f_new = diag * f + (-n_x + 1j * n_y) * off * np.conj(g)
            g_new = (n_x - 1j * n_y) * off * np.conj(f) + diag * g
            f, g = f_new, g_new
            # Global phase: delta*|e><e| = SU(2) part * exp(-i delta tau/2)
            probe_global_phase *= np.exp(-1j * delta_total * tau / 2)

    return f, g, probe_global_phase


def detuning_sensitivity(seq, M=None, psi0=None, delta=0.0, eps=1e-7):
    """
    Compute |partial_delta <M>|^2 for a three-level clock sequence.

    The laser detuning delta shifts ALL segments uniformly via
        H_delta = delta * |e><e|.
    This is equivalent to the traceless probe-subspace form delta/2*(|e><e|-|g><g|)
    plus a global probe phase exp(-i delta tau/2) per segment, which matters
    because the clock state |m> is not part of the probe subspace and therefore
    does not acquire this phase.
    The sensitivity at the chosen operating point is found by central
    finite differences on the exact CK propagator:
        partial_delta <M> approx (<M>(delta+eps) - <M>(delta-eps)) / (2 eps).

    Default observable M = sigma_y^{gm} = -i|g><m| + i|m><g| and
    default initial state psi0 = (|g> + |m>)/sqrt(2).

    Parameters
    ----------
    seq : MultiLevelPulseSequence
        Three-level probe sequence.
    M : np.ndarray, shape (d, d), optional
        Measurement observable.  Default: sigma_y on the |g>-|m> clock
        transition.
    psi0 : np.ndarray, shape (d,), optional
        Initial state vector.  Default: (|g> + |m>)/sqrt(2).
    delta : float
        Operating-point detuning at which to evaluate the slope (default 0).
    eps : float
        Finite-difference step size in detuning units (default 1e-7).

    Returns
    -------
    dM_ddelta : float
        partial_delta <M> at the operating point.
    sens_sq : float
        |partial_delta <M>|^2.
    """
    d   = seq.dim
    i_g, i_e = seq.subspace._levels
    i_m = next(i for i in range(d) if i not in (i_g, i_e))

    if psi0 is None:
        psi0 = np.zeros(d, dtype=complex)
        psi0[i_g] = 1.0 / np.sqrt(2)
        psi0[i_m] = 1.0 / np.sqrt(2)
    else:
        psi0 = np.asarray(psi0, dtype=complex)

    if M is None:
        M = np.zeros((d, d), dtype=complex)
        M[i_g, i_m] = -1j   # sigma_y^{gm}
        M[i_m, i_g] =  1j
    else:
        M = np.asarray(M, dtype=complex)

    def expectation(delta_extra):
        f, g, probe_phase = _probe_ck_at_delta(seq, delta_extra)
        psi_f = np.empty(d, dtype=complex)
        # Apply global probe phase: H_delta = delta*|e><e| shifts only |e>,
        # so the probe subspace picks up exp(-i delta tau/2) relative to |m>.
        psi_f[i_g] = probe_phase * (f               * psi0[i_g] + 1j * g     * psi0[i_e])
        psi_f[i_e] = probe_phase * (1j * np.conj(g) * psi0[i_g] + np.conj(f) * psi0[i_e])
        psi_f[i_m] = psi0[i_m]
        return float(np.real(psi_f.conj() @ M @ psi_f))

    dM_ddelta = (expectation(delta + eps) - expectation(delta - eps)) / (2.0 * eps)
    return dM_ddelta, dM_ddelta ** 2


def detuning_sensitivity_2level(seq, m_hat=None, r0=None, delta=0.0, eps=1e-7):
    """
    Compute |partial_delta <M>|^2 for a two-level qubit sequence.

    The laser detuning delta shifts ALL segments uniformly.
    <M>(delta) = m_hat . R(delta) . r0  where R(delta) is the SO(3) rotation
    matrix corresponding to the probe SU(2) propagator at total detuning delta.
    The slope is found via central finite differences.

    Parameters
    ----------
    seq : ContinuousPulseSequence or InstantaneousPulseSequence
        Two-level qubit pulse sequence.
    m_hat : array_like, shape (3,), optional
        Bloch vector of measurement observable M = m_hat . sigma.
        Default (0, 1, 0) = sigma_y.
    r0 : array_like, shape (3,), optional
        Bloch vector of initial state psi0.
        Default (1, 0, 0) for equal superposition (|0>+|1>)/sqrt(2).
    delta : float
        Operating-point detuning (default 0).
    eps : float
        Finite-difference step size (default 1e-7).

    Returns
    -------
    dM_ddelta : float
        partial_delta <M> at the operating point.
    sens_sq : float
        |partial_delta <M>|^2.
    """
    if m_hat is None:
        m_hat = np.array([0., 1., 0.])
    if r0 is None:
        r0 = np.array([1., 0., 0.])
    m_hat = np.asarray(m_hat, dtype=float)
    r0    = np.asarray(r0,    dtype=float)

    def expectation(delta_extra):
        f, g, _ = _probe_ck_at_delta(seq, delta_extra)
        # SU(2) → SO(3): U^dag sigma_j U = sum_k R_jk sigma_k
        a, b = np.real(f), np.imag(f)
        c, d_g = np.real(g), np.imag(g)
        R = np.array([
            [a**2 + c**2 - b**2 - d_g**2,  2*(c*d_g - a*b),            2*(a*d_g + b*c)          ],
            [2*(a*b + c*d_g),               a**2 + d_g**2 - b**2 - c**2, 2*(b*d_g - a*c)          ],
            [2*(b*c - a*d_g),               2*(a*c + b*d_g),             a**2 + b**2 - c**2 - d_g**2],
        ])
        return float(np.dot(m_hat, R @ r0))

    dM_ddelta = (expectation(delta + eps) - expectation(delta - eps)) / (2.0 * eps)
    return dM_ddelta, dM_ddelta ** 2


def _I_exp_local(omega, tau):
    """Analytic integral int_0^tau exp(-i*omega*s) ds.

    = (1 - exp(-i*omega*tau)) / (i*omega),  with limit tau as omega -> 0.
    """
    omega = np.asarray(omega, dtype=float)
    result = np.zeros(len(omega), dtype=complex)
    small = np.abs(omega) < 1e-12
    large = ~small
    result[small] = tau
    result[large] = (1.0 - np.exp(-1j * omega[large] * tau)) / (1j * omega[large])
    return result


def _I_cos_local(omega, Omega, tau):
    """Analytic integral int_0^tau cos(Omega*s) * exp(-i*omega*s) ds.

    = (1/2) * [I_exp_local(omega - Omega, tau) + I_exp_local(omega + Omega, tau)]
    """
    return 0.5 * (_I_exp_local(omega - Omega, tau) + _I_exp_local(omega + Omega, tau))


def _I_sin_local(omega, Omega, tau):
    """Analytic integral int_0^tau sin(Omega*s) * exp(-i*omega*s) ds.

    = (1/2i) * [I_exp_local(omega - Omega, tau) - I_exp_local(omega + Omega, tau)]
    """
    return 0.5j * (_I_exp_local(omega + Omega, tau) - _I_exp_local(omega - Omega, tau))


def analytic_filter(seq, omega_array, m_y=1.0):
    """
    Compute Fe(omega) via exact analytic Fourier integrals of the QSP recurrences.

    Uses the closed-form expressions for phi_j(s) = F_j*(s)*G_j(s) and
    chi_j(s) = |G_j(s)|^2 within each segment, derived from the recurrence
    relations:

      Equiangular (continuous drive, n_z=0, delta=0), with a = f_{j-1}, b = g_{j-1},
      phi_drive = arctan2(n_y, n_x):

          phi_j(s) = conj(a)*b * cos(Omega*s)
                   + [exp(-i*phi)*conj(a)^2 - exp(+i*phi)*b^2]/2 * sin(Omega*s)

          chi_j(s) = 1/2 + (|b|^2-|a|^2)/2 * cos(Omega*s)
                   + Re[exp(+i*phi)*a*b] * sin(Omega*s)

      Pulsed QSP free evolution (Omega=0, delta=0): phi_j and chi_j are
      *constant* (the detuning phase cancels in F*G):

          phi_j(s) = conj(a)*b
          chi_j(s) = |b|^2

    Both cases unify to the same formulas with Omega=0 for free evolution.
    The Fourier integrals are evaluated exactly using I_cos, I_sin, I_exp
    (see _I_cos_local, _I_sin_local, _I_exp_local).

    Valid for any frequency grid, produces smooth curves on dense log-spaced
    grids extending to arbitrarily low frequency.

    Parameters
    ----------
    seq : MultiLevelPulseSequence
    omega_array : array_like
        Frequencies at which to evaluate Fe (can be log-spaced).
    m_y : float
        y-component of measurement direction (typically 1.0).

    Returns
    -------
    omega_array : np.ndarray
    Fe : np.ndarray
        e-noise filter function Fe = m_y^2*|Chi|^2.
    """
    from .multilevel import (
        MultiLevelFreeEvolution,
        MultiLevelContinuousPulse,
        MultiLevelInstantPulse,
    )

    omega_array = np.asarray(omega_array, dtype=float)
    n_w = len(omega_array)
    # Evaluate at +omega and -omega to build the correct filter kernel:
    #   H(w) = Re[F(T)]*Chi(w) + G(T)/2*Phi(w) + conj(G(T))/2*conj(Phi(-w))
    omega_both = np.concatenate([omega_array, -omega_array])
    Phi = np.zeros(2 * n_w, dtype=complex)
    Chi = np.zeros(2 * n_w, dtype=complex)
    _oa = omega_both   # alias used in loop body

    a = 1.0 + 0j   # f_{j-1}; equals F(T) after loop
    b = 0.0 + 0j   # g_{j-1}; equals G(T) after loop
    t_start = 0.0

    for elem in seq.elements:
        if isinstance(elem, MultiLevelInstantPulse):
            # Zero-duration: update CK amplitudes, no Fourier contribution.
            theta = elem.angle
            c_h, s_h = np.cos(theta / 2), np.sin(theta / 2)
            axis = elem.axis
            if np.allclose(axis, [1, 0, 0]):
                a, b = (a * c_h - np.conj(b) * s_h,
                        b * c_h + np.conj(a) * s_h)
            elif np.allclose(axis, [0, 1, 0]):
                a, b = (a * c_h - 1j * np.conj(b) * s_h,
                        b * c_h + 1j * np.conj(a) * s_h)
            else:
                U = elem.subspace_unitary()
                a, b = (U[0, 0] * a + U[0, 1] / 1j * np.conj(b),
                        U[1, 0] / 1j * np.conj(a) + U[1, 1] * b)

        elif isinstance(elem, MultiLevelFreeEvolution):
            tau = elem.tau
            if tau < 1e-15:
                continue
            # Omega=0: phi_j = conj(a)*b (constant), chi_j = |b|^2 (constant).
            # Unified formula with Omega=0: I_cos(w,0,tau)=I_exp(w,tau), I_sin=0.
            phase = np.exp(-1j * _oa * t_start)
            I_e = _I_exp_local(_oa, tau)
            Phi += phase * np.conj(a) * b * I_e
            Chi += phase * (abs(b) ** 2) * I_e
            # CK amplitudes unchanged for delta=0; phase factor if delta != 0.
            delta = elem.delta
            if abs(delta) > 1e-15:
                ph = np.exp(1j * delta * tau / 2)
                a, b = ph * a, ph * b
            t_start += tau

        elif isinstance(elem, MultiLevelContinuousPulse):
            tau = elem.tau
            if tau < 1e-15:
                continue
            Omega = elem.omega
            n_x, n_y, n_z = elem.axis
            delta = elem.delta

            if abs(n_z) < 1e-12 and abs(delta) < 1e-12:
                # Pure xy-plane drive, no detuning: exact analytic formulas.
                phi_d = np.arctan2(n_y, n_x)          # drive phase
                ep = np.exp(-1j * phi_d)               # e^{-i*phi}
                em = np.exp(+1j * phi_d)               # e^{+i*phi}

                # Coefficients for phi_j(s) = P_phi*cos(Omega*s) + Q_phi*sin(Omega*s)
                P_phi = np.conj(a) * b
                Q_phi = 0.5 * (ep * np.conj(a) ** 2 - em * b ** 2)

                # Coefficients for chi_j(s) = A*1 + B*cos(Omega*s) + C*sin(Omega*s)
                A_chi = 0.5 * (abs(a) ** 2 + abs(b) ** 2)   # = 0.5 by normalization
                B_chi = 0.5 * (abs(b) ** 2 - abs(a) ** 2)
                C_chi = float(np.real(em * a * b))           # real scalar

                phase = np.exp(-1j * _oa * t_start)
                I_c = _I_cos_local(_oa, Omega, tau)
                I_s = _I_sin_local(_oa, Omega, tau)
                I_e = _I_exp_local(_oa, tau)

                Phi += phase * (P_phi * I_c + Q_phi * I_s)
                Chi += phase * (A_chi * I_e + B_chi * I_c + C_chi * I_s)

                # Update CK amplitudes at end of segment.
                c_h = np.cos(Omega * tau / 2)
                s_h = np.sin(Omega * tau / 2)
                a, b = (c_h * a - ep * s_h * np.conj(b),
                        c_h * b + ep * s_h * np.conj(a))

            else:
                # General case (n_z != 0 or delta != 0): use dense time sampling.
                rabi = elem.effective_rabi
                rz = (delta + n_z * Omega) / rabi
                rperp_neg = (-n_x + 1j * n_y) * Omega / rabi
                a0, b0 = a, b
                t0 = t_start
                n_local = max(512, int(tau * rabi * 16 / (2 * np.pi)) + 64)
                ts = np.linspace(0, tau, n_local, endpoint=False)
                dt = tau / n_local
                c_r = np.cos(rabi * ts / 2)
                s_r = np.sin(rabi * ts / 2)
                F_s = (c_r + 1j * rz * s_r) * a0 + rperp_neg * s_r * np.conj(b0)
                G_s = np.conj(rperp_neg) * s_r * np.conj(a0) + (c_r + 1j * rz * s_r) * b0
                phi_s = np.conj(F_s) * G_s
                chi_s = np.abs(G_s) ** 2
                phas = np.exp(-1j * np.outer(_oa, t0 + ts))
                Phi += (phas @ phi_s) * dt
                Chi += (phas @ chi_s) * dt
                # Update CK amplitudes via subspace unitary.
                U = elem.subspace_unitary()
                a, b = (U[0, 0] * a + U[0, 1] / 1j * np.conj(b),
                        U[1, 0] / 1j * np.conj(a) + U[1, 1] * b)

            t_start += tau

    # After the loop: a = F(T), b = G(T)
    # H(omega) = Re[F(T)]*Chi(omega) + (G(T)*Phi(omega) + G(T)*.conj()*Phi(-omega)*.conj()) / 2
    # Split Chi and Phi into +omega and -omega halves.
    Chi_pos  = Chi[:n_w]      # Chi evaluated at +omega_array
    Chi_neg  = Chi[n_w:]      # Chi evaluated at -omega_array  (unused but symmetric check)
    Phi_pos  = Phi[:n_w]      # Phi evaluated at +omega_array
    Phi_neg  = Phi[n_w:]      # Phi evaluated at -omega_array
    H = np.real(a) * Chi_pos + 0.5 * (b * Phi_pos + np.conj(b) * np.conj(Phi_neg))
    Fe = np.abs(H) ** 2
    return omega_array, Fe


def raised_cosine_filter(seq, omega_array, m_y=1.0, n_per_segment=1024):
    """
    Compute Fe(omega) for a raised-cosine Rabi envelope version of seq.

    Each continuous drive segment j is reshaped from a square (constant Ω)
    pulse to a raised-cosine envelope with the **same total rotation angle**
    Θ_j = Ω_j * τ_j::

        Ω_j(t) = (Θ_j / τ_j) * [1 − cos(2π(t−t_{j−1})/τ_j)]

    so the mean Rabi frequency equals the original Ω_j and the CK amplitudes
    at segment boundaries are **identical** to those of the square-pulse
    sequence (the signal slope is preserved).

    Free-evolution and instant-pulse elements are unchanged.

    Implementation: dense midpoint-quadrature with ``n_per_segment`` points.
    For validation of the Jacobi–Anger closed form, see
    ``raised_cosine_filter_analytic``.

    Parameters
    ----------
    seq : MultiLevelPulseSequence
    omega_array : array_like
    m_y : float
    n_per_segment : int
        Number of quadrature points per continuous-drive segment (default 1024).

    Returns
    -------
    omega_array : np.ndarray
    Fe : np.ndarray
    """
    from .multilevel import (
        MultiLevelFreeEvolution,
        MultiLevelContinuousPulse,
        MultiLevelInstantPulse,
    )

    omega_array = np.asarray(omega_array, dtype=float)
    n_w = len(omega_array)
    Phi = np.zeros(n_w, dtype=complex)
    Chi = np.zeros(n_w, dtype=complex)

    a = 1.0 + 0j   # f_{j-1}
    b = 0.0 + 0j   # g_{j-1}
    t_start = 0.0

    for elem in seq.elements:
        if isinstance(elem, MultiLevelInstantPulse):
            theta = elem.angle
            c_h, s_h = np.cos(theta / 2), np.sin(theta / 2)
            axis = elem.axis
            if np.allclose(axis, [1, 0, 0]):
                a, b = (a * c_h - np.conj(b) * s_h,
                        b * c_h + np.conj(a) * s_h)
            elif np.allclose(axis, [0, 1, 0]):
                a, b = (a * c_h - 1j * np.conj(b) * s_h,
                        b * c_h + 1j * np.conj(a) * s_h)
            else:
                U = elem.subspace_unitary()
                a, b = (U[0, 0] * a + U[0, 1] / 1j * np.conj(b),
                        U[1, 0] / 1j * np.conj(a) + U[1, 1] * b)

        elif isinstance(elem, MultiLevelFreeEvolution):
            tau = elem.tau
            if tau < 1e-15:
                continue
            phase = np.exp(-1j * omega_array * t_start)
            I_e = _I_exp_local(omega_array, tau)
            Phi += phase * np.conj(a) * b * I_e
            Chi += phase * (abs(b) ** 2) * I_e
            delta = elem.delta
            if abs(delta) > 1e-15:
                ph = np.exp(1j * delta * tau / 2)
                a, b = ph * a, ph * b
            t_start += tau

        elif isinstance(elem, MultiLevelContinuousPulse):
            tau = elem.tau
            if tau < 1e-15:
                continue
            n_x, n_y, _ = elem.axis
            phi_d = np.arctan2(n_y, n_x)
            ep = np.exp(-1j * phi_d)   # e^{-i*phi_d}
            Omega_mean = elem.omega    # original (mean) Rabi frequency
            # Total rotation angle: Θ = Ω_mean * τ  (same for square and RC)

            # Midpoint quadrature over [0, τ]
            n_pts = n_per_segment
            ts = (np.arange(n_pts) + 0.5) * (tau / n_pts)   # midpoints
            dt = tau / n_pts

            # Raised-cosine cumulative area:
            #   h(s) = (Θ/τ)*s  −  (Θ/2π)*sin(2π*s/τ)
            # ∫_0^τ h'(s) ds = Θ  (total rotation preserved)
            Theta = Omega_mean * tau
            hs = Omega_mean * ts - (Theta / (2.0 * np.pi)) * np.sin(2.0 * np.pi * ts / tau)

            # CK amplitudes at local time s, starting from (a, b):
            #   F(s) = cos(h/2)*a − e^{-iφ}*sin(h/2)*conj(b)
            #   G(s) = cos(h/2)*b + e^{+iφ}*sin(h/2)*conj(a)
            c_hs = np.cos(hs / 2.0)
            s_hs = np.sin(hs / 2.0)
            F_s = c_hs * a - ep * s_hs * np.conj(b)
            G_s = c_hs * b + ep * s_hs * np.conj(a)

            phi_s = np.conj(F_s) * G_s    # F*(s)*G(s)
            chi_s = np.abs(G_s) ** 2      # |G(s)|^2

            # Contribution to Phi and Chi
            phas = np.exp(-1j * np.outer(omega_array, t_start + ts))
            Phi += (phas @ phi_s) * dt
            Chi += (phas @ chi_s) * dt

            # End-of-segment CK update: h(τ) = Θ (sin(2π) = 0)
            c_end = np.cos(Theta / 2.0)
            s_end = np.sin(Theta / 2.0)
            a, b = (c_end * a - ep * s_end * np.conj(b),
                    c_end * b + ep * s_end * np.conj(a))

            t_start += tau

    Fe = m_y ** 2 * np.abs(Chi) ** 2
    return omega_array, Fe


def _jacobi_anger_integrals(Theta, tau, omega_array, n_terms=30):
    """
    Jacobi–Anger analytic integrals for the raised-cosine envelope.

    Computes the complex integral

        Ĩ_c(ω) + i*Ĩ_s(ω)  =  ∫_0^τ exp(i*h(s)) * exp(−iωs) ds

    where h(s) = (Θ/τ)s − (Θ/2π)*sin(2πs/τ) is the raised-cosine
    cumulative area, using the Jacobi–Anger expansion

        exp(−iz·sin(ψ)) = Σ_n  J_n(z) * exp(−inψ)

    to give (paper Eq. JA_result):

        Ĩ_c + i·Ĩ_s = τ * Σ_n  J_n(Θ/2π) * sinc(ξ_n) * exp(i*ξ_n)

    where  ξ_n = [(Θ/2π − n)*π − ω*τ/2].

    The sum is truncated at |n| ≤ n_terms (Bessel functions decay super-
    exponentially once |n| ≫ |Θ/2π|).

    Parameters
    ----------
    Theta : float
        Total rotation angle of the segment.
    tau : float
        Segment duration.
    omega_array : np.ndarray, shape (n_w,)
    n_terms : int
        Truncation order for the Bessel series.

    Returns
    -------
    I_cs : np.ndarray, shape (n_w,), complex
        Ĩ_c(ω) + i*Ĩ_s(ω) at each frequency.
    """
    from scipy.special import jn

    omega_array = np.asarray(omega_array, dtype=float)
    z = Theta / (2.0 * np.pi)     # argument of Bessel functions
    ns = np.arange(-n_terms, n_terms + 1)   # integers summed over

    # ξ_n = (z − n)*π − ω*τ/2,  shape (n_terms, n_w)
    xi = (z - ns[:, None]) * np.pi - 0.5 * tau * omega_array[None, :]   # (2n+1, n_w)

    # sinc with limit: sin(x)/x → 1 as x → 0
    with np.errstate(invalid='ignore', divide='ignore'):
        sinc_xi = np.where(np.abs(xi) < 1e-14, 1.0, np.sin(xi) / xi)

    # Bessel coefficients: J_n(z), shape (2n+1,)
    Jn = np.array([jn(int(n), z) for n in ns], dtype=float)

    # Sum: τ * Σ_n J_n(z) * sinc(ξ_n) * exp(i*ξ_n)
    I_cs = tau * np.sum(Jn[:, None] * sinc_xi * np.exp(1j * xi), axis=0)
    return I_cs


def raised_cosine_filter_analytic(seq, omega_array, m_y=1.0,
                                   n_terms=30, n_chi_pts=512):
    """
    Compute Fe(omega) for raised-cosine envelopes using the Jacobi–Anger
    analytic formula for the probe channel Φ(ω), and dense quadrature for
    the clock channel χ(ω).

    This implements Eq. (JA_result) from the paper (Sec.~\\ref{sec:pulseshaping})
    for the probe contribution, and validates it against ``raised_cosine_filter``
    (which uses pure quadrature throughout).

    Parameters
    ----------
    seq : MultiLevelPulseSequence
    omega_array : array_like
    m_y : float
    n_terms : int
        Bessel truncation order for Jacobi–Anger (default 30).
    n_chi_pts : int
        Quadrature points per segment for the clock-channel integral (default 512).

    Returns
    -------
    omega_array : np.ndarray
    Fe : np.ndarray
    Phi : np.ndarray, complex
        Probe path-function integral (useful for validation).
    Chi : np.ndarray, complex
        Clock channel integral (useful for validation).
    """
    from .multilevel import (
        MultiLevelFreeEvolution,
        MultiLevelContinuousPulse,
        MultiLevelInstantPulse,
    )

    omega_array = np.asarray(omega_array, dtype=float)
    n_w = len(omega_array)
    Phi = np.zeros(n_w, dtype=complex)
    Chi = np.zeros(n_w, dtype=complex)

    a = 1.0 + 0j
    b = 0.0 + 0j
    t_start = 0.0

    for elem in seq.elements:
        if isinstance(elem, MultiLevelInstantPulse):
            theta = elem.angle
            c_h, s_h = np.cos(theta / 2), np.sin(theta / 2)
            axis = elem.axis
            if np.allclose(axis, [1, 0, 0]):
                a, b = (a * c_h - np.conj(b) * s_h,
                        b * c_h + np.conj(a) * s_h)
            elif np.allclose(axis, [0, 1, 0]):
                a, b = (a * c_h - 1j * np.conj(b) * s_h,
                        b * c_h + 1j * np.conj(a) * s_h)
            else:
                U = elem.subspace_unitary()
                a, b = (U[0, 0] * a + U[0, 1] / 1j * np.conj(b),
                        U[1, 0] / 1j * np.conj(a) + U[1, 1] * b)

        elif isinstance(elem, MultiLevelFreeEvolution):
            tau = elem.tau
            if tau < 1e-15:
                continue
            phase = np.exp(-1j * omega_array * t_start)
            I_e = _I_exp_local(omega_array, tau)
            Phi += phase * np.conj(a) * b * I_e
            Chi += phase * (abs(b) ** 2) * I_e
            delta = elem.delta
            if abs(delta) > 1e-15:
                ph = np.exp(1j * delta * tau / 2)
                a, b = ph * a, ph * b
            t_start += tau

        elif isinstance(elem, MultiLevelContinuousPulse):
            tau = elem.tau
            if tau < 1e-15:
                continue
            n_x, n_y, _ = elem.axis
            phi_d = np.arctan2(n_y, n_x)
            ep = np.exp(-1j * phi_d)
            Omega_mean = elem.omega
            Theta = Omega_mean * tau

            phase = np.exp(-1j * omega_array * t_start)

            # ── Probe channel: Jacobi–Anger analytic formula ─────────────────
            # The path function within segment j is:
            #   φ_j(s) = P_phi * cos(h(s)) + Q_phi * sin(h(s))
            # so its Fourier contribution is:
            #   Φ_j(ω) = phase * [P_phi * Ĩ_c(ω) + Q_phi * Ĩ_s(ω)]
            # where Ĩ_c = ∫ cos(h) e^{-iωs} ds and Ĩ_s = ∫ sin(h) e^{-iωs} ds.
            #
            # The Jacobi–Anger expansion gives:
            #   I_cs_pos = ∫ e^{+ih(s)} e^{-iωs} ds  = Ĩ_c + i*Ĩ_s
            #   I_cs_neg = ∫ e^{-ih(s)} e^{-iωs} ds  = Ĩ_c - i*Ĩ_s
            # so:
            #   Ĩ_c = (I_cs_pos + I_cs_neg) / 2
            #   Ĩ_s = (I_cs_pos - I_cs_neg) / (2i)
            # which gives the compact form:
            #   P*Ĩ_c + Q*Ĩ_s = (P - iQ)/2 * I_cs_pos + (P + iQ)/2 * I_cs_neg
            I_cs_pos = _jacobi_anger_integrals( Theta, tau, omega_array, n_terms)
            I_cs_neg = _jacobi_anger_integrals(-Theta, tau, omega_array, n_terms)

            P_phi = np.conj(a) * b
            Q_phi = 0.5 * (ep * np.conj(a) ** 2 - np.conj(ep) * b ** 2)
            Phi += phase * (0.5 * (P_phi - 1j * Q_phi) * I_cs_pos
                          + 0.5 * (P_phi + 1j * Q_phi) * I_cs_neg)

            # ── Clock channel: dense quadrature ───────────────────────────────
            ts = (np.arange(n_chi_pts) + 0.5) * (tau / n_chi_pts)
            dt = tau / n_chi_pts
            hs = Omega_mean * ts - (Theta / (2.0 * np.pi)) * np.sin(2.0 * np.pi * ts / tau)
            c_hs = np.cos(hs / 2.0)
            s_hs = np.sin(hs / 2.0)
            G_s = c_hs * b + ep * s_hs * np.conj(a)
            chi_s = np.abs(G_s) ** 2
            phas = np.exp(-1j * np.outer(omega_array, t_start + ts))
            Chi += (phas @ chi_s) * dt

            # ── End-of-segment CK update ──────────────────────────────────────
            c_end = np.cos(Theta / 2.0)
            s_end = np.sin(Theta / 2.0)
            a, b = (c_end * a - ep * s_end * np.conj(b),
                    c_end * b + ep * s_end * np.conj(a))

            t_start += tau

    Fe = m_y ** 2 * np.abs(Chi) ** 2
    return omega_array, Fe, Phi, Chi


def three_level_noise_variance(Fe, Ff, frequencies, S_e, S_f,
                               omega_cutoff=None, T=None):
    """
    Compute total noise variance <dM^2> from filter functions and PSDs.

    <dM^2> = (1/2pi) int [S_be(w) Fe(w) + S_bf(w) Ff(w)] dw

    Parameters
    ----------
    Fe : np.ndarray
        e-noise filter function.
    Ff : np.ndarray
        f-noise filter function.
    frequencies : np.ndarray
        Angular frequencies (positive, one-sided).
    S_e : callable or np.ndarray
        Noise PSD for beta_e. If callable, evaluated at frequencies.
    S_f : callable or np.ndarray
        Noise PSD for beta_f. If callable, evaluated at frequencies.

    omega_cutoff : float or None, optional
        Lower integration cutoff in angular-frequency units.  If None and
        ``T`` is provided, uses the Fourier limit 2*pi/T.  If both are None,
        integrates over the full supplied grid.
    T : float or None, optional
        Sequence duration used to resolve the default cutoff when
        ``omega_cutoff`` is None.

    Returns
    -------
    float
        Total noise variance.
    """
    w = np.asarray(frequencies)

    Se_vals = S_e(w) if callable(S_e) else np.asarray(S_e)
    Sf_vals = S_f(w) if callable(S_f) else np.asarray(S_f)

    omega_min = resolve_omega_cutoff(T, omega_cutoff) if T is not None or omega_cutoff is not None else 0.0
    mask = w >= omega_min
    if np.count_nonzero(mask) < 2:
        return 0.0
    integrand = Se_vals * Fe + Sf_vals * Ff
    return float(simpson(integrand[mask], x=w[mask]) / (2 * np.pi))
