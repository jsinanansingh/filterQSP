"""
Three-level clock filter functions.

Computes the noise variance <dM^2> for a three-level Lambda clock system where:
- QSP control acts on {|g>, |e>} subspace (probe transition)
- Initial state is (|g> + |f>)/sqrt(2)
- Measurement observable M = m . sigma_gf acts on {|g>, |f>} manifold
- Two independent noise sources: beta_e(t) on |e><e| and beta_f(t) on |f><f|

The variance decomposes as (Eq 75 from draft):
    <dM^2> = (1/hbar^2) int dw/2pi [S_be(w) Fe(w) + S_bf(w) Ff(w)]
F
with two filter functions:
- Fe(w) = 1/2 * (1 + 2*m_z*m_x) * |Phi(w)|^2  + (m_x^2 + m_y^2) * |int_0^T dt sum_j W_j(t) |G_j(t)|^2 e^{-iwt}|^2 (Eq 78)
- Ff(w) = (m_z^2 - 1) * (1 - cos(w*T)) / w^2          (Eq 76)
- Phi(w) = int_0^T dt sum_j W_j(t) F*_j(t) G_j(t) e^{-iwt}  (Eq 79)
"""

import numpy as np
from scipy.integrate import simpson, quad


def Ff_analytic(frequencies, T, m_z):
    """
    Compute the f-noise filter function Ff(w).

    Protocol-independent filter function for noise on the |f> state.
        Ff(w) = (m_z^2 - 1) * (1 - cos(w*T)) / w^2

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
        result[small] = (m_z**2 - 1) * T**2 / 2
    if np.any(large):
        result[large] = (m_z**2 - 1) * (1 - np.cos(w[large] * T)) / w[large]**2

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
                           m_x=0.0, m_y=0.0, m_z=0.0):
    """
    Compute three-level clock filter functions via FFT.

    Samples phi(t) = F*(t) G(t) and chi(t) = |G(t)|^2 on a time grid,
    zero-pads, and FFTs to obtain Phi(w) and Chi(w). Then constructs:

        Fe(w) = 1/2*(1 + 2*m_z*m_x)*|Phi(w)|^2 + (m_x^2 + m_y^2)*|Chi(w)|^2

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
    Fe : np.ndarray
        e-noise filter function at each frequency.
    Ff : np.ndarray
        f-noise filter function at each frequency.
    Phi : np.ndarray
        Core spectral quantity Phi(w) at each frequency.
    """
    T = seq.total_duration()
    dt = T / n_samples
    subspace = seq.subspace
    elements = seq.elements
    d = seq.dim

    # Precompute segment boundaries and cumulative unitaries for efficiency
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

            # Project to 2x2 subspace
            U_sub = subspace.project_operator(U_t)

            # Extract Cayley-Klein parameters: U_sub = [[f, ig], [ig*, f*]]
            f_t = U_sub[0, 0]
            g_t = U_sub[0, 1] / 1j

            phi_samples[idx] = np.conj(f_t) * g_t
            chi_samples[idx] = np.abs(g_t)**2

    # Zero-pad and FFT
    n_padded = pad_factor * n_samples
    phi_padded = np.zeros(n_padded, dtype=complex)
    phi_padded[:n_samples] = phi_samples
    chi_padded = np.zeros(n_padded, dtype=float)
    chi_padded[:n_samples] = chi_samples

    # FFT convention: Phi(w) = int phi(t) e^{-iwt} dt ~ dt * sum phi[n] e^{-iwt_n}
    Phi_fft = np.fft.fft(phi_padded) * dt
    Chi_fft = np.fft.fft(chi_padded) * dt
    freqs = 2 * np.pi * np.fft.fftfreq(n_padded, d=dt)

    # Positive frequencies only
    pos_mask = freqs > 0
    freqs_pos = freqs[pos_mask]
    Phi_pos = Phi_fft[pos_mask]
    Chi_pos = Chi_fft[pos_mask]

    # Construct Fe(w) = 1/2*(1 + 2*m_z*m_x)*|Phi|^2 + (m_x^2 + m_y^2)*|Chi|^2
    Fe = (0.5 * (1 + 2 * m_z * m_x) * np.abs(Phi_pos)**2
          + (m_x**2 + m_y**2) * np.abs(Chi_pos)**2)

    # Construct Ff(w)
    Ff = Ff_analytic(freqs_pos, T, m_z)

    return freqs_pos, Fe, Ff, Phi_pos


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

            # r(t) = ⟨ψ₀|[M, H̃]|ψ₀⟩  (M and H̃ are Hermitian → r is imaginary)
            MH = M @ H_tilde
            commutator = MH - MH.conj().T        # = [M, H̃]
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
        Components of measurement direction in {|g>, |f>} Bloch sphere.

    Returns
    -------
    frequencies : np.ndarray
        Angular frequencies (same as input).
    Fe : np.ndarray
        e-noise filter function at each frequency.
    Ff : np.ndarray
        f-noise filter function at each frequency.
    Phi : np.ndarray
        Core spectral quantity Phi(w) at each frequency.
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

    # Construct Fe(w) = 1/2*(1+2*m_z*m_x)*|Phi|^2 + (m_x^2+m_y^2)*|Chi|^2
    Fe = (0.5 * (1 + 2 * m_z * m_x) * np.abs(Phi)**2
          + (m_x**2 + m_y**2) * np.abs(Chi)**2)

    # Construct Ff(w)
    Ff = Ff_analytic(frequencies, T, m_z)

    return frequencies, Fe, Ff, Phi


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

    The laser detuning acts on every time-evolution segment as
        H_delta = delta/2 * (|e><e| - |g><g|)
    giving the probe SU(2) update
        diag(exp(+i delta tau/2),  exp(-i delta tau/2))
    for free evolution of duration tau.  Continuous-pulse elements use the
    full off-resonant Rabi formula with delta_total = element.delta + delta_extra.
    Instantaneous pulses are delta-independent.

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
        Final Cayley-Klein amplitudes of the probe subspace.
    """
    # Local imports to avoid circular dependencies
    from quantum_pulse_suite.core.multilevel import (
        MultiLevelFreeEvolution, MultiLevelInstantPulse, MultiLevelContinuousPulse,
    )
    from quantum_pulse_suite.core.pulse_sequence import (
        FreeEvolution, InstantaneousPulse, ContinuousPulse,
    )

    f, g = 1.0 + 0j, 0.0 + 0j

    for element in seq.elements:

        # ── Free evolution ───────────────────────────────────────────────────
        if isinstance(element, (MultiLevelFreeEvolution, FreeEvolution)):
            delta_total = element.delta + delta_extra
            tau = element.tau
            # U_free = diag(exp(+i delta_total tau/2),  exp(-i delta_total tau/2))
            # Both f and g pick up the same phase factor (see compute_polynomials):
            phase = np.exp(1j * delta_total * tau / 2)
            f, g = phase * f, phase * g

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

    return f, g


def detuning_sensitivity(seq, M=None, psi0=None, delta=0.0, eps=1e-7):
    """
    Compute |partial_delta <M>|^2 for a three-level clock sequence.

    The laser detuning delta shifts ALL segments uniformly via
        H_delta = delta/2 * (|e><e| - |g><g|).
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
        f, g = _probe_ck_at_delta(seq, delta_extra)
        psi_f = np.empty(d, dtype=complex)
        psi_f[i_g] = f               * psi0[i_g] + 1j * g           * psi0[i_e]
        psi_f[i_e] = 1j * np.conj(g) * psi0[i_g] + np.conj(f)       * psi0[i_e]
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
        f, g = _probe_ck_at_delta(seq, delta_extra)
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


def three_level_noise_variance(Fe, Ff, frequencies, S_e, S_f):
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

    Returns
    -------
    float
        Total noise variance.
    """
    w = np.asarray(frequencies)

    Se_vals = S_e(w) if callable(S_e) else np.asarray(S_e)
    Sf_vals = S_f(w) if callable(S_f) else np.asarray(S_f)

    integrand = Se_vals * Fe + Sf_vals * Ff
    return float(simpson(integrand, x=w) / (2 * np.pi))
