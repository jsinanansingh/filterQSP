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
