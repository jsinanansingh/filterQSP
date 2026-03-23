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
    # 3-level Lambda clock: signal = -Im[probe_phase * F(T)]  (probe_phase ≈ 1 at delta=0)
    # Sensitivity kernel: h(t) = Re[F(T)] * chi(t) - Re[conj(G(T)) * phi(t)]
    # => H(omega) = Re[F(T)]*Chi(omega) - (conj(b)*Phi(omega) + b*conj(Phi(-omega))) / 2
    # Sign and b/conj(b) swap verified by DC-consistency: H(0)^2 = sens_sq for all sequences.
    # Split Chi and Phi into +omega and -omega halves.
    Chi_pos  = Chi[:n_w]      # Chi evaluated at +omega_array
    Chi_neg  = Chi[n_w:]      # Chi evaluated at -omega_array  (unused but symmetric check)
    Phi_pos  = Phi[:n_w]      # Phi evaluated at +omega_array
    Phi_neg  = Phi[n_w:]      # Phi evaluated at -omega_array
    H = np.real(a) * Chi_pos - 0.5 * (np.conj(b) * Phi_pos + b * np.conj(Phi_neg))
    Fe = np.abs(H) ** 2
    return omega_array, Fe


def gps_shaped_filter(envelope_fn, T, omega_mean, method='direct',
                      n_samples=16384, pad_factor=4, seq=None, m_y=1.0):
    """
    Filter function Fe(ω) for a single-polarisation shaped GPS drive.

    Because all H(t) = Ω(t)/2 · σ_x^{ge} commute, the propagator is exactly

        U(t) = exp(−i·θ(t)/2 · σ_x),   θ(t) = ∫₀ᵗ Ω(s) ds

    and the sensitivity trajectory r(t) = ⟨ψ₀|[M_I(T), A_e(t)]|ψ₀⟩ reduces to

        r(t) = (i/2) · ( cos(Θ/2) − cos(θ(t) − Θ/2) )

    where Θ = θ(T) = Ω_mean·T is the total rotation angle.  Fe(ω) = |FT[r]|².

    **Validity**: only for δ = 0 (no laser detuning).  With δ ≠ 0 the
    Hamiltonians no longer commute; use method='piecewise' in that case.

    Parameters
    ----------
    envelope_fn : callable
        ``envelope_fn(t_array, T, omega_mean) -> Ω(t_array)``.
        The Rabi-frequency envelope as a function of time.
    T : float
        Total sequence duration.
    omega_mean : float
        Mean Rabi frequency (sets the total rotation Θ = omega_mean * T).
    method : {'direct', 'piecewise'}
        ``'direct'``
            Evaluates θ(t) by cumulative midpoint-rectangle integration of
            the continuous Ω(t), then computes r(t) analytically and FFTs.
            No piecewise-constant approximation of the envelope.
        ``'piecewise'``
            Delegates to ``fft_three_level_filter(seq, ...)``.  Requires
            ``seq`` to be a pre-built ``MultiLevelPulseSequence``.  Use for
            cross-validation or when δ ≠ 0.
    n_samples : int
        Uniform time samples for the direct FFT (default 16384) or time
        samples passed to fft_three_level_filter for the piecewise method.
    pad_factor : int
        Zero-padding factor for the FFT (default 4).
    seq : MultiLevelPulseSequence or None
        Required when method='piecewise'.
    m_y : float
        Observable weight passed to fft_three_level_filter (piecewise only).

    Returns
    -------
    freqs : ndarray
        Positive angular frequencies (rad/s).
    Fe : ndarray
        Filter function |FT[r(t)]|² at each frequency.
    """
    if method == 'piecewise':
        if seq is None:
            raise ValueError("method='piecewise' requires seq to be provided.")
        freqs, Fe, _, _ = fft_three_level_filter(
            seq, n_samples=n_samples, pad_factor=pad_factor, m_y=m_y)
        return freqs, Fe

    if method != 'direct':
        raise ValueError(f"method must be 'direct' or 'piecewise', got {method!r}")

    # ------------------------------------------------------------------
    # Direct method: continuous envelope, no piecewise approximation.
    # ------------------------------------------------------------------
    dt = T / n_samples
    # Evaluate envelope at interval midpoints for midpoint-rectangle rule
    t_mid = (np.arange(n_samples) + 0.5) * dt
    Omega_vals = np.asarray(envelope_fn(t_mid, T, omega_mean), dtype=float)

    # Cumulative rotation angle θ at the end of each interval (midpoint rule)
    theta_end = np.cumsum(Omega_vals) * dt
    # θ at each midpoint: average of the surrounding endpoints
    theta_start = np.concatenate([[0.0], theta_end[:-1]])
    theta = 0.5 * (theta_start + theta_end)   # θ(t_mid[j])

    # Total rotation angle Θ = θ(T)
    Theta = theta_end[-1]

    # Analytic sensitivity trajectory (Eq. derived from commuting H(t)):
    #   r(t) = (i/2) · (cos(Θ/2) − cos(θ(t) − Θ/2))
    r = 0.5j * (np.cos(Theta * 0.5) - np.cos(theta - Theta * 0.5))

    # Zero-pad and FFT
    n_padded = pad_factor * n_samples
    r_padded = np.zeros(n_padded, dtype=complex)
    r_padded[:n_samples] = r
    r_fft = np.fft.fft(r_padded) * dt
    freqs_all = 2.0 * np.pi * np.fft.fftfreq(n_padded, d=dt)

    pos = freqs_all > 0
    return freqs_all[pos], np.abs(r_fft[pos]) ** 2

