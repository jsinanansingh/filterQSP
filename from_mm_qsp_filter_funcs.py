import numpy as np
from scipy.linalg import expm

# Define Pauli matrices
sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
pauli = [sigma_x, sigma_y, sigma_z]

def R(n, theta):
    """Rotation operator using matrix exponential"""
    # n . {sigma_x, sigma_y, sigma_z} = n[0]*sigma_x + n[1]*sigma_y + n[2]*sigma_z
    sigma_dot = sum(n[i] * pauli[i] for i in range(3))
    return expm(1j * sigma_dot * theta / 2)

def axis(theta, phi):
    """Convert spherical coordinates to Cartesian unit vector"""
    return np.array([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta)
    ])

# Define standard axes
x = axis(np.pi/2, 0)
y = axis(np.pi/2, np.pi/2)
z = axis(0, 0)

def h(Omega, n, Delta):
    """Hamiltonian vector"""
    return Omega * n + Delta * np.array([0, 0, 1])

def u(Omega, n, tau, Delta):
    """Time evolution operator"""
    return R(h(Omega, n, Delta), tau)

def q(l, Gamma):
    """Product of evolution operators up to index l"""
    n = len(Gamma)
    if l > 0:
        mats = [u(*params) for params in Gamma[:l]]
    else:
        mats = [u(0, np.array([0, 0, 0]), 0, 0)]
    
    # Compute product from right to left (reverse order for matrix multiplication)
    result = mats[-1]
    for mat in reversed(mats[:-1]):
        result = mat @ result
    return result

def ul(l, Gamma, t):
    """Evolution operator for l-th segment at time t"""
    Omega, n, tau, Delta = Gamma[l-1]  # l is 1-indexed in Mathematica
    tau_s = [0] + [g[2] for g in Gamma]  # Prepend 0 to tau values
    
    t_offset = t - sum(tau_s[:l])
    return u(Omega, n, t_offset, Delta)

def gl(Gamma, t):
    """Step functions indicating which segment is active at time t"""
    n = len(Gamma)
    tau_s = [0] + [g[2] for g in Gamma]
    
    def f(l):
        # UnitStep in Mathematica: 1 if arg >= 0, else 0
        t_start = sum(tau_s[:l+1])
        t_end = tau_s[l+1] + sum(tau_s[:l+1])
        step1 = 1 if t >= t_start else 0
        step2 = 1 if t >= t_end else 0
        return step1 - step2
    
    return np.array([f(l) for l in range(n)])

def bt(B, Gamma):
    """Transform B vector through full evolution"""
    n = len(Gamma)
    q_n = q(n, Gamma)
    
    # B . {sigma_x, sigma_y, sigma_z}
    B_sigma = sum(B[i] * pauli[i] for i in range(3))
    
    # Conjugate transformation
    sigma = q_n.conj().T @ B_sigma @ q_n
    
    # Project onto Pauli basis: 1/2 Tr[sigma . pauli_i]
    return np.array([0.5 * np.trace(sigma @ p).real for p in pauli])

def Rz(Gamma, t):
    """Time-dependent expectation value of rotated Pauli matrices"""
    n = len(Gamma)
    g = gl(Gamma, t)
    
    # Sum over active segments
    sigma = np.zeros((2, 2), dtype=complex)
    for l in range(n):
        if g[l] != 0:
            q_prev = q(l, Gamma)  # q[l-1] in Mathematica (0-indexed)
            ul_t = ul(l+1, Gamma, t)  # ul uses 1-indexing
            
            # Conjugate transformation
            term = q_prev.conj().T @ ul_t.conj().T @ sigma_z @ ul_t @ q_prev
            sigma += g[l] * term
    
    # Project onto Pauli basis
    return np.array([0.5 * np.trace(sigma @ p).real for p in pauli])


def ROmega_t(Gamma, t):
    """Time-dependent expectation value with drive field n"""
    n = len(Gamma)
    g = gl(Gamma, t)
    
    # Sum over active segments
    sigma = np.zeros((2, 2), dtype=complex)
    for l in range(n):
        if g[l] != 0:
            q_prev = q(l, Gamma)
            ul_t = ul(l+1, Gamma, t)
            
            # n . {sigma_x, sigma_y, sigma_z} where n is Gamma[l][1]
            n_sigma = sum(Gamma[l][1][i] * pauli[i] for i in range(3))
            
            # Conjugate transformation
            term = q_prev.conj().T @ ul_t.conj().T @ n_sigma @ ul_t @ q_prev
            sigma += g[l] * term
    
    # Project onto Pauli basis
    return np.array([0.5 * np.trace(sigma @ p) for p in pauli])


def R_Phi(Gamma, omega):
    """Fourier transform of time derivative of Rz"""
    from scipy.integrate import quad
    
    # Compute total duration
    T_total = sum(g[2] for g in Gamma)
    
    def integrand_real(t, i):
        # Derivative of Rz[Gamma, t][i]
        dt = 1e-8
        deriv = (Rz(Gamma, t + dt)[i] - Rz(Gamma, t - dt)[i]) / (2 * dt)
        return np.real(deriv * np.exp(-1j * omega * t))
    
    def integrand_imag(t, i):
        dt = 1e-8
        deriv = (Rz(Gamma, t + dt)[i] - Rz(Gamma, t - dt)[i]) / (2 * dt)
        return np.imag(deriv * np.exp(-1j * omega * t))
    
    result = np.zeros(3, dtype=complex)
    for i in range(3):
        real_part, _ = quad(integrand_real, 0, T_total, args=(i,))
        imag_part, _ = quad(integrand_imag, 0, T_total, args=(i,))
        result[i] = real_part + 1j * imag_part
    
    return result


def R_Omega(Gamma, omega):
    """Fourier transform of ROmega_t"""
    from scipy.integrate import quad
    
    # Compute total duration
    T_total = sum(g[2] for g in Gamma)
    
    def integrand_real(t, i):
        return np.real(ROmega_t(Gamma, t)[i] * np.exp(-1j * omega * t))
    
    def integrand_imag(t, i):
        return np.imag(ROmega_t(Gamma, t)[i] * np.exp(-1j * omega * t))
    
    result = np.zeros(3, dtype=complex)
    for i in range(3):
        real_part, _ = quad(integrand_real, 0, T_total, args=(i,))
        imag_part, _ = quad(integrand_imag, 0, T_total, args=(i,))
        result[i] = real_part + 1j * imag_part
    
    return result


def F_Phi(Gamma, omega, B):
    """Filter function for phase noise using R_Phi"""
    rw = R_Phi(Gamma, omega)
    # Conjugate[rw] . Transpose[rw] - Conjugate[B . rw] * B . rw
    return np.conj(rw) @ rw - np.conj(B @ rw) * (B @ rw)


def F_Omega(Gamma, omega, B):
    """Filter function for amplitude noise using R_Omega"""
    rw = R_Omega(Gamma, omega)
    return np.conj(rw) @ rw - np.conj(B @ rw) * (B @ rw)


def F(Gamma, omega, B, Rw_func):
    """Generic filter function with custom Rw function
    
    Args:
        Gamma: pulse sequence parameters
        omega: frequency
        B: projection vector
        Rw_func: function that takes (Gamma, omega) and returns response
    """
    rw = Rw_func(Gamma, omega)
    return np.conj(rw) @ rw - np.conj(B @ rw) * (B @ rw)


# Example usage:
if __name__ == "__main__":
    # Example: single rotation pulse
    Gamma_example = [
        (1.0, np.array([1, 0, 0]), np.pi, 0.5),  # (Omega, n, tau, Delta)
    ]
    
    print("Pauli X matrix:")
    print(sigma_x)
    print("\nRotation around x-axis by pi/2:")
    print(R(x, np.pi/2))
    print("\nbt with B=[0,0,1]:")
    print(bt(np.array([0, 0, 1]), Gamma_example))
    print("\nROmega_t at t=pi/2:")
    print(ROmega_t(Gamma_example, np.pi/2))
    
    # Note: Fourier transform functions can be slow for numerical integration
    # For faster computation, consider using FFT on discretized time grid
    
    # Reproduce the Mathematica plots
    print("\n" + "="*50)
    print("Computing filter functions for different sequences...")
    print("="*50)
    
    import matplotlib.pyplot as plt
    
    # Define omega range (log-spaced for efficiency)
    omega_vals = np.logspace(-1, 2, 100)  # 0.1 to 100
    
    # GPS-1: Single pulse with Omega=1
    print("Computing GPS-1 (Omega=1)...")
    Gamma_gps1 = [(1, x, 2*np.pi, 0)]
    B_gps1 = bt(np.array([0, 1, 0]), Gamma_gps1)
    fgps1_phi = np.array([F_Phi(Gamma_gps1, w, B_gps1) / w**2 for w in omega_vals])
    
    # GPS-8: Single pulse with Omega=8
    print("Computing GPS-8 (Omega=8)...")
    Gamma_gps8 = [(8, x, 2*np.pi, 0)]
    B_gps8 = bt(np.array([0, 1, 0]), Gamma_gps8)
    fgps8_phi = np.array([F_Phi(Gamma_gps8, w, B_gps8) / w**2 for w in omega_vals])
    
    # Rabi: Single pulse with detuning
    print("Computing Rabi sequence...")
    Gamma_rabi = [(0.5, x, 2*np.pi, 3/4/2)]
    B_rabi = bt(np.array([0, 0, 1]), Gamma_rabi)
    frabi_phi = np.array([F_Phi(Gamma_rabi, w, B_rabi) / w**2 for w in omega_vals])
    
    # Ramsey: Three-pulse sequence
    print("Computing Ramsey sequence...")
    tRam = 1e-2
    Delta = 1/4
    Gamma_ram = [
        (np.pi/tRam, x, tRam, 0),
        (0, np.array([0, 0, 0]), 2*np.pi, Delta),
        (np.pi/tRam, -x, tRam, 0)
    ]
    B_ram = bt(np.array([0, 1, 0]), Gamma_ram)
    fram_phi = np.array([F_Phi(Gamma_ram, w, B_ram) / w**2 for w in omega_vals])
    
    # Create log-log plot
    plt.figure(figsize=(10, 7))
    plt.loglog(omega_vals, np.abs(fgps1_phi), 'b-', label='GPS-1 (Ω=1)', linewidth=2)
    plt.loglog(omega_vals, np.abs(fgps8_phi), 'r-', label='GPS-8 (Ω=8)', linewidth=2)
    plt.loglog(omega_vals, np.abs(frabi_phi), 'g-', label='Rabi (Δ≠0)', linewidth=2)
    plt.loglog(omega_vals, np.abs(fram_phi), 'm-', label='Ramsey', linewidth=2)
    
    plt.xlabel('ω (frequency)', fontsize=12)
    plt.ylabel('F_Φ(ω) / ω²', fontsize=12)
    plt.title('Filter Functions for Different Pulse Sequences', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.ylim(1e-8, 1e2)
    plt.xlim(omega_vals[0], omega_vals[-1])
    
    plt.tight_layout()
    plt.savefig('filter_functions.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved as 'filter_functions.png'")
    plt.show()