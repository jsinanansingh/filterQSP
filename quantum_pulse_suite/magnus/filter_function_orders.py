"""
Higher-order filter function computations via Magnus expansion.

This module provides filter function calculations beyond first order,
capturing non-linear noise effects and cross-correlations.

Theory
------
The standard filter function F(omega) describes the linear response to noise.
Higher-order terms capture:

1. Second-order detuning noise: How fluctuations in delta couple at O(delta^2)
2. Second-order amplitude noise: How Rabi frequency fluctuations couple at O(Omega^2)
3. Cross terms: Coupling between different noise channels

The variance of an observable B due to noise is:

    Var(B) = integral |F^(1)(omega)|^2 S(omega) d omega / (2 pi)       [First order]
           + integral integral F^(2)(omega1, omega2) S(omega1) S(omega2) ... [Second order]
           + ...

References
----------
- Green et al., "Arbitrary quantum control of qubits in the presence of
  universal noise" (2013) - Multi-frequency filter functions
- Cywiński et al., "How to enhance dephasing time in superconducting qubits"
  (2008) - Higher-order dephasing
"""

from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class FilterFunctionOrder(ABC):
    """
    Abstract base class for filter function calculations at different orders.

    Each order captures different physics:
    - First order: Linear response to noise
    - Second order: Quadratic response, noise cross-correlations
    - Higher orders: Non-Gaussian effects, strong noise regime
    """

    @property
    @abstractmethod
    def order(self) -> int:
        """The order of this filter function."""
        pass

    @abstractmethod
    def compute(self, frequencies: np.ndarray) -> np.ndarray:
        """
        Compute the filter function at given frequencies.

        Parameters
        ----------
        frequencies : np.ndarray
            Angular frequencies omega

        Returns
        -------
        np.ndarray
            Filter function values
        """
        pass

    @abstractmethod
    def noise_susceptibility(self, frequencies: np.ndarray,
                             psd_func: Callable[[np.ndarray], np.ndarray]) -> float:
        """
        Compute the noise susceptibility (contribution to variance).

        Parameters
        ----------
        frequencies : np.ndarray
            Frequencies for integration
        psd_func : callable
            Power spectral density S(omega)

        Returns
        -------
        float
            Contribution to observable variance
        """
        pass


class FirstOrderFilterFunction(FilterFunctionOrder):
    """
    First-order (standard) filter function.

    This wraps the existing filter function implementations to fit
    the FilterFunctionOrder interface.

    Parameters
    ----------
    ff : FilterFunction
        Existing filter function calculator from core module
    """

    def __init__(self, ff: 'FilterFunction'):
        self._ff = ff

    @property
    def order(self) -> int:
        return 1

    def compute(self, frequencies: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute first-order filter function.

        Returns
        -------
        tuple
            (Fx, Fy, Fz) filter function components
        """
        return self._ff.filter_function(frequencies)

    def noise_susceptibility(self, frequencies: np.ndarray,
                             psd_func: Optional[Callable] = None) -> np.ndarray:
        """
        Compute |F(omega)|^2.

        If psd_func is provided, returns integral of |F|^2 * S.
        Otherwise returns |F|^2 array.
        """
        susceptibility = self._ff.noise_susceptibility(frequencies)

        if psd_func is not None:
            from scipy.integrate import simpson
            psd_values = psd_func(frequencies)
            integrand = susceptibility * psd_values
            return simpson(integrand, x=frequencies) / (2 * np.pi)

        return susceptibility


class SecondOrderFilterFunction(FilterFunctionOrder):
    """
    Second-order filter function for quadratic noise response.

    This captures effects like:
    - Quadratic dephasing: Var(B) ~ integral |F^(2)|^2 S^2
    - Cross-frequency correlations
    - Non-linear pulse imperfections

    Parameters
    ----------
    sequence_params : list
        Pulse sequence parameters
    noise_type : str
        Type of noise: 'dephasing' or 'amplitude'

    Notes
    -----
    The second-order filter function F^(2)(omega1, omega2) is generally
    a function of two frequencies, representing how noise at omega1 and
    omega2 jointly affect the observable.

    For single-frequency evaluation, we typically use:
    - Diagonal: F^(2)(omega, omega) for self-correlation
    - Integrated: integral F^(2)(omega, omega') d omega'

    TODO: Full implementation of second-order terms. Currently placeholder.
    """

    def __init__(self, sequence_params: list, noise_type: str = 'dephasing'):
        self._params = sequence_params
        self._noise_type = noise_type
        self._computed = False
        self._cache = {}

    @property
    def order(self) -> int:
        return 2

    def compute(self, frequencies: np.ndarray) -> np.ndarray:
        """
        Compute second-order filter function (diagonal approximation).

        Returns F^(2)(omega, omega) - the self-correlation term.

        Parameters
        ----------
        frequencies : np.ndarray
            Angular frequencies

        Returns
        -------
        np.ndarray
            Second-order filter function values

        Notes
        -----
        PLACEHOLDER IMPLEMENTATION
        Returns zeros until proper formulas are implemented.

        The full formula for second-order dephasing involves:
        F^(2)_dephasing(omega) = integral of terms like
            [U^dag(t1) sigma_z U(t1), U^dag(t2) sigma_z U(t2)]
        which requires tracking the unitary trajectory.
        """
        # TODO: Implement proper second-order filter function
        # This requires:
        # 1. Computing U(t) trajectory
        # 2. Transforming noise operators to interaction picture
        # 3. Computing nested integrals of commutators

        # Placeholder: return zeros
        return np.zeros_like(frequencies)

    def compute_two_frequency(self, omega1: np.ndarray,
                               omega2: np.ndarray) -> np.ndarray:
        """
        Compute full two-frequency second-order filter function.

        F^(2)(omega1, omega2) captures correlations between noise
        at different frequencies.

        Parameters
        ----------
        omega1, omega2 : np.ndarray
            Two frequency arrays (can be meshgrid)

        Returns
        -------
        np.ndarray
            F^(2)(omega1, omega2) matrix

        Notes
        -----
        PLACEHOLDER IMPLEMENTATION
        """
        # TODO: Implement two-frequency filter function
        shape = np.broadcast_shapes(omega1.shape, omega2.shape)
        return np.zeros(shape, dtype=complex)

    def noise_susceptibility(self, frequencies: np.ndarray,
                             psd_func: Optional[Callable] = None) -> Union[np.ndarray, float]:
        """
        Compute second-order noise susceptibility.

        For second order, this involves a double integral:
        integral integral |F^(2)(omega1, omega2)|^2 S(omega1) S(omega2) d omega1 d omega2

        Parameters
        ----------
        frequencies : np.ndarray
            Frequencies for integration
        psd_func : callable, optional
            Power spectral density

        Returns
        -------
        float or np.ndarray
            Second-order contribution to variance
        """
        F2 = self.compute(frequencies)

        if psd_func is not None:
            # For diagonal approximation: integral |F^(2)(omega)|^2 S(omega)^2
            from scipy.integrate import simpson
            psd_values = psd_func(frequencies)
            integrand = np.abs(F2)**2 * psd_values**2
            return simpson(integrand, x=frequencies) / (2 * np.pi)**2

        return np.abs(F2)**2


class SecondOrderDephasingFilter(SecondOrderFilterFunction):
    """
    Second-order filter function specifically for dephasing (detuning) noise.

    For a pulse sequence with time-dependent detuning delta(t) = delta_0 + epsilon(t),
    the second-order term captures how fluctuations epsilon(t) contribute
    quadratically to phase accumulation.

    Physical interpretation:
    - First order: Phase error ~ integral epsilon(t) * g(t) dt
    - Second order: Phase error ~ integral integral epsilon(t1) epsilon(t2) * K(t1,t2) dt1 dt2

    where g(t) is the sensitivity function and K(t1,t2) is the second-order kernel.

    TODO: Implement formulas from Cywiński et al. or similar references.
    """

    def __init__(self, sequence_params: list):
        super().__init__(sequence_params, noise_type='dephasing')

    def compute(self, frequencies: np.ndarray) -> np.ndarray:
        """
        Compute second-order dephasing filter function.

        PLACEHOLDER: Returns estimate based on first-order scaling.

        The actual formula involves:
        F^(2)_z(omega) ~ d/d(delta) F^(1)_z(omega, delta) |_{delta=0}
        """
        # TODO: Implement proper second-order dephasing formula
        # Placeholder: scale first-order by frequency (rough approximation)
        return np.zeros_like(frequencies)


class SecondOrderAmplitudeFilter(SecondOrderFilterFunction):
    """
    Second-order filter function for amplitude (Rabi frequency) noise.

    For a pulse with Omega(t) = Omega_0 + epsilon(t), this captures
    how Rabi frequency fluctuations contribute quadratically.

    Physical effects:
    - Pulse area errors that don't average out
    - Rotation axis wobble
    - Dressed state energy shifts

    TODO: Implement formulas specific to amplitude noise.
    """

    def __init__(self, sequence_params: list):
        super().__init__(sequence_params, noise_type='amplitude')

    def compute(self, frequencies: np.ndarray) -> np.ndarray:
        """
        Compute second-order amplitude filter function.

        PLACEHOLDER IMPLEMENTATION
        """
        # TODO: Implement proper second-order amplitude formula
        return np.zeros_like(frequencies)


# =============================================================================
# Combined multi-order filter function
# =============================================================================

class MultiOrderFilterFunction:
    """
    Combined filter function calculator supporting multiple orders.

    This class manages filter functions at different orders and combines
    them for total noise susceptibility calculations.

    Parameters
    ----------
    ff_first : FilterFunction
        First-order filter function
    max_order : int
        Maximum order to include (1, 2, or 3)

    Examples
    --------
    >>> ff = sequence.get_filter_function_calculator()
    >>> multi_ff = MultiOrderFilterFunction(ff, max_order=2)
    >>> total_var = multi_ff.total_variance(frequencies, psd_func)
    """

    def __init__(self, ff_first: 'FilterFunction', max_order: int = 1):
        self.max_order = max_order
        self._orders: Dict[int, FilterFunctionOrder] = {}

        # First order from existing implementation
        self._orders[1] = FirstOrderFilterFunction(ff_first)

        # Higher orders (placeholders for now)
        if max_order >= 2:
            # TODO: Pass proper parameters
            self._orders[2] = SecondOrderFilterFunction([], 'dephasing')

    def get_order(self, n: int) -> FilterFunctionOrder:
        """Get filter function calculator for order n."""
        if n not in self._orders:
            raise ValueError(f"Order {n} not available. Max order: {self.max_order}")
        return self._orders[n]

    def total_susceptibility(self, frequencies: np.ndarray,
                             psd_func: Callable[[np.ndarray], np.ndarray]) -> float:
        """
        Compute total noise susceptibility including all orders.

        Parameters
        ----------
        frequencies : np.ndarray
            Frequencies for integration
        psd_func : callable
            Noise power spectral density

        Returns
        -------
        float
            Total variance contribution from all orders
        """
        total = 0.0
        for order, ff in self._orders.items():
            total += ff.noise_susceptibility(frequencies, psd_func)
        return total
