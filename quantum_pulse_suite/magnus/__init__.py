"""
Magnus expansion module for computing filter functions to arbitrary order.

This module provides tools for computing higher-order corrections to the
filter function using the Magnus expansion of the time evolution operator.

Theory
------
The time-ordered exponential U(t) = T exp(-i int_0^t H(t') dt') can be written as:

    U(t) = exp(Omega(t))

where Omega(t) = Omega_1(t) + Omega_2(t) + Omega_3(t) + ...

First order (standard filter function):
    Omega_1 = -i int_0^t H(t') dt'

Second order:
    Omega_2 = -1/2 int_0^t dt1 int_0^t1 dt2 [H(t1), H(t2)]

Higher orders involve nested commutators with known recursive formulas.

For noise sensitivity, we expand H(t) = H_0(t) + epsilon * V(t) where V is
the noise operator. The filter function emerges from the first-order term,
while higher orders capture non-linear noise effects.

References
----------
- Magnus, W. (1954). "On the exponential solution of differential equations
  for a linear operator." Comm. Pure Appl. Math. 7, 649-673.
- Blanes et al. (2009). "The Magnus expansion and some of its applications."
  Physics Reports 470, 151-238.
"""

from .expansion import (
    MagnusExpansion,
    first_order_term,
    second_order_term,
)

from .filter_function_orders import (
    FilterFunctionOrder,
    FirstOrderFilterFunction,
    SecondOrderFilterFunction,
)

__all__ = [
    'MagnusExpansion',
    'first_order_term',
    'second_order_term',
    'FilterFunctionOrder',
    'FirstOrderFilterFunction',
    'SecondOrderFilterFunction',
]
