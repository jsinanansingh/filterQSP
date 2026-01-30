"""
Analysis module for quantum pulse sequence suite.

This module exports classes for numerical evolution and trajectory analysis.
"""

from .batch_numerical import (
    bloch_vector_from_operator,
    NumericalEvolution,
    TrajectoryAnalyzer,
)

__all__ = [
    'bloch_vector_from_operator',
    'NumericalEvolution',
    'TrajectoryAnalyzer',
]
