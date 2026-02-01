# This module provides geometric operations for basin coordinates.
# It re-exports canonical functions - NO LOCAL RE-IMPLEMENTATIONS.

import numpy as np
from .canonical import fisher_rao_distance, frechet_mean, validate_basin  # noqa: F401 (re-exported)

__all__ = ['fisher_rao_distance', 'frechet_mean', 'validate_basin', 'to_simplex', 'bhattacharyya_coefficient']


def to_simplex(p):
    """Projects a vector to the probability simplex."""
    p = np.abs(p)
    return p / p.sum()


def bhattacharyya_coefficient(p, q):
    """Compute the Bhattacharyya coefficient between two probability distributions."""
    return np.sum(np.sqrt(p * q))
