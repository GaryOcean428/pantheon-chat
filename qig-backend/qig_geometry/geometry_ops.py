# This module provides geometric operations for basin coordinates.
# It should contain the geometric operations required by other modules.

import numpy as np
from .canonical import fisher_rao_distance


def to_simplex(p):
    """Projects a vector to the probability simplex."""
    p = np.abs(p)
    return p / p.sum()


def frechet_mean(basins):
    """Compute the Fréchet mean of a set of basins.
    
    This is a simplified implementation using the arithmetic mean in sqrt-space.
    """
    if len(basins) == 0:
        raise ValueError("Cannot compute Fréchet mean of empty set")
    
    # Convert to sqrt-space
    sqrt_basins = [np.sqrt(np.abs(b)) for b in basins]
    
    # Compute arithmetic mean in sqrt-space
    mean_sqrt = np.mean(sqrt_basins, axis=0)
    
    # Convert back to probability space
    mean = mean_sqrt ** 2
    
    # Normalize to simplex
    return to_simplex(mean)


def bhattacharyya_coefficient(p, q):
    """Compute the Bhattacharyya coefficient between two probability distributions."""
    return np.sum(np.sqrt(p * q))
