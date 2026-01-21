# This file was created to resolve a ModuleNotFoundError.
# It should contain the geometric operations required by other modules.

import numpy as np
from qig_geometry import fisher_rao_distance
from qig_geometry import canonical_frechet_mean as frechet_mean

def to_simplex(p):
    """Projects a vector to the probability simplex."""
    p = np.abs(p)
    return p / p.sum()

def fisher_rao_distance(p, q):
    """Compute the Fisher-Rao distance between two probability distributions."""
    return np.arccos(np.sum(np.sqrt(p * q)))

def frechet_mean(basins):
    """Compute the Fréchet mean of a set of basins."""
    # This is a simplified implementation. A proper implementation would involve
    # an iterative optimization process.
    return to_simplex(frechet_mean(basins))

def bhattacharyya_coefficient(p, q):
    """Compute the Bhattacharyya coefficient between two probability distributions."""
    return np.sum(np.sqrt(p * q))
