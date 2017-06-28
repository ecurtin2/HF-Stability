from scipy import special
import numpy as np


def f2D(k, k_fermi):
    """Return the value of the function f2D from Guiliani and Vignale, pg 81."""
    y = np.linalg.norm(k) / k_fermi
    if y <= 1.0:
        return special.ellipe(y)
    else:
        return y * (special.ellipe(1.0 / y) - (1.0 - (1.0 / y**2)) * special.ellipk(1.0 / y))


def f3D(k, k_fermi):
    """Return the value of the function f3D from Guiliani and Vignale, pg 81."""
    y = np.linalg.norm(k) / k_fermi
    return 0.5 + (1.0 - y**2) / (4 * y) * np.log((1.0 + y) / (1.0 - y))


def exchange_energy(k, k_fermi, n_dimensions):
    """Return the analytic exchange energy for state with momentum k."""
    N = - 2.0 * k_fermi / np.pi
    if n_dimensions == 2:
        return N * f2D(k, k_fermi)
    elif n_dimensions == 3:
        return N * f3D(k, k_fermi)
    else:
        raise ValueError('Only 2 and 3 dimensions currently supported, got {}.'.format(str(n_dimensions)))


def kinetic_energy(k):
    return np.sum(k**2) / 2.0  


def total_energy(k, k_fermi, n_dimensions):
    return kinetic_energy(k) + exchange_energy(k, k_fermi, n_dimensions)