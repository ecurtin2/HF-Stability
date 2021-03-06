import constants

import functools
import numpy as np
from scipy.special import expi


class TwoElectronIntegral(object):
    """An object for controlling the Two Electron Integrals.

    This contains functions to evaluate all the various two electron integrals that will be needed in the
    evaluation of the code. It controls which exact two electron integral is to be used and controls the normalization.
    The member function self.eval(k1, k2, k3, k4) is the exposed method that can be called to evaluate the two electron
    integral.
    """

    def __init__(self, parameters):
        self.eval = __class__._two_eri_not_set
        self.parameters = parameters
        self.prefactor = None
        if not hasattr(self.parameters, 'volume'):
            raise AttributeError('Parameters must have volume defined before instantiating.')
        self._set_eval(safe=self.parameters.safe_eri)

    def _set_eval(self, safe=True):
        """Update two electron integral with information from states object.

        Once this is called, the member function self.eval(*args) can be used to evaluate the two
        electron integral."""

        if self.parameters.n_dimensions == 1:
            if hasattr(self.parameters, "delta_fxn_magnitude"):
                self.eval = functools.partial(self._1d_delta_potential,
                                              self.parameters.delta_fxn_magnitude)
            elif hasattr(self.parameters, "cylinder_radius"):
                self.eval = functools.partial(self._1d_coulomb_potential
                                              , self.parameters.cylinder_radius)
            else:
                raise AttributeError('For 1D case, must set either delta_fxn_magnitude or cylinder_radius'
                                     + ' attributes in parameters instance.')

        elif self.parameters.n_dimensions == 2:
            # CORRECT ONE:
            self.prefactor = np.pi * 2.0 / self.parameters.volume
            # INCORRECT
            # prefactor = np.pi / self.parameters.volume
            self.eval = functools.partial(self._2d, self.prefactor)
        elif self.parameters.n_dimensions == 3:
            self.prefactor = np.pi * 4.0 / self.parameters.volume
            self.eval = functools.partial(self._3d, self.prefactor)
        else:
            raise ValueError('parameters must have attribute n_dimensions = 1, 2 or 3.')

        if safe:
            self.eval = self.eval_safe(self.eval)

    def eval_safe(self, two_electron_func):
        """Return the value of the two electron repulsion integral iff the states conserve momentum.
        returns <1 2 | 1 / r_12 | 3 4>

        :param two_electron_func: function taking (k1, k2, k3, k4) and returning the value of the two electron
        integral.
        """
        def wrapper(k1, k2, k3, k4, *args, **kwargs):
            k = k1 + k2 - (k3 + k4)
            k = self.parameters.to_first_brillouin_zone(k)
            if np.linalg.norm(k) < constants.SMALL_NUMBER:
                return two_electron_func(k1, k2, k3, k4, *args, **kwargs)
            else:
                return 0.0
        return wrapper

    @staticmethod
    def _two_eri_not_set(*args, **kwargs):
        """Dummy function to raise AttributError if two_eri was not set."""
        raise AttributeError('Two electron integral not set properly, maybe need to update with '
                             + 'states?')

    def _1d_delta_potential(self, V0, k1, k2, k3, k4):
        """Return the value of the two electron integral in one dimension with delta function potential.

        V(r12) = delta(r12) * V0
        """
        return V0

    def _1d_coulomb_potential(self, cylinder_radius, k1, k2, k3, k4):
        """Return the value of the two electron integral in one dimension for the given radius of cylinder.

        See appendix of Guiliani & Vignale Quantum Theory of the Electron Liquid for details.
        """
        k = self.parameters.to_first_brillouin_zone(k1 - k3)
        arg = k**2 * cylinder_radius**2

        # arg = 30.0 evaluates to -3.02 e-15 and only gets smaller as arg increases. This leads to some
        # numerical issues so set to 0.
        if np.any(arg < constants.SMALL_NUMBER) or np.any(arg > 30.0):
            return 0.0
        else:
            val = np.exp(arg) * expi(-arg)
            return val

    def _2d(self, prefactor, k1, k2, k3, k4):
        """Return the value of the two electron integral in two dimensions."""
        # denominator = np.sum(np.abs(k1 - k3))
        k = self.parameters.to_first_brillouin_zone(k1 - k3)
        denominator = np.linalg.norm(k)
        if denominator < constants.SMALL_NUMBER:
            return 0.0
        else:
            return prefactor / denominator

    def _3d(self, prefactor, k1, k2, k3, k4):
        """Return the value of the two electron integral in three dimensions."""
        k = self.parameters.to_first_brillouin_zone(k1 - k3)
        denominator = np.sum(k**2)

        if denominator < constants.SMALL_NUMBER:
            return 0.0
        else:
            return prefactor / denominator
