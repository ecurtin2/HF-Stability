import numpy as np

import constants
import excitations
import prettyprint
import states
import twoERI


class Parameters(object):
    """Object for the physical parameters of the system, like wigner-seitz Radius (rs) and number
    of dimensions. """

    def __init__(self, *, rs=1.2, n_k_points=12, n_dimensions=2, instability_type='cRHF2cUHF',
                 cylinder_radius=None, delta_fxn_magnitude=None, safe_eri=True):
        """Create the parameters object for the given parameters.

        :param rs: The wigner seitz radius, in 1 / bohr
        :type rs: float
        :param n_k_points: The number of reciprocal space grid points. Must be > 2
        :type n_k_points: int
        :param n_dimensions: Number of physical dimensions: 1, 2 or 3.
        :type n_dimensions: int
        :param instability_type: The type of instability to be investigated. Must be one of the strings in the set __class__.instabilities_supported.
        :type instability_type: string
        :param cylinder_radius: The radius of the cylinder for the pseudo - 1 dimensional system. This is meant to be small such that motion in the radial dimension is 'frozen'. See the appendix of Guiliani and Vignale for a discussion on this.
        :type cylinder_radius: float
        :param delta_fxn_magnitude: The multiplicative constant for the delta function potential in 1 dimension. ( v(r12) = delta_fxn_magnitude * delta(r12) )
        :type delta_fxn_magnitude: float
        :param safe_eri: Whether to check for momentum conservation explicitly in the two ERI.
        :type safe_eri: bool
        """

        self.rs = rs
        self.n_k_points = n_k_points
        self.n_dimensions = n_dimensions
        self.safe_eri = safe_eri

        if self.n_dimensions == 1:
            if (cylinder_radius is None) and (delta_fxn_magnitude is None):
                raise ValueError(
                    'In one dimension, must specify exactly one of cylinder_radius or delta_fxn_magnitude, got Neither.')
            elif (cylinder_radius is not None) and (delta_fxn_magnitude is not None):
                raise ValueError(
                    'In one dimension, must specify exactly one of cylinder_radius or delta_fxn_magnitude, got Both.')
            elif (cylinder_radius is not None) and (delta_fxn_magnitude is None):
                self.cylinder_radius = cylinder_radius
            elif (cylinder_radius is None) and (delta_fxn_magnitude is not None):
                self.delta_fxn_magnitude = delta_fxn_magnitude

        self.instability_type = instability_type
        self.k_fermi = __class__._calc_k_fermi(rs=self.rs, n_dimensions=self.n_dimensions)
        self.k_max = (2.0 + 1E-8) * self.k_fermi  # A bit offset from 2 helps with
                                                  # not getting states exactly at kmax.

        # Make a grid 1 point too big then remove the last. This gives the desired
        # number of points in the grid, while removing equivalent points at
        # opposite ends of the first Brillouin zone.
        self.k_grid = np.linspace(-self.k_max, self.k_max, self.n_k_points + 1)[:-1]
        self.k_grid_spacing = np.diff(self.k_grid)[0]

        self.states = states.States(self)
        self.n_electrons = 2 * self.states.n_occupied
        self.volume = __class__._calc_volume(self.rs, self.n_electrons, self.n_dimensions)

        self.eri = twoERI.TwoElectronIntegral(self)
        self.states.calc_energies()
        self.states.sort_by_energy()

        self.excitations = excitations.Excitations(self)

    def __str__(self):
        s = prettyprint.header("Parameters")
        slist = [str(key) + ' : ' + str(val) for key, val in self.__dict__.items()
                 if not (isinstance(val, states.States)
                         or isinstance(val, excitations.Excitations))]

        return s + '\n'.join(slist)

    @staticmethod
    def _calc_volume(rs, n_electrons, n_dimensions):
        """Calculate the volume of one unit cell."""
        if n_dimensions == 1:
            vol = n_electrons * 2.0 * rs
        elif n_dimensions == 2:
            vol = n_electrons * np.pi * rs**2
        elif n_dimensions == 3:
            vol = n_electrons * 4.0 / 3.0 * np.pi * rs**3
        else:
            raise AttributeError('self.n_dimensions has an invalid value: ', n_electrons, ' expected 1 2 or 3.')
        return vol

    @staticmethod
    def _calc_k_fermi(rs, n_dimensions):

        if n_dimensions == 1:
            return np.pi / (4.0 * rs)
        elif n_dimensions == 2:
            return 2.0 ** 0.5 / rs
        elif n_dimensions == 3:
            return (9.0 * np.pi / 4.0) ** (1.0 / 3.0) / rs
        else:
            raise ValueError('n_dimensions must be one of: 1, 2, 3 \nbut ' + str(n_dimensions) + ' was given.')

    def to_first_brillouin_zone(self, k):
        """Translate k to first brillouin zone, defined by k_max, in place

         :param k: Momentum vector of a state
         :type k: np.ndarray(len = n_dimensions)
         """

        # Note the equivalence of boolean array and 0, 1 array.
        k += 2.0 * self.k_max * (k < (-self.k_max - constants.SMALL_NUMBER))
        k -= 2.0 * self.k_max * (k > (self.k_max - constants.SMALL_NUMBER))
        return k