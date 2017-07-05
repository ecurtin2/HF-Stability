import itertools

import numpy as np

import slepc_wrapper
import twoERI


class OrbitalHessian(object):
    """Object to contain information about the Orbital Hessian, AKA the Stability matrix or electronic Hessian.
    Currently is implemented as numpy data structures but if needed should be relatively painless to use other
    data structures. Hey it's almost like it's actually encapsulated or something.
    """

    # How big is the matrix for each different case? This x number of excitations.
    case_size = {
        'cRHF2cRHF': 2,
        'cRHF2cUHF': 2,
        'cUHF2cUHF': 4,
        'cUHF2cGHF': 4
    }

    def __init__(self, parameters):
        """Use a parameters instance to create the orbital hessian."""
        self.parameters = parameters
        self.states = parameters.states
        self.excitations = parameters.excitations
        self.size = __class__.case_size[self.parameters.instability_type] * self.excitations.n
        self.A = A(self)
        self.B = B(self)

    def as_numpy(self):
        """Return the OrbitalHessian instance as a numpy array.
        No simplifications are made for the various symmetries, so the entire H matrix is generated always."""
        H = slepc_wrapper.PetscMatWrapper.row_generator_to_numpy(self.row_generator,
                                                                 self.size, self.size)
        return H

    def row_generator(self, i_row):
        N = self.size // 2

        # Switched the order of row_gen iterator in each case. This
        # may be better for contiguous access, but I'm not sure. It likely
        # is not significant.
        if i_row < N:
            A_gen = self.A.row_generator(i_row, offset=0)
            B_gen = self.B.row_generator(i_row, offset=N)
            row_gen = itertools.chain(A_gen, B_gen)
        else:
            i_row -= N
            A_gen = self.A.row_generator(i_row, offset=N)
            B_gen = self.B.row_generator(i_row, offset=0)
            row_gen = itertools.chain(B_gen, A_gen)
        return row_gen

    def as_PETSc(self):
        """Create and return reference to Petsc Sparse Parallel Matrix."""
        H = slepc_wrapper.PetscMatWrapper(self.size, self.size,
                                                       self.row_generator)
        return H

    def get_conserving_virtual(self, k_occ, k_vir, k_second_occ):
        """Given momenta of excitation k_occ -> k_vir and an occupied momenta, return the second virtual state.

        The pair of excitations k_occ -> k_vir and k_occ_2 -> k_vir_2
        :param k_occ: Momentum of occupied state.
        :param k_vir: The momenta of virtual state.
        :type k_occ: np.ndarray(len=Ndim)
        :type k_vir: np.ndarray(len=Ndim)
        :type k_occ_2: np.ndarray(len=Ndim)
        :returns k: The momentum of the virtual state of the momentum conserving second excitation.
        """
        k = k_occ + k_vir - k_second_occ
        twoERI.to_first_brillouin_zone(k, self.parameters.k_max)

        indices = np.round(k / self.parameters.k_grid_spacing)
        exc_label = self.excitations.label_from_indices(indices)
        return exc_label, k

    def lowest_eigenvalue(self, method='Numpy'):
        """Return the lowest eigenvalue of the Orbital Hessian

        Again currently only builds the full matrix and diagonalizes.
        """
        methods = {'SLEPc_Sparse': self._slepc_sparse_lowest_eigenvalue
                  ,'Numpy': self._numpy_lowest_eigenvalue}
        if method not in methods.keys():
            raise ValueError('Method is not available. Given = ' +
                             '{given}, must be one of {must}.'.format(given=method
                                                                      , must=set(methods.keys())))
        return methods[method]()

    def _slepc_sparse_lowest_eigenvalue(self):
        """Return the lowest eigenvalue using a sparse parallel PETSc Matrix."""
        H = self.as_PETSc()
        eps = slepc_wrapper.SlepcEPSWrapper(H)
        print(eps)
        return

    def _numpy_lowest_eigenvalue(self):
        """Return the lowest eigenvalue using dense numpy."""
        return np.amin(np.linalg.eigvals(self.as_numpy()))


class AorB(object):

    def __init__(self, orbitalhessian):
        """Abstract base class for orbital hessian sub-matrices A and B."""

        self.orbital_hessian = orbitalhessian
        self.parameters = orbitalhessian.parameters
        self.states = orbitalhessian.states
        self.excitations = orbitalhessian.excitations
        self.elmnt_from_momenta = self._gen_elmnt_function()
        self.n_rows = self.parameters.excitations.n
        self.n_cols = self.parameters.excitations.n

    def as_numpy(self):
        """Return the matrix in the form of a numpy array."""
        return slepc_wrapper.PetscMatWrapper.row_generator_to_numpy(self.row_generator,
                                                               self.n_rows, self.n_cols)

    def row_generator(self, i_row, offset=0):
        """Return a generator for the i'th row that makes (column index, value) pairs for nonzero elements."""
        raise NotImplementedError

    def momentum_conserving_pairs(self, k_i, k_a):
        """Given one excitation, return a generator for all complementary momentum conserving excitations."""
        raise NotImplementedError

    def _gen_elmnt_function(self):
        """Return the member function to get the matrix element based on instability type."""
        matrix_elmnt_dic = {
            'cRHF2cRHF': self.singlet_elmnt,
            'cRHF2cUHF': self.triplet_elmnt
        }
        return matrix_elmnt_dic[self.parameters.instability_type]

    def singlet_elmnt(self, ki, ka, kj, kb):
        """Return the value of the singlet matrix element for the ki -> ka, kj -> kb excitations.

        :param ki: Momentum of occupied state i
        :type ki: np.ndarray(len = # of dimensions)
        :param ka: Momentum of virtual state a
        :type ka: np.ndarray(len = # of dimensions)
        :param kj: Momentum of occupied state j
        :type kj: np.ndarray(len = # of dimensions)
        :param kb: Momentum of virtual state b
        :type kb: np.ndarray(len = # of dimensions)
        :returns: (float) the value of the matrix element.
        """
        raise NotImplementedError

    def triplet_elmnt(self, ki, ka, kj, kb):
        """Return the value of the triplet matrix element for the ki -> ka, kj -> kb excitations.

        :param ki: Momentum of occupied state i
        :type ki: np.ndarray(len = # of dimensions)
        :param ka: Momentum of virtual state a
        :type ka: np.ndarray(len = # of dimensions)
        :param kj: Momentum of occupied state j
        :type kj: np.ndarray(len = # of dimensions)
        :param kb: Momentum of virtual state b
        :type kb: np.ndarray(len = # of dimensions)
        :returns: (float) the value of the matrix element.
        """
        raise NotImplementedError


class A(AorB):

    def __init__(self, orbitalhessian):
        super().__init__(orbitalhessian)

    def momentum_conserving_pairs(self, ki, ka):
        """yield the momentum conserving excitation.

        This applies to the following + their complex conjugates
        <ka kj | ki kb>
        <ka kj | kb ki>
        <kj ka | ki kb>
        <kj ka | kb ki>

        :param ki: Occupied state momentum
        :type ki: np.ndarray(n_dimensions)
        :param ka: Virtual state momentum
        :type ka: np.ndarray(n_dimensions)
        """
        for j, kj in enumerate(self.states.occupied_momenta):
            kb = ka + kj - ki
            self.parameters.to_first_brillouin_zone(kb)
            if self.states.is_k_vec_virtual(kb):
                t = self.excitations.label_from_momenta(kj, kb)
                yield t, kj, kb

    def row_generator(self, i_row, offset=0):
        """Return a generator for the i'th row that makes (column index, value) pairs for nonzero elements."""

        def it():
            (k_i, k_a) = self.excitations.momenta[i_row]
            for i_col, k_j, k_b in self.momentum_conserving_pairs(k_i, k_a):
                val = self.elmnt_from_momenta(ki=k_i, ka=k_a, kj=k_j, kb=k_b)
                if i_row == i_col:
                    val += self.excitations.energies[i_row]
                yield (i_col + offset, val)
        return it()

    def singlet_elmnt(self, ki, ka, kj, kb):
        return 2.0 * self.parameters.eri.eval(ka, kj, ki, kb) - self.parameters.eri.eval(ka, kj, kb, ki)

    def triplet_elmnt(self, ki, ka, kj, kb):
        return - self.parameters.eri.eval(ka, kj, kb, ki)


class B(AorB):

    def __init__(self, orbitalhessian):
        super().__init__(orbitalhessian)

    def momentum_conserving_pairs(self, ki, ka):
        """yield the momentum conserving excitation.

        This applies to the following + their complex conjugates
        <ka kb | ki kj>
        <ka kb | kj ki>
        <kb ka | ki kj>
        <kb ka | kj ki>

        :param ki: Occupied state momentum
        :type ki: np.ndarray(n_dimensions)
        :param ka: Virtual state momentum
        :type ka: np.ndarray(n_dimensions)
        """
        for j, kj in enumerate(self.states.occupied_momenta):
            kb = ki + kj - ka
            self.parameters.to_first_brillouin_zone(kb)
            if self.states.is_k_vec_virtual(kb):
                t = self.excitations.label_from_momenta(kj, kb)
                yield t, kj, kb

    def row_generator(self, i_row, offset=0):
        """Return a generator for the i'th row that makes (column index, value) pairs for nonzero elements."""
        def it():
            (k_i, k_a) = self.excitations.momenta[i_row]
            for i_col, k_j, k_b in self.momentum_conserving_pairs(k_i, k_a):
                yield (i_col + offset, self.elmnt_from_momenta(ki=k_i, ka=k_a, kj=k_j, kb=k_b))
        return it()

    def singlet_elmnt(self, ki, ka, kj, kb):
        return 2.0 * self.parameters.eri.eval(ka, kb, ki, kj) - self.parameters.eri.eval(ka, kb, kj, ki)

    def triplet_elmnt(self, ki, ka, kj, kb):

        val = -self.parameters.eri.eval(k1=ka, k2=kb, k3=kj, k4=ki)
        return val