import itertools
import time
import sys

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
        try:
            self.size = __class__.case_size[self.parameters.instability_type] * self.excitations.n
        except KeyError:
            raise ValueError("Instability type '{type}' not supported, must be one of {set}.".format(
                type=self.parameters.instability_type, set=set(self.case_size.keys())
            ))
        self.A = A(self)
        self.B = B(self)
        self.timings = {}
        self.PETSc_mat = None

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
        self.PETSc_mat = H
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
        t = time.time()
        H = self.as_PETSc()
        self.timings['PETSc Build'] = time.time() - t

        eps = slepc_wrapper.SlepcEPSWrapper(H)
        eps.set_initial_space(n_initial=10, which='identity')
        t = time.time()
        eps.solve()
        self.timings['SLEPc Solve'] = time.time() - t

        eigs = eps.get_eigenpairs()
        eigs_real = [val[0] for val in eigs]
        return min(eigs_real)

    def _numpy_lowest_eigenvalue(self):
        """Return the lowest eigenvalue using dense numpy."""
        t = time.time()
        ary = self.as_numpy()
        self.timings['Numpy Build'] = time.time() - t

        t = time.time()
        val = np.amin(np.linalg.eigvals(ary))
        self.timings['Numpy Diagonalization'] = time.time() - t
        return val


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

        def a_row_iterator():
            k_i, k_a = np.split(self.excitations._momenta[i_row], 2)
            for i_col, k_j, k_b in self.momentum_conserving_pairs(k_i, k_a):
                val = self.elmnt_from_momenta(k_i, k_a, k_j, k_b)
                if i_row == i_col:
                    val += self.excitations.energies[i_row]
                yield (i_col + offset, val)
        return a_row_iterator()

    def fast_row_generator(self, i_row, offset=0):
        """Fast version of row generator

        Logically equivalent to row_generator.
        """
        k_i, k_a = np.split(self.excitations._momenta[i_row], 2)

        #  All possible momentum conserving virtual states
        kj_ary = self.parameters.states.occupied_momenta
        kb_ary = (k_a - k_i)[np.newaxis, :] + kj_ary
        self.parameters.to_first_brillouin_zone(kb_ary)
        mask = np.linalg.norm(kb_ary, axis=1) > self.parameters.k_fermi
        kt_ary = np.hstack((kj_ary, kb_ary))
        #  Keep only excitations with actual virtual states
        kt_ary = kt_ary[mask]
        idx = np.zeros(len(kt_ary), dtype=np.uint32)

        row_vals = np.zeros(len(kt_ary), dtype=np.float64)
        # this isn't optimal.
        for i, kt in enumerate(kt_ary):
            j = self.excitations._label_from_momenta[tuple(np.round(kt, 5))]
            if j == i_row:
                row_vals[i] = self.excitations.energies[i]
            idx[i] = j


        # only called once should be ok
        two_e_kakjkikb = self.elmnt_from_momenta(k_a, 0, k_i, 0)

        denoms = np.linalg.norm(self.parameters.to_first_brillouin_zone((k_a - k_i)[np.newaxis, :]), axis=0)

        print('denom = ', denoms)
        sys.exit()

        val = self.elmnt_from_momenta(k_i, k_a, k_j, k_b)
        if i_row == i_col:
            val += self.excitations.energies[i_row]
        return (i_col + offset, val)

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
        def b_row_iterator():
            k_i, k_a = np.split(self.excitations._momenta[i_row], 2)
            for i_col, k_j, k_b in self.momentum_conserving_pairs(k_i, k_a):
                yield (i_col + offset, self.elmnt_from_momenta(k_i, k_a, k_j, k_b))
        return b_row_iterator()

    def singlet_elmnt(self, ki, ka, kj, kb):
        return 2.0 * self.parameters.eri.eval(ka, kb, ki, kj) - self.parameters.eri.eval(ka, kb, kj, ki)

    def triplet_elmnt(self, ki, ka, kj, kb):

        val = -self.parameters.eri.eval(ka, kb, kj, ki)
        return val