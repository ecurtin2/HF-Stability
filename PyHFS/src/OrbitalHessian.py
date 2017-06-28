import numpy as np


import twoERI


class OrbitalHessian(object):

    case_size = {
        'cRHF2cRHF': 2,
        'cRHF2cUHF': 2,
        'cUHF2cUHF': 4,
        'cUHF2cGHF': 4
    }

    def __init__(self, parameters):
        self.parameters = parameters
        self.states = parameters.states
        self.excitations = parameters.excitations
        self.size = __class__.case_size[self.parameters.instability_type] * self.excitations.n
        self.A = A(self)
        self.B = B(self)

    def as_numpy(self):
        H = np.zeros((self.size, self.size))

        N = self.size // 2
        H[:N, :N] = self.A.as_numpy()
        H[:N, N:] = self.B.as_numpy()
        H[N:, :N] = self.B.as_numpy()
        H[N:, N:] = self.A.as_numpy()
        return H

    def get_conserving_virtual(self, k_occ, k_vir, k_second_occ):
        k = k_occ + k_vir - k_second_occ
        twoERI.to_first_brillouin_zone(k, self.parameters.k_max)

        indices = np.round(k / self.parameters.k_grid_spacing)
        exc_label = self.excitations.label_from_indices(indices)
        return exc_label, k

    def lowest_eigenvalue(self):
        return np.amin(np.linalg.eigvals(self.as_numpy()))


class AorB(object):

    def __init__(self, orbitalhessian):
        self.orbital_hessian = orbitalhessian
        self.parameters = orbitalhessian.parameters
        self.states = orbitalhessian.states
        self.excitations = orbitalhessian.excitations
        self.elmnt_from_momenta = self._gen_elmnt_function()

    def as_numpy(self):
        raise NotImplementedError

    def momentum_conserving_pairs(self, k_i, k_a):
        raise NotImplementedError

    def _gen_elmnt_function(self):
        matrix_elmnt_dic = {
            'cRHF2cRHF': self.singlet_elmnt,
            'cRHF2cUHF': self.triplet_elmnt
        }
        return matrix_elmnt_dic[self.parameters.instability_type]

    def singlet_elmnt(self, ki, ka, kj, kb):
        raise NotImplementedError

    def triplet_elmnt(self, ki, ka, kj, kb):
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

    def as_numpy(self):
        a = np.diag(self.excitations.energies)
        for s, (k_i, k_a) in enumerate(self.excitations.momenta):
            for t, k_j, k_b in self.momentum_conserving_pairs(k_i, k_a):
                a[s, t] += self.elmnt_from_momenta(ki=k_i, ka=k_a, kj=k_j, kb=k_b)
        return a

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

    def as_numpy(self):
        b = np.zeros((self.excitations.n, self.excitations.n))
        for s, (k_i, k_a) in enumerate(self.excitations.momenta):
            for t, k_j, k_b in self.momentum_conserving_pairs(k_i, k_a):
                b[s, t] = self.elmnt_from_momenta(ki=k_i, ka=k_a, kj=k_j, kb=k_b)
        return b

    def singlet_elmnt(self, ki, ka, kj, kb):
        return 2.0 * self.parameters.eri.eval(ka, kb, ki, kj) - self.parameters.eri.eval(ka, kb, kj, ki)

    def triplet_elmnt(self, ki, ka, kj, kb):

        val = -self.parameters.eri.eval(k1=ka, k2=kb, k3=kj, k4=ki)
        # k = self.parameters.to_first_brillouin_zone(kj-ka)
        # print("within triplet", np.linalg.norm(k))
        # print("B triplet_elmnt", ka, kb, kj, ki, val)
        return val