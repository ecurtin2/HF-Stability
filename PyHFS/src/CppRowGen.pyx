#cython: wraparound=False, boundscheck=False, initializedcheck=False, cdivision=True

import numpy as np
cimport numpy as cnp

from libc.math cimport sqrt
from libc.math cimport round


def inverse_mapper(ary):
    """Given array of NxM entries, make a N^M MD array corresponding where MDary[i, j ...] = index of original."""
    imax = np.max(ary)
    M = ary.shape[1]
    
    if M == 1:
        inv_map = - np.ones((imax + 1), dtype=np.int32)
        for idx, i in enumerate(ary):
                inv_map[i] = idx
        return inv_map
    
    if M == 2:
        inv_map = - np.ones((imax + 1, imax + 1), dtype=np.int32)
        for idx, (i, j) in enumerate(ary):
                inv_map[i, j] = idx
        return inv_map
    
    if M == 3:
        inv_map = - np.ones((imax + 1, imax + 1, imax + 1), dtype=np.int32)
        for idx, (i, j, k) in enumerate(ary):
                inv_map[i, j, k] = idx
        return inv_map
    

cdef class CppRowGen:
    cdef:
        double k_fermi_c, k_max_c, first_bz_high_bound_c, first_bz_low_bound_c, two_eri_prefactor_c
        double brillouin_zone_width_c, k_grid_spacing_c, small_number_c
        cnp.int32_t n_dimensions_c, n_excitations_c, n_occupied_c, n_k_points_c
        cnp.double_t[:] values_c
        cnp.int32_t[:] indices_c
        cnp.double_t[:] exc_energies_c
        cnp.double_t[:, :] ki_ary_c, ka_ary_c, k_occ_ary_c
        cnp.int32_t[:, :] indices_to_virtual_c, exc_idx_from_states_c

        cnp.double_t[:] ki
        cnp.double_t[:] kj
        cnp.double_t[:] ka
        cnp.double_t[:] kb

        cnp.int32_t[:] b_indices
        
    
    def __init__(self, params, small_number):
        self.k_fermi_c = params.k_fermi
        self.k_max_c = params.k_max
        self.brillouin_zone_width_c = 2.0 * self.k_max_c
        self.k_grid_spacing_c = params.k_grid_spacing
        
        self.small_number_c = small_number
        self.two_eri_prefactor_c = params.eri.prefactor
        self.first_bz_high_bound_c = self.k_max_c - self.small_number_c
        self.first_bz_low_bound_c = - self.k_max_c - self.small_number_c
        
        self.n_dimensions_c = params.n_dimensions
        self.n_excitations_c = params.excitations.n
        self.n_occupied_c = params.states.n_occupied
        self.n_k_points_c = params.n_k_points
    
        ki, ka = np.split(params.excitations._momenta, 2, axis=1)
    
        self.ki_ary_c = ki
        self.ka_ary_c = ka
        self.k_occ_ary_c = params.states.occupied_momenta
        self.exc_energies_c = params.excitations.energies
        
        self.indices_to_virtual_c = inverse_mapper(params.states.virtual_indices)
        self.exc_idx_from_states_c = inverse_mapper(params.excitations.indices)
        
        # Allocate once and read the values, then reuse memory
        self.indices_c = np.zeros(params.excitations.n, dtype=np.int32)
        self.values_c = np.zeros(params.excitations.n, dtype=np.float)
        
    cdef double two_eri_2d(self, cnp.double_t[:] k1, cnp.double_t[:] k3):
        cdef double denom = 0.0
        for i in range(k1.shape[0]):
            denom += (k1[i] - k3[i]) * (k1[i] - k3[i])
        denom = sqrt(denom)
        if denom < self.small_number_c:
            return 0.0
        else:
            return self.two_eri_prefactor_c / denom

    cdef cnp.int32_t b_from_indices(self, cnp.int32_t[:] indices):
        return self.indices_to_virtual_c[indices[0], indices[1]]
        
    cdef void to_first_brillouin_zone(self, cnp.double_t[:] k):
        cdef cnp.int32_t i
        for i in range(k.shape[0]):           
            if k[i] > self.first_bz_high_bound_c:
                k[i] -= self.brillouin_zone_width_c
            elif k[i] < self.first_bz_low_bound_c:
                k[i] += self.brillouin_zone_width_c
        
        
    @property
    def indices(self):
        #  casting to array from memoryview should be cheap
        return np.asarray(self.indices_c)
    
    @property
    def values(self):
        #  casting to array from memoryview should be cheap
        return np.asarray(self.values_c)
    
    def generate(self, cnp.int32_t row_index, cnp.int32_t offset=0):
        """Fill indices and values with corresponding to the matrix row row_index with column offset."""
        cdef:
            cnp.double_t[:] ki = np.zeros(self.n_dimensions_c)
            cnp.double_t[:] kj = np.zeros(self.n_dimensions_c)
            cnp.double_t[:] ka = np.zeros(self.n_dimensions_c)
            cnp.double_t[:] kb = np.zeros(self.n_dimensions_c)
            
            cnp.int32_t[:] b_indices = np.zeros(self.n_dimensions_c, dtype=np.int32)
            cnp.int32_t count, b, col_index

            
        for i in range(self.n_dimensions_c):
            ki[i] = self.ki_ary_c[row_index, i]
            ka[i] = self.ka_ary_c[row_index, i]

        count = 0
        for occ_idx in range(self.n_occupied_c):
            for j in range(self.n_dimensions_c):
                kj[j] = self.k_occ_ary_c[occ_idx, j]
                kb[j] = ka[j] + kj[j] - ki[j] 

            self.to_first_brillouin_zone(self.kb)

            norm = 0.0
            for j in range(self.n_dimensions_c):
                norm += kb[j] * kb[j]

            norm = sqrt(norm)

            # Only continue if actually virtual
            if norm > self.k_fermi_c:
                for k in range(self.n_occupied_c):
                    b_indices[k] = <int>round((kb[k] + self.k_max_c) / self.k_grid_spacing_c)
                b = self.b_from_indices(b_indices)
                #assert(b >= 0, 'One of the values of b is negative, implying that the map from'
                #       + ' indices to b was not accessed properly.')

                # the column index is the excitation label from j -> b
                col_index = self.exc_idx_from_states_c[occ_idx, b]
                # diagonal is the excitation energy
                if col_index == row_index:
                    self.values_c[count] += self.exc_energies_c[row_index]

                self.values_c[count] += - self.two_eri_2d(ki, kj)
                self.indices_c[count] = col_index + offset
                count += 1
        self.values_c.shape[0] = count
        self.indices_c.shape[0] = count