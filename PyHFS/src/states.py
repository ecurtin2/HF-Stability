import itertools

import numpy as np

from analytic_energies import total_energy
import prettyprint
import constants


class States(object):

    def __init__(self, parameters):
        self.parameters = parameters

        # Initialize to avoid append penalties.
        self.len = self.parameters.n_k_points ** self.parameters.n_dimensions
        self.indices = np.zeros((self.len, self.parameters.n_dimensions), dtype=np.uint32)
        self.energies = np.zeros(self.len)
        self.set_indices()
        self.momenta = self.parameters.k_grid[self.indices]
        print(self.indices)
        print(self.momenta)

        self.is_occupied = None
        self.occupied_indices = None
        self.occupied_momenta = None
        self.occupied_energies = None
        self.n_occupied = None

        self.is_virtual = None
        self.virtual_indices = None
        self.virtual_momenta = None
        self.virtual_energies = None
        self.n_virtual = None

        self.two_ERI_func = None

        self.determine_occ_and_vir()

    def calc_energies(self):
        kinetic_energy = np.sum(self.momenta**2, axis=1) / 2.0
        exchange_energy = np.zeros(self.len)
        for i, k in enumerate(self.momenta):
            exchange_energy[i] = - sum([self.parameters.eri.eval(k, occ, occ, k) for occ in self.occupied_momenta])
        self.energies = kinetic_energy + exchange_energy

    def determine_occ_and_vir(self):
        self.is_virtual = np.apply_along_axis(self.is_k_vec_virtual, axis=1, arr=self.momenta)
        self.is_occupied = np.logical_not(self.is_virtual)

        occ_subset = np.where(self.is_occupied)[0]

        self.occupied_indices = self.indices[occ_subset]
        self.occupied_momenta = self.parameters.k_grid[self.occupied_indices]
        self.occupied_energies = self.energies[occ_subset]
        self.n_occupied = len(self.occupied_indices)

        vir_subset = np.where(np.logical_not(self.is_occupied))
        self.virtual_indices = self.indices[vir_subset]
        self.virtual_momenta = self.parameters.k_grid[self.virtual_indices]
        self.virtual_energies = self.energies[vir_subset]
        self.n_virtual = len(self.virtual_indices)

    def sort_by_energy(self):
        """Ascending sort by energies in place."""
        idx = np.argsort(self.energies)
        self.energies = self.energies[idx]
        self.momenta = self.momenta[idx]
        self.indices = self.indices[idx]
        self.determine_occ_and_vir() # Rerun with sorted

    def set_indices(self):
        iter = range(self.parameters.n_k_points)
        count = 0

        # Get the indices of states only if the non-first dimensions are below the fermi level.
        for idx in itertools.product(iter, repeat=self.parameters.n_dimensions):

            if self.parameters.n_dimensions == 1:
                self.indices[count] = idx
                count += 1
            else:
                k = self.parameters.k_grid[list(idx)[1:]]
                if np.any(np.abs(k) < (self.parameters.k_fermi + constants.SMALL_NUMBER)):
                    self.indices[count] = idx
                    count += 1

        self.indices = self.indices[:count]
        self.len = count

    def get_is_occupied(self):
        """Return """
        norms = np.apply(self.is_k_vec_virtual, axis=1, arr=self.momenta)
        return norms < self.parameters.k_fermi

    def is_k_vec_virtual(self, k_vec):
        return np.linalg.norm(k_vec) > self.parameters.k_fermi

    def occ_vir_label_from_momenta(self, kocc, kvir):
        """Given the momenta of the excitation, return the index of the corresponding excitation"""
        occ_grid_indices = np.round((kocc + self.parameters.k_max)
                                    / self.parameters.k_grid_spacing).astype(np.uint32)
        vir_grid_indices = np.round((kvir + self.parameters.k_max)
                                    / self.parameters.k_grid_spacing).astype(np.uint32)

        occ_label = self.occ_indices_to_label[tuple(occ_grid_indices)]
        vir_label = self.vir_indices_to_label[tuple(vir_grid_indices)]

        return occ_label, vir_label

    def __str__(self):
        """Specify all states in list, split into occupied and virtual."""
        ijk = ['nx', 'ny', 'nz'][:self.parameters.n_dimensions]
        k_labels = ['kx', 'ky', 'kz'][:self.parameters.n_dimensions]
        col_header = (3 * ' ').join(ijk) + 12 * ' ' + (8 * ' ').join(k_labels) + '\n'

        s = ''
        s += prettyprint.header('Occupied States')
        s += "Number of Occupied States: " + str(self.n_occupied) + '\n'
        s += col_header
        for idx, k, e in zip(self.occupied_indices, self.occupied_momenta, self.occupied_energies):
            s += (str(e) + ' '
                  #+ str(total_energy(k, self.parameters.k_fermi, self.parameters.n_dimensions))
                 + str(idx) + ' ' + str(k) + ' \n')

        s += prettyprint.header('Virtual States')
        s += "Number of Virtual States: " + str(self.n_virtual) + '\n'
        s += col_header
        for idx, k, e in zip(self.virtual_indices, self.virtual_momenta, self.virtual_energies):
            s += (str(e) + ' '
                  #+ str(total_energy(k, self.parameters.k_fermi, self.parameters.n_dimensions))
                 + str(idx) + ' ' + str(k) + ' \n')

        return s