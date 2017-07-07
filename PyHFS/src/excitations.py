import itertools

import numpy as np

import prettyprint


class Excitations(object):
    """Class for the excitations of the system."""

    def __init__(self, parameters):
        self.parameters = parameters
        self.states = parameters.states
        self.n = 0
        max_n_excitations = self.states.n_occupied * self.states.n_virtual
        self.indices = np.zeros((max_n_excitations, 2), dtype=np.uint32)
        self._momenta = np.zeros((max_n_excitations, 2 * self.parameters.n_dimensions))
        self.energies = np.zeros(max_n_excitations)
        self._label_from_momenta = {}
        self.find_x_excitations()

    @property
    def momenta(self):
        return ((k[:self.parameters.n_dimensions], k[self.parameters.n_dimensions:])
                for k in self._momenta)

    def find_all_excitations(self):
        """Find excitations from all combinations of occupied and virtual states."""
        count = 0
        for i, occ in enumerate(self.states.occupied_indices):
            for a, vir in enumerate(self.states.virtual_indices):
                self.indices[count] = [i, a]
                k = np.hstack((self.states.occupied_momenta[i],
                               self.states.virtual_momenta[a]))
                self._momenta[count] = k
                self._label_from_momenta[tuple(np.round(k, 5))] = count
                self.energies[count] = (self.states.virtual_energies[a]
                                        - self.states.occupied_energies[i])
                count += 1

        self.indices = self.indices[:count]
        self.energies = self.energies[:count]
        self._momenta = self._momenta[:count]
        self.n = count

    def find_x_excitations(self):
        """Find only excitations which excite in the x dimension."""
        count = 0
        for i, occ in enumerate(self.states.occupied_indices):
            for a, vir in enumerate(self.states.virtual_indices):
                # For all combinations that are along the same x axis
                if np.all(occ[1:] == vir[1:]):
                    self.indices[count] = [i, a]
                    k = np.hstack((self.states.occupied_momenta[i],
                                   self.states.virtual_momenta[a]))
                    self._momenta[count] = k
                    self.energies[count] = (self.states.virtual_energies[a]
                                            - self.states.occupied_energies[i])
                    count += 1
        self.indices = self.indices[:count]
        self.energies = self.energies[:count]
        self._momenta = self._momenta[:count]
        self.n = count

        idx = np.argsort(self.energies)
        self.indices = self.indices[idx]
        self.energies = self.energies[idx]
        self._momenta = self._momenta[idx]
        self._label_from_momenta = {tuple(np.round(k, 5)): i for i, k in enumerate(self._momenta)}

    def labels_from_momenta_ary(self, k):

        return self._label_from_momenta[tuple(np.round(k, 5))]

    def label_from_momenta(self, ki, ka):
        """Return the label for the excitation from ki -> ka.

        The label is the index describing an excitation: excitation = excitations[label]
        This term was chosen to avoid name conflict with index, which describes the indices of the
        reciprocal space grid.
        """
        k = tuple(np.round(np.hstack((ki, ka)), 5))
        return self._label_from_momenta[k]

    def __str__(self):
        s = prettyprint.header("Excitations")
        s += 'Number of excitations : ' + str(self.n) + '\n'
        for idx, e, k in zip(self.indices, self.energies, self._momenta):
            s += str(idx) + '   ' + str(e) + '   ' + str(k) + '\n'
        return s


