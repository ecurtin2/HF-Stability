################################################################################
#                                   C++ Land is Here 
################################################################################
from libcpp cimport bool
import itertools
import math
from scipy import special as sp
import numpy as np
cimport numpy as np
include "cyarma.pyx"

cdef extern from "stdint.h" nogil:
    ctypedef unsigned long long uint64_t
	
#C++ class
#Note only methods/attributes that want python access need to be here.
cdef extern from "stability.h" namespace "HFStability":
    cdef cppclass HEG:
        HEG() except +
        #Attributes
        double bzone_length, vol, rs, kf, kmax, fermi_energy
        double two_e_const, deltaK
        uint64_t Nocc, Nvir, Nexc, N_elec, ndim, Nk
        mat states          #arma::mat wrapped by cyarma
        vec energies, exc_energies, kgrid
        umat excitations, 
        uvec occ2state, vir2state  #arma::umat not native to cyarma I added it
        mat occ_states, vir_states

        #Methods
        double min_eigval(long, long, long, long, long, long, bool, 
                            double, double*)
        double energy(long long unsigned int)
        double two_electron_3d(double[], double[], double[])
        double two_electron_2d(double[], double[], double[])
        void calc_energies_2d()
        void calc_energies_3d()

################################################################################
#                             Python Land is Here
################################################################################
#Python interface to c++ class
cdef class PyHEG:
    """This is a class docstring"""
    cdef HEG* c_HEG
    def __cinit__(self):
        self.c_HEG = new HEG()
    def __dealloc__(self):
        del self.c_HEG

    #All constants of calculation specified by __init__
    def __init__(self, ndim=3, rs=1.0, Nk=4):
        """This is an init docstring"""
        self.rs = float(rs)
        self.ndim = int(ndim)
        self.Nk = int(Nk)
        self.get_resulting_params()

    def get_resulting_params(self):
        """Given rs, ndim, Nk, determine other parameters

        Description:
                Does nothing if all 3 aren't defined, which shouldn't
                ever be possible.
        Args:
                No args
        Returns:
                Nothing
        Raises:
                No exceptions
                

        """
        #It looks like the variables default to 0, not undefined?
        if self.rs == 0 or self.ndim == 0 or self.Nk == 0:
            return None
        try:
            self.rs
            self.ndim
            self.Nk
        except:
            return None #Don't try to calc resulting params until all defined!
        else:
            pass #Continue on if it checks out

        if self.ndim == 1:
            self.kf = np.pi / (4 * self.rs)
        elif self.ndim == 2:
            self.kf = 2**0.5 / self.rs
        elif self.ndim == 3:
            self.kf = (9 * np.pi / 4)**(1./3.) * (1. / self.rs) 


        self.kmax = 2.0 * self.kf
        self.fermi_energy = 0.5 * self.kf**2
        #brillioun zone is from - pi/a .. pi/a
        self.bzone_length =  2.0 * self.kmax
        direct_length = self.bzone_length / (2.0 * np.pi)

        # states is list of tuples each of length ndim containing the 
        # coordinates in k-space of each state
        self.kgrid = np.linspace(-self.kmax, self.kmax, self.Nk)
        self.deltaK = self.kgrid[1] - self.kgrid[0]
        states = list(itertools.product(self.kgrid, repeat=self.ndim))

        #Separating into occ and vir by momentum
        occ_states = []
        vir_states = []
        occ2state = []
        vir2state = []
        for index, state in enumerate(states):
            k = np.linalg.norm(state)
            if k < self.kf + 10e-8:
                occ2state.append(index) 
                occ_states.append(state)
            else:
                vir2state.append(index) 
                vir_states.append(state)

        self.Nocc = len(occ_states)
        self.Nvir = len(vir_states)
        self.states = np.asfortranarray(states, dtype=np.float64)
        self.occ_states = np.asfortranarray(occ_states, dtype=np.float64)
        self.vir_states = np.asfortranarray(vir_states, dtype=np.float64)
        self.occ2state = np.asfortranarray(occ2state, dtype=np.uint64)
        self.vir2state = np.asfortranarray(vir2state, dtype=np.uint64)
        #RHF ONLY
        self.N_elec = 2 * self.Nocc

        if self.ndim == 3:
            self.vol = self.N_elec * 4.0 / 3.0 * np.pi * (self.rs**3)
            self.two_e_const = 4.0 * np.pi / self.vol 
        elif self.ndim == 2:
            self.vol = self.N_elec * np.pi * (self.rs**2)
            self.two_e_const = 2.0 * np.pi / self.vol 
        elif self.ndim == 1:
            self.vol = self.N_elec * 2.0 * self.rs

    def calc_energy(self):
        if self.ndim == 2:
            self.c_HEG.calc_energies_2d()
        elif self.ndim == 3:
            self.c_HEG.calc_energies_3d()

    def calc_excitations(self):
        exc = []
        if self.ndim == 2:
            for i, occ_state in enumerate(self.occ_states):
                matches = np.where(self.vir_states[:,0] == occ_state[0])
                exc.append([(i,j) for j in matches[0]])
        elif self.ndim == 3:
            for i, occ_state in enumerate(self.occ_states):
                matches = np.where(np.all(self.vir_states[:, :2] == occ_state[:2], axis=1))
                exc.append([(i,j) for j in matches[0]])

        self.excitations = np.asfortranarray(list(itertools.chain.from_iterable(exc)), dtype=np.uint64)

    def find_virs2(self):
        x_exc = (self.kgrid + self.kmax)[1:]         #all potential +x excitations within 1st BZ
        all_exc = np.zeros((self.Nk - 1, self.ndim)) # -1 excludes the occ_state
        all_exc[:,0] = x_exc                       #only consider +x,  y and z are zero
        num_add = self.Nk - 1                       # max number of excitations in 1D
        my_ones = np.ones((num_add), dtype=int)
        N = self.Nocc * num_add
        occ_idx = np.zeros((N), dtype=np.uint64)
        vir = np.zeros((N, self.ndim), dtype=np.float64)
        i1 = 0
        i2 = num_add
        for index, state in enumerate(self.occ_states):
            a = all_exc + state
            b = index * my_ones
            occ_idx[i1:i2] = b
            vir[i1:i2] = a
            i1 += num_add
            i2 += num_add
        vir_norms = np.sqrt((vir*vir).sum(axis=1))  #norm of each row
        idx= np.where((vir_norms > self.kf+10E-8) & (vir_norms <= self.kmax+10E-8))
        vir = vir[idx]  # keep only those above fermi but below cutoff
        occ_idx = occ_idx[idx]  #this is the occupied state that generated the vir
        return occ_idx, self.k_to_index(vir)

    def unique_rows2(self, data):                 
        """Return only unique rows
        see http://stackoverflow.com/questions/31097247/remove-duplicate-rows-of-a-numpy-array"""
        sorted_idx = np.lexsort(data.T)
        sorted_data =  data[sorted_idx,:]
        row_mask = np.append([True],np.any(np.diff(sorted_data,axis=0),1))
        new_idx = 0
        vir_list = [0]
        for i in row_mask[1:]:
            if i:
                new_idx += 1
            vir_list.append(new_idx)

        vir_list = np.asarray(vir_list, dtype=int)
        return sorted_data[row_mask], vir_list, sorted_idx

    def find_exc2(self):
        occ_idx, vir = self.find_virs2()
        vir_states, exc_virs, idx2 = self.unique_rows2(vir)
        occ_idx = occ_idx[idx2]
        exc = np.column_stack((occ_idx, exc_virs))
        return exc

    def find_virs(self):
        x_exc = (self.kgrid + self.kmax)[1:]         #all potential +x excitations within 1st BZ
        all_exc = np.zeros((self.Nk - 1, self.ndim)) # -1 excludes the occ_state
        all_exc[:,0] = x_exc                         #only consider +x,  y and z are zero
        first_state = self.occ_states[0]
        vir = all_exc + first_state                  # adds k of first state to each possible exc
        my_ones = np.ones((self.Nk-1), dtype=np.uint64)
        occ_idx = my_ones * 0
        for index, state in enumerate(self.occ_states[1:]): ####THIS LOOP IS A BOTTLENECK
            vir = np.vstack((vir, all_exc + state)) #do for the rest, and put into 1 array
            occ_idx = np.append(occ_idx, (index+1) * my_ones)
        vir_norms = np.sqrt((vir*vir).sum(axis=1))  #norm of each row
        idx = np.where((vir_norms > self.kf+10E-8) & (vir_norms <= self.kmax+10E-8))
        vir = vir[idx]  # keep only those above fermi but below cutoff
        occ_idx = occ_idx[idx]  #this is the occupied state that generated the vir
        return occ_idx, self.k_to_index(vir)

    def unique_rows(self, data):                 
        """Return only unique rows
        see http://stackoverflow.com/questions/31097247/remove-duplicate-rows-of-a-numpy-array"""
        sorted_idx = np.lexsort(data.T)
        sorted_data =  data[sorted_idx,:]
        row_mask = np.append([True],np.any(np.diff(sorted_data,axis=0),1))
        new_idx = 0
        vir_list = [0]
        for i in row_mask[1:]:
                if i:
                    new_idx += 1
                vir_list.append(new_idx)

        vir_list = np.asfortranarray(vir_list, dtype=np.uint64)
        return sorted_data[row_mask], vir_list, sorted_idx

    def find_exc(self):
        occ_idx, vir = self.find_virs()
        vir_states, exc_virs, idx2 = self.unique_rows(vir)
        occ_idx = occ_idx[idx2]
        exc = np.column_stack((occ_idx, exc_virs))
        return exc
                    
#        exc_energies = []
#        for exc in self.excitations:
#            exc_energy = (self.energies[self.vir2state[exc[1]]] 
#                         -self.energies[self.occ2state[exc[0]]])
#            exc_energies.append(exc_energy)
#        self.exc_energies = np.asfortranarray(exc_energies, dtype=np.float64)

    def f2D(self, y):
        if y <= 1.0:
            #scipy and guiliani/vignale define K and E differently, x -> x*x
            return sp.ellipe(y*y)
        else:
            #scipy and guiliani/vignale define K and E differently, x -> x*x
            x = 1.0 / y
            return y * (sp.ellipe(x*x) - (1.0 - x*x) * sp.ellipk(x*x))

    def f3D(self, y):
        if y < 10e-10:
            return 1.0
        return 0.5 + (1 - y*y) / (4*y) * math.log(abs((1+y) / (1-y)))

    def analytic_exch(self, k):
        const = -2.0 * self.kf / math.pi
        if self.ndim == 2:
            return const * self.f2D(k/self.kf)
        elif self.ndim == 3:
            return const * self.f3D(k/self.kf)

    def analytic_energy(self, k):
        x = np.linalg.norm(k)  #works on k of any dimension
        return (x*x / 2.0) + self.analytic_exch(x)

    def kin(self, int i):
        return 0.5 * np.linalg.norm(self.states[i]) ** 2

    def k_to_index(self, array):
        deltaK = self.kgrid[1] - self.kgrid[0]
        idx = np.rint(((array + self.kmax) / deltaK)).astype(np.uint64)
        assert np.all(np.isclose(self.kgrid[idx], array))
        return idx


#    def min_eigval(self, 
#                   np.ndarray[double, ndim = 2] GUESS_VECS not None,
#                   long MAX_ITS = 20, 
#                   long NUM_OF_ROOTS = 1, 
#                   long MAX_SUB_SIZE = 2000, 
#                   long BLOCK_SIZE   = 1,
#                   bool DOTEST=False, 
#                   double TOLERANCE  =  10E-8):
#
#       GUESS_VECS = np.asfortranarray(GUESS_VECS)
#
#       cdef long N, num_of_roots,  num_of_guess, max_its, max_sub_size
#       N = GUESS_VECS.shape[0]
#       num_of_roots = NUM_OF_ROOTS
#       num_of_guess = GUESS_VECS.shape[1]
#       max_its = MAX_ITS
#       max_sub_size = MAX_SUB_SIZE
#       block_size = BLOCK_SIZE
#       
#       cdef bool dotest = DOTEST	
#       
#       cdef double tol = TOLERANCE
#       cdef double* guess_vecs = &GUESS_VECS[0,0]
#       
#       return self.c_HEG.min_eigval(N, max_its, max_sub_size, num_of_roots, 
#          num_of_guess, block_size, dotest, tol, guess_vecs)

    #############################################################################
    #       Define properties, allows for setting/getting them in python        #
    #############################################################################
    
    def get_rs(self):
        """(float) Get/set Wigner-Seitz Radius. Setting checks type.
            Setting ndim calls get_resulting_params to update all
            dependent variables appropriately. """
        return self.c_HEG.rs
    def set_rs(self, value):
        self.c_HEG.rs = float(value)
        self.get_resulting_params()
    rs = property(get_rs, set_rs)

    def get_ndim(self):
        """(int) Get/set number of dimensions. Setting checks type.
            Setting ndim calls get_resulting_params to update all
            dependent variables appropriately. 
        """
        return self.c_HEG.ndim
    def set_ndim(self, value):
        self.c_HEG.ndim = int(value)
        self.get_resulting_params()
    ndim = property(get_ndim, set_ndim)

    def get_Nk(self):
        """(int) Get/set number of k-points. Setting checks type.
            Setting ndim calls get_resulting_params to update all
            dependent variables appropriately. """
        return self.c_HEG.Nk
    def set_Nk(self, value):
        self.c_HEG.Nk = int(value)
        self.get_resulting_params()
    Nk = property(get_Nk, set_Nk)

    def get_kf(self):
        """(float) Get/set Fermi wavenumber. Setting checks type."""
        return self.c_HEG.kf
    def set_kf(self, value):
        self.c_HEG.kf = float(value)
    kf = property(get_kf, set_kf)

    def get_kmax(self):
        """(float) Get/set Fermi wavenumber. Setting checks type."""
        return self.c_HEG.kmax
    def set_kmax(self, value):
        self.c_HEG.kmax = float(value)
    kmax = property(get_kmax, set_kmax)

    def get_two_e_const(self):
        """(float) Get/set Fermi wavenumber. Setting checks type."""
        return self.c_HEG.two_e_const
    def set_two_e_const(self, value):
        self.c_HEG.two_e_const = float(value)
    two_e_const = property(get_two_e_const, set_two_e_const)

    def get_fermi_energy(self):
        """(float) Get/set Fermi energy. Setting checks type."""
        return self.c_HEG.fermi_energy
    def set_fermi_energy(self, value):
        self.c_HEG.fermi_energy = float(value)
    fermi_energy = property(get_fermi_energy, set_fermi_energy)

    def get_N_elec(self):
        """(int) Get/set number of electrons. Setting checks type."""
        return self.c_HEG.N_elec
    def set_N_elec(self, value):
        self.c_HEG.N_elec = int(value)
    N_elec = property(get_N_elec, set_N_elec)


    def get_Nocc(self):
        """(int) Get/set number of occupied states. Setting checks type."""
        return self.c_HEG.Nocc
    def set_Nocc(self, value):
        self.c_HEG.Nocc = int(value)
    Nocc = property(get_Nocc, set_Nocc)

    def get_Nvir(self):
        """(int) Get/set number of virtual states. Setting checks type."""
        return self.c_HEG.Nvir
    def set_Nvir(self, value):
        self.c_HEG.Nvir = int(value)
    Nvir = property(get_Nvir, set_Nvir)

    def get_Nexc(self):
        """(int) Get/set number of virtual states. Setting checks type."""
        return self.c_HEG.Nexc
    def set_Nexc(self, value):
        self.c_HEG.Nexc = int(value)
    Nexc = property(get_Nexc, set_Nexc)

    def get_bzone_length(self):
        """(float) Get/set length of brillioun zone. Setting checks type.
	    This is the entire length from -pi/a .. pi/a. """
        return self.c_HEG.bzone_length
    def set_bzone_length(self, value):
        self.c_HEG.bzone_length = float(value)
    bzone_length = property(get_bzone_length, set_bzone_length)

    def get_vol(self):
        """(float) Get/set direct lattice volume. Setting checks type."""
        return self.c_HEG.vol
    def set_vol(self, double value):
        self.c_HEG.vol = float(value)
    vol = property(get_vol, set_vol)

    def get_states(self):
        return mat_to_numpy(self.c_HEG.states)
    def set_states(self, 
                        np.ndarray[double, ndim=2, mode="fortran"]
                        inp_states not None):
        self.c_HEG.states = numpy_to_mat_d(inp_states)
    states = property(get_states, set_states)

    def get_occ_states(self):
        return mat_to_numpy(self.c_HEG.occ_states)
    def set_occ_states(self, 
                        np.ndarray[double, ndim=2, mode='fortran']
                        inp_occ_states not None):
        self.c_HEG.occ_states = numpy_to_mat_d(inp_occ_states)
    occ_states = property(get_occ_states, set_occ_states)

    def get_vir_states(self):
        return mat_to_numpy(self.c_HEG.vir_states)
    def set_vir_states(self, 
                        np.ndarray[double, ndim=2, mode="fortran"]
                        inp_vir_states not None):
        self.c_HEG.vir_states = numpy_to_mat_d(inp_vir_states)
    vir_states = property(get_vir_states, set_vir_states)

    def get_excitations(self):
        return umat_to_numpy(self.c_HEG.excitations)
    def set_excitations(self,
                        np.ndarray[long long unsigned int, ndim=2,mode="fortran"]
                        inp_excitations not None):
        self.c_HEG.excitations = numpy_to_umat_d(inp_excitations)
    excitations = property(get_excitations, set_excitations)
    
    def get_energies(self):
        return vec_to_numpy(self.c_HEG.energies)
    def set_energies(self, 
                        np.ndarray[double, ndim=1]
                        inp_energies not None):
        self.c_HEG.energies = numpy_to_vec_d(inp_energies)
    energies = property(get_energies, set_energies)

    def get_occ2state(self):
        return uvec_to_numpy(self.c_HEG.occ2state)
    def set_occ2state(self, 
                        np.ndarray[long long unsigned int, ndim=1]
                        inp_occ2state not None):
        self.c_HEG.occ2state = numpy_to_uvec_d(inp_occ2state)
    occ2state = property(get_occ2state, set_occ2state)

    def get_vir2state(self):
        return uvec_to_numpy(self.c_HEG.vir2state)
    def set_vir2state(self, 
                        np.ndarray[long long unsigned int, ndim=1]
                        inp_vir2state not None):
        self.c_HEG.vir2state = numpy_to_uvec_d(inp_vir2state)
    vir2state = property(get_vir2state, set_vir2state)


    def get_exc_energies(self):
        return vec_to_numpy(self.c_HEG.exc_energies)
    def set_exc_energies(self, 
                        np.ndarray[double, ndim=1]
                        inp_exc_energies not None):
        self.c_HEG.exc_energies = numpy_to_vec_d(inp_exc_energies)
    exc_energies = property(get_exc_energies, set_exc_energies)

    def get_kgrid(self):
        return vec_to_numpy(self.c_HEG.kgrid)
    def set_kgrid(self, 
                        np.ndarray[double, ndim=1]
                        inp_kgrid not None):
        self.c_HEG.kgrid = numpy_to_vec_d(inp_kgrid)
    kgrid = property(get_kgrid, set_kgrid)

    def get_deltaK(self):
        return self.c_HEG.deltaK
    def set_deltaK(self, value):
        self.c_HEG.deltaK = float(value)
    deltaK = property(get_deltaK, set_deltaK)
