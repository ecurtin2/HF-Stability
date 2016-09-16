#C++ class
#Note only methods/attributes that want python access need to be here.
cdef extern from "stability.h" namespace "HFStability":
    cdef cppclass HEG:
        HEG() except +
        #Attributes
        double bzone_length, vol, rs, kf, kmax, fermi_energy
        double two_e_const, deltaK
        long long unsigned int Nocc, Nvir, Nexc, N_elec, ndim, Nk
        vec  occ_energies, vir_energies, exc_energies, kgrid
        umat occ_states, vir_states, excitations, 

        #Methods
        void calc_energy_wrap(bool) # True = vir, else = occ 
        void calc_exc_energy()
