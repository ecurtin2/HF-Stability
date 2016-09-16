#This file was automatically generated from stability.husing cpp_header_to_pyxheader!
cdef extern from "stability.h" namespace "HFStability":
    cdef cppclass HEG:
        #Attributes
        double  bzone_length, vol, rs, kf, kmax, fermi_energy
        double  two_e_const, deltaK
        long long unsigned int    Nocc, Nvir, Nexc, N_elec, ndim, Nk
        vec  occ_energies, vir_energies, exc_energies, kgrid
        umat occ_states, vir_states, excitations
        #Methods
        vec& mat_vec_prod_2d(vec)
        void   calc_energy_wrap(bool)
        void   calc_exc_energy()
        #2d
        void   calc_energies_2d(umat&, vec&)
        double two_electron_2d(double[], double[])
        double exchange_2d(umat&, long long unsigned int)
        long long unsigned int get_k_to_idx(double[])
        void get_vir_states_inv_2d()
        #3d
        void   calc_energies_3d(umat&, vec&)
        double exchange_3d(umat&, long long unsigned int)
        double two_electron_3d(double[], double[])
