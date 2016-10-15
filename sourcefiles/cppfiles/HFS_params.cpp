#include "HFS_params.h"

namespace HFS{
    double  bzone_length, vol, rs, kf, kmax, fermi_energy;
    double  two_e_const, deltaK;
    arma::uword Nocc, Nvir, Nexc, N_elec, Nk;
    int ndim;
    arma::vec  occ_energies, vir_energies, exc_energies, kgrid;
    arma::umat occ_states, vir_states, excitations;
    arma::umat vir_N_to_1_mat, inv_exc_mat;
}
