#ifndef HFS_params_included
#define HFS_params_included

#ifndef PI
    #define PI 3.14159265358979323846264338327
#endif
#ifndef SMALLNUMBER
    #define SMALLNUMBER 1E-10
#endif
#include "armadillo"

namespace HFS {
    extern double  bzone_length, vol, rs, kf, kmax, fermi_energy;
    extern double  two_e_const, deltaK, Total_Calculation_Time;
    extern std::string Computation_Starttime;
    extern std::string OutputFileName;
    extern arma::uword Nocc, Nvir, Nexc, N_elec;
    extern unsigned ndim, Nk;
    extern arma::vec  occ_energies, vir_energies, exc_energies, kgrid;
    extern arma::umat occ_states, vir_states, excitations;
    extern arma::umat vir_N_to_1_mat, inv_exc_mat;
}
#endif // HFS_params_included
