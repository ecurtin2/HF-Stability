#ifndef HFS_params_calc_included
#define HFS_params_calc_included

#include "HFS_params.hpp"
#include "HFS_base_funcs.hpp"

namespace HFS {
    extern void calc_params();
    extern void calc_kf();
    extern void calc_vol_and_two_e_const();
    extern void calc_occ_states();
    extern void calc_occ_energies();
    extern void calc_vir_energies();
    extern void calc_energies(arma::umat&, arma::vec&);
    extern void calc_excitations();
    extern void calc_exc_energy();
    extern void calc_vir_N_to_1_mat();
    extern void calc_inv_exc_mat();
}
#endif // HFS_params_calc_included
