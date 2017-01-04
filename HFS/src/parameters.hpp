/** @file parameters.hpp
@author Evan Curtin
@version Revision 0.1
@brief Header including extern declarations for global parameters.
@details Definitions are in parameters.cpp
@date Wednesday, 04 Jan, 2017
*/

#ifndef HFS_params_included
#define HFS_params_included

#ifndef PI
    #define PI 3.14159265358979323846264338327
#endif
#ifndef SMALLNUMBER
    #define SMALLNUMBER 1E-12
#endif
#include "armadillo"

/** \namespace HFS
    \brief Global parameters. Used mostly for traceability of parameters.
*/
namespace HFS {

    extern double  bzone_length, vol, rs, kf, kmax, fermi_energy, cond_number;
    extern double  two_e_const, deltaK, Total_Calculation_Time;
    extern std::string Computation_Starttime;
    extern std::string OutputFileName;
    extern std::string mycase;
    extern arma::uword Nocc, Nvir, Nexc, N_elec, Nmat;
    extern unsigned Nk, ground_state_degeneracy;
    extern arma::vec  occ_energies, vir_energies, exc_energies, kgrid;
    extern arma::umat occ_states, vir_states, excitations;
    #if NDIM == 2
        extern arma::umat vir_N_to_1_mat;
    #elif NDIM == 3
        extern arma::ucube vir_N_to_1_mat;
    #endif
    extern arma::umat inv_exc_mat;
    extern void (*MatVecProduct_func)(arma::vec& v, arma::vec& Mv);
    extern double (*Matrix_func)(arma::uword i, arma::uword j);

    extern unsigned dav_its;
    extern arma::vec dav_vals;
    extern unsigned num_guess_evecs;
    extern unsigned Dav_blocksize;
    extern unsigned Dav_Num_evals;
    extern unsigned Dav_nconv;
    extern double Dav_tol;
    extern double Dav_final_val;
    extern unsigned Dav_maxits;
    extern unsigned Dav_minits;
    extern unsigned Dav_maxsubsize;
    extern double Dav_time;

}
#endif // HFS_params_included
