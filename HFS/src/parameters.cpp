#include "parameters.hpp"

namespace HFS{
    double  bzone_length, vol, rs, kf, kmax, fermi_energy, cond_number;
    double  two_e_const, deltaK, Total_Calculation_Time;
    std::string Computation_Starttime;
    std::string OutputFileName;
    std::string mycase;
    arma::uword Nocc, Nvir, Nexc, N_elec, Nmat;
    unsigned Nk, ground_state_degeneracy;
    arma::vec  occ_energies, vir_energies, exc_energies, kgrid;
    arma::umat occ_states, vir_states, excitations;
    #if NDIM == 2
        arma::umat vir_N_to_1_mat;
    #elif NDIM == 3
        arma::ucube vir_N_to_1_mat;
    #endif // NDIM
    arma::umat inv_exc_mat;
    void (*MatVecProduct_func)(arma::vec& v, arma::vec& Mv);
    double (*Matrix_func)(arma::uword i, arma::uword j);
    unsigned dav_its;
    arma::vec dav_vals;
    unsigned num_guess_evecs;
    unsigned Dav_blocksize;
    unsigned Dav_Num_evals;
    unsigned Dav_nconv;
    double Dav_tol;
    double Dav_final_val;
    unsigned Dav_maxits;
    unsigned Dav_minits;
    unsigned Dav_maxsubsize;
    double Dav_time;
}
