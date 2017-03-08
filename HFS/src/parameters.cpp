#include "parameters.hpp"


namespace HFS{
    scalar bzone_length;
    scalar vol;
    scalar rs;
    scalar kf;
    scalar kmax;
    scalar fermi_energy;
    scalar cond_number;
    scalar two_e_const;
    scalar deltaK;
    scalar Total_Calculation_Time;
    std::string Computation_Starttime;
    std::string OutputFileName;
    std::string mycase;
    uint Nocc;
    uint Nvir;
    uint Nexc;
    uint N_elec;
    uint Nmat;
    uint Nk;
    uint ground_state_degeneracy;
    arma::vec occ_energies;
    arma::vec vir_energies;
    arma::vec exc_energies;
    arma::vec kgrid;
    arma::umat occ_states;
    arma::umat vir_states;
    arma::umat excitations;
    #if NDIM == 2
      arma::umat vir_N_to_1_mat;
    #elif NDIM == 3
      arma::ucube vir_N_to_1_mat;
    #endif // NDIM;
    arma::umat inv_exc_mat;
    void (*MatVecProduct_func)(arma::vec& v, arma::vec& Mv);
    arma::mat (*Matrix_generator)();
    uint dav_its;
    arma::vec dav_vals;
    uint num_guess_evecs;
    uint Dav_blocksize;
    uint Dav_Num_evals;
    uint Dav_nconv;
    scalar Dav_tol;
    scalar Dav_final_val;
    uint Dav_maxits;
    uint Dav_minits;
    uint Dav_maxsubsize;
    scalar Dav_time;
}
