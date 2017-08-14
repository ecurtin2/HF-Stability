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
    scalar twoE_parameter_1dCase;
    bool use_delta_1D;
    std::string computation_started;
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
    # if NDIM == 1
      arma::uvec vir_N_to_1_mat;
    # endif
    # if NDIM == 2
      arma::umat vir_N_to_1_mat;
    # endif
    # if NDIM == 3
      arma::ucube vir_N_to_1_mat;
    #endif
    arma::umat inv_exc_mat;
    void (*MatVecProduct_func)(arma::vec& v, arma::vec& Mv);
    arma::mat (*Matrix_generator)();
    arma::vec exact_evals;
    uint dav_its;
    arma::vec dav_vals;
    uint num_guess_evecs;
    uint dav_blocksize;
    uint dav_num_evals;
    uint dav_nconv;
    scalar dav_tol;
    scalar dav_min_eval;
    uint dav_maxits;
    uint Dav_minits;
    uint dav_max_subsize;
    scalar dav_time;
    int N_MV_PROD;
}
