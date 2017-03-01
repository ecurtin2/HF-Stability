#include "parameters.hpp"

namespace HFS{
    double bzone_length;
    double vol;
    double rs;
    double kf;
    double kmax;
    double fermi_energy;
    double cond_number;
    double two_e_const;
    double deltaK;
    double Total_Calculation_Time;
    std::string Computation_Starttime;
    std::string OutputFileName;
    std::string mycase;
    arma::uword Nocc;
    arma::uword Nvir;
    arma::uword Nexc;
    arma::uword N_elec;
    arma::uword Nmat;
    unsigned Nk;
    unsigned ground_state_degeneracy;
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
