#ifndef HFS_matrix_utils_included
#define HFS_matrix_utils_included

#include "HFS_params.hpp"
#include "HFS_base_funcs.hpp"
#include "HFS_params_calc.hpp"

namespace HFS{

    // 3H
    extern arma::vec matvec_prod_3A(arma::vec&);
    extern arma::vec matvec_prod_3B(arma::vec&);
    extern void matvec_prod_3H(arma::vec&, arma::vec& Mv);
    extern double calc_3A(arma::uword, arma::uword);
    extern double calc_3B(arma::uword, arma::uword);
    extern double calc_3H(arma::uword, arma::uword);

    // Hprime
    extern arma::vec matvec_prod_Aprime(arma::vec&);
    extern arma::vec matvec_prod_Bprime(arma::vec&);
    extern void matvec_prod_Hprime(arma::vec&, arma::vec& Mv);
    extern double calc_Aprime(arma::uword s, arma::uword t);
    extern double calc_Bprime(arma::uword s, arma::uword t);
    extern double calc_Hprime(arma::uword s, arma::uword t);

    // 1H
    extern double calc_1B(arma::uword, arma::uword);
    extern double calc_1A(arma::uword, arma::uword);

    // Utilities
    extern arma::uword kb_j_to_t(arma::vec&, arma::uword);
    extern void set_case_opts();
}

#endif // HFS_matrix_utils_included
