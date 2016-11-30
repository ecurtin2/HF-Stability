#ifndef HFS_matrix_utils_included
#define HFS_matrix_utils_included

#include "HFS_params.hpp"
#include "HFS_base_funcs.hpp"
#include "HFS_params_calc.hpp"

namespace HFS{
    extern arma::vec matvec_prod_3A(arma::vec&);
    extern arma::vec matvec_prod_3B(arma::vec&);
    extern arma::vec matvec_prod_3H(arma::vec&);
    extern void void_matvec_prod_3H(arma::vec& v, arma::vec& Mv);
    extern void matvec_prod_Hprime(arma::vec& v, arma::vec& Mv);
    extern void matvec_prod_Aprime(arma::vec& v, arma::vec& Mv);
    extern void matvec_prod_Bprime(arma::vec& v, arma::vec& Mv);
    extern double calc_1B(arma::uword, arma::uword);
    extern double calc_3B(arma::uword, arma::uword);
    extern double calc_1A(arma::uword, arma::uword);
    extern double calc_3A(arma::uword, arma::uword);
    extern double calc_3H(arma::uword, arma::uword);
    extern arma::uword kb_j_to_t(arma::vec&, arma::uword);  // only used in matvec_prod_3A & 3B
}

#endif // HFS_matrix_utils_included
