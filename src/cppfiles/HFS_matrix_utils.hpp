#ifndef HFS_matrix_utils_included
#define HFS_matrix_utils_included

#include "HFS_params.hpp"
#include "HFS_base_funcs.hpp"
#include "HFS_params_calc.hpp"

namespace HFS{

    // 3H
    extern void Mv_3A(arma::vec&, arma::vec& Mv);
    extern void Mv_3B(arma::vec&, arma::vec& Mv);
    extern void Mv_3H(arma::vec&, arma::vec& Mv);
    extern double calc_3A(arma::uword, arma::uword);
    extern double calc_3B(arma::uword, arma::uword);
    extern double calc_3H(arma::uword, arma::uword);

    // Hprime
    extern void Mv_Hprime(arma::vec&v, arma::vec& Mv);
    extern void Mv_Aprime(arma::vec&v, arma::vec& Mv);
    extern void Mv_Bprime(arma::vec&v, arma::vec& Mv);
    extern void Mv_AprimeDiag(arma::vec& v, arma::vec& Mv);
    extern void Mv_AprimeOffDiag(arma::vec& v, arma::vec& Mv);
    extern void Mv_BprimeDiag(arma::vec& v, arma::vec& Mv);
    extern void Mv_BprimeOffDiag(arma::vec& v, arma::vec& Mv);
    extern double calc_Aprime(arma::uword s, arma::uword t);
    extern double calc_Bprime(arma::uword s, arma::uword t);
    extern double calc_Hprime(arma::uword s, arma::uword t);

    // 1H
    extern double calc_1B(arma::uword, arma::uword);
    extern double calc_1A(arma::uword, arma::uword);

    // Utilities
    extern arma::uword kb_j_to_t(arma::vec&, arma::uword);
    extern void set_case_opts();
    extern void Factorize2by2Mv(arma::vec& v, arma::vec& Mv
                         , void (*Av)(arma::vec&, arma::vec&)
                         , void (*Bv)(arma::vec&, arma::vec&)
                         , void (*Cv)(arma::vec&, arma::vec&)
                         , void (*Dv)(arma::vec&, arma::vec&)
                         );

}

#endif // HFS_matrix_utils_included
