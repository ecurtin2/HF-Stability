#ifndef HFS_base_funcs_included
#define HFS_base_funcs_included

#include "HFS_params.hpp"

namespace HFS{
    extern double exchange(arma::umat&, arma::uword);
    extern double two_electron(arma::vec&, arma::vec&);
    extern double two_electron_safe(arma::vec&, arma::vec&, arma::vec&, arma::vec&); // checks momentum conserve, used in calc_matrix(i,j)
    extern void to_first_BZ(arma::vec&);
    extern bool is_occ(double);
    extern arma::uvec k_to_index(arma::vec&);
    extern arma::umat k_to_index(arma::mat&);
    extern arma::vec occ_idx_to_k(arma::uword);
    extern arma::vec vir_idx_to_k(arma::uword);
    extern int KronDelta(arma::uword, arma::uword);
    extern std::vector<arma::vec> st_to_kikakjkb(arma::uword s, arma::uword t);
}

#endif // HFS_base_funcs_included
