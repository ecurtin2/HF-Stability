#ifndef HFS_base_funcs_included
#define HFS_base_funcs_included

#include "HFS_params.hpp"

namespace HFS{
    extern double exchange(arma::umat&, arma::uword);
    extern double two_electron_safe(arma::vec&, arma::vec&, arma::vec&, arma::vec&); // checks momentum conserve, used in calc_matrix(i,j)
    inline void to_first_BZ(arma::vec& k){
        // Translate to first brillioun zone, defined on the
        // interval [-pi/a .. pi/a)
        for (unsigned i = 0; i < NDIM; ++i) {
            if (k[i] < -HFS::kmax - SMALLNUMBER) {
                k[i] += HFS::bzone_length;
            } else if (k[i] > HFS::kmax - SMALLNUMBER) {
                k[i] -= HFS::bzone_length;
            }
        }
    }
    extern double two_electron(arma::vec& k1, arma::vec& k2);

    inline bool is_occ(double k){
        return (k < (HFS::kf));
    }
    extern arma::uvec k_to_index(arma::vec& k);
    extern arma::umat k_to_index(arma::mat&);
    extern arma::vec occ_idx_to_k(arma::uword);
    extern arma::vec vir_idx_to_k(arma::uword);
    extern int KronDelta(arma::uword, arma::uword);
    extern std::vector<arma::vec> st_to_kikakjkb(arma::uword s, arma::uword t);
}

#endif // HFS_base_funcs_included
