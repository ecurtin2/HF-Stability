#include "HFS_base_funcs.hpp"

namespace HFS {

    double exchange(arma::umat& inp_states, arma::uword i) {

        double exch = 0.0;
        arma::vec ki(NDIM), k2(NDIM);
        for (unsigned j = 0; j < NDIM; ++j) {
            ki(j) = HFS::kgrid(inp_states(i, j));
        }
        for (arma::uword k = 0; k < HFS::Nocc; ++k) {
            for (unsigned j = 0; j < NDIM; ++j) {
                k2(j) = HFS::kgrid(HFS::occ_states(k, j));
            }
            exch += HFS::two_electron(ki, k2);
        }
        exch *= -1.0;
        return exch;
    }

    double two_electron(arma::vec& k1, arma::vec& k2) {
        arma::vec k = k1 - k2;
        HFS::to_first_BZ(k);
        double norm = arma::norm(k);
        if (norm < SMALLNUMBER) {
            return 0.0;
        }else{
            #if NDIM == 2
                return HFS::two_e_const / norm;
            #elif NDIM == 3
                return HFS::two_e_const / (norm * norm);
            #endif
        }
    }

    double two_electron_safe(arma::vec& k1, arma::vec& k2, arma::vec& k3, arma::vec& k4) {
        // Same as two_electron, except checks for momentum conservation
        // In the other, conservation is assumed
        arma::vec k(NDIM);

        k = k1 + k2 - k3 - k4;
        HFS::to_first_BZ(k);

        // If not momentum conserving:
        if (arma::any(arma::abs(k) > SMALLNUMBER)) {
                return 0.0;
        }

        k =  k1 - k3;
        HFS::to_first_BZ(k);
        double norm = arma::norm(k);

        if (norm < SMALLNUMBER) {
            return 0.0;
        }else{
            return HFS::two_e_const / std::pow(norm, NDIM - 1);
        }
    }

    arma::uvec k_to_index(arma::vec& k) {
        arma::vec idx = arma::round((k + HFS::kmax) / HFS::deltaK);
        arma::uvec indices = arma::conv_to<arma::uvec>::from(idx);
        return indices;
    }

    arma::umat k_to_index(arma::mat& k) {
        arma::mat idx = arma::round((k + HFS::kmax) / HFS::deltaK);
        arma::umat indices = arma::conv_to<arma::umat>::from(idx);
        return indices;
    }

    arma::vec occ_idx_to_k(arma::uword idx) {
        arma::vec k(NDIM);
        for (unsigned i = 0; i < NDIM; ++i) {
            k[i] = HFS::kgrid(HFS::occ_states(idx, i));
        }
        return k;
    }

    arma::vec vir_idx_to_k(arma::uword idx) {
        arma::vec k(NDIM);
        for (unsigned i=0; i < NDIM; ++i) {
            k[i] = HFS::kgrid(vir_states(idx, i));
        }
        return k;

    }

    int KronDelta(arma::uword i, arma::uword j) {
        /* Returns 1 if i == j, else 0 */

        int val = 0;
        if (i == j) {
            val = 1;
        }
        return val;
    }

    std::vector<arma::vec> st_to_kikakjkb(arma::uword s, arma::uword t){
        /* Given excitation indices s and t, return a vector
           containing armadillo vectors of momentum. The order
           of the returned vectors is ki, ka, kj, kb
           where s: i -> a and t: j -> b */
        arma::uword i = HFS::excitations(s, 0);
        arma::uword a = HFS::excitations(s, 1);
        arma::uword j = HFS::excitations(t, 0);
        arma::uword b = HFS::excitations(t, 1);
        arma::vec ki = HFS::occ_idx_to_k(i);
        arma::vec kj = HFS::occ_idx_to_k(j);
        arma::vec ka = HFS::vir_idx_to_k(a);
        arma::vec kb = HFS::vir_idx_to_k(b);
        std::vector<arma::vec> klist = {ki, ka, kj, kb};
        return klist;
    }

}
