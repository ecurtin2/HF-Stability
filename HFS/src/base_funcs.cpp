/** @file base_funcs.cpp
   @author Evan Curtin
   @version Revision 0.1
   @brief Implementation for low-level functions used throughout the codebase.

   @date Wednesday, 05 Jan, 2017
 */


#include "base_funcs.hpp"
#include <cmath>
# if NDIM == 1
#include <boost/math/special_functions/expint.hpp>
# endif  // NDIM

namespace HFS {

scalar exchange(const arma::umat& states, const uint i) {

        scalar exch = 0.0;
        arma::vec ki(NDIM), k2(NDIM);
        for (uint j = 0; j < NDIM; ++j) {
                ki(j) = HFS::kgrid(states(j, i));
        }
        for (uint k = 0; k < HFS::Nocc; ++k) {
                for (uint j = 0; j < NDIM; ++j) {
                        k2(j) = HFS::kgrid(HFS::occ_states(j, k));
                }
                exch += HFS::twoElectron(ki, k2);
        }
        exch *= -1.0;
        return exch;
}

scalar twoElectron(const arma::vec& k1, const arma::vec& k2) {
        std::array<scalar, NDIM> k_ary;
        for (unsigned i = 0; i < NDIM; ++i) {
                k_ary[i] = k1[i] - k2[i];
        }

        HFS::toFirstBrillouinZone(k_ary);
        scalar norm = 0.0;
        for (auto k : k_ary) {
                norm += k * k;
        }
        norm = sqrt(norm);
        if (norm < SMALLNUMBER) {
                return 0.0;
        }else{
            # if NDIM == 1
                return exp(norm * norm * HFS::two_e_const * HFS::two_e_const)
                       * boost::math::expint(-norm * norm * HFS::two_e_const * HFS::two_e_const);
            # elif NDIM == 2
                return HFS::two_e_const / norm;
            # elif NDIM == 3
                return HFS::two_e_const / (norm * norm);
            #endif
        }

}

scalar twoElectronSafe(const arma::vec& k1
                      ,const arma::vec& k2
                      ,const arma::vec& k3
                      ,const arma::vec& k4) {
        // Same as two_electron, except checks for momentum conservation
        // In the other, conservation is assumed
        arma::vec k(NDIM);

        k = k1 + k2 - k3 - k4;
        HFS::toFirstBrillouinZone(k);

        // If not momentum conserving:
        if (arma::any(arma::abs(k) > SMALLNUMBER)) {
                return 0.0;
        }

        k =  k1 - k3;
        HFS::toFirstBrillouinZone(k);
        scalar norm = arma::norm(k);

        if (norm < SMALLNUMBER) {
                return 0.0;
        }else{
            # if NDIM == 1
                return exp(norm * norm * HFS::two_e_const * HFS::two_e_const)
                       * boost::math::expint(-norm * norm * HFS::two_e_const * HFS::two_e_const);
            # elif NDIM == 2
                return HFS::two_e_const / norm;
            # elif NDIM == 3
                return HFS::two_e_const / (norm * norm);
            #endif
        }
}

void kToIndex(const arma::vec& k, arma::uvec& idx) {
        for (uint i = 0; i < NDIM; ++i) {
                idx[i] = std::round((k[i] + HFS::kmax) / HFS::deltaK);
        }
}

arma::umat kToIndex(const arma::mat& k) {
        arma::mat idx = arma::round((k + HFS::kmax) / HFS::deltaK);
        arma::umat indices = arma::conv_to<arma::umat>::from(idx);
        return indices;
}

void occIndexToK(const uint idx, arma::vec&k) {
        for (uint i = 0; i < NDIM; ++i) {
                k[i] = HFS::kgrid(HFS::occ_states(i, idx));
        }
}

arma::vec virIndexToK(const uint idx) {
        arma::vec k(NDIM);
        for (uint i=0; i < NDIM; ++i) {
                k[i] = HFS::kgrid(vir_states(i, idx));
        }
        return k;

}

int kroneckerDelta(const uint i, const uint j) {
        /* Returns 1 if i == j, else 0 */

        int val = 0;
        if (i == j) {
                val = 1;
        }
        return val;
}

std::vector<arma::vec> stToKiKaKjKb(const uint s, const uint t){
        /* Given excitation indices s and t, return a vector
           containing armadillo vectors of momentum. The order
           of the returned vectors is ki, ka, kj, kb
           where s: i -> a and t: j -> b */
        uint i = HFS::excitations(0, s);
        uint a = HFS::excitations(1, s);
        uint j = HFS::excitations(0, t);
        uint b = HFS::excitations(1, t);
        arma::vec ki(NDIM); HFS::occIndexToK(i, ki);
        arma::vec kj(NDIM); HFS::occIndexToK(j, kj);
        arma::vec ka = HFS::virIndexToK(a);
        arma::vec kb = HFS::virIndexToK(b);
        std::vector<arma::vec> klist = {ki, ka, kj, kb};
        return klist;
}

}
