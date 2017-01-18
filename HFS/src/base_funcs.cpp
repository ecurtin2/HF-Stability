/** @file base_funcs.cpp
@author Evan Curtin
@version Revision 0.1
@brief Implementation for low-level functions used throughout the codebase.

@date Wednesday, 05 Jan, 2017
*/


#include "base_funcs.hpp"

namespace HFS {

    double exchange(arma::umat& states, arma::uword i) {

        double exch = 0.0;
        arma::vec ki(NDIM), k2(NDIM);
        for (unsigned j = 0; j < NDIM; ++j) {
            ki(j) = HFS::kgrid(states(j, i));
        }
        for (arma::uword k = 0; k < HFS::Nocc; ++k) {
            for (unsigned j = 0; j < NDIM; ++j) {
                k2(j) = HFS::kgrid(HFS::occ_states(j, k));
            }
            exch += HFS::twoElectron(ki, k2);
        }
        exch *= -1.0;
        return exch;
    }

    double twoElectron(arma::vec& k1, arma::vec& k2) {
        arma::vec k = k1 - k2;
        HFS::toFirstBrillouinZone(k);
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

    double twoElectronSafe(arma::vec& k1, arma::vec& k2, arma::vec& k3, arma::vec& k4) {
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
        double norm = arma::norm(k);

        if (norm < SMALLNUMBER) {
            return 0.0;
        }else{
            return HFS::two_e_const / std::pow(norm, NDIM - 1);
        }
    }
/*
    arma::uvec kToIndex(arma::vec& k) {
        arma::vec idx = arma::round((k + HFS::kmax) / HFS::deltaK);
        arma::uvec indices = arma::conv_to<arma::uvec>::from(idx);
        return indices;
    }
*/
    void kToIndex(arma::vec& k, arma::uvec& idx) {
        for (unsigned i = 0; i < NDIM; ++i) {
            idx[i] = std::round((k[i] + HFS::kmax) / HFS::deltaK);
        }
    }



    arma::umat kToIndex(arma::mat& k) {
        arma::mat idx = arma::round((k + HFS::kmax) / HFS::deltaK);
        arma::umat indices = arma::conv_to<arma::umat>::from(idx);
        return indices;
    }

    void occIndexToK(arma::uword idx, arma::vec&k) {
        for (unsigned i = 0; i < NDIM; ++i) {
            k[i] = HFS::kgrid(HFS::occ_states(i, idx));
        }
    }

    arma::vec virIndexToK(arma::uword idx) {
        arma::vec k(NDIM);
        for (unsigned i=0; i < NDIM; ++i) {
            k[i] = HFS::kgrid(vir_states(i, idx));
        }
        return k;

    }

    int kroneckerDelta(arma::uword i, arma::uword j) {
        /* Returns 1 if i == j, else 0 */

        int val = 0;
        if (i == j) {
            val = 1;
        }
        return val;
    }

    std::vector<arma::vec> stToKiKaKjKb(arma::uword s, arma::uword t){
        /* Given excitation indices s and t, return a vector
           containing armadillo vectors of momentum. The order
           of the returned vectors is ki, ka, kj, kb
           where s: i -> a and t: j -> b */
        arma::uword i = HFS::excitations(0, s);
        arma::uword a = HFS::excitations(1, s);
        arma::uword j = HFS::excitations(0, t);
        arma::uword b = HFS::excitations(1, t);
        arma::vec ki(NDIM); HFS::occIndexToK(i, ki);
        arma::vec kj(NDIM); HFS::occIndexToK(j, kj);
        arma::vec ka = HFS::virIndexToK(a);
        arma::vec kb = HFS::virIndexToK(b);
        std::vector<arma::vec> klist = {ki, ka, kj, kb};
        return klist;
    }

}
