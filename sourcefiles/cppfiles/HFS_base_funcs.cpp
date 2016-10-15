#include "HFS_base_funcs.h"

namespace HFS {

    double exchange(arma::umat& inp_states, arma::uword i) {

        double exch = 0.0;
        arma::vec ki(ndim), k2(ndim);
        for (int j = 0; j < HFS::ndim; ++j) {
            ki(j) = HFS::kgrid(inp_states(i, j));
        }
        for (arma::uword k = 0; k < HFS::Nocc; ++k) {
            for (int j = 0; j < HFS::ndim; ++j) {
                k2(j) = HFS::kgrid(HFS::occ_states(k, j));
            }
            exch += HFS::two_electron(ki, k2);
        }
        exch *= -1.0;
        return exch;
    }

    double two_electron(arma::vec& k1, arma::vec& k2) {
        double norm = 0.0;
        arma::vec k(HFS::ndim);
        k = k1 - k2;

        HFS::to_first_BZ(k);
        norm = arma::norm(k);
        if (norm < 10E-10) {
            return 0.0;
        }else{
            return HFS::two_e_const / std::pow(norm, HFS::ndim - 1);
        }
    }

    double two_electron_check(arma::vec& k1, arma::vec& k2, arma::vec& k3, arma::vec& k4) {
        // Same as two_electron, except checks for momentum conservation
        // In the other, conservation is assumed
        arma::vec k(HFS::ndim);

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
            return HFS::two_e_const / std::pow(norm, HFS::ndim - 1);
            //return HFS::two_e_const / norm;   //  < 1% speedup in davidson, keep general
        }
    }

    void to_first_BZ(arma::vec& k) {
        // Translate to first brillioun zone, defined on the
        // interval [-pi/a .. pi/a)

        for (int i = 0; i < HFS::ndim; ++i) {
            if (k[i] < -HFS::kmax - SMALLNUMBER) {
                k[i] += HFS::bzone_length;
            }else if (k[i] > HFS::kmax - SMALLNUMBER) {
                k[i] -= HFS::bzone_length;
            }
        }
    }

    bool is_vir(double k) {
            return (k <= HFS::kf + SMALLNUMBER);
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
        arma::vec k(HFS::ndim);
        for (int i = 0; i < HFS::ndim; ++i) {
            k[i] = HFS::kgrid(HFS::occ_states(idx, i));
        }
        return k;
    }

    arma::vec vir_idx_to_k(arma::uword idx) {
        arma::vec k(HFS::ndim);
        for (int i=0; i < HFS::ndim; ++i) {
            k[i] = HFS::kgrid(vir_states(idx, i));
        }
        return k;

    }

}
