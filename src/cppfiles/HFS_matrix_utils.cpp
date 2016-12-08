#include "HFS_matrix_utils.hpp"
#include <map>

namespace HFS {

    // Start 1H Functions

    double calc_1A(arma::uword s, arma::uword t) {
        arma::uword i = HFS::excitations(s, 0);
        arma::uword a = HFS::excitations(s, 1);
        arma::uword j = HFS::excitations(t, 0);
        arma::uword b = HFS::excitations(t, 1);
        arma::vec ki(HFS::ndim), kj(HFS::ndim), ka(HFS::ndim), kb(HFS::ndim);
        for (unsigned idx = 0; idx < HFS::ndim; ++idx) {
            ki[idx] = HFS::kgrid(HFS::occ_states(i, idx));
            kj[idx] = HFS::kgrid(HFS::occ_states(j, idx));
            ka[idx] = HFS::kgrid(HFS::vir_states(a, idx));
            kb[idx] = HFS::kgrid(HFS::vir_states(b, idx));
        }
        double val = 0.0;
        if ((i == j) && (a == b)) {
            val = HFS::exc_energies(s);
        }
        val += 2.0 * HFS::two_electron_safe(ka, kj, ki, kb) - HFS::two_electron_safe(ka, kj, kb, ki);
        return val;
    }

    double calc_1B(arma::uword s, arma::uword t) {
        arma::uword i =  HFS::excitations(s, 0);
        arma::uword a =  HFS::excitations(s, 1);
        arma::uword j =  HFS::excitations(t, 0);
        arma::uword b =  HFS::excitations(t, 1);
        arma::vec ki(HFS::ndim), kj(HFS::ndim), ka(HFS::ndim), kb(HFS::ndim);
        for (unsigned idx = 0; idx < HFS::ndim; ++idx) {
            ki[idx] = HFS::kgrid(HFS::occ_states(i, idx));
            kj[idx] = HFS::kgrid(HFS::occ_states(j, idx));
            ka[idx] = HFS::kgrid(HFS::vir_states(a, idx));
            kb[idx] = HFS::kgrid(HFS::vir_states(b, idx));
        }
        return 2.0 * two_electron_safe(ka, kb, ki, kj) - two_electron_safe(ka, kb, kj, ki);
    }

    // End 1H functions
    // Start 3H functions

    double calc_3B(arma::uword s, arma::uword t) {
        std::vector<arma::vec> klist(4);
        arma::vec ki(HFS::ndim), kj(HFS::ndim), ka(HFS::ndim), kb(HFS::ndim);
        klist = HFS::st_to_kikakjkb(s, t);
        ki = klist[0];
        ka = klist[1];
        kj = klist[2];
        kb = klist[3];
        return -1.0 * HFS::two_electron_safe(ka, kb, kj, ki);
    }

    double calc_3A(arma::uword s, arma::uword t) {
        std::vector<arma::vec> klist(4);
        arma::vec ki(HFS::ndim), kj(HFS::ndim), ka(HFS::ndim), kb(HFS::ndim);
        klist = HFS::st_to_kikakjkb(s, t);
        ki = klist[0];
        ka = klist[1];
        kj = klist[2];
        kb = klist[3];

        double val = 0.0;
        if (s == t) {
            val = HFS::exc_energies(s);
        }
        val += -1.0 * HFS::two_electron_safe(ka, kj, kb, ki);
        return val;
    }

    double calc_3H(arma::uword i, arma::uword j) {
        if (i < HFS::Nexc) {
            if (j < HFS::Nexc) {
                // First quadrant
                return HFS::calc_3A(i, j);
            }else{
                // Second quadrant
                return HFS::calc_3B(i, j - HFS::Nexc);
            }
        }else{
            if (j < HFS::Nexc) {
                // Third quadrant
                return HFS::calc_3B(i - HFS::Nexc, j);
            }else{
                // Fourth quadrant
                return HFS::calc_3A(i - HFS::Nexc, j - HFS::Nexc);
            }
        }
    }

    arma::vec matvec_prod_3A(arma::vec& v) {
        arma::vec Mv(HFS::Nexc, arma::fill::zeros);
        for (arma::uword s = 0; s < HFS::Nexc; ++s) {
            arma::uword i = HFS::excitations(s, 0), a = HFS::excitations(s, 1);
            arma::vec ki(HFS::ndim), ka(HFS::ndim);
            ki = HFS::occ_idx_to_k(i);
            ka = HFS::vir_idx_to_k(a);
            for (arma::uword j = 0; j < HFS::Nocc; ++j) {
                arma::vec kj(HFS::ndim), kb(HFS::ndim);
                kj = HFS::occ_idx_to_k(j);
                kb = ka + kj - ki; // Momentum conservation for <aj|bi>
                HFS::to_first_BZ(kb);
                if (arma::norm(kb) > (HFS::kf + SMALLNUMBER)) {
                    // only if momentum conserving state is virtual
                    arma::uword t = HFS::kb_j_to_t(kb, j);
                    if (s == t) {
                        Mv(s) += HFS::exc_energies(s) * v(t);
                    } else {
                        Mv(s) += -1.0 * HFS::two_electron(ka, kb) * v(t);
                    }

                }
            }
        }
        return Mv;
    }

    arma::vec matvec_prod_3B(arma::vec& v) {
        arma::vec Mv(HFS::Nexc, arma::fill::zeros);
        for (arma::uword s = 0; s < HFS::Nexc; ++s) {
            arma::uword i = HFS::excitations(s, 0), a = HFS::excitations(s, 1);
            arma::vec ki(HFS::ndim), ka(HFS::ndim);
            ki = HFS::occ_idx_to_k(i);
            ka = HFS::vir_idx_to_k(a);
            for (arma::uword j = 0; j < HFS::Nocc; ++j) {
                arma::vec kj(HFS::ndim), kb(HFS::ndim);
                kj = HFS::occ_idx_to_k(j);
                kb = kj + ki - ka; // Momentum conservation for <ab|ji>
                HFS::to_first_BZ(kb);
                if (arma::norm(kb) > (HFS::kf + SMALLNUMBER)) {
                    // only if momentum conserving state is virtual
                    arma::uword t = HFS::kb_j_to_t(kb, j);
                    Mv(s) += -1.0 * HFS::two_electron(ka, kj) * v(t);
                }
            }
        }
        return Mv;
    }

    void matvec_prod_3H(arma::vec& v, arma::vec& Mv) {
        /*  The matrix-vector multiplication | A  B | |v1|  =  | Mv1 |
                                             | B* A*| |v2|     | Mv2 |
            Factors into 4 matrix vector multiplications (x, y are vectors; A, B are matrices)
            Mv1 = A*v1 + B*v2
            Mv2 = B*v1 + A*v2
        */
        arma::vec v1 = v.head(HFS::Nexc);
        arma::vec v2 = v.tail(HFS::Nexc);

        arma::vec Mv1 = HFS::matvec_prod_3A(v1) + HFS::matvec_prod_3B(v2);
        arma::vec Mv2 = HFS::matvec_prod_3B(v1) + HFS::matvec_prod_3A(v2);
        Mv = arma::join_cols(Mv1, Mv2);
    }

    // End 3H functions

    // Start Hprime functions

    arma::vec matvec_prod_Aprime_diag(arma::vec& v){
        arma::vec Mv(HFS::Nexc, arma::fill::zeros);
        for (arma::uword s = 0; s < HFS::Nexc; ++s) {
            arma::uword i = HFS::excitations(s, 0), a = HFS::excitations(s, 1);
            arma::vec ki(HFS::ndim), ka(HFS::ndim);
            ki = HFS::occ_idx_to_k(i);
            ka = HFS::vir_idx_to_k(a);
            for (arma::uword j = 0; j < HFS::Nocc; ++j) {
                arma::vec kj(HFS::ndim), kb(HFS::ndim);
                kj = HFS::occ_idx_to_k(j);
                kb = ka + kj - ki; // Momentum conservation for <aj|ib> or <aj|bi>
                HFS::to_first_BZ(kb);
                if (arma::norm(kb) > (HFS::kf + SMALLNUMBER)) {
                    // only if momentum conserving state is virtual
                    arma::uword t = HFS::kb_j_to_t(kb, j);
                    if (s == t) {
                        Mv(s) += HFS::exc_energies(s) * v(t);
                    } else {
                        Mv(s) += (HFS::two_electron(ka, ki) - HFS::two_electron(ka, kb)) * v(t);
                    }

                }
            }
        }
        return Mv;
    }

    arma::vec matvec_prod_Aprime_offdiag(arma::vec& v) {
        arma::vec Mv(HFS::Nexc, arma::fill::zeros);
        for (arma::uword s = 0; s < HFS::Nexc; ++s) {
            arma::uword i = HFS::excitations(s, 0), a = HFS::excitations(s, 1);
            arma::vec ki(HFS::ndim), ka(HFS::ndim);
            ki = HFS::occ_idx_to_k(i);
            ka = HFS::vir_idx_to_k(a);
            for (arma::uword j = 0; j < HFS::Nocc; ++j) {
                arma::vec kj(HFS::ndim), kb(HFS::ndim);
                kj = HFS::occ_idx_to_k(j);
                kb = ka + kj - ki; // Momentum conservation for <aj|ib>
                HFS::to_first_BZ(kb);
                if (arma::norm(kb) > (HFS::kf + SMALLNUMBER)) {
                    // only if momentum conserving state is virtual
                    arma::uword t = HFS::kb_j_to_t(kb, j);
                    Mv(s) += HFS::two_electron(ka, ki) * v(t);
                }
            }
        }
        return Mv;
    }

    arma::vec matvec_prod_Aprime(arma::vec& v) {
        arma::vec v1 = v.head(HFS::Nexc);
        arma::vec v2 = v.tail(HFS::Nexc);

        arma::vec Mv1 = HFS::matvec_prod_Aprime_diag(v1) + HFS::matvec_prod_Aprime_offdiag(v2);
        arma::vec Mv2 = HFS::matvec_prod_Aprime_offdiag(v1) + HFS::matvec_prod_Aprime_diag(v2);

        arma::vec Mv = arma::join_cols(Mv1, Mv2);
        return Mv;
    }

    arma::vec matvec_prod_Bprime_diag(arma::vec& v){
        arma::vec Mv(HFS::Nexc, arma::fill::zeros);
        for (arma::uword s = 0; s < HFS::Nexc; ++s) {
            arma::uword i = HFS::excitations(s, 0), a = HFS::excitations(s, 1);
            arma::vec ki(HFS::ndim), ka(HFS::ndim);
            ki = HFS::occ_idx_to_k(i);
            ka = HFS::vir_idx_to_k(a);
            for (arma::uword j = 0; j < HFS::Nocc; ++j) {
                arma::vec kj(HFS::ndim), kb(HFS::ndim);
                kj = HFS::occ_idx_to_k(j);
                kb = kj + ki - ka; // Momentum conservation for <ab|ij> or <ab|ji>
                HFS::to_first_BZ(kb);
                if (arma::norm(kb) > (HFS::kf + SMALLNUMBER)) {
                    // only if momentum conserving state is virtual
                    arma::uword t = HFS::kb_j_to_t(kb, j);
                    Mv(s) += (HFS::two_electron(ka, ki) - HFS::two_electron(ka, kj)) * v(t);
                }
            }
        }
        return Mv;
    }

    arma::vec matvec_prod_Bprime_offdiag(arma::vec& v) {
        arma::vec Mv(HFS::Nexc, arma::fill::zeros);
        for (arma::uword s = 0; s < HFS::Nexc; ++s) {
            arma::uword i = HFS::excitations(s, 0), a = HFS::excitations(s, 1);
            arma::vec ki(HFS::ndim), ka(HFS::ndim);
            ki = HFS::occ_idx_to_k(i);
            ka = HFS::vir_idx_to_k(a);
            for (arma::uword j = 0; j < HFS::Nocc; ++j) {
                arma::vec kj(HFS::ndim), kb(HFS::ndim);
                kj = HFS::occ_idx_to_k(j);
                kb = kj + ki - ka; // Momentum conservation for <ab|ji>
                HFS::to_first_BZ(kb);
                if (arma::norm(kb) > (HFS::kf + SMALLNUMBER)) {
                    // only if momentum conserving state is virtual
                    arma::uword t = HFS::kb_j_to_t(kb, j);
                    Mv(s) += HFS::two_electron(ka, ki) * v(t);
                }
            }
        }
        return Mv;
    }

    arma::vec matvec_prod_Bprime(arma::vec& v) {
        arma::vec v1 = v.head(HFS::Nexc);
        arma::vec v2 = v.tail(HFS::Nexc);

        arma::vec Mv1 = HFS::matvec_prod_Bprime_diag(v1) + HFS::matvec_prod_Bprime_offdiag(v2);
        arma::vec Mv2 = HFS::matvec_prod_Bprime_offdiag(v1) + HFS::matvec_prod_Bprime_diag(v2);

        arma::vec Mv = arma::join_cols(Mv1, Mv2);
        return Mv;
    }

    void matvec_prod_Hprime(arma::vec& v, arma::vec& Mv) {
        /*  The matrix-vector multiplication | A  B | |v1|  =  | Mv1 |
                                             | B* A*| |v2|     | Mv2 |
            Factors into 4 matrix vector multiplications
            Mv1 = A*v1 + B*v2
            Mv2 = B*v1 + A*v2
        */
        arma::vec v1 = v.head(2*HFS::Nexc);
        arma::vec v2 = v.tail(2*HFS::Nexc);

        arma::vec Mv1 = HFS::matvec_prod_Aprime(v1) + HFS::matvec_prod_Bprime(v2);
        arma::vec Mv2 = HFS::matvec_prod_Bprime(v1) + HFS::matvec_prod_Aprime(v2);
        Mv = arma::join_cols(Mv1, Mv2);
    }

    double calc_Hprime(arma::uword s, arma::uword t) {
        // Hprime is 4Nexc x 4Nexc

        arma::uword N = 2 * HFS::Nexc;
        double Hprime;

        if ((s < N) && (t < N)) {         // top left
            Hprime = HFS::calc_Aprime(s, t);
        } else if ((s < N) && (t >= N)) { // top right
            Hprime = HFS::calc_Bprime(s, t - N);
        } else if ((s >= N) && (t < N)) { // bottom left
            Hprime = HFS::calc_Bprime(s - N, t);
        } else if ((s >= N) && (t >= N)) { // bottom right
            Hprime = HFS::calc_Aprime(s - N, t - N);
        }
        return Hprime;
    }

    double calc_Aprime(arma::uword s, arma::uword t) {
        double Aprime;
        arma::uword N = HFS::Nexc;
        std::vector<arma::vec> klist(4);
        arma::vec ki(HFS::ndim), kj(HFS::ndim), ka(HFS::ndim), kb(HFS::ndim);

        if ((s < N) && (t < N)) { // top left
            klist = HFS::st_to_kikakjkb(s, t);
            ki = klist[0];
            ka = klist[1];
            kj = klist[2];
            kb = klist[3];

            if (s == t) {
                Aprime = HFS::KronDelta(s, t) * HFS::exc_energies(s);
            } else {
                Aprime = HFS::two_electron_safe(ka, kj, ki, kb) - HFS::two_electron_safe(ka, kj, kb, ki);
            }

        } else if ((s < N) && (t >= N)) { // top right
            klist = HFS::st_to_kikakjkb(s, t - N);
            ki = klist[0];
            ka = klist[1];
            kj = klist[2];
            kb = klist[3];
            Aprime = HFS::two_electron_safe(ka, kj, ki, kb);

        } else if ((s >= N) && (t < N)) { // bottom left
            klist = HFS::st_to_kikakjkb(s - N, t);
            ki = klist[0];
            ka = klist[1];
            kj = klist[2];
            kb = klist[3];
            Aprime = HFS::two_electron_safe(ka, kj, ki, kb);

        } else if ((s >= N) && (t >= N)) { // bottom right
            klist = HFS::st_to_kikakjkb(s - N, t - N);
            ki = klist[0];
            ka = klist[1];
            kj = klist[2];
            kb = klist[3];
            if (s == t) {
                Aprime = HFS::KronDelta(s, t) * HFS::exc_energies(s - N);
            } else {
                Aprime = HFS::two_electron_safe(ka, kj, ki, kb) - HFS::two_electron_safe(ka, kj, kb, ki);
            }
        }
        return Aprime;
    }

    double calc_Bprime(arma::uword s, arma::uword t) {
        double Bprime;
        arma::uword N = HFS::Nexc;
        std::vector<arma::vec> klist(4);
        arma::vec ki(HFS::ndim), kj(HFS::ndim), ka(HFS::ndim), kb(HFS::ndim);

        if ((s < N) && (t < N)) { // top left
            klist = HFS::st_to_kikakjkb(s, t);
            ki = klist[0];
            ka = klist[1];
            kj = klist[2];
            kb = klist[3];
            Bprime = HFS::two_electron_safe(ka, kb, ki, kj) - HFS::two_electron_safe(ka, kb, kj, ki);

        } else if ((s < N) && (t >= N)) { // top right
            klist = HFS::st_to_kikakjkb(s, t - N);
            ki = klist[0];
            ka = klist[1];
            kj = klist[2];
            kb = klist[3];
            Bprime = HFS::two_electron_safe(ka, kb, ki, kj);

        } else if ((s >= N) && (t < N)) { // bottom left
            klist = HFS::st_to_kikakjkb(s - N, t);
            ki = klist[0];
            ka = klist[1];
            kj = klist[2];
            kb = klist[3];
            Bprime = HFS::two_electron_safe(ka, kb, ki, kj);

        } else if ((s >= N) && (t >= N)) { // bottom right
            klist = HFS::st_to_kikakjkb(s - N, t - N);
            ki = klist[0];
            ka = klist[1];
            kj = klist[2];
            kb = klist[3];
            Bprime = HFS::two_electron_safe(ka, kb, ki, kj) - HFS::two_electron_safe(ka, kb, kj, ki);
        }
        return Bprime;
    }


    // End Hprime functions

    arma::uword kb_j_to_t(arma::vec& kb, arma::uword j) {
        arma::uvec b_N_idx =  k_to_index(kb);
        arma::uword b = HFS::vir_N_to_1_mat(b_N_idx(0), b_N_idx(1));
        arma::uword t = HFS::inv_exc_mat(j, b);
        return t;
    }

    void set_case_opts() {

        if (HFS::mycase == "cRHF2cUHF") {
            HFS::MatVecProduct_func = HFS::matvec_prod_3H;
            HFS::Matrix_func = HFS::calc_3H;
            HFS::Nmat = 2 * HFS::Nexc;

        } else if (HFS::mycase == "cUHF2cUHF") {
            HFS::MatVecProduct_func = HFS::matvec_prod_Hprime;
            HFS::Matrix_func = HFS::calc_Hprime;
            HFS::Nmat = 4 * HFS::Nexc;

        } else {
            std::cout << "Invalid HFS::mycase, use one of: cRHF2cUHF  cRHF2cUHF" << std::endl;
            exit(EXIT_FAILURE);
        }

    }

} // Namespace HFS
