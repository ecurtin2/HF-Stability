#include "HFS_matrix_utils.hpp"
#include <map>

namespace HFS {

    // Start 1H Functions

    double calc_1A(arma::uword s, arma::uword t) {
        arma::uword i = HFS::excitations(s, 0);
        arma::uword a = HFS::excitations(s, 1);
        arma::uword j = HFS::excitations(t, 0);
        arma::uword b = HFS::excitations(t, 1);
        arma::vec ki(NDIM), kj(NDIM), ka(NDIM), kb(NDIM);
        for (unsigned idx = 0; idx < NDIM; ++idx) {
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
        arma::vec ki(NDIM), kj(NDIM), ka(NDIM), kb(NDIM);
        for (unsigned idx = 0; idx < NDIM; ++idx) {
            ki[idx] = HFS::kgrid(HFS::occ_states(i, idx));
            kj[idx] = HFS::kgrid(HFS::occ_states(j, idx));
            ka[idx] = HFS::kgrid(HFS::vir_states(a, idx));
            kb[idx] = HFS::kgrid(HFS::vir_states(b, idx));
        }
        return 2.0 * two_electron_safe(ka, kb, ki, kj) - two_electron_safe(ka, kb, kj, ki);
    }

    // End 1H functions

    // Start 3H functions
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

    double calc_3A(arma::uword s, arma::uword t) {
        std::vector<arma::vec> klist(4);
        arma::vec ki(NDIM), kj(NDIM), ka(NDIM), kb(NDIM);
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

    double calc_3B(arma::uword s, arma::uword t) {
        std::vector<arma::vec> klist(4);
        arma::vec ki(NDIM), kj(NDIM), ka(NDIM), kb(NDIM);
        klist = HFS::st_to_kikakjkb(s, t);
        ki = klist[0];
        ka = klist[1];
        kj = klist[2];
        kb = klist[3];
        return -1.0 * HFS::two_electron_safe(ka, kb, kj, ki);
    }

    void Mv_3H(arma::vec& v, arma::vec& Mv) {
        Mv.zeros();
        Factorize2by2Mv(v, Mv, Mv_3A, Mv_3B, Mv_3B, Mv_3A);
    }

    void Mv_3A(arma::vec& v, arma::vec& Mv) {
        for (arma::uword s = 0; s < HFS::Nexc; ++s) {
            arma::uword i = HFS::excitations(s, 0), a = HFS::excitations(s, 1);
            arma::vec ki(NDIM), ka(NDIM);
            ki = HFS::occ_idx_to_k(i);
            ka = HFS::vir_idx_to_k(a);
            for (arma::uword j = 0; j < HFS::Nocc; ++j) {
                arma::vec kj(NDIM), kb(NDIM);
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
    }

    void Mv_3B(arma::vec& v, arma::vec& Mv) {
        for (arma::uword s = 0; s < HFS::Nexc; ++s) {
            arma::uword i = HFS::excitations(s, 0), a = HFS::excitations(s, 1);
            arma::vec ki(NDIM), ka(NDIM);
            ki = HFS::occ_idx_to_k(i);
            ka = HFS::vir_idx_to_k(a);
            for (arma::uword j = 0; j < HFS::Nocc; ++j) {
                arma::vec kj(NDIM), kb(NDIM);
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
    }

    // End 3H functions

    // Start Hprime functions

    double calc_Hprime(arma::uword s, arma::uword t) {
        // Hprime is 4Nexc x 4Nexc

        arma::uword N = 2 * HFS::Nexc;
        double Hprime = 0.0;

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
        double Aprime = 0.0;
        arma::uword N = HFS::Nexc;
        std::vector<arma::vec> klist(4);
        arma::vec ki(NDIM), kj(NDIM), ka(NDIM), kb(NDIM);

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
        double Bprime = 0.0;
        arma::uword N = HFS::Nexc;
        std::vector<arma::vec> klist(4);
        arma::vec ki(NDIM), kj(NDIM), ka(NDIM), kb(NDIM);

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

    void Mv_Hprime(arma::vec& v, arma::vec& Mv) {
        Mv.zeros();
        Factorize2by2Mv(v, Mv, Mv_Aprime, Mv_Bprime, Mv_Bprime, Mv_Aprime);
    }

    void Mv_Aprime(arma::vec& v, arma::vec& Mv) {
        Factorize2by2Mv(v, Mv, Mv_AprimeDiag, Mv_AprimeOffDiag, Mv_AprimeOffDiag, Mv_AprimeDiag);
    }

    void Mv_Bprime(arma::vec&v, arma::vec& Mv) {
        Factorize2by2Mv(v, Mv, Mv_BprimeDiag, Mv_BprimeOffDiag, Mv_BprimeOffDiag, Mv_BprimeDiag);
    }

    void Mv_AprimeDiag(arma::vec& v, arma::vec& Mv) {
        for (arma::uword s = 0; s < HFS::Nexc; ++s) {
            arma::uword i = HFS::excitations(s, 0), a = HFS::excitations(s, 1);
            arma::vec ki(NDIM), ka(NDIM);
            ki = HFS::occ_idx_to_k(i);
            ka = HFS::vir_idx_to_k(a);
            for (arma::uword j = 0; j < HFS::Nocc; ++j) {
                arma::vec kj(NDIM), kb(NDIM);
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
    }

    void Mv_AprimeOffDiag(arma::vec& v, arma::vec& Mv) {
        for (arma::uword s = 0; s < HFS::Nexc; ++s) {
            arma::uword i = HFS::excitations(s, 0), a = HFS::excitations(s, 1);
            arma::vec ki(NDIM), ka(NDIM);
            ki = HFS::occ_idx_to_k(i);
            ka = HFS::vir_idx_to_k(a);
            for (arma::uword j = 0; j < HFS::Nocc; ++j) {
                arma::vec kj(NDIM), kb(NDIM);
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
    }

    void Mv_BprimeDiag(arma::vec& v, arma::vec& Mv) {
        for (arma::uword s = 0; s < HFS::Nexc; ++s) {
            arma::uword i = HFS::excitations(s, 0), a = HFS::excitations(s, 1);
            arma::vec ki(NDIM), ka(NDIM);
            ki = HFS::occ_idx_to_k(i);
            ka = HFS::vir_idx_to_k(a);
            for (arma::uword j = 0; j < HFS::Nocc; ++j) {
                arma::vec kj(NDIM), kb(NDIM);
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
    }

    void Mv_BprimeOffDiag(arma::vec& v, arma::vec& Mv) {
        for (arma::uword s = 0; s < HFS::Nexc; ++s) {
            arma::uword i = HFS::excitations(s, 0), a = HFS::excitations(s, 1);
            arma::vec ki(NDIM), ka(NDIM);
            ki = HFS::occ_idx_to_k(i);
            ka = HFS::vir_idx_to_k(a);
            for (arma::uword j = 0; j < HFS::Nocc; ++j) {
                arma::vec kj(NDIM), kb(NDIM);
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
    }


    // End Hprime functions

    void Factorize2by2Mv(arma::vec& v, arma::vec& Mv
                         , void (*Av)(arma::vec&, arma::vec&)
                         , void (*Bv)(arma::vec&, arma::vec&)
                         , void (*Cv)(arma::vec&, arma::vec&)
                         , void (*Dv)(arma::vec&, arma::vec&)
                         ) {
        /*
            Given | A B | | v1 | = | Mv1 |
                  | C D | | v2 | = | Mv2 |

            Factorize into four matrix vector products,
            Mv1 = Av1 + Bv2
            Mv2 = Cv1 + Dv2

            This is done without copy.
            Mv is assumed to be initialized to zero for all elements.
            The matrix vector products must not initialize, and must
            add in-place.
        */

        arma::vec v1 = arma::vec(&v[0], v.n_elem/2, false, true);
        arma::vec v2 = arma::vec(&v[v.n_elem/2], v.n_elem/2, false, true);
        arma::vec Mv1 = arma::vec(&Mv[0], v.n_elem/2, false, true);
        arma::vec Mv2 = arma::vec(&Mv[Mv.n_elem/2], v.n_elem/2, false, true);

        Av(v1, Mv1);
        Bv(v2, Mv1);
        Cv(v1, Mv2);
        Dv(v2, Mv2);
    }

    arma::uword kb_j_to_t(arma::vec& kb, arma::uword j) {
        arma::uvec b_N_idx =  k_to_index(kb);
        arma::uword b = HFS::vir_N_to_1_mat(b_N_idx(0), b_N_idx(1));
        arma::uword t = HFS::inv_exc_mat(j, b);
        return t;
    }

    void set_case_opts() {

        if (HFS::mycase == "cRHF2cUHF") {
            HFS::MatVecProduct_func = HFS::Mv_3H;
            HFS::Matrix_func = HFS::calc_3H;
            HFS::Nmat = 2 * HFS::Nexc;
        } else if (HFS::mycase == "cUHF2cUHF") {
            HFS::MatVecProduct_func = HFS::Mv_Hprime;
            HFS::Matrix_func = HFS::calc_Hprime;
            HFS::Nmat = 4 * HFS::Nexc;

        } else {
            std::cout << "Invalid HFS::mycase, use one of: cRHF2cUHF  cRHF2cUHF" << std::endl;
            exit(EXIT_FAILURE);
        }

    }

} // Namespace HFS
