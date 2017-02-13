#include "matrix_utils.hpp"

namespace HFS {

    // Start 1H Functions

    double calcFromIndices1A(arma::uword s, arma::uword t) {
        arma::uword i = HFS::excitations(0, s);
        arma::uword a = HFS::excitations(1, s);
        arma::uword j = HFS::excitations(0, t);
        arma::uword b = HFS::excitations(1, t);
        arma::vec ki(NDIM), kj(NDIM), ka(NDIM), kb(NDIM);
        for (unsigned idx = 0; idx < NDIM; ++idx) {
            ki[idx] = HFS::kgrid(HFS::occ_states(idx, i));
            kj[idx] = HFS::kgrid(HFS::occ_states(idx, j));
            ka[idx] = HFS::kgrid(HFS::vir_states(idx, a));
            kb[idx] = HFS::kgrid(HFS::vir_states(idx, b));
        }
        double val = 0.0;
        if ((i == j) && (a == b)) {
            val = HFS::exc_energies(s);
        }
        val += 2.0 * HFS::twoElectronSafe(ka, kj, ki, kb) - HFS::twoElectronSafe(ka, kj, kb, ki);
        return val;
    }

    double calcFromIndices1B(arma::uword s, arma::uword t) {
        arma::uword i =  HFS::excitations(0, s);
        arma::uword a =  HFS::excitations(1, s);
        arma::uword j =  HFS::excitations(0, t);
        arma::uword b =  HFS::excitations(1, t);
        arma::vec ki(NDIM), kj(NDIM), ka(NDIM), kb(NDIM);
        for (unsigned idx = 0; idx < NDIM; ++idx) {
            ki[idx] = HFS::kgrid(HFS::occ_states(idx, i));
            kj[idx] = HFS::kgrid(HFS::occ_states(idx, j));
            ka[idx] = HFS::kgrid(HFS::vir_states(idx, a));
            kb[idx] = HFS::kgrid(HFS::vir_states(idx, b));
        }
        return 2.0 * twoElectronSafe(ka, kb, ki, kj) - twoElectronSafe(ka, kb, kj, ki);
    }

    // End 1H functions

    // Start 3H functions
    double calcFromIndices3H(arma::uword i, arma::uword j) {
        if (i < HFS::Nexc) {
            if (j < HFS::Nexc) {
                // First quadrant
                return HFS::calcFromIndices3A(i, j);
            }else{
                // Second quadrant
                return HFS::calcFromIndices3B(i, j - HFS::Nexc);
            }
        }else{
            if (j < HFS::Nexc) {
                // Third quadrant
                return HFS::calcFromIndices3B(i - HFS::Nexc, j);
            }else{
                // Fourth quadrant
                return HFS::calcFromIndices3A(i - HFS::Nexc, j - HFS::Nexc);
            }
        }
    }

    double calcFromIndices3A(arma::uword s, arma::uword t) {
        std::vector<arma::vec> klist(4);
        arma::vec ki(NDIM), kj(NDIM), ka(NDIM), kb(NDIM);
        klist = HFS::stToKiKaKjKb(s, t);
        ki = klist[0];
        ka = klist[1];
        kj = klist[2];
        kb = klist[3];

        double val = 0.0;
        if (s == t) {
            val = HFS::exc_energies(s);
        }
        val += -1.0 * HFS::twoElectronSafe(ka, kj, kb, ki);
        return val;
    }

    double calcFromIndices3B(arma::uword s, arma::uword t) {
        std::vector<arma::vec> klist(4);
        arma::vec ki(NDIM), kj(NDIM), ka(NDIM), kb(NDIM);
        klist = HFS::stToKiKaKjKb(s, t);
        ki = klist[0];
        ka = klist[1];
        kj = klist[2];
        kb = klist[3];
        return -1.0 * HFS::twoElectronSafe(ka, kb, kj, ki);
    }

    void matrixVectorProduct3H(arma::vec& v, arma::vec& Mv) {
        Mv.zeros();
        //factorize2by2MatrixVectorProduct(v, Mv, matrixVectorProduct3A, matrixVectorProduct3B, matrixVectorProduct3B, matrixVectorProduct3A);
        matrixVectorProduct3A(v, Mv);
        matrixVectorProduct3B(v, Mv);
    }

    void matrixVectorProduct3A(arma::vec& v, arma::vec& Mv) {

        for (arma::uword s = 0; s < HFS::Nexc; ++s) {
            arma::uword i = HFS::excitations(0, s), a = HFS::excitations(1, s);
            arma::vec ki(NDIM), ka(NDIM);
            HFS::occIndexToK(i, ki);
            ka = HFS::virIndexToK(a);
            //#pragma omp parallel for
            for (arma::uword j = 0; j < HFS::Nocc; ++j) {
                arma::vec kj(NDIM), kb(NDIM);
                HFS::occIndexToK(j, kj);
                kb = ka + kj - ki; // Momentum conservation for <aj|bi>
                HFS::toFirstBrillouinZone(kb);
                if (arma::norm(kb) > (HFS::kf + SMALLNUMBER)) {
                    // only if momentum conserving state is virtual
                    arma::uword t = HFS::calcTfromKbAndJ(kb, j);
                    if (s == t) {
                        Mv(s) += HFS::exc_energies(s) * v(t);
                    } else {
                        Mv(s) += -1.0 * HFS::twoElectron(ka, kb) * v(t);
                    }
                }
            }
        }
    }

    void matrixVectorProduct3B(arma::vec& v, arma::vec& Mv) {

        for (arma::uword s = 0; s < HFS::Nexc; ++s) {
            arma::uword i = HFS::excitations(0, s), a = HFS::excitations(1, s);
            arma::vec ki(NDIM), ka(NDIM);
            HFS::occIndexToK(i, ki);
            ka = HFS::virIndexToK(a);
            //#pragma omp parallel for
            for (arma::uword j = 0; j < HFS::Nocc; ++j) {
                arma::vec kj(NDIM), kb(NDIM);
                HFS::occIndexToK(j, kj);
                kb = kj + ki - ka; // Momentum conservation for <ab|ji>
                HFS::toFirstBrillouinZone(kb);
                if (arma::norm(kb) > (HFS::kf + SMALLNUMBER)) {
                    // only if momentum conserving state is virtual
                    arma::uword t = HFS::calcTfromKbAndJ(kb, j);
                    Mv(s) += 1.0 * HFS::twoElectron(ka, kj) * v(t);
                }
            }
        }
    }

    // End 3H functions

    // Start Hprime functions

    double calcFromIndicesHprime(arma::uword s, arma::uword t) {
        // Hprime is 4Nexc x 4Nexc

        arma::uword N = 2 * HFS::Nexc;
        double Hprime = 0.0;

        if ((s < N) && (t < N)) {         // top left
            Hprime = HFS::calcFromIndicesAprime(s, t);
        } else if ((s < N) && (t >= N)) { // top right
            Hprime = HFS::calcFromIndicesBprime(s, t - N);
        } else if ((s >= N) && (t < N)) { // bottom left
            Hprime = HFS::calcFromIndicesBprime(s - N, t);
        } else if ((s >= N) && (t >= N)) { // bottom right
            Hprime = HFS::calcFromIndicesAprime(s - N, t - N);
        }
        return Hprime;
    }

    double calcFromIndicesAprime(arma::uword s, arma::uword t) {
        double Aprime = 0.0;
        arma::uword N = HFS::Nexc;
        std::vector<arma::vec> klist(4);
        arma::vec ki(NDIM), kj(NDIM), ka(NDIM), kb(NDIM);

        if ((s < N) && (t < N)) { // top left
            klist = HFS::stToKiKaKjKb(s, t);
            ki = klist[0];
            ka = klist[1];
            kj = klist[2];
            kb = klist[3];

            if (s == t) {
                Aprime = HFS::kroneckerDelta(s, t) * HFS::exc_energies(s);
            } else {
                Aprime = HFS::twoElectronSafe(ka, kj, ki, kb) - HFS::twoElectronSafe(ka, kj, kb, ki);
            }

        } else if ((s < N) && (t >= N)) { // top right
            klist = HFS::stToKiKaKjKb(s, t - N);
            ki = klist[0];
            ka = klist[1];
            kj = klist[2];
            kb = klist[3];
            Aprime = HFS::twoElectronSafe(ka, kj, ki, kb);

        } else if ((s >= N) && (t < N)) { // bottom left
            klist = HFS::stToKiKaKjKb(s - N, t);
            ki = klist[0];
            ka = klist[1];
            kj = klist[2];
            kb = klist[3];
            Aprime = HFS::twoElectronSafe(ka, kj, ki, kb);

        } else if ((s >= N) && (t >= N)) { // bottom right
            klist = HFS::stToKiKaKjKb(s - N, t - N);
            ki = klist[0];
            ka = klist[1];
            kj = klist[2];
            kb = klist[3];
            if (s == t) {
                Aprime = HFS::kroneckerDelta(s, t) * HFS::exc_energies(s - N);
            } else {
                Aprime = HFS::twoElectronSafe(ka, kj, ki, kb) - HFS::twoElectronSafe(ka, kj, kb, ki);
            }
        }
        return Aprime;
    }

    double calcFromIndicesBprime(arma::uword s, arma::uword t) {
        double Bprime = 0.0;
        arma::uword N = HFS::Nexc;
        std::vector<arma::vec> klist(4);
        arma::vec ki(NDIM), kj(NDIM), ka(NDIM), kb(NDIM);

        if ((s < N) && (t < N)) { // top left
            klist = HFS::stToKiKaKjKb(s, t);
            ki = klist[0];
            ka = klist[1];
            kj = klist[2];
            kb = klist[3];
            Bprime = HFS::twoElectronSafe(ka, kb, ki, kj) - HFS::twoElectronSafe(ka, kb, kj, ki);

        } else if ((s < N) && (t >= N)) { // top right
            klist = HFS::stToKiKaKjKb(s, t - N);
            ki = klist[0];
            ka = klist[1];
            kj = klist[2];
            kb = klist[3];
            Bprime = HFS::twoElectronSafe(ka, kb, ki, kj);

        } else if ((s >= N) && (t < N)) { // bottom left
            klist = HFS::stToKiKaKjKb(s - N, t);
            ki = klist[0];
            ka = klist[1];
            kj = klist[2];
            kb = klist[3];
            Bprime = HFS::twoElectronSafe(ka, kb, ki, kj);

        } else if ((s >= N) && (t >= N)) { // bottom right
            klist = HFS::stToKiKaKjKb(s - N, t - N);
            ki = klist[0];
            ka = klist[1];
            kj = klist[2];
            kb = klist[3];
            Bprime = HFS::twoElectronSafe(ka, kb, ki, kj) - HFS::twoElectronSafe(ka, kb, kj, ki);
        }
        return Bprime;
    }

    void matrixVectorProductHprime(arma::vec& v, arma::vec& Mv) {
        Mv.zeros();
        factorize2by2MatrixVectorProduct(v, Mv, matrixVectorProductAprime, matrixVectorProductBprime, matrixVectorProductBprime, matrixVectorProductAprime);
    }

    void matrixVectorProductAprime(arma::vec& v, arma::vec& Mv) {
        factorize2by2MatrixVectorProduct(v, Mv, matrixVectorProductAprimeDiag, matrixVectorProductAprimeOffDiag, matrixVectorProductAprimeOffDiag, matrixVectorProductAprimeDiag);
    }

    void matrixVectorProductBprime(arma::vec&v, arma::vec& Mv) {
        factorize2by2MatrixVectorProduct(v, Mv, matrixVectorProductBprimeDiag, matrixVectorProductBprimeOffDiag, matrixVectorProductBprimeOffDiag, matrixVectorProductBprimeDiag);
    }

    void matrixVectorProductAprimeDiag(arma::vec& v, arma::vec& Mv) {
        for (arma::uword s = 0; s < HFS::Nexc; ++s) {
            arma::uword i = HFS::excitations(0, s), a = HFS::excitations(1, s);
            arma::vec ki(NDIM), ka(NDIM);
            HFS::occIndexToK(i, ki);
            ka = HFS::virIndexToK(a);
            for (arma::uword j = 0; j < HFS::Nocc; ++j) {
                arma::vec kj(NDIM), kb(NDIM);
                HFS::occIndexToK(j, kj);
                kb = ka + kj - ki; // Momentum conservation for <aj|ib> or <aj|bi>
                HFS::toFirstBrillouinZone(kb);
                if (arma::norm(kb) > (HFS::kf + SMALLNUMBER)) {
                    // only if momentum conserving state is virtual
                    arma::uword t = HFS::calcTfromKbAndJ(kb, j);
                    if (s == t) {
                        Mv(s) += HFS::exc_energies(s) * v(t);
                    } else {
                        Mv(s) += (HFS::twoElectron(ka, ki) - HFS::twoElectron(ka, kb)) * v(t);
                    }

                }
            }
        }
    }

    void matrixVectorProductAprimeOffDiag(arma::vec& v, arma::vec& Mv) {
        for (arma::uword s = 0; s < HFS::Nexc; ++s) {
            arma::uword i = HFS::excitations(0, s), a = HFS::excitations(1, s);
            arma::vec ki(NDIM), ka(NDIM);
            HFS::occIndexToK(i, ki);
            ka = HFS::virIndexToK(a);
            for (arma::uword j = 0; j < HFS::Nocc; ++j) {
                arma::vec kj(NDIM), kb(NDIM);
                HFS::occIndexToK(j, kj);
                kb = ka + kj - ki; // Momentum conservation for <aj|ib>
                HFS::toFirstBrillouinZone(kb);
                if (arma::norm(kb) > (HFS::kf + SMALLNUMBER)) {
                    // only if momentum conserving state is virtual
                    arma::uword t = HFS::calcTfromKbAndJ(kb, j);
                    Mv(s) += HFS::twoElectron(ka, ki) * v(t);
                }
            }
        }
    }

    void matrixVectorProductBprimeDiag(arma::vec& v, arma::vec& Mv) {
        for (arma::uword s = 0; s < HFS::Nexc; ++s) {
            arma::uword i = HFS::excitations(0, s), a = HFS::excitations(1, s);
            arma::vec ki(NDIM), ka(NDIM);
            HFS::occIndexToK(i, ki);
            ka = HFS::virIndexToK(a);
            for (arma::uword j = 0; j < HFS::Nocc; ++j) {
                arma::vec kj(NDIM), kb(NDIM);
                HFS::occIndexToK(j, kj);
                kb = kj + ki - ka; // Momentum conservation for <ab|ij> or <ab|ji>
                HFS::toFirstBrillouinZone(kb);
                if (arma::norm(kb) > (HFS::kf + SMALLNUMBER)) {
                    // only if momentum conserving state is virtual
                    arma::uword t = HFS::calcTfromKbAndJ(kb, j);
                    Mv(s) += (HFS::twoElectron(ka, ki) - HFS::twoElectron(ka, kj)) * v(t);
                }
            }
        }
    }

    void matrixVectorProductBprimeOffDiag(arma::vec& v, arma::vec& Mv) {
        for (arma::uword s = 0; s < HFS::Nexc; ++s) {
            arma::uword i = HFS::excitations(0, s), a = HFS::excitations(1, s);
            arma::vec ki(NDIM), ka(NDIM);
            HFS::occIndexToK(i, ki);
            ka = HFS::virIndexToK(a);
            for (arma::uword j = 0; j < HFS::Nocc; ++j) {
                arma::vec kj(NDIM), kb(NDIM);
                HFS::occIndexToK(j, kj);
                kb = kj + ki - ka; // Momentum conservation for <ab|ji>
                HFS::toFirstBrillouinZone(kb);
                if (arma::norm(kb) > (HFS::kf + SMALLNUMBER)) {
                    // only if momentum conserving state is virtual
                    arma::uword t = HFS::calcTfromKbAndJ(kb, j);
                    Mv(s) += HFS::twoElectron(ka, ki) * v(t);
                }
            }
        }
    }


    // End Hprime functions

    void factorize2by2MatrixVectorProduct(arma::vec& v, arma::vec& Mv
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

    arma::uword calcTfromKbAndJ(arma::vec& kb, arma::uword j) {
        //arma::uvec b_N_idx =  kToIndex(kb);
        arma::uvec b_N_idx(NDIM); kToIndex(kb, b_N_idx);
        #if NDIM == 2
            arma::uword b = HFS::vir_N_to_1_mat(b_N_idx(0), b_N_idx(1));
        #elif NDIM == 3
            arma::uword b = HFS::vir_N_to_1_mat(b_N_idx(0), b_N_idx(1), b_N_idx(2));
        #endif // NDIM
        arma::uword t = HFS::inv_exc_mat(j, b);
        return t;
    }

    void setMatrixPropertiesFromCase() {

        if (HFS::mycase == "cRHF2cUHF") {
            HFS::MatVecProduct_func = HFS::matrixVectorProduct3H;
            HFS::Matrix_func = HFS::calcFromIndices3H;
            HFS::Nmat = HFS::Nexc;
        } else if (HFS::mycase == "cUHF2cUHF") {
            HFS::MatVecProduct_func = HFS::matrixVectorProductHprime;
            HFS::Matrix_func = HFS::calcFromIndicesHprime;
            HFS::Nmat = 4 * HFS::Nexc;

        } else {
            std::cout << "Invalid HFS::mycase, use one of: cRHF2cUHF  cRHF2cUHF" << std::endl;
            exit(EXIT_FAILURE);
        }

    }

} // Namespace HFS
