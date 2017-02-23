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
        matrixVectorProduct3B(v, Mv); // adds in place, so this is Av + Bv
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

        double Hprime = 0.0;
        if ((s < HFS::Nmat) && (t < HFS::Nmat)) {         // top left
            Hprime = HFS::calcFromIndicesAprime(s, t);
        } else if ((s < HFS::Nmat) && (t >= HFS::Nmat)) { // top right
            Hprime = HFS::calcFromIndicesBprime(s, t - HFS::Nmat);
        } else if ((s >= HFS::Nmat) && (t < HFS::Nmat)) { // bottom left
            Hprime = HFS::calcFromIndicesBprime(s - HFS::Nmat, t);
        } else if ((s >= HFS::Nmat) && (t >= HFS::Nmat)) { // bottom right
            Hprime = HFS::calcFromIndicesAprime(s - HFS::Nmat, t - HFS::Nmat);
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

    // Start H Functions
    double calcFromIndicesH(arma::uword s, arma::uword t) {
        double Hprime = 0.0;
        arma::uword N = HFS::Nmat / 2;

        if ((s < N) && (t < N)) {         // top left
            Hprime = HFS::calcFromIndicesA(s, t);
        } else if ((s < N) && (t >= N)) { // top right
            Hprime = HFS::calcFromIndicesB(s, t - N);
        } else if ((s >= N) && (t < N)) { // bottom left
            Hprime = HFS::calcFromIndicesB(s - N, t);
        } else if ((s >= N) && (t >= N)) { // bottom right
            Hprime = HFS::calcFromIndicesA(s - N, t - N);
        }
        return Hprime;
    }

    double calcFromIndicesA(arma::uword s, arma::uword t) {
        /*
        | M1 0  0  M2 |
        | 0  M3 0  0  |
        | 0  0  M3 0  |
        | M2 0  0  M1 |

        Where
          M1 = (e_a - e_i)Delta(ij)Delta(ab) + (aj||ib)
          M2 = (aj|ib)
          M3 = (e_a - e_i)Delta(ij)Delta(ab) - (aj|bi)
        */

        double A = 0.0;
        arma::uword N = HFS::Nmat / 8;

        if ((s < N) && (t < N))   {  // M1 top left
            A = HFS::calcFromIndicesA_M1(s, t);
        } else if ((s < N) && (t >= 3*N)) { // M2 top right
            A = HFS::calcFromIndicesA_M2(s, t - 3*N);
        } else if ((s >= N) && (s < 2*N) && (t >= N) && (t < 2*N)) { // M3 center-top-left
            A = HFS::calcFromIndicesA_M3(s - N, t - N);
        } else if ((s >= 2*N) && (s < 3*N) && (t >= 2*N) && (t < 3*N)) { // M3 center-bottom-right
            A = HFS::calcFromIndicesA_M3(s - 2*N, t - 2*N);
        } else if ((s < N) && (t >= 3*N)) { // M2 bottom left
            A = HFS::calcFromIndicesA_M2(s - 3*N, t);
        } else if ((s >= 3*N) && (t >= 3*N)) { // M1 bottom right
            A = HFS::calcFromIndicesA_M1(s - 3*N, t - 3*N);
        } else {
            A = 0;
        }
        return A;
    }

    double calcFromIndicesB(arma::uword s, arma::uword t) {
        double B = 0.0;
        arma::uword N = HFS::Nmat / 8;

        if ((s < N) && (t < N))   {  // M1 top left
            B = HFS::calcFromIndicesB_M1(s, t);
        } else if ((s < N) && (t >= 3*N)) { // M2 top right
            B = HFS::calcFromIndicesB_M2(s, t - 3*N);
        } else if ((s >= N) && (s < 2*N) && (t >= 2*N) && (t < 3*N)) { // M3 center-top-right
            B = HFS::calcFromIndicesB_M3(s - N, t - 2*N);
        } else if ((s >= 2*N) && (s < 3*N) && (t >= N) && (t < 2*N)) { // M3 center-bottom-left
            B = HFS::calcFromIndicesB_M3(s - 2*N, t - N);
        } else if ((s < N) && (t >= 3*N)) { // M2 bottom left
            B = HFS::calcFromIndicesB_M2(s - 3*N, t);
        } else if ((s >= 3*N) && (t >= 3*N)) { // M1 bottom right
            B = HFS::calcFromIndicesB_M1(s - 3*N, t - 3*N);
        } else {
            B = 0;
        }
        return B;
    }

    double calcFromIndicesA_M1(arma::uword s, arma::uword t) {
        std::vector<arma::vec> klist(4);
        klist = HFS::stToKiKaKjKb(s, t);
        arma::vec ki(NDIM), kj(NDIM), ka(NDIM), kb(NDIM);
        ki = klist[0];
        ka = klist[1];
        kj = klist[2];
        kb = klist[3];

        double A1;
        A1 = HFS::kroneckerDelta(s, t) * HFS::exc_energies(s)
            + HFS::twoElectronSafe(ka, kj, ki, kb) - HFS::twoElectronSafe(ka, kj, kb, ki);
        return A1;
    }

    double calcFromIndicesA_M2(arma::uword s, arma::uword t) {
        std::vector<arma::vec> klist(4);
        klist = HFS::stToKiKaKjKb(s, t);
        arma::vec ki(NDIM), kj(NDIM), ka(NDIM), kb(NDIM);
        ki = klist[0];
        ka = klist[1];
        kj = klist[2];
        kb = klist[3];

        double A2 = HFS::twoElectronSafe(ka, kj, ki, kb);
        return A2;
    }

    double calcFromIndicesA_M3(arma::uword s, arma::uword t) {
        std::vector<arma::vec> klist(4);
        klist = HFS::stToKiKaKjKb(s, t);
        arma::vec ki(NDIM), kj(NDIM), ka(NDIM), kb(NDIM);
        ki = klist[0];
        ka = klist[1];
        kj = klist[2];
        kb = klist[3];

        double A3;
        A3 = HFS::kroneckerDelta(s, t) * HFS::exc_energies(s) - HFS::twoElectronSafe(ka, kj, kb, ki);
        return A3;
    }

    double calcFromIndicesB_M1(arma::uword s, arma::uword t) {
        std::vector<arma::vec> klist(4);
        klist = HFS::stToKiKaKjKb(s, t);
        arma::vec ki(NDIM), kj(NDIM), ka(NDIM), kb(NDIM);
        ki = klist[0];
        ka = klist[1];
        kj = klist[2];
        kb = klist[3];

        double B1 = HFS::twoElectronSafe(ka, kb, ki, kj) - HFS::twoElectronSafe(ka, kb, kj, ki);
        return B1;
    }

    double calcFromIndicesB_M2(arma::uword s, arma::uword t) {
        std::vector<arma::vec> klist(4);
        klist = HFS::stToKiKaKjKb(s, t);
        arma::vec ki(NDIM), kj(NDIM), ka(NDIM), kb(NDIM);
        ki = klist[0];
        ka = klist[1];
        kj = klist[2];
        kb = klist[3];

        double B2 = HFS::twoElectronSafe(ka, kb, ki, kj);
        return B2;
    }

    double calcFromIndicesB_M3(arma::uword s, arma::uword t) {
        std::vector<arma::vec> klist(4);
        klist = HFS::stToKiKaKjKb(s, t);
        arma::vec ki(NDIM), kj(NDIM), ka(NDIM), kb(NDIM);
        ki = klist[0];
        ka = klist[1];
        kj = klist[2];
        kb = klist[3];

        double B2 = - HFS::twoElectronSafe(ka, kb, kj, ki);
        return B2;
    }

    void matrixVectorProductH(arma::vec& v, arma::vec& Mv) {
        Mv.zeros();
        factorize2by2MatrixVectorProduct(v, Mv, matrixVectorProductA,
                                                matrixVectorProductB,
                                                matrixVectorProductB,
                                                matrixVectorProductA);
    }

    void matrixVectorProductA(arma::vec& v, arma::vec& Mv) {
        /*
        | M1 0  0  M2 |
        | 0  M3 0  0  |
        | 0  0  M3 0  |
        | M2 0  0  M1 |

        Where
          M1 = (e_a - e_i)Delta(ij)Delta(ab) + (aj||ib)
          M2 = (aj|ib)
          M3 = (e_a - e_i)Delta(ij)Delta(ab) - (aj|bi)
        */

        factorizeA(v, Mv, matrixVectorProductA_M1, matrixVectorProductA_M2, matrixVectorProductA_M3);
    }

    void matrixVectorProductB(arma::vec& v, arma::vec& Mv) {
        /*
        | M1 0  0  M2 |
        | 0  0  M3 0  |
        | 0  M3 0  0  |
        | M2 0  0  M1 |

        Where
          M1 = (ab||ij)
          M2 = (ab|ij)
          M3 = -(ab|ji)
        */
        factorizeB(v, Mv, matrixVectorProductB_M1
                        , matrixVectorProductB_M2
                        , matrixVectorProductB_M3);
    }

    void matrixVectorProductA_M1(arma::vec& v, arma::vec& Mv) {
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

    void matrixVectorProductA_M2(arma::vec& v, arma::vec& Mv) {
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
                    Mv(s) += (HFS::twoElectron(ka, ki)) * v(t);
                }
            }
        }
    }

    void matrixVectorProductA_M3(arma::vec& v, arma::vec& Mv) {
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
                        Mv(s) += (-HFS::twoElectron(ka, kb)) * v(t);
                    }
                }
            }
        }
    }

    void matrixVectorProductB_M1(arma::vec& v, arma::vec& Mv) {
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

    void matrixVectorProductB_M2(arma::vec& v, arma::vec& Mv) {
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

    void matrixVectorProductB_M3(arma::vec& v, arma::vec& Mv) {
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
                    Mv(s) += -HFS::twoElectron(ka, kj) * v(t);
                }
            }
        }
    }

    void factorizeA(arma::vec& v, arma::vec& Mv
                    ,void (*M1v)(arma::vec&, arma::vec&)
                    ,void (*M2v)(arma::vec&, arma::vec&)
                    ,void (*M3v)(arma::vec&, arma::vec&)) {
        /*
        | M1 0  0  M2 | |v1|     |M1v1 + M2v4|
        | 0  M3 0  0  | |v2|  =  |    M3v2   |
        | 0  0  M3 0  | |v3|     |    M3v3   |
        | M2 0  0  M1 | |v4|     |M2v1 + M1v4|
        */

        arma::uword N = v.n_elem/4;

        // Split v into 4 subvectors (no copy)
        arma::vec v1 = arma::vec(&v[0]  ,     N, false, true);
        arma::vec v2 = arma::vec(&v[N]  , 2 * N, false, true);
        arma::vec v3 = arma::vec(&v[2*N], 3 * N, false, true);
        arma::vec v4 = arma::vec(&v[3*N], 4 * N, false, true);

        // Split Mv into 4 subvectors (no copy)
        arma::vec Mv1 = arma::vec(&v[0]  ,     N, false, true);
        arma::vec Mv2 = arma::vec(&v[N]  , 2 * N, false, true);
        arma::vec Mv3 = arma::vec(&v[2*N], 3 * N, false, true);
        arma::vec Mv4 = arma::vec(&v[3*N], 4 * N, false, true);

        M1v(v1, Mv1);
        M2v(v4, Mv1);

        M3v(v2, Mv2);

        M3v(v3, Mv3);

        M2v(v1, Mv4);
        M1v(v4, Mv4);
    }

    void factorizeB(arma::vec& v, arma::vec& Mv
                    ,void (*M1v)(arma::vec&, arma::vec&)
                    ,void (*M2v)(arma::vec&, arma::vec&)
                    ,void (*M3v)(arma::vec&, arma::vec&)) {
        /*
        | M1 0  0  M2 | |v1|     |M1v1 + M2v4|
        | 0  0  M3  0 | |v2|  =  |    M3v3   |
        | 0  M3 0  0  | |v3|     |    M3v2   |
        | M2 0  0  M1 | |v4|     |M2v1 + M1v4|
        */

        arma::uword N = v.n_elem/4;

        // Split v into 4 subvectors (no copy)
        arma::vec v1 = arma::vec(&v[0]  ,     N, false, true);
        arma::vec v2 = arma::vec(&v[N]  , 2 * N, false, true);
        arma::vec v3 = arma::vec(&v[2*N], 3 * N, false, true);
        arma::vec v4 = arma::vec(&v[3*N], 4 * N, false, true);

        // Split Mv into 4 subvectors (no copy)
        arma::vec Mv1 = arma::vec(&v[0]  ,     N, false, true);
        arma::vec Mv2 = arma::vec(&v[N]  , 2 * N, false, true);
        arma::vec Mv3 = arma::vec(&v[2*N], 3 * N, false, true);
        arma::vec Mv4 = arma::vec(&v[3*N], 4 * N, false, true);

        M1v(v1, Mv1);
        M2v(v4, Mv1);

        M3v(v3, Mv2);

        M3v(v2, Mv3);

        M2v(v1, Mv4);
        M1v(v4, Mv4);
    }

    // End H Functions

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
        Bv(v2, Mv1); // add to Mv1 in place
        Cv(v1, Mv2);
        Dv(v2, Mv2); // add to mv2 in place
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
        } else if (HFS::mycase == "cRHF2cGHF") {
            HFS::MatVecProduct_func = HFS::matrixVectorProductH;
            HFS::Matrix_func = HFS::calcFromIndicesH;
            HFS::Nmat = 8 * HFS::Nexc;
        } else {
            std::cout << "Invalid HFS::mycase, use one of: cRHF2cUHF  cRHF2cUHF" << std::endl;
            exit(EXIT_FAILURE);
        }

    }

} // Namespace HFS
