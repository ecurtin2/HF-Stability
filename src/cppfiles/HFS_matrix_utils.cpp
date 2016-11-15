#include "HFS_matrix_utils.hpp"

namespace HFS {

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

    double calc_3B(arma::uword s, arma::uword t) {
        arma::uword i = HFS::excitations(s, 0);
        arma::uword a = HFS::excitations(s, 1);
        arma::uword j = HFS::excitations(t, 0);
        arma::uword b = HFS::excitations(t, 1);
        arma::vec ki(HFS::ndim), kj(HFS::ndim), ka(HFS::ndim), kb(HFS::ndim);
        for (unsigned idx = 0; idx < ndim; ++idx) {
            ki[idx] = HFS::kgrid(HFS::occ_states(i, idx));
            kj[idx] = HFS::kgrid(HFS::occ_states(j, idx));
            ka[idx] = HFS::kgrid(HFS::vir_states(a, idx));
            kb[idx] = HFS::kgrid(HFS::vir_states(b, idx));
        }
        return -1.0 * HFS::two_electron_safe(ka, kb, kj, ki);
    }

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

    double calc_3A(arma::uword s, arma::uword t) {
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
                return HFS::calc_3A(i-HFS::Nexc, j-HFS::Nexc);
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
                if (arma::norm(kb) > (HFS::kf + 10E-8)) {
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

    arma::vec matvec_prod_3H(arma::vec& v) {
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
        arma::vec Mv = arma::join_cols(Mv1, Mv2);
        return Mv;
    }

    void void_matvec_prod_3H(arma::vec& v, arma::vec&Mv) {
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

    arma::uword kb_j_to_t(arma::vec& kb, arma::uword j) {
        arma::uvec b_N_idx =  k_to_index(kb);
        arma::uword b = HFS::vir_N_to_1_mat(b_N_idx(0), b_N_idx(1));
        arma::uword t = HFS::inv_exc_mat(j, b);
        return t;
    }

    void matvec_prod_Hprime(arma::vec& v, arma::vec& Mv) {
        //MUST BE CALLED WITH MV of LENGTH 8Nexc
        /*  The matrix-vector multiplication | A'  B' | |v1|  =  | Mv1 |
                                             | B'* A'*| |v2|     | Mv2 |
            Factors into 4 matrix vector multiplications (A, B are matrices)
            Mv1 = A'v1  + B'v2
            Mv2 = B'*v1 + A'*v2

            This routine is structured to factorize the matrix-vector product to be easier
            to understand and program, whilst avoiding any copying of vectors, in anticipation
            that they will become large.
        */

        // Each matrix is 4*Nexc, the vector has 8Nexc elements.
        arma::vec v1(&v             , 4*HFS::Nexc, false, true);    // First 4Nexc elements of v
        arma::vec v2(&v(4*HSF::Nexc), 4*HFS::Nexc, false, true);    // Last 4Nexc elements of v
        arma::vec Mv1(&Mv             , 4*HFS::Nexc, false, true);  // First 4Nexc elements of Mv
        arma::vec Mv2(&Mv(4*HSF::Nexc), 4*HFS::Nexc, false, true);  // Last 4Nexc elements of Mv

        /* Each matrix vector product routine *adds* the
           matrix vector product inplace to the given Mv
           return vector. Thus the Mv must be
           initialized, which initializes all Mv1, Mv2. */
        Mv.zeros();
        HFS::matvec_prod_Aprime(v1, Mv1);
        HFS::matvec_prod_Bprime(v2, Mv1);

        HFS::matvec_prod_Bprime(v1, Mv2);
        HFS::matvec_prod_Aprime(v2, Mv2);
        // At this point all values in Mv are filled.

    }
}

    arma::vec matvec_prod_Aprime(arma::vec& v, arma::vec& Mv) {
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
                if (arma::norm(kb) > (HFS::kf + 10E-8)) {
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
