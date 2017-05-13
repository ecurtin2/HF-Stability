#include "matrix_vectorproducts.hpp"

namespace HFS {
namespace Matrix {
namespace MatrixVectorProduct {

void TripletH(const arma::vec& v, arma::vec& Mv) {
    Mv.zeros();
    std::vector<std::tuple<uint, uint, void (*)(const arma::vec&, arma::vec&)>> MvList (4);

    MvList[0] = std::make_tuple(0, 0, E_delta_st_minus_aj_bi);
    MvList[1] = std::make_tuple(0, 1, minus_ab_ji);
    MvList[2] = std::make_tuple(1, 0, minus_ab_ji);
    MvList[3] = std::make_tuple(1, 1, E_delta_st_minus_aj_bi);
    ApplyMvFxnsToSubMatrices(v, Mv, 2, MvList);
}

void SingletH(const arma::vec& v, arma::vec& Mv) {
    Mv.zeros();
    std::vector<std::tuple<uint, uint, void (*)(const arma::vec&, arma::vec&)>> MvList (4);

    MvList[0] = std::make_tuple(0, 0, E_delta_st_plus_2aj_ib_minus_ajbi);
    MvList[1] = std::make_tuple(0, 1, minus_abji_plus_2ab_ij);
    MvList[2] = std::make_tuple(1, 0, minus_abji_plus_2ab_ij);
    MvList[3] = std::make_tuple(1, 1, E_delta_st_plus_2aj_ib_minus_ajbi);
    ApplyMvFxnsToSubMatrices(v, Mv, 2, MvList);
}

void Hprime(const arma::vec& v, arma::vec& Mv) {
    Mv.zeros();
    std::vector<std::tuple<uint, uint, void (*)(const arma::vec&, arma::vec&)>> MvList (4);

    MvList[0] = std::make_tuple(0, 0, Aprime);
    MvList[1] = std::make_tuple(0, 1, Bprime);
    MvList[2] = std::make_tuple(1, 0, Bprime);
    MvList[3] = std::make_tuple(1, 1, Aprime);
    ApplyMvFxnsToSubMatrices(v, Mv, 2, MvList);
}

void Aprime(const arma::vec& v, arma::vec& Mv) {
    std::vector<std::tuple<uint, uint, void (*)(const arma::vec&, arma::vec&)>> MvList (4);

    MvList[0] = std::make_tuple(0, 0, E_delta_st_plus_aj_ib_antisym);
    MvList[1] = std::make_tuple(0, 1, aj_ib);
    MvList[2] = std::make_tuple(1, 0, aj_ib);
    MvList[3] = std::make_tuple(1, 1, E_delta_st_plus_aj_ib_antisym);
    ApplyMvFxnsToSubMatrices(v, Mv, 2, MvList);
}

void Bprime(const arma::vec& v, arma::vec& Mv) {
    std::vector<std::tuple<uint, uint, void (*)(const arma::vec&, arma::vec&)>> MvList (4);

    MvList[0] = std::make_tuple(0, 0, ab_ij_antisym);
    MvList[1] = std::make_tuple(0, 1, ab_ij);
    MvList[2] = std::make_tuple(1, 0, ab_ij);
    MvList[3] = std::make_tuple(1, 1, ab_ij_antisym);
    ApplyMvFxnsToSubMatrices(v, Mv, 2, MvList);
}


void H(const arma::vec& v, arma::vec& Mv) {
    Mv.zeros();
    std::vector<std::tuple<uint, uint, void (*)(const arma::vec&, arma::vec&)>> MvList (4);

    MvList[0] = std::make_tuple(0, 0, A);
    MvList[1] = std::make_tuple(0, 1, B);
    MvList[2] = std::make_tuple(1, 0, B);
    MvList[3] = std::make_tuple(1, 1, A);
    ApplyMvFxnsToSubMatrices(v, Mv, 2, MvList);
}

void A(const arma::vec& v, arma::vec& Mv) {
    std::vector<std::tuple<uint, uint, void (*)(const arma::vec&, arma::vec&)>> MvList (6);

    MvList[0] = std::make_tuple(0, 0, E_delta_st_plus_aj_ib_antisym);
    MvList[1] = std::make_tuple(0, 3, aj_ib);
    MvList[2] = std::make_tuple(1, 1, E_delta_st_minus_aj_bi);
    MvList[3] = std::make_tuple(2, 2, E_delta_st_minus_aj_bi);
    MvList[4] = std::make_tuple(3, 0, aj_ib);
    MvList[5] = std::make_tuple(3, 3, E_delta_st_plus_aj_ib_antisym);
    ApplyMvFxnsToSubMatrices(v, Mv, 4, MvList);
}

void B(const arma::vec& v, arma::vec& Mv) {
    std::vector<std::tuple<uint, uint, void (*)(const arma::vec&, arma::vec&)>> MvList (6);

    MvList[0] = std::make_tuple(0, 0, ab_ij_antisym);
    MvList[1] = std::make_tuple(0, 3, ab_ij);
    MvList[2] = std::make_tuple(1, 2, minus_ab_ji);
    MvList[3] = std::make_tuple(2, 1, minus_ab_ji);
    MvList[4] = std::make_tuple(3, 0, ab_ij);
    MvList[5] = std::make_tuple(3, 3, ab_ij_antisym);
    ApplyMvFxnsToSubMatrices(v, Mv, 4, MvList);
}


void E_delta_st_plus_aj_ib_antisym(const arma::vec& v, arma::vec& Mv) {
    for (uint s = 0; s < Nexc; ++s) {
        uint i = excitations(0, s), a = excitations(1, s);
        arma::vec ki(NDIM), ka(NDIM);
        occIndexToK(i, ki);
        ka = virIndexToK(a);
        for (uint j = 0; j < Nocc; ++j) {
            arma::vec kj(NDIM), kb(NDIM);
            occIndexToK(j, kj);
            kb = ka + kj - ki; // Momentum conservation for <aj|ib> or <aj|bi>
            toFirstBrillouinZone(kb);
            if (arma::norm(kb) > (kf + SMALLNUMBER)) {
                // only if momentum conserving state is virtual
                uint t = Matrix::calcTfromKbAndJ(kb, j);
                Mv(s) += (kroneckerDelta(s, t) * exc_energies(s)
                        + twoElectron(ka, ki) - twoElectron(ka, kb)) * v(t);
            }
        }
    }
}

void E_delta_st_plus_2aj_ib_minus_ajbi(const arma::vec& v, arma::vec& Mv) {
    for (uint s = 0; s < Nexc; ++s) {
        uint i = excitations(0, s), a = excitations(1, s);
        arma::vec ki(NDIM), ka(NDIM);
        occIndexToK(i, ki);
        ka = virIndexToK(a);
        for (uint j = 0; j < Nocc; ++j) {
            arma::vec kj(NDIM), kb(NDIM);
            occIndexToK(j, kj);
            kb = ka + kj - ki; // Momentum conservation for <aj|ib> or <aj|bi>
            toFirstBrillouinZone(kb);
            if (arma::norm(kb) > (kf + SMALLNUMBER)) {
                // only if momentum conserving state is virtual
                uint t = Matrix::calcTfromKbAndJ(kb, j);
                Mv(s) += (kroneckerDelta(s, t) * exc_energies(s)
                        + 2 * twoElectron(ka, ki) - twoElectron(ka, kb)) * v(t);
            }
        }
    }
}

void aj_ib(const arma::vec& v, arma::vec& Mv) {
    for (uint s = 0; s < Nexc; ++s) {
        uint i = excitations(0, s), a = excitations(1, s);
        arma::vec ki(NDIM), ka(NDIM);
        occIndexToK(i, ki);
        ka = virIndexToK(a);
        for (uint j = 0; j < Nocc; ++j) {
            arma::vec kj(NDIM), kb(NDIM);
            occIndexToK(j, kj);
            kb = ka + kj - ki; // Momentum conservation for <aj|ib> or <aj|bi>
            toFirstBrillouinZone(kb);
            if (arma::norm(kb) > (kf + SMALLNUMBER)) {
                // only if momentum conserving state is virtual
                uint t = Matrix::calcTfromKbAndJ(kb, j);
                Mv(s) += (twoElectron(ka, ki)) * v(t);
            }
        }
    }
}

void E_delta_st_minus_aj_bi(const arma::vec& v, arma::vec& Mv) {
    for (uint s = 0; s < Nexc; ++s) {
        uint i = excitations(0, s), a = excitations(1, s);
        arma::vec ki(NDIM), ka(NDIM);
        occIndexToK(i, ki);
        ka = virIndexToK(a);
        for (uint j = 0; j < Nocc; ++j) {
            arma::vec kj(NDIM), kb(NDIM);
            occIndexToK(j, kj);
            kb = ka + kj - ki; // Momentum conservation for <aj|ib> or <aj|bi>
            toFirstBrillouinZone(kb);
            if (arma::norm(kb) > (kf + SMALLNUMBER)) {
                // only if momentum conserving state is virtual
                uint t = Matrix::calcTfromKbAndJ(kb, j);
                if (s == t) {
                    Mv(s) += exc_energies(s) * v(t);
                } else {
                    Mv(s) += (-twoElectron(ka, kb)) * v(t);
                }
            }
        }
    }
}

void ab_ij_antisym(const arma::vec& v, arma::vec& Mv) {
    for (uint s = 0; s < Nexc; ++s) {
        uint i = excitations(0, s), a = excitations(1, s);
        arma::vec ki(NDIM), ka(NDIM);
        occIndexToK(i, ki);
        ka = virIndexToK(a);
        for (uint j = 0; j < Nocc; ++j) {
            arma::vec kj(NDIM), kb(NDIM);
            occIndexToK(j, kj);
            kb = kj + ki - ka; // Momentum conservation for <ab|ij> or <ab|ji>
            toFirstBrillouinZone(kb);
            if (arma::norm(kb) > (kf + SMALLNUMBER)) {
                // only if momentum conserving state is virtual
                uint t = Matrix::calcTfromKbAndJ(kb, j);
                Mv(s) += (twoElectron(ka, ki) - twoElectron(ka, kj)) * v(t);
            }
        }
    }
}

void minus_abji_plus_2ab_ij(const arma::vec& v, arma::vec& Mv) {
    for (uint s = 0; s < Nexc; ++s) {
        uint i = excitations(0, s), a = excitations(1, s);
        arma::vec ki(NDIM), ka(NDIM);
        occIndexToK(i, ki);
        ka = virIndexToK(a);
        for (uint j = 0; j < Nocc; ++j) {
            arma::vec kj(NDIM), kb(NDIM);
            occIndexToK(j, kj);
            kb = kj + ki - ka; // Momentum conservation for <ab|ij> or <ab|ji>
            toFirstBrillouinZone(kb);
            if (arma::norm(kb) > (kf + SMALLNUMBER)) {
                // only if momentum conserving state is virtual
                uint t = Matrix::calcTfromKbAndJ(kb, j);
                Mv(s) += (2 * twoElectron(ka, ki) - twoElectron(ka, kj)) * v(t);
            }
        }
    }
}

void ab_ij(const arma::vec& v, arma::vec& Mv) {
    for (uint s = 0; s < Nexc; ++s) {
        uint i = excitations(0, s), a = excitations(1, s);
        arma::vec ki(NDIM), ka(NDIM);
        occIndexToK(i, ki);
        ka = virIndexToK(a);
        for (uint j = 0; j < Nocc; ++j) {
            arma::vec kj(NDIM), kb(NDIM);
            occIndexToK(j, kj);
            kb = kj + ki - ka; // Momentum conservation for <ab|ji>
            toFirstBrillouinZone(kb);
            if (arma::norm(kb) > (kf + SMALLNUMBER)) {
                // only if momentum conserving state is virtual
                uint t = Matrix::calcTfromKbAndJ(kb, j);
                Mv(s) += twoElectron(ka, ki) * v(t);
            }
        }
    }
}

void minus_ab_ji(const arma::vec& v, arma::vec& Mv) {
    for (uint s = 0; s < Nexc; ++s) {
        uint i = excitations(0, s), a = excitations(1, s);
        arma::vec ki(NDIM), ka(NDIM);
        occIndexToK(i, ki);
        ka = virIndexToK(a);
        for (uint j = 0; j < Nocc; ++j) {
            arma::vec kj(NDIM), kb(NDIM);
            occIndexToK(j, kj);
            kb = kj + ki - ka; // Momentum conservation for <ab|ji>
            toFirstBrillouinZone(kb);
            if (arma::norm(kb) > (kf + SMALLNUMBER)) {
                // only if momentum conserving state is virtual
                uint t = Matrix::calcTfromKbAndJ(kb, j);
                Mv(s) += -twoElectron(ka, kj) * v(t);
            }
        }
    }
}

void ApplyMvFxnsToSubMatrices(
               const arma::vec& v,
               arma::vec& Mv,
               uint Ndivisions,
               std::vector<std::tuple<uint, uint, void (*)(const arma::vec&, arma::vec&)>> MvList
              ) {
     /** \brief Apply in-place Matrix-Vector product functions to sub-matrices.
      *
      * Assumes each function DOES NOT INITIALIZE and ADDS the contribution to Mv IN-PLACE.
      * DOES NOT INITIALIZE Mv.
      * \param v Vector to be multiplied by matrix.
      * \param Mv Vector to store
      * \param Ndivisions Number of sections per row/column. Assumed equal.
      * \param locs
      * \param Mvfuncs
      *
      */

    assert((MvList.size() <= Ndivisions*Ndivisions) && "Mvlist Too big for # of divisions.");

    uint N = v.n_elem / Ndivisions; // Size of each subview

    for (auto& MvL : MvList) {
        uint irow = std::get<0>(MvL);
        uint icol = std::get<1>(MvL);
        void (*MvFunc)(const arma::vec&, arma::vec&) = std::get<2>(MvL);

        scalar* Mv_ptr = &Mv[irow * N];
        arma::vec Mv_subview = arma::vec(Mv_ptr, N, false, true);

        scalar* v_ptr  = (scalar*) &v[icol * N];
        /* This technically violates const correctness. Do not use v_ptr for anything but initializing
           a const vec */
        const arma::vec v_subview = arma::vec(v_ptr, N, false, true);

        MvFunc(v_subview, Mv_subview);
    }
}

} // namespace MatrixVectorProduct


uint calcTfromKbAndJ(const arma::vec& kb, uint j) {
    arma::uvec b_N_uint(NDIM);
    kToIndex(kb, b_N_uint);
    # if NDIM == 1
        uint b = vir_N_to_1_mat(b_N_uint(0));
    # endif
    # if NDIM == 2
        uint b = vir_N_to_1_mat(b_N_uint(0), b_N_uint(1));
    # endif
    # if NDIM == 3
        uint b = vir_N_to_1_mat(b_N_uint(0), b_N_uint(1), b_N_uint(2));
    #endif
    uint t = inv_exc_mat(j, b);
    return t;
}

void setMatrixPropertiesFromCase() {

    if (mycase == "cRHF2cRHF") {
        MatVecProduct_func = MatrixVectorProduct::SingletH;
        Matrix_generator = Matrix::Gen::SingletH;
        Nmat = 2 * Nexc;
    } else if (mycase == "cRHF2cUHF") {
        MatVecProduct_func = MatrixVectorProduct::TripletH;
        Matrix_generator = Matrix::Gen::TripletH;
        Nmat = 2 * Nexc;
    } else if (mycase == "cUHF2cUHF") {
        MatVecProduct_func = MatrixVectorProduct::Hprime;
        Matrix_generator = Matrix::Gen::Hprime;
        Nmat = 4 * Nexc;
    } else if (mycase == "cRHF2cGHF") {
        MatVecProduct_func = MatrixVectorProduct::H;
        Matrix_generator = Matrix::Gen::H;
        Nmat = 8 * Nexc;
    } else {
        std::cout << "Invalid mycase, use one of: cRHF2cUHF, cUHF2cUHF, cRHF2cGHF" << std::endl;
        exit(EXIT_FAILURE);
    }

}

} // namespace Matrix
} // namespace HFS
