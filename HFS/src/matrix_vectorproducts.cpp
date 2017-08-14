#include "matrix_vectorproducts.hpp"

namespace HFS {
namespace Matrix {
namespace MatrixVectorProduct {

void TripletH(arma::vec& v, arma::vec& Mv) {
    Mv.zeros();
    std::vector<std::tuple<uint, uint, void (*)(arma::vec&, arma::vec&)>> MvList (4);

    MvList[0] = std::make_tuple(0, 0, E_delta_st_minus_aj_bi);
    MvList[1] = std::make_tuple(0, 1, minus_ab_ji);
    MvList[2] = std::make_tuple(1, 0, minus_ab_ji);
    MvList[3] = std::make_tuple(1, 1, E_delta_st_minus_aj_bi);
    ApplyMvFxnsToSubMatrices(v, Mv, 2, MvList);
}


void SingletH(arma::vec& v, arma::vec& Mv) {
    Mv.zeros();
    std::vector<std::tuple<uint, uint, void (*)(arma::vec&, arma::vec&)>> MvList (4);

    MvList[0] = std::make_tuple(0, 0, E_delta_st_plus_2aj_ib_minus_ajbi);
    MvList[1] = std::make_tuple(0, 1, minus_abji_plus_2ab_ij);
    MvList[2] = std::make_tuple(1, 0, minus_abji_plus_2ab_ij);
    MvList[3] = std::make_tuple(1, 1, E_delta_st_plus_2aj_ib_minus_ajbi);
    ApplyMvFxnsToSubMatrices(v, Mv, 2, MvList);
}

void Hprime(arma::vec& v, arma::vec& Mv) {
    Mv.zeros();
    std::vector<std::tuple<uint, uint, void (*)(arma::vec&, arma::vec&)>> MvList (4);

    MvList[0] = std::make_tuple(0, 0, Aprime);
    MvList[1] = std::make_tuple(0, 1, Bprime);
    MvList[2] = std::make_tuple(1, 0, Bprime);
    MvList[3] = std::make_tuple(1, 1, Aprime);
    ApplyMvFxnsToSubMatrices(v, Mv, 2, MvList);
}

void Aprime(arma::vec& v, arma::vec& Mv) {
    std::vector<std::tuple<uint, uint, void (*)(arma::vec&, arma::vec&)>> MvList (4);

    MvList[0] = std::make_tuple(0, 0, E_delta_st_plus_aj_ib_antisym);
    MvList[1] = std::make_tuple(0, 1, aj_ib);
    MvList[2] = std::make_tuple(1, 0, aj_ib);
    MvList[3] = std::make_tuple(1, 1, E_delta_st_plus_aj_ib_antisym);
    ApplyMvFxnsToSubMatrices(v, Mv, 2, MvList);
}

void Bprime(arma::vec& v, arma::vec& Mv) {
    std::vector<std::tuple<uint, uint, void (*)(arma::vec&, arma::vec&)>> MvList (4);

    MvList[0] = std::make_tuple(0, 0, ab_ij_antisym);
    MvList[1] = std::make_tuple(0, 1, ab_ij);
    MvList[2] = std::make_tuple(1, 0, ab_ij);
    MvList[3] = std::make_tuple(1, 1, ab_ij_antisym);
    ApplyMvFxnsToSubMatrices(v, Mv, 2, MvList);
}


void H(arma::vec& v, arma::vec& Mv) {
    Mv.zeros();
    std::vector<std::tuple<uint, uint, void (*)(arma::vec&, arma::vec&)>> MvList (4);

    MvList[0] = std::make_tuple(0, 0, A);
    MvList[1] = std::make_tuple(0, 1, B);
    MvList[2] = std::make_tuple(1, 0, B);
    MvList[3] = std::make_tuple(1, 1, A);
    ApplyMvFxnsToSubMatrices(v, Mv, 2, MvList);
}

void A(arma::vec& v, arma::vec& Mv) {
    std::vector<std::tuple<uint, uint, void (*)(arma::vec&, arma::vec&)>> MvList (6);

    MvList[0] = std::make_tuple(0, 0, E_delta_st_plus_aj_ib_antisym);
    MvList[1] = std::make_tuple(0, 3, aj_ib);
    MvList[2] = std::make_tuple(1, 1, E_delta_st_minus_aj_bi);
    MvList[3] = std::make_tuple(2, 2, E_delta_st_minus_aj_bi);
    MvList[4] = std::make_tuple(3, 0, aj_ib);
    MvList[5] = std::make_tuple(3, 3, E_delta_st_plus_aj_ib_antisym);
    ApplyMvFxnsToSubMatrices(v, Mv, 4, MvList);
}

void B(arma::vec& v, arma::vec& Mv) {
    std::vector<std::tuple<uint, uint, void (*)(arma::vec&, arma::vec&)>> MvList (6);

    MvList[0] = std::make_tuple(0, 0, ab_ij_antisym);
    MvList[1] = std::make_tuple(0, 3, ab_ij);
    MvList[2] = std::make_tuple(1, 2, minus_ab_ji);
    MvList[3] = std::make_tuple(2, 1, minus_ab_ji);
    MvList[4] = std::make_tuple(3, 0, ab_ij);
    MvList[5] = std::make_tuple(3, 3, ab_ij_antisym);
    ApplyMvFxnsToSubMatrices(v, Mv, 4, MvList);
}


void E_delta_st_plus_aj_ib_antisym(arma::vec& v, arma::vec& Mv) {
    for (uint s = 0; s < HFS::Nexc; ++s) {
        uint i = HFS::excitations(0, s), a = HFS::excitations(1, s);
        arma::vec ki(NDIM), ka(NDIM);
        HFS::occIndexToK(i, ki);
        ka = HFS::virIndexToK(a);
        for (uint j = 0; j < HFS::Nocc; ++j) {
            arma::vec kj(NDIM), kb(NDIM);
            HFS::occIndexToK(j, kj);
            kb = ka + kj - ki; // Momentum conservation for <aj|ib> or <aj|bi>
            HFS::toFirstBrillouinZone(kb);
            if (arma::norm(kb) > (HFS::kf + SMALLNUMBER)) {
                // only if momentum conserving state is virtual
                uint t = HFS::Matrix::calcTfromKbAndJ(kb, j);
                Mv(s) += (HFS::kroneckerDelta(s, t) * HFS::exc_energies(s)
                        + HFS::twoElectron(ka, ki) - HFS::twoElectron(ka, kb)) * v(t);
            }
        }
    }
}

void E_delta_st_plus_2aj_ib_minus_ajbi(arma::vec& v, arma::vec& Mv) {
    for (uint s = 0; s < HFS::Nexc; ++s) {
        uint i = HFS::excitations(0, s), a = HFS::excitations(1, s);
        arma::vec ki(NDIM), ka(NDIM);
        HFS::occIndexToK(i, ki);
        ka = HFS::virIndexToK(a);
        for (uint j = 0; j < HFS::Nocc; ++j) {
            arma::vec kj(NDIM), kb(NDIM);
            HFS::occIndexToK(j, kj);
            kb = ka + kj - ki; // Momentum conservation for <aj|ib> or <aj|bi>
            HFS::toFirstBrillouinZone(kb);
            if (arma::norm(kb) > (HFS::kf + SMALLNUMBER)) {
                // only if momentum conserving state is virtual
                uint t = HFS::Matrix::calcTfromKbAndJ(kb, j);
                Mv(s) += (HFS::kroneckerDelta(s, t) * HFS::exc_energies(s)
                        + 2 * HFS::twoElectron(ka, ki) - HFS::twoElectron(ka, kb)) * v(t);
            }
        }
    }
}

void aj_ib(arma::vec& v, arma::vec& Mv) {
    for (uint s = 0; s < HFS::Nexc; ++s) {
        uint i = HFS::excitations(0, s), a = HFS::excitations(1, s);
        arma::vec ki(NDIM), ka(NDIM);
        HFS::occIndexToK(i, ki);
        ka = HFS::virIndexToK(a);
        for (uint j = 0; j < HFS::Nocc; ++j) {
            arma::vec kj(NDIM), kb(NDIM);
            HFS::occIndexToK(j, kj);
            kb = ka + kj - ki; // Momentum conservation for <aj|ib> or <aj|bi>
            HFS::toFirstBrillouinZone(kb);
            if (arma::norm(kb) > (HFS::kf + SMALLNUMBER)) {
                // only if momentum conserving state is virtual
                uint t = HFS::Matrix::calcTfromKbAndJ(kb, j);
                Mv(s) += (HFS::twoElectron(ka, ki)) * v(t);
            }
        }
    }
}

void E_delta_st_minus_aj_bi(arma::vec& v, arma::vec& Mv) {
    for (uint s = 0; s < HFS::Nexc; ++s) {
        uint i = HFS::excitations(0, s), a = HFS::excitations(1, s);
        arma::vec ki(NDIM), ka(NDIM);
        HFS::occIndexToK(i, ki);
        ka = HFS::virIndexToK(a);
        for (uint j = 0; j < HFS::Nocc; ++j) {
            arma::vec kj(NDIM), kb(NDIM);
            HFS::occIndexToK(j, kj);
            kb = ka + kj - ki; // Momentum conservation for <aj|ib> or <aj|bi>
            HFS::toFirstBrillouinZone(kb);
            if (arma::norm(kb) > (HFS::kf + SMALLNUMBER)) {
                // only if momentum conserving state is virtual
                uint t = HFS::Matrix::calcTfromKbAndJ(kb, j);
                if (s == t) {
                    Mv(s) += HFS::exc_energies(s) * v(t);
                } else {
                    Mv(s) += (-HFS::twoElectron(ka, kb)) * v(t);
                }
            }
        }
    }
}

void ab_ij_antisym(arma::vec& v, arma::vec& Mv) {
    for (uint s = 0; s < HFS::Nexc; ++s) {
        uint i = HFS::excitations(0, s), a = HFS::excitations(1, s);
        arma::vec ki(NDIM), ka(NDIM);
        HFS::occIndexToK(i, ki);
        ka = HFS::virIndexToK(a);
        for (uint j = 0; j < HFS::Nocc; ++j) {
            arma::vec kj(NDIM), kb(NDIM);
            HFS::occIndexToK(j, kj);
            kb = kj + ki - ka; // Momentum conservation for <ab|ij> or <ab|ji>
            HFS::toFirstBrillouinZone(kb);
            if (arma::norm(kb) > (HFS::kf + SMALLNUMBER)) {
                // only if momentum conserving state is virtual
                uint t = HFS::Matrix::calcTfromKbAndJ(kb, j);
                Mv(s) += (HFS::twoElectron(ka, ki) - HFS::twoElectron(ka, kj)) * v(t);
            }
        }
    }
}

void minus_abji_plus_2ab_ij(arma::vec& v, arma::vec& Mv) {
    for (uint s = 0; s < HFS::Nexc; ++s) {
        uint i = HFS::excitations(0, s), a = HFS::excitations(1, s);
        arma::vec ki(NDIM), ka(NDIM);
        HFS::occIndexToK(i, ki);
        ka = HFS::virIndexToK(a);
        for (uint j = 0; j < HFS::Nocc; ++j) {
            arma::vec kj(NDIM), kb(NDIM);
            HFS::occIndexToK(j, kj);
            kb = kj + ki - ka; // Momentum conservation for <ab|ij> or <ab|ji>
            HFS::toFirstBrillouinZone(kb);
            if (arma::norm(kb) > (HFS::kf + SMALLNUMBER)) {
                // only if momentum conserving state is virtual
                uint t = HFS::Matrix::calcTfromKbAndJ(kb, j);
                Mv(s) += (2 * HFS::twoElectron(ka, ki) - HFS::twoElectron(ka, kj)) * v(t);
            }
        }
    }
}

void ab_ij(arma::vec& v, arma::vec& Mv) {
    for (uint s = 0; s < HFS::Nexc; ++s) {
        uint i = HFS::excitations(0, s), a = HFS::excitations(1, s);
        arma::vec ki(NDIM), ka(NDIM);
        HFS::occIndexToK(i, ki);
        ka = HFS::virIndexToK(a);
        for (uint j = 0; j < HFS::Nocc; ++j) {
            arma::vec kj(NDIM), kb(NDIM);
            HFS::occIndexToK(j, kj);
            kb = kj + ki - ka; // Momentum conservation for <ab|ji>
            HFS::toFirstBrillouinZone(kb);
            if (arma::norm(kb) > (HFS::kf + SMALLNUMBER)) {
                // only if momentum conserving state is virtual
                uint t = HFS::Matrix::calcTfromKbAndJ(kb, j);
                Mv(s) += HFS::twoElectron(ka, ki) * v(t);
            }
        }
    }
}

void minus_ab_ji(arma::vec& v, arma::vec& Mv) {
    for (uint s = 0; s < HFS::Nexc; ++s) {
        uint i = HFS::excitations(0, s), a = HFS::excitations(1, s);
        arma::vec ki(NDIM), ka(NDIM);
        HFS::occIndexToK(i, ki);
        ka = HFS::virIndexToK(a);
        for (uint j = 0; j < HFS::Nocc; ++j) {
            arma::vec kj(NDIM), kb(NDIM);
            HFS::occIndexToK(j, kj);
            kb = kj + ki - ka; // Momentum conservation for <ab|ji>
            HFS::toFirstBrillouinZone(kb);
            if (arma::norm(kb) > (HFS::kf + SMALLNUMBER)) {
                // only if momentum conserving state is virtual
                uint t = HFS::Matrix::calcTfromKbAndJ(kb, j);
                Mv(s) += -HFS::twoElectron(ka, kj) * v(t);
            }
        }
    }
}

void ApplyMvFxnsToSubMatrices(
               arma::vec& v,
               arma::vec& Mv,
               uint Ndivisions,
               std::vector<std::tuple<uint, uint, void (*)(arma::vec&, arma::vec&)>> MvList
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
        void (*MvFunc)(arma::vec&, arma::vec&) = std::get<2>(MvL);

        arma::vec Mv_subview = arma::vec(&Mv[irow*N], N, false, true);
        arma::vec v_subview = arma::vec(&v[icol*N], N, false, true);
        MvFunc(v_subview, Mv_subview);
    }
}

} // namespace MatrixVectorProduct


uint calcTfromKbAndJ(arma::vec& kb, uint j) {
    arma::uvec b_N_uint(NDIM);
    kToIndex(kb, b_N_uint);
    # if NDIM == 1
        uint b = HFS::vir_N_to_1_mat(b_N_uint(0));
    # endif
    # if NDIM == 2
        uint b = HFS::vir_N_to_1_mat(b_N_uint(0), b_N_uint(1));
    # endif
    # if NDIM == 3
        uint b = HFS::vir_N_to_1_mat(b_N_uint(0), b_N_uint(1), b_N_uint(2));
    #endif
    uint t = HFS::inv_exc_mat(j, b);
    return t;
}

void setMatrixPropertiesFromCase() {

    if (HFS::mycase == "cRHF2cRHF") {
        HFS::MatVecProduct_func = MatrixVectorProduct::SingletH;
        HFS::Matrix_generator = HFS::Matrix::Gen::SingletH;
        HFS::Nmat = 2 * HFS::Nexc;
    } else if (HFS::mycase == "cRHF2cUHF") {
        HFS::MatVecProduct_func = MatrixVectorProduct::TripletH;
        HFS::Matrix_generator = HFS::Matrix::Gen::TripletH;
        HFS::Nmat = 2 * HFS::Nexc;
    } else if (HFS::mycase == "cUHF2cUHF") {
        HFS::MatVecProduct_func = MatrixVectorProduct::Hprime;
        HFS::Matrix_generator = HFS::Matrix::Gen::Hprime;
        HFS::Nmat = 4 * HFS::Nexc;
    } else if (HFS::mycase == "cRHF2cGHF") {
        HFS::MatVecProduct_func = MatrixVectorProduct::H;
        HFS::Matrix_generator = HFS::Matrix::Gen::H;
        HFS::Nmat = 8 * HFS::Nexc;
    } else {
        std::cout << "Invalid HFS::mycase, use one of: cRHF2cUHF, cUHF2cUHF, cRHF2cGHF" << std::endl;
        exit(EXIT_FAILURE);
    }

}

} // namespace Matrix
} // namespace HFS
