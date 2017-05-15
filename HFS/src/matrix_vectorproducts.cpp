#include "matrix_vectorproducts.hpp"
#include <functional>

MatrixVectorProduct::MatrixVectorProduct(PhysicalParams Params) {
    if (Params.mycase == "cRHF2cRHF") {
        Mvfunc = &MatrixVectorProduct::SingletH;
        } else if (Params.mycase == "cRHF2cUHF") {
            Mvfunc = &MatrixVectorProduct::TripletH;
        } else if (Params.mycase == "cUHF2cUHF") {
            Mvfunc = &MatrixVectorProduct::Hprime;
        } else if (Params.mycase == "cRHF2cGHF") {
            Mvfunc = &MatrixVectorProduct::H;
        } else {
            std::cout << "Invalid mycase, use one of: cRHF2cUHF, cUHF2cUHF, cRHF2cGHF" << std::endl;
            exit(EXIT_FAILURE);
    }
}
//MvFunc MatrixVectorProduct::get_Mvfunc() {
 //   using std::placeholders::_1;
 //   std::function <void (const arma::vec&, arma::vec&)> Mv
 //     = std::bind( &MatrixVectorProduct::Mvfunc, this, _1 );
 //   Mv = [&] () { (this->*Mvfunc); };
//    return Mv;
//}

void MatrixVectorProduct::TripletH(const arma::vec& v, arma::vec& Mv) {
    Mv.zeros();

    auto fp = std::bind(&MatrixVectorProduct::E_delta_st_minus_aj_bi, this, std::placeholders::_1, std::placeholders::_2);
    //std::vector<Loc2MemberFxnMap> MvMap (4);

    /*
    MvMapVec.resize(4);

    MvMapVec.push_back(Loc2MemberFxnMap(this, 0, 0, this.E_delta_st_minus_aj_bi));
    MvList[1] = std::make_tuple(0, 1, minus_ab_ji);
    MvList[2] = std::make_tuple(1, 0, minus_ab_ji);
    MvList[3] = std::make_tuple(1, 1, E_delta_st_minus_aj_bi);
    ApplyMvFxnsToSubMatrices(v, Mv, 2, MvList);
    */
    ApplyMvFxnToSubMatrix(v, Mv, 0, 0, 2, fp);
}

void MatrixVectorProduct::SingletH(const arma::vec& v, arma::vec& Mv) {
    Mv.zeros();
    std::vector<std::tuple<uint, uint, void (MatrixVectorProduct::*)(const arma::vec&, arma::vec&)>> MvList (4);

    MvList[0] = std::make_tuple(0, 0, E_delta_st_plus_2aj_ib_minus_ajbi);
    MvList[1] = std::make_tuple(0, 1, minus_abji_plus_2ab_ij);
    MvList[2] = std::make_tuple(1, 0, minus_abji_plus_2ab_ij);
    MvList[3] = std::make_tuple(1, 1, E_delta_st_plus_2aj_ib_minus_ajbi);
    ApplyMvFxnsToSubMatrices(v, Mv, 2, MvList);
}

void MatrixVectorProduct::Hprime(const arma::vec& v, arma::vec& Mv) {
    Mv.zeros();
    std::vector<std::tuple<uint, uint, void (MatrixVectorProduct::*)(const arma::vec&, arma::vec&)>> MvList (4);

    MvList[0] = std::make_tuple(0, 0, Aprime);
    MvList[1] = std::make_tuple(0, 1, Bprime);
    MvList[2] = std::make_tuple(1, 0, Bprime);
    MvList[3] = std::make_tuple(1, 1, Aprime);
    ApplyMvFxnsToSubMatrices(v, Mv, 2, MvList);
}

void MatrixVectorProduct::Aprime(const arma::vec& v, arma::vec& Mv) {
    std::vector<std::tuple<uint, uint, void (MatrixVectorProduct::*)(const arma::vec&, arma::vec&)>> MvList (4);

    MvList[0] = std::make_tuple(0, 0, E_delta_st_plus_aj_ib_antisym);
    MvList[1] = std::make_tuple(0, 1, aj_ib);
    MvList[2] = std::make_tuple(1, 0, aj_ib);
    MvList[3] = std::make_tuple(1, 1, E_delta_st_plus_aj_ib_antisym);
    ApplyMvFxnsToSubMatrices(v, Mv, 2, MvList);
}

void MatrixVectorProduct::Bprime(const arma::vec& v, arma::vec& Mv) {
    std::vector<std::tuple<uint, uint, void (MatrixVectorProduct::*)(const arma::vec&, arma::vec&)>> MvList (4);

    MvList[0] = std::make_tuple(0, 0, ab_ij_antisym);
    MvList[1] = std::make_tuple(0, 1, ab_ij);
    MvList[2] = std::make_tuple(1, 0, ab_ij);
    MvList[3] = std::make_tuple(1, 1, ab_ij_antisym);
    ApplyMvFxnsToSubMatrices(v, Mv, 2, MvList);
}


void MatrixVectorProduct::H(const arma::vec& v, arma::vec& Mv) {
    Mv.zeros();
    std::vector<std::tuple<uint, uint, void (MatrixVectorProduct::*)(const arma::vec&, arma::vec&)>> MvList (4);

    MvList[0] = std::make_tuple(0, 0, A);
    MvList[1] = std::make_tuple(0, 1, B);
    MvList[2] = std::make_tuple(1, 0, B);
    MvList[3] = std::make_tuple(1, 1, A);
    ApplyMvFxnsToSubMatrices(v, Mv, 2, MvList);
}

void MatrixVectorProduct::A(const arma::vec& v, arma::vec& Mv) {
    std::vector<std::tuple<uint, uint, void (MatrixVectorProduct::*)(const arma::vec&, arma::vec&)>> MvList (6);

    MvList[0] = std::make_tuple(0, 0, E_delta_st_plus_aj_ib_antisym);
    MvList[1] = std::make_tuple(0, 3, aj_ib);
    MvList[2] = std::make_tuple(1, 1, E_delta_st_minus_aj_bi);
    MvList[3] = std::make_tuple(2, 2, E_delta_st_minus_aj_bi);
    MvList[4] = std::make_tuple(3, 0, aj_ib);
    MvList[5] = std::make_tuple(3, 3, E_delta_st_plus_aj_ib_antisym);
    ApplyMvFxnsToSubMatrices(v, Mv, 4, MvList);
}

void MatrixVectorProduct::B(const arma::vec& v, arma::vec& Mv) {
    std::vector<std::tuple<uint, uint, void (MatrixVectorProduct::*)(const arma::vec&, arma::vec&)>> MvList (6);

    MvList[0] = std::make_tuple(0, 0, ab_ij_antisym);
    MvList[1] = std::make_tuple(0, 3, ab_ij);
    MvList[2] = std::make_tuple(1, 2, minus_ab_ji);
    MvList[3] = std::make_tuple(2, 1, minus_ab_ji);
    MvList[4] = std::make_tuple(3, 0, ab_ij);
    MvList[5] = std::make_tuple(3, 3, ab_ij_antisym);
    ApplyMvFxnsToSubMatrices(v, Mv, 4, MvList);
}


void MatrixVectorProduct::E_delta_st_plus_aj_ib_antisym(const arma::vec& v, arma::vec& Mv) {
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

void MatrixVectorProduct::E_delta_st_plus_2aj_ib_minus_ajbi(const arma::vec& v, arma::vec& Mv) {
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

void MatrixVectorProduct::aj_ib(const arma::vec& v, arma::vec& Mv) {
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

void MatrixVectorProduct::E_delta_st_minus_aj_bi(const arma::vec& v, arma::vec& Mv) {
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

void MatrixVectorProduct::ab_ij_antisym(const arma::vec& v, arma::vec& Mv) {
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

void MatrixVectorProduct::minus_abji_plus_2ab_ij(const arma::vec& v, arma::vec& Mv) {
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

void MatrixVectorProduct::ab_ij(const arma::vec& v, arma::vec& Mv) {
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

void MatrixVectorProduct::minus_ab_ji(const arma::vec& v, arma::vec& Mv) {
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

void MatrixVectorProduct::ApplyMvFxnToSubMatrix(const arma::vec& v, arma::vec& Mv,
                                                uint irow, uint icol,
                                                uint Ndivisions, MvPtr MvFunc) {
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

    uint N = v.n_elem / Ndivisions; // Size of each subview
    scalar* Mv_ptr = &Mv[irow * N];
    arma::vec Mv_subview = arma::vec(Mv_ptr, N, false, true);

    scalar* v_ptr  = (scalar*) &v[icol * N];
    /* This technically violates const correctness. Do not use v_ptr for anything but initializing
       a const vec */
    const arma::vec v_subview = arma::vec(v_ptr, N, false, true);
    MvFunc(v_subview, Mv_subview);
    }
}

void MatrixVectorProduct::ApplyMvFxnsToSubMatrices(
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

uint MatrixVectorProduct::calcTfromKbAndJ(const arma::vec& kb, uint j) {
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

} // namespace Matrix
