# include "matrix_gen.hpp"

arma::mat Matrix::Gen::H() {
    std::vector<std::pair<uint, uint>> locs(6);
    std::vector<scalar (*)(uint, uint)> funcs(6);

    // Make A
     locs[0] = std::make_pair(0, 0);
    funcs[0] = Matrix::Gen::A_E_delta_ij_delta_ab_plus_aj_ib_antisym;

     locs[1] = std::make_pair(0, 3);
    funcs[1] = Matrix::Gen::A_aj_ib;

     locs[2] = std::make_pair(1, 1);
    funcs[2] = Matrix::Gen::A_E_delta_ij_delta_ab_minus_aj_bi;

     locs[3] = std::make_pair(2, 2);
    funcs[3] = Matrix::Gen::A_E_delta_ij_delta_ab_minus_aj_bi;

     locs[4] = std::make_pair(3, 0);
    funcs[4] = Matrix::Gen::A_aj_ib;

     locs[5] = std::make_pair(3, 3);
    funcs[5] = Matrix::Gen::A_E_delta_ij_delta_ab_plus_aj_ib_antisym;

    arma::mat A = buildMatrixFromFunctionList(Nmat / 2, 4, locs, funcs);

    // Make B
     locs[0] = std::make_pair(0, 0);
    funcs[0] = Matrix::Gen::B_ab_ij_antisym;

     locs[1] = std::make_pair(0, 3);
    funcs[1] = Matrix::Gen::B_ab_ij;

     locs[2] = std::make_pair(1, 2);
    funcs[2] = Matrix::Gen::B_minus_ab_ji;

     locs[3] = std::make_pair(2, 1);
    funcs[3] = Matrix::Gen::B_minus_ab_ji;

     locs[4] = std::make_pair(3, 0);
    funcs[4] = Matrix::Gen::B_ab_ij;

     locs[5] = std::make_pair(3, 3);
    funcs[5] = Matrix::Gen::B_ab_ij_antisym;

    arma::mat B = buildMatrixFromFunctionList(Nmat / 2, 4, locs, funcs);

    return Matrix::Gen::buildHFromAandB(A, B);
}

arma::mat Matrix::Gen::Hprime() {
    std::vector<std::pair<uint, uint>> locs(4);
    std::vector<scalar (*)(uint, uint)> funcs(4);

    // Make A
     locs[0] = std::make_pair(0, 0);
    funcs[0] = Matrix::Gen::A_E_delta_ij_delta_ab_plus_aj_ib_antisym;

     locs[1] = std::make_pair(0, 1);
    funcs[1] = Matrix::Gen::A_aj_ib;

     locs[2] = std::make_pair(1, 0);
    funcs[2] = Matrix::Gen::A_aj_ib;

     locs[3] = std::make_pair(1, 1);
    funcs[3] = Matrix::Gen::A_E_delta_ij_delta_ab_plus_aj_ib_antisym;

    arma::mat A = buildMatrixFromFunctionList(Nmat / 2, 2, locs, funcs);

    // Make B
     locs[0] = std::make_pair(0, 0);
    funcs[0] = Matrix::Gen::B_ab_ij_antisym;

     locs[1] = std::make_pair(0, 1);
    funcs[1] = Matrix::Gen::B_ab_ij;

     locs[2] = std::make_pair(1, 0);
    funcs[2] = Matrix::Gen::B_ab_ij;

     locs[3] = std::make_pair(1, 1);
    funcs[3] = Matrix::Gen::B_ab_ij_antisym;

    arma::mat B = buildMatrixFromFunctionList(Nmat / 2, 2, locs, funcs);

    return Matrix::Gen::buildHFromAandB(A, B);

}

arma::mat Matrix::Gen::TripletH() {
    uint Ndivisions = 2;
    std::vector<std::pair<uint, uint>> locs(4);
    std::vector<scalar (*)(uint, uint)> funcs(4);

    locs[0] = std::make_pair(0, 0);
    funcs[0] = Matrix::Gen::A_E_delta_ij_delta_ab_minus_aj_bi;

    locs[1] = std::make_pair(0, 1);
    funcs[1] = Matrix::Gen::B_minus_ab_ji;

    locs[2] = std::make_pair(1, 0);
    funcs[2] = Matrix::Gen::B_minus_ab_ji;

    locs[3] = std::make_pair(1, 1);
    funcs[3] = Matrix::Gen::A_E_delta_ij_delta_ab_minus_aj_bi;
    return buildMatrixFromFunctionList(Nmat, Ndivisions, locs, funcs);
}

arma::mat Matrix::Gen::SingletH() {
    uint Ndivisions = 2;
    std::vector<std::pair<uint, uint>> locs(4);
    std::vector<scalar (*)(uint, uint)> funcs(4);

    locs[0] = std::make_pair(0, 0);
    funcs[0] = Matrix::Gen::A_E_delta_ij_delta_ab_plus_2aj_ib_minus_ajbi;

    locs[1] = std::make_pair(0, 1);
    funcs[1] = Matrix::Gen::B_minus_abji_plus_2abij;

    locs[2] = std::make_pair(1, 0);
    funcs[2] = Matrix::Gen::B_minus_abji_plus_2abij;

    locs[3] = std::make_pair(1, 1);
    funcs[3] = Matrix::Gen::A_E_delta_ij_delta_ab_plus_2aj_ib_minus_ajbi;
    return buildMatrixFromFunctionList(Nmat, Ndivisions, locs, funcs);
}

scalar Matrix::Gen::A_E_delta_ij_delta_ab_plus_aj_ib_antisym(uint s, uint t) {
    std::vector<arma::vec> klist(4);
    klist = stToKiKaKjKb(s, t);
    arma::vec ki(NDIM), kj(NDIM), ka(NDIM), kb(NDIM);
    ki = klist[0];
    ka = klist[1];
    kj = klist[2];
    kb = klist[3];

    scalar A1 = kroneckerDelta(s, t) * exc_energies(s)
        + twoElectronSafe(ka, kj, ki, kb) - twoElectronSafe(ka, kj, kb, ki);
    return A1;
}

scalar Matrix::Gen::A_E_delta_ij_delta_ab_plus_2aj_ib_minus_ajbi(uint s, uint t) {
    std::vector<arma::vec> klist(4);
    klist = stToKiKaKjKb(s, t);
    arma::vec ki(NDIM), kj(NDIM), ka(NDIM), kb(NDIM);
    ki = klist[0];
    ka = klist[1];
    kj = klist[2];
    kb = klist[3];

    scalar Ast = kroneckerDelta(s, t) * exc_energies(s)
        + 2 * twoElectronSafe(ka, kj, ki, kb) - twoElectronSafe(ka, kj, kb, ki);
    return Ast;
}

scalar Matrix::Gen::B_minus_abji_plus_2abij(uint s, uint t) {
    std::vector<arma::vec> klist(4);
    klist = stToKiKaKjKb(s, t);
    arma::vec ki(NDIM), kj(NDIM), ka(NDIM), kb(NDIM);
    ki = klist[0];
    ka = klist[1];
    kj = klist[2];
    kb = klist[3];

    scalar Bst = 2 * twoElectronSafe(ka, kb, ki, kj) - twoElectronSafe(ka, kb, kj, ki);
    return Bst;
}

scalar Matrix::Gen::A_aj_ib(uint s, uint t) {
    std::vector<arma::vec> klist(4);
    klist = stToKiKaKjKb(s, t);
    arma::vec ki(NDIM), kj(NDIM), ka(NDIM), kb(NDIM);
    ki = klist[0];
    ka = klist[1];
    kj = klist[2];
    kb = klist[3];

    scalar A2 = twoElectronSafe(ka, kj, ki, kb);
    return A2;
}

scalar Matrix::Gen::A_E_delta_ij_delta_ab_minus_aj_bi(uint s, uint t) {
    std::vector<arma::vec> klist(4);
    klist = stToKiKaKjKb(s, t);
    arma::vec ki(NDIM), kj(NDIM), ka(NDIM), kb(NDIM);
    ki = klist[0];
    ka = klist[1];
    kj = klist[2];
    kb = klist[3];

    scalar A3;
    A3 = kroneckerDelta(s, t) * exc_energies(s) - twoElectronSafe(ka, kj, kb, ki);
    return A3;
}

scalar Matrix::Gen::B_ab_ij_antisym(uint s, uint t) {
    std::vector<arma::vec> klist(4);
    klist = stToKiKaKjKb(s, t);
    arma::vec ki(NDIM), kj(NDIM), ka(NDIM), kb(NDIM);
    ki = klist[0];
    ka = klist[1];
    kj = klist[2];
    kb = klist[3];

    scalar B1 = twoElectronSafe(ka, kb, ki, kj) - twoElectronSafe(ka, kb, kj, ki);
    return B1;
}

scalar Matrix::Gen::B_ab_ij(uint s, uint t) {
    std::vector<arma::vec> klist(4);
    klist = stToKiKaKjKb(s, t);
    arma::vec ki(NDIM), kj(NDIM), ka(NDIM), kb(NDIM);
    ki = klist[0];
    ka = klist[1];
    kj = klist[2];
    kb = klist[3];

    scalar B2 = twoElectronSafe(ka, kb, ki, kj);
    return B2;
}

scalar Matrix::Gen::B_minus_ab_ji(uint s, uint t) {
    std::vector<arma::vec> klist(4);
    klist = stToKiKaKjKb(s, t);
    arma::vec ki(NDIM), kj(NDIM), ka(NDIM), kb(NDIM);
    ki = klist[0];
    ka = klist[1];
    kj = klist[2];
    kb = klist[3];

    scalar B2 = - twoElectronSafe(ka, kb, kj, ki);
    return B2;
}

arma::mat Matrix::Gen::buildMatrixFromFunctionList(uint N, uint Ndivisions,
                 std::vector<std::pair<uint, uint>> locs,
                 std::vector<scalar (*)(uint, uint)> Mfuncs) {
     /** \brief Apply in-place Matrix-Vector product functions to sub-matrices.
      *
      * Assumes each function DOES NOT INITIALIZE and ADDS the contribution to Mv IN-PLACE.
      * Assumes equal square matrix partitions of a square matrix.
      *
      * \param Ndivisions Number of sections per row/column.
      * \param locs
      * \param Mfuncs
      *
      */



    assert((locs.size() <= Ndivisions*Ndivisions) && "Too many divisions for matrix.");
    assert((Mfuncs.size() <= Ndivisions*Ndivisions) && "Too many functions for # of divisions.");
    assert((Mfuncs.size() == locs.size()) && "Unequal # of functions and locations.");

    arma::mat Matrix(N, N, arma::fill::zeros);
    uint Nsub = N / Ndivisions; // Size of each subsection

    for (uint i = 0; i < locs.size(); ++i) {
        uint irow = std::get<0>(locs[i]);
        uint icol = std::get<1>(locs[i]);
        for (uint s = 0; s < Nsub; ++s) {
            for (uint t = 0; t < Nsub; ++t) {
                Matrix(s + irow*Nsub, t + icol*Nsub) = Mfuncs[i](s, t);
            }
        }
    }
    return Matrix;
}

arma::mat Matrix::Gen::buildHFromAandB(arma::mat& A, arma::mat& B) {
    arma::mat Htop    = arma::join_rows(A, B);
    arma::mat Hbottom = arma::join_rows(B, A);
    return arma::join_cols(Htop, Hbottom);
}

