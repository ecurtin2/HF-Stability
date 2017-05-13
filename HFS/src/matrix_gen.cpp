#include "matrix_gen.hpp"
#include "utils.hpp"

MatrixManager::MatrixManager(const PhysicalParams& params) : Params(params) {

    if (Params.mycase == "cRHF2cRHF") {
        Matrix_generator = &MatrixManager::SingletH;
        Nmat = 2 * Params.Nexc;
    } else if (Params.mycase == "cRHF2cUHF") {
        Matrix_generator = &MatrixManager::TripletH;
        Nmat = 2 * Params.Nexc;
    } else if (Params.mycase == "cUHF2cUHF") {
        Matrix_generator = &MatrixManager::Hprime;
        Nmat = 4 * Params.Nexc;
    } else if (Params.mycase == "cRHF2cGHF") {
        Matrix_generator = &MatrixManager::H;
        Nmat = 8 * Params.Nexc;
    } else {
        std::cout << "Invalid mycase, use one of: cRHF2cUHF, cUHF2cUHF, cRHF2cGHF" << std::endl;
        exit(EXIT_FAILURE);
    }
}

arma::mat MatrixManager::getMatrix() {
    return (this->*Matrix_generator)();
}

uint MatrixManager::getNmat() {
    return Nmat;
}

arma::mat MatrixManager::H() {
    std::vector<std::pair<uint, uint>> locs(6);
    std::vector<scalar (MatrixManager::*)(uint, uint)> funcs(6);

    // Make A
     locs[0] = std::make_pair(0, 0);
    funcs[0] = &MatrixManager::A_E_delta_ij_delta_ab_plus_aj_ib_antisym;

     locs[1] = std::make_pair(0, 3);
    funcs[1] = &MatrixManager::A_aj_ib;

     locs[2] = std::make_pair(1, 1);
    funcs[2] = &MatrixManager::A_E_delta_ij_delta_ab_minus_aj_bi;

     locs[3] = std::make_pair(2, 2);
    funcs[3] = &MatrixManager::A_E_delta_ij_delta_ab_minus_aj_bi;

     locs[4] = std::make_pair(3, 0);
    funcs[4] = &MatrixManager::A_aj_ib;

     locs[5] = std::make_pair(3, 3);
    funcs[5] = &MatrixManager::A_E_delta_ij_delta_ab_plus_aj_ib_antisym;

    arma::mat A = buildMatrixFromFunctionList(Nmat / 2, 4, locs, funcs);

    // Make B
     locs[0] = std::make_pair(0, 0);
    funcs[0] = &MatrixManager::B_ab_ij_antisym;

     locs[1] = std::make_pair(0, 3);
    funcs[1] = &MatrixManager::B_ab_ij;

     locs[2] = std::make_pair(1, 2);
    funcs[2] = &MatrixManager::B_minus_ab_ji;

     locs[3] = std::make_pair(2, 1);
    funcs[3] = &MatrixManager::B_minus_ab_ji;

     locs[4] = std::make_pair(3, 0);
    funcs[4] = &MatrixManager::B_ab_ij;

     locs[5] = std::make_pair(3, 3);
    funcs[5] = &MatrixManager::B_ab_ij_antisym;

    arma::mat B = buildMatrixFromFunctionList(Nmat / 2, 4, locs, funcs);

    return MatrixManager::buildHFromAandB(A, B);
}

arma::mat MatrixManager::Hprime() {
    std::vector<std::pair<uint, uint>> locs(4);
    std::vector<scalar (MatrixManager::*)(uint, uint)> funcs(6);

    // Make A
     locs[0] = std::make_pair(0, 0);
    funcs[0] = &MatrixManager::A_E_delta_ij_delta_ab_plus_aj_ib_antisym;

     locs[1] = std::make_pair(0, 1);
    funcs[1] = &MatrixManager::A_aj_ib;

     locs[2] = std::make_pair(1, 0);
    funcs[2] = &MatrixManager::A_aj_ib;

     locs[3] = std::make_pair(1, 1);
    funcs[3] = &MatrixManager::A_E_delta_ij_delta_ab_plus_aj_ib_antisym;

    arma::mat A = buildMatrixFromFunctionList(Nmat / 2, 2, locs, funcs);

    // Make B
     locs[0] = std::make_pair(0, 0);
    funcs[0] = &MatrixManager::B_ab_ij_antisym;

     locs[1] = std::make_pair(0, 1);
    funcs[1] = &MatrixManager::B_ab_ij;

     locs[2] = std::make_pair(1, 0);
    funcs[2] = &MatrixManager::B_ab_ij;

     locs[3] = std::make_pair(1, 1);
    funcs[3] = &MatrixManager::B_ab_ij_antisym;

    arma::mat B = buildMatrixFromFunctionList(Nmat / 2, 2, locs, funcs);

    return MatrixManager::buildHFromAandB(A, B);

}

arma::mat MatrixManager::TripletH() {
    uint Ndivisions = 2;
    std::vector<std::pair<uint, uint>> locs(4);
    std::vector<scalar (MatrixManager::*)(uint, uint)> funcs(6);

    locs[0] = std::make_pair(0, 0);
    funcs[0] = &MatrixManager::A_E_delta_ij_delta_ab_minus_aj_bi;

    locs[1] = std::make_pair(0, 1);
    funcs[1] = &MatrixManager::B_minus_ab_ji;

    locs[2] = std::make_pair(1, 0);
    funcs[2] = &MatrixManager::B_minus_ab_ji;

    locs[3] = std::make_pair(1, 1);
    funcs[3] = &MatrixManager::A_E_delta_ij_delta_ab_minus_aj_bi;
    return buildMatrixFromFunctionList(Nmat, Ndivisions, locs, funcs);
}

arma::mat MatrixManager::SingletH() {
    uint Ndivisions = 2;
    std::vector<std::pair<uint, uint>> locs(4);
    std::vector<scalar (MatrixManager::*)(uint, uint)> funcs(6);

    locs[0] = std::make_pair(0, 0);
    funcs[0] = &MatrixManager::A_E_delta_ij_delta_ab_plus_2aj_ib_minus_ajbi;

    locs[1] = std::make_pair(0, 1);
    funcs[1] = &MatrixManager::B_minus_abji_plus_2abij;

    locs[2] = std::make_pair(1, 0);
    funcs[2] = &MatrixManager::B_minus_abji_plus_2abij;

    locs[3] = std::make_pair(1, 1);
    funcs[3] = &MatrixManager::A_E_delta_ij_delta_ab_plus_2aj_ib_minus_ajbi;
    return buildMatrixFromFunctionList(Nmat, Ndivisions, locs, funcs);
}

scalar MatrixManager::A_E_delta_ij_delta_ab_plus_aj_ib_antisym(uint s, uint t) {
    std::vector<arma::vec> klist(4);
    klist = Params.stToKiKaKjKb(s, t);
    arma::vec ki(NDIM), kj(NDIM), ka(NDIM), kb(NDIM);
    ki = klist[0];
    ka = klist[1];
    kj = klist[2];
    kb = klist[3];

    scalar A1 = utils::kroneckerDelta(s, t) * Params.exc_energies(s)
        + Params.twoElectronSafe(ka, kj, ki, kb) - Params.twoElectronSafe(ka, kj, kb, ki);
    return A1;
}

scalar MatrixManager::A_E_delta_ij_delta_ab_plus_2aj_ib_minus_ajbi(uint s, uint t) {
    std::vector<arma::vec> klist(4);
    klist = Params.stToKiKaKjKb(s, t);
    arma::vec ki(NDIM), kj(NDIM), ka(NDIM), kb(NDIM);
    ki = klist[0];
    ka = klist[1];
    kj = klist[2];
    kb = klist[3];

    scalar Ast = utils::kroneckerDelta(s, t) * Params.exc_energies(s)
        + 2 * Params.twoElectronSafe(ka, kj, ki, kb) - Params.twoElectronSafe(ka, kj, kb, ki);
    return Ast;
}

scalar MatrixManager::B_minus_abji_plus_2abij(uint s, uint t) {
    std::vector<arma::vec> klist(4);
    klist = Params.stToKiKaKjKb(s, t);
    arma::vec ki(NDIM), kj(NDIM), ka(NDIM), kb(NDIM);
    ki = klist[0];
    ka = klist[1];
    kj = klist[2];
    kb = klist[3];

    scalar Bst = 2 * Params.twoElectronSafe(ka, kb, ki, kj) - Params.twoElectronSafe(ka, kb, kj, ki);
    return Bst;
}

scalar MatrixManager::A_aj_ib(uint s, uint t) {
    std::vector<arma::vec> klist(4);
    klist = Params.stToKiKaKjKb(s, t);
    arma::vec ki(NDIM), kj(NDIM), ka(NDIM), kb(NDIM);
    ki = klist[0];
    ka = klist[1];
    kj = klist[2];
    kb = klist[3];

    scalar A2 = Params.twoElectronSafe(ka, kj, ki, kb);
    return A2;
}

scalar MatrixManager::A_E_delta_ij_delta_ab_minus_aj_bi(uint s, uint t) {
    std::vector<arma::vec> klist(4);
    klist = Params.stToKiKaKjKb(s, t);
    arma::vec ki(NDIM), kj(NDIM), ka(NDIM), kb(NDIM);
    ki = klist[0];
    ka = klist[1];
    kj = klist[2];
    kb = klist[3];

    scalar A3;
    A3 = utils::kroneckerDelta(s, t) * Params.exc_energies(s) - Params.twoElectronSafe(ka, kj, kb, ki);
    return A3;
}

scalar MatrixManager::B_ab_ij_antisym(uint s, uint t) {
    std::vector<arma::vec> klist(4);
    klist = Params.stToKiKaKjKb(s, t);
    arma::vec ki(NDIM), kj(NDIM), ka(NDIM), kb(NDIM);
    ki = klist[0];
    ka = klist[1];
    kj = klist[2];
    kb = klist[3];

    scalar B1 = Params.twoElectronSafe(ka, kb, ki, kj) - Params.twoElectronSafe(ka, kb, kj, ki);
    return B1;
}

scalar MatrixManager::B_ab_ij(uint s, uint t) {
    std::vector<arma::vec> klist(4);
    klist = Params.stToKiKaKjKb(s, t);
    arma::vec ki(NDIM), kj(NDIM), ka(NDIM), kb(NDIM);
    ki = klist[0];
    ka = klist[1];
    kj = klist[2];
    kb = klist[3];

    scalar B2 = Params.twoElectronSafe(ka, kb, ki, kj);
    return B2;
}

scalar MatrixManager::B_minus_ab_ji(uint s, uint t) {
    std::vector<arma::vec> klist(4);
    klist = Params.stToKiKaKjKb(s, t);
    arma::vec ki(NDIM), kj(NDIM), ka(NDIM), kb(NDIM);
    ki = klist[0];
    ka = klist[1];
    kj = klist[2];
    kb = klist[3];

    scalar B2 = - Params.twoElectronSafe(ka, kb, kj, ki);
    return B2;
}

arma::mat MatrixManager::buildMatrixFromFunctionList(uint N, uint Ndivisions,
                 std::vector<std::pair<uint, uint>> locs,
                 std::vector<scalar (MatrixManager::*)(uint, uint)> Mfuncs) {
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
                // syntax for calling member function pointer
                Matrix(s + irow*Nsub, t + icol*Nsub) = (this->*Mfuncs[i])(s, t);
            }
        }
    }
    return Matrix;
}

arma::mat MatrixManager::buildHFromAandB(arma::mat& A, arma::mat& B) {
    arma::mat Htop    = arma::join_rows(A, B);
    arma::mat Hbottom = arma::join_rows(B, A);
    return arma::join_cols(Htop, Hbottom);
}

