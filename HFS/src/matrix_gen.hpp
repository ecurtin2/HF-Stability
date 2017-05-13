/** @file matrix_gen.hpp
@author Evan Curtin
@version Revision 0.1
@brief Functions for matrix generation.
@detail The matrix names are in accordance with Seeger & Pople: doi=10.1063/1.434318
@date Wednesday, 04 Jan, 2017
*/

#ifndef HFS_MATRIX_GEN_INCLUDED
#define HFS_MATRIX_GEN_INCLUDED

#include "assert.h"
#include "parameters.hpp"

class MatrixManager {

public:
    MatrixManager(const PhysicalParams& params);
    arma::mat getMatrix();
    uint getNmat();

private:
    uint Nmat;
    arma::mat (MatrixManager::*Matrix_generator)();
    PhysicalParams Params;
    arma::mat buildMatrixFromFunctionList(uint N, uint Ndivisions,
                                         std::vector<std::pair<uint, uint>> locs,
                                         std::vector<scalar (MatrixManager::*)(uint, uint)> Mfuncs);
    arma::mat TripletH();
    arma::mat SingletH();
    arma::mat Hprime();
    arma::mat H();
    arma::mat buildHFromAandB(arma::mat& A, arma::mat& B);
    scalar A_E_delta_ij_delta_ab_plus_aj_ib_antisym(uint s, uint t);
    scalar A_E_delta_ij_delta_ab_plus_2aj_ib_minus_ajbi(uint s, uint t);
    scalar A_aj_ib(uint s, uint t);
    scalar A_E_delta_ij_delta_ab_minus_aj_bi(uint s, uint t);
    scalar B_ab_ij_antisym(uint s, uint t);
    scalar B_minus_abji_plus_2abij(uint s, uint t);
    scalar B_ab_ij(uint s, uint t);
    scalar B_minus_ab_ji(uint s, uint t);
};


#endif // HFS_MATRIX_GEN_INCLUDED
