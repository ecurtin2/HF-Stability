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
#include "base_funcs.hpp"
#include "calc_parameters.hpp"


namespace HFS {
namespace Matrix {
namespace Gen {

    extern arma::mat buildMatrixFromFunctionList(unsigned N,
                                                 unsigned Ndivisions,
                                                 std::vector<std::pair<unsigned, unsigned>> locs,
                                                 std::vector<double (*)(arma::uword, arma::uword)> Mfuncs);
    extern arma::mat TripletH();
    extern arma::mat Hprime();
    extern arma::mat H();
    extern arma::mat buildHFromAandB(arma::mat& A, arma::mat& B);

    extern double A_E_delta_ij_delta_ab_plus_aj_ib_antisym(arma::uword s, arma::uword t);
    extern double A_aj_ib(arma::uword s, arma::uword t);
    extern double A_E_delta_ij_delta_ab_minus_aj_bi(arma::uword s, arma::uword t);
    extern double B_ab_ij_antisym(arma::uword s, arma::uword t);
    extern double B_ab_ij(arma::uword s, arma::uword t);
    extern double B_minus_ab_ji(arma::uword s, arma::uword t);

} // Gen
} // Matrix
} // HFS

#endif // HFS_MATRIX_GEN_INCLUDED
