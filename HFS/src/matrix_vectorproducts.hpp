/** @file matrix_utils.hpp
@author Evan Curtin
@version Revision 0.1
@brief Functions for matrix-vector products, matrix generation, and dependencies.
@detail The matrix names are in accordance with Seeger & Pople: doi=10.1063/1.434318
@date Wednesday, 04 Jan, 2017
*/

#ifndef HFS_MATRIX_VECTORPRODUCTS_INCLUDED
#define HFS_MATRIX_VECTORPRODUCTS_INCLUDED

#include "parameters.hpp"
#include "base_funcs.hpp"
#include "calc_parameters.hpp"
#include "matrix_gen.hpp"

namespace HFS{
namespace Matrix {

extern void setMatrixPropertiesFromCase();
extern uint calcTfromKbAndJ(const arma::vec&, uint);

namespace MatrixVectorProduct {
    extern void ApplyMvFxnsToSubMatrices(const arma::vec& v, arma::vec& Mv, uint Ndivisions,
                 std::vector<std::tuple<uint, uint, void (*)(const arma::vec&, arma::vec&)>> MvList
                 );

    extern void E_delta_st_plus_aj_ib_antisym(const arma::vec& v, arma::vec& Mv);
    extern void E_delta_st_minus_aj_bi(const arma::vec& v, arma::vec& Mv);
    extern void E_delta_st_plus_2aj_ib_minus_ajbi(const arma::vec& v, arma::vec& Mv);
    extern void aj_ib(const arma::vec& v, arma::vec& Mv);
    extern void ab_ij_antisym(const arma::vec& v, arma::vec& Mv);
    extern void minus_abji_plus_2ab_ij(const arma::vec& v, arma::vec& Mv);
    extern void ab_ij(const arma::vec& v, arma::vec& Mv);
    extern void minus_ab_ji(const arma::vec& v, arma::vec& Mv);

    extern void TripletH(const arma::vec& v, arma::vec& Mv);
    extern void Hprime(const arma::vec& v, arma::vec& Mv);
    extern void Aprime(const arma::vec& v, arma::vec& Mv);
    extern void Bprime(const arma::vec& v, arma::vec& Mv);
    extern void H(const arma::vec& v, arma::vec& Mv);
    extern void A(const arma::vec& v, arma::vec& Mv);
    extern void B(const arma::vec& v, arma::vec& Mv);

}  // MatrixVectorProduct
} // Matrix
}; // HFS

#endif // HFS_matrix_utils_included
