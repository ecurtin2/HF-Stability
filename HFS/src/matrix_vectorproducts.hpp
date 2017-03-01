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
extern arma::uword calcTfromKbAndJ(arma::vec&, arma::uword);

namespace MatrixVectorProduct {
    extern void ApplyMvFxnsToSubMatrices(arma::vec& v, arma::vec& Mv, unsigned Ndivisions,
                 std::vector<std::tuple<unsigned, unsigned, void (*)(arma::vec&, arma::vec&)>> MvList
                 );

    extern void E_delta_st_plus_aj_ib_antisym(arma::vec& v, arma::vec& Mv);
    extern void E_delta_st_minus_aj_bi(arma::vec& v, arma::vec& Mv);
    extern void aj_ib(arma::vec& v, arma::vec& Mv);
    extern void ab_ij_antisym(arma::vec& v, arma::vec& Mv);
    extern void ab_ij(arma::vec& v, arma::vec& Mv);
    extern void minus_ab_ji(arma::vec& v, arma::vec& Mv);

    extern void TripletH(arma::vec& v, arma::vec& Mv);
    extern void Hprime(arma::vec& v, arma::vec& Mv);
    extern void Aprime(arma::vec& v, arma::vec& Mv);
    extern void Bprime(arma::vec& v, arma::vec& Mv);
    extern void H(arma::vec& v, arma::vec& Mv);
    extern void A(arma::vec& v, arma::vec& Mv);
    extern void B(arma::vec& v, arma::vec& Mv);

}  // MatrixVectorProduct
} // Matrix
}; // HFS

#endif // HFS_matrix_utils_included
