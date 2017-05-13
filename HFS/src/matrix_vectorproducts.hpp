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
#include <functional>
typedef void (*MvFunc)(const arma::vec&, arma::vec&);

class MatrixVectorProduct {

public:
    MatrixVectorProduct(PhysicalParams Params);
    MvFunc get_Mvfunc();

private:
    void (MatrixVectorProduct::*Mvfunc)(const arma::vec& v, arma::vec& Mv);
    void ApplyMvFxnsToSubMatrices(const arma::vec& v, arma::vec& Mv, uint Ndivisions,
                 std::vector<std::tuple<uint, uint, void (*)(const arma::vec&, arma::vec&)>> MvList
                 );
    uint calcTfromKbAndJ(const arma::vec&, uint);

    void E_delta_st_plus_aj_ib_antisym(const arma::vec& v, arma::vec& Mv);
    void E_delta_st_minus_aj_bi(const arma::vec& v, arma::vec& Mv);
    void E_delta_st_plus_2aj_ib_minus_ajbi(const arma::vec& v, arma::vec& Mv);
    void aj_ib(const arma::vec& v, arma::vec& Mv);
    void ab_ij_antisym(const arma::vec& v, arma::vec& Mv);
    void minus_abji_plus_2ab_ij(const arma::vec& v, arma::vec& Mv);
    void ab_ij(const arma::vec& v, arma::vec& Mv);
    void minus_ab_ji(const arma::vec& v, arma::vec& Mv);

    void SingletH(const arma::vec& v, arma::vec& Mv);
    void TripletH(const arma::vec& v, arma::vec& Mv);
    void Hprime(const arma::vec& v, arma::vec& Mv);
    void Aprime(const arma::vec& v, arma::vec& Mv);
    void Bprime(const arma::vec& v, arma::vec& Mv);
    void H(const arma::vec& v, arma::vec& Mv);
    void A(const arma::vec& v, arma::vec& Mv);
    void B(const arma::vec& v, arma::vec& Mv);

};  // MatrixVectorProduct

#endif // HFS_MATRIX_VECTORPRODUCTS_INCLUDED
