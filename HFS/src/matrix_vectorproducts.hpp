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

typedef void (*MvPtr)(const arma::vec&, arma::vec&); // Fucntion pointer to Mv



class MatrixVectorProduct {

typedef void (MatrixVectorProduct::*MvMemberFxnPtr)(const arma::vec&, arma::vec&);
typedef void (MvMemberFxn)(const arma::vec&, arma::vec&);
typedef std::vector<std::tuple<uint, uint, MvMemberFxnPtr>> Loc2FxnMap;

class Loc2MemberFxnMap {
    uint i;
    uint j;
    MvMemberFxnPtr Mv;
    MatrixVectorProduct& self;

    Loc2MemberFxnMap(MatrixVectorProduct& self_in, uint i_in, uint j_in, MvMemberFxnPtr Mv_in)
         : self (self_in) {
        i  = i_in;
        j  = j_in;
        Mv = Mv_in;
    }
};


public:
    MatrixVectorProduct(PhysicalParams Params);
    MvPtr get_Mvfunc();

private:
    MvMemberFxnPtr Mvfunc;
    std::vector<Loc2MemberFxnMap> MvMapVec;
    void ApplyMvFxnsToSubMatrices(const arma::vec& v, arma::vec& Mv, uint Ndivisions,
                                  std::vector<Loc2MemberFxnMap> MvList);
    void ApplyMvFxnToSubMatrix(const arma::vec& v, arma::vec& Mv,
                            uint irow, uint icol,
                            uint Ndivisions, MvMemberFxnPtr MvFunc);

    uint calcTfromKbAndJ;

    MvMemberFxn
        E_delta_st_plus_aj_ib_antisym,
        minus_abji_plus_2ab_ij,
        minus_ab_ji,
        E_delta_st_minus_aj_bi,
        E_delta_st_plus_2aj_ib_minus_ajbi,
        aj_ib,
        ab_ij, ab_ij_antisym,
        SingletH, TripletH,
        Hprime, Aprime, Bprime,
        H, A, B;


};  // MatrixVectorProduct



#endif // HFS_MATRIX_VECTORPRODUCTS_INCLUDED
