/** @file matrix_utils.hpp
@author Evan Curtin
@version Revision 0.1
@brief Functions for matrix-vector products, matrix generation, and dependencies.
@detail The matrix names are in accordance with Seeger & Pople: doi=10.1063/1.434318
@date Wednesday, 04 Jan, 2017
*/

#ifndef HFS_matrix_utils_included
#define HFS_matrix_utils_included

#include "parameters.hpp"
#include "base_funcs.hpp"
#include "calc_parameters.hpp"

namespace HFS{

    // 3H
    extern void matrixVectorProduct3A(arma::vec&, arma::vec& Mv);
    extern void matrixVectorProduct3B(arma::vec&, arma::vec& Mv);
    extern void matrixVectorProduct3H(arma::vec&, arma::vec& Mv);
    extern double calcFromIndices3A(arma::uword, arma::uword);
    extern double calcFromIndices3B(arma::uword, arma::uword);
    extern double calcFromIndices3H(arma::uword, arma::uword);

    // Hprime
    extern void matrixVectorProductHprime(arma::vec&v, arma::vec& Mv);
    extern void matrixVectorProductAprime(arma::vec&v, arma::vec& Mv);
    extern void matrixVectorProductBprime(arma::vec&v, arma::vec& Mv);
    extern void matrixVectorProductAprimeDiag(arma::vec& v, arma::vec& Mv);
    extern void matrixVectorProductAprimeOffDiag(arma::vec& v, arma::vec& Mv);
    extern void matrixVectorProductBprimeDiag(arma::vec& v, arma::vec& Mv);
    extern void matrixVectorProductBprimeOffDiag(arma::vec& v, arma::vec& Mv);
    extern double calcFromIndicesAprime(arma::uword s, arma::uword t);
    extern double calcFromIndicesBprime(arma::uword s, arma::uword t);
    extern double calcFromIndicesHprime(arma::uword s, arma::uword t);

    // 1H
    extern double calcFromIndices1B(arma::uword, arma::uword);
    extern double calcFromIndices1A(arma::uword, arma::uword);

    // H
    extern double calcFromIndicesH(arma::uword s, arma::uword t);
    extern double calcFromIndicesA(arma::uword s, arma::uword t);
    extern double calcFromIndicesB(arma::uword s, arma::uword t);
    extern double calcFromIndicesA_M1(arma::uword s, arma::uword t);
    extern double calcFromIndicesA_M2(arma::uword s, arma::uword t);
    extern double calcFromIndicesA_M3(arma::uword s, arma::uword t);
    extern double calcFromIndicesB_M1(arma::uword s, arma::uword t);
    extern double calcFromIndicesB_M2(arma::uword s, arma::uword t);
    extern double calcFromIndicesB_M3(arma::uword s, arma::uword t);
    extern void matrixVectorProductH(arma::vec& v, arma::vec& Mv);
    extern void matrixVectorProductA(arma::vec& v, arma::vec& Mv);
    extern void matrixVectorProductB(arma::vec& v, arma::vec& Mv);
    extern void matrixVectorProductA_M1(arma::vec& v, arma::vec& Mv);
    extern void matrixVectorProductA_M2(arma::vec& v, arma::vec& Mv);
    extern void matrixVectorProductA_M3(arma::vec& v, arma::vec& Mv);
    extern void matrixVectorProductB_M1(arma::vec& v, arma::vec& Mv);
    extern void matrixVectorProductB_M2(arma::vec& v, arma::vec& Mv);
    extern void matrixVectorProductB_M3(arma::vec& v, arma::vec& Mv);
    extern void factorizeA(arma::vec& v, arma::vec& Mv
                    ,void (*M1v)(arma::vec&, arma::vec&)
                    ,void (*M2v)(arma::vec&, arma::vec&)
                    ,void (*M3v)(arma::vec&, arma::vec&));
    void factorizeB(arma::vec& v, arma::vec& Mv
                    ,void (*M1v)(arma::vec&, arma::vec&)
                    ,void (*M2v)(arma::vec&, arma::vec&)
                    ,void (*M3v)(arma::vec&, arma::vec&));


    // Utilities
    extern arma::uword calcTfromKbAndJ(arma::vec&, arma::uword);
    extern void setMatrixPropertiesFromCase();
    extern void factorize2by2MatrixVectorProduct(arma::vec& v, arma::vec& Mv
                         , void (*Av)(arma::vec&, arma::vec&)
                         , void (*Bv)(arma::vec&, arma::vec&)
                         , void (*Cv)(arma::vec&, arma::vec&)
                         , void (*Dv)(arma::vec&, arma::vec&)
                         );
    /** \brief
     *
     * \param
     * \param
     * \return
     *
     */


}

#endif // HFS_matrix_utils_included
