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

    // Utilities
    extern arma::uword calcTfromKbAndJ(arma::vec&, arma::uword);
    extern void setMatrixPropertiesFromCase();
    extern void factorize2by2MatrixVectorProduct(arma::vec& v, arma::vec& Mv
                         , void (*Av)(arma::vec&, arma::vec&)
                         , void (*Bv)(arma::vec&, arma::vec&)
                         , void (*Cv)(arma::vec&, arma::vec&)
                         , void (*Dv)(arma::vec&, arma::vec&)
                         );

}

#endif // HFS_matrix_utils_included
