/** @file debug.hpp
@author Evan Curtin
@version Revision 0.1
@brief Header including prototypes to assist in debugging and profiling.
@details The definitions are in debug.cpp.
@date Wednesday, 04 Jan, 2017
*/

#ifndef HFS_DEBUG_INCLUDED
#define HFS_DEBUG_INCLUDED

#include <assert.h>
#include "parameters.hpp"
#include "matrix_vectorproducts.hpp"
#include "matrix_gen.hpp"

scalar full_diag_min; /**< The minimum eigenvalue determined by full diagonalization of the stability matrix. Used to debug the Davidson algorithm.*/
scalar mv_time; /**< The time taken to execute one call of the matrix-vector product function. */
scalar full_diag_time; /**< The time taken to diagonalize the matrix using armadillo's eig_sym. */
bool agreesWithFullDiag();
/**< \brief Checks that Davidson's Algorithm is getting the lowest eigenvalue.
The full matrix is built and diagonalized, which is slow for Nk > 20 or so. The
minimum of the eigenvalues is compared to the value returned by Davidson's algorithm
and is said to be equal if they are the same within SMALLNUMBER
@return true if |DavidsonValue - MinimumEigenvalue| < SMALLNUMBER, otherwise false
@see buildMatrixFromFunction()
@see SMALLNUMBER
*/

bool matrixVectorProductWorks(const arma::mat& M, void (*Mv_func)(const arma::vec& v, arma::vec& Mv));
/**< \brief Checks that matrix vector product function gets the same result as armadillo's matrix multiply.
The full matrix is built and multiplied by a random vector. T
and is said to be equal if they are the same within SMALLNUMBER
@return true if |Mv_armadillo - Mv_function| < SMALLNUMBER element-wise, Otherwise false.
@see MatVecProduct_func
@see SMALLNUMBER
*/

bool everything_works();
/**< \brief Checks matrix-vector product and Davidson's algorithm.
If NDEBUG is defined, it won't do anything. Will throw exceptions if there is an issue.
@return true always, will throw an exception if one of the functions fails.
@see davidsonAgreesWithFullDiag()
@see matrixVectorProductWorks()
*/

double timeMatrixVectorProduct(void (* const Mv_func)(const arma::vec&, arma::vec&), uint N);
/**< \brief Times the matrix vector product execution.
Sets the value of mv_time.
@return The matrix.
@see Mv_time
*/

#endif // HFS_debug_included
