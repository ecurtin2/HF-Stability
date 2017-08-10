/** @file SLEPcWrapper.hpp
@author Evan Curtin
@version Revision 0.1
@brief Bindings for SLEPc
@details Made to simplify and abstract the SLEPc function calls.
@date Wednesday, 04 Jan, 2017
*/

#ifndef SLEPC_WRAPPER_INCLUDED
#define SLEPC_WRAPPER_INCLUDED

#include <cstdio>
#include <armadillo>
#include <vector>
#include <slepceps.h>
#include <petscblaslapack.h>

#include "parameters.hpp"
#include "matrix_gen.hpp"




/** \namespace SLEPc
    \brief Classes/Functions using the SLEPc library.
*/
namespace SLEPc {

    extern PetscErrorCode Petsc_MatVecProd(Mat matrix, Vec x, Vec y);
    extern PetscErrorCode Petsc_MatDiags(Mat M, Vec diag);
    extern PetscErrorCode Petsc_Mv_TripletH(Mat M, Vec v, Vec Mv);
    extern void (*matvec_product)(arma::vec&, arma::vec&);
    class EpS {
        public:
            PetscErrorCode                     ierr;
            EPS                                eps;
            EPSType                            eps_type;
            PetscMPIInt                        nprocs=1;
            Mat                                matrix;
            PetscInt                           N, nconv, niter, BlockSize=1, nguess=1, Nevals=1, maxits;
            PetscScalar                             tol=1E-5;
            std::vector<PetscScalar>                iVals, rVals;
            std::vector< std::vector<PetscScalar> > iVecs, rVecs;

            EpS(PetscInt Ninput, void (*matvec_product)(arma::vec&, arma::vec&), int argc=0, char** argv = (char**)"eps_monitor");
            ~EpS();
            PetscErrorCode SetInitialSpace(std::vector<std::vector<PetscScalar>> vecs);
            PetscErrorCode EPSContext ();
            PetscErrorCode PETSCMatShellCreate(Mat& matrix);
            PetscErrorCode SetFromOptions();
            PetscErrorCode SetDimensions(PetscInt num_evals, PetscInt max_subspace_size);
            PetscErrorCode SetBlockSize(PetscInt blocksize);
            PetscErrorCode SetTol(PetscScalar tolerance, int max_it=200);
            PetscErrorCode Solve();
            void PrintEvals(const char* format="%10.5f\n");
            void PrintEvecs(const char* format="%10.5f");
            PetscErrorCode print();

    }; // Class EpS
}; // Namespace SLEPc

#endif // SLEPC_WRAPPER_INCLUDED
