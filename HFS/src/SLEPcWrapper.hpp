/** @file SLEPcWrapper.hpp
@author Evan Curtin
@version Revision 0.1
@brief Bindings for SLEPc
@details Made to simplify and abstract the SLEPc function calls.
@date Wednesday, 04 Jan, 2017
*/

#ifndef SLEPC_WRAPPER_INCLUDED
#define SLEPC_WRAPPER_INCLUDED

#include <slepceps.h>
#include <cstdio>
#include <petscblaslapack.h>
#include <armadillo>
#include <vector>

/** \namespace SLEPc
    \brief Classes/Functions using the SLEPc library.
*/
namespace SLEPc {
    extern PetscErrorCode Petsc_MatVecProd(Mat matrix, Vec x, Vec y);
    extern void (*matvec_product)(arma::vec&, arma::vec&);
    class EpS {
        public:
            PetscErrorCode                     ierr;
            EPS                                eps;
            EPSType                            eps_type;
            PetscMPIInt                        nprocs=1;
            Mat                                matrix;
            PetscInt                           N, nconv, niter, BlockSize=1, nguess=1, Nevals=1, maxits;
            double                             tol=1E-5;
            std::vector<double>                iVals, rVals;
            std::vector< std::vector<double> > iVecs, rVecs;
            int                                argc=1;
            char*                              args = (char*)"eps_monitor";
            char**                             argv=&args;


            EpS(PetscInt Ninput, void (*matvec_product)(arma::vec&, arma::vec&));
            ~EpS();
            PetscErrorCode SetInitialSpace(std::vector<std::vector<double>> vecs);
            PetscErrorCode EPSContext ();
            PetscErrorCode PETSCMatShellCreate(Mat& matrix);
            PetscErrorCode SetFromOptions();
            PetscErrorCode SetDimensions(PetscInt num_evals, PetscInt max_subspace_size);
            PetscErrorCode SetBlockSize(PetscInt blocksize);
            PetscErrorCode SetTol(double tolerance, int max_it=200);
            PetscErrorCode Solve();
            void PrintEvals(const char* format="%10.5f\n");
            void PrintEvecs(const char* format="%10.5f");
            PetscErrorCode print();

    }; // Class EpS
}; // Namespace SLEPc

#endif // SLEPC_WRAPPER_INCLUDED
