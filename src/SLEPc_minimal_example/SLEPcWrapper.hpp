#include <slepceps.h>
#include <cstdio>
#include <petscblaslapack.h>
#include <armadillo>
#include <vector>


namespace SLEPc {
    PetscErrorCode Petsc_MatVecProd(Mat matrix, Vec x, Vec y);
    void (*matvec_product)(arma::vec&, arma::vec&);
    class EpS {
        public:
            PetscErrorCode                     ierr;
            EPS                                eps; 
            EPSType                            eps_type;
            PetscMPIInt                        size;
            Mat                                matrix;
            PetscInt                           N, nconv, niter, BlockSize=1, nguess=1, Nevals=1;
            double                             tol=1E-5; 
            std::vector<double>                iVals, rVals;
            std::vector< std::vector<double> > iVecs, rVecs;


            EpS(int argc, char **argv, PetscInt Ninput, void (*matvec_product)(arma::vec&, arma::vec&)) {
                static char help[] = "Solves the same eigenproblem as in example ex2, but using a shell matrix. "
                                     "The problem is a standard symmetric eigenproblem corresponding to the 2-D Laplacian operator.\n\n"
                                     "The command line options are:\n"
                                     "  -n <n>, where <n> = number of grid subdivisions in both x and y dimensions.\n\n";

                N = Ninput;
                SLEPc::matvec_product = matvec_product;
                SlepcInitialize(&argc, &argv, (char*)0, help);
                MPI_Comm_size(PETSC_COMM_WORLD, &size);
                MatCreateShell(PETSC_COMM_WORLD, N, N, N, N, &N, &matrix);
                PETSCMatCreate(matrix);
                EPSCreate(PETSC_COMM_WORLD, &eps);
                EPSContext();
            }

            PetscErrorCode clean() {
                ierr = EPSDestroy(&eps);    CHKERRQ(ierr);
                ierr = MatDestroy(&matrix); CHKERRQ(ierr);
                ierr = SlepcFinalize();
                return ierr;
            }


            PetscErrorCode SetInitialSpace(std::vector<std::vector<double>> vecs) {
                
                int Nvecs = vecs.size();

                std::vector<Vec> petsc_vecs(Nvecs);
                std::vector<Vec*> petsc_vecpointers(Nvecs);
                std::vector<PetscInt> indices(N); 

                for (int i = 0; i < N; ++i) {
                    indices[i] = i;
                }

                for (int i = 0; i < Nvecs; ++i) {
                    MatCreateVecs(matrix, &petsc_vecs[i], NULL);
                    VecSet(petsc_vecs[i], 0.0); 
                    VecSetValues(petsc_vecs[i], Nvecs, &indices[0], &vecs[i][0], INSERT_VALUES);
                    petsc_vecpointers[i] = &petsc_vecs[i];
                }

                nguess = Nvecs;
                ierr = EPSSetInitialSpace(eps, Nvecs, *petsc_vecpointers.begin());
                return ierr;
            }

            PetscErrorCode EPSContext () {
                ierr = EPSSetOperators(eps, matrix, NULL);              CHKERRQ(ierr);  // Set Operators, null = non-general eigevalue problem
                ierr = EPSSetProblemType(eps, EPS_HEP);                 CHKERRQ(ierr);  // Hermitian eigenvalue?
                ierr = EPSSetWhichEigenpairs(eps, EPS_SMALLEST_REAL);   CHKERRQ(ierr);  // Set default searching
                ierr = EPSSetType(eps, EPSJD);                          CHKERRQ(ierr);  // Set default solver to Jacobi-Davidson
                return ierr;
            }


            PetscErrorCode PETSCMatCreate(Mat &matrix) {
                // only matvec prod supported
                ierr = MatSetFromOptions(matrix);                                                        CHKERRQ(ierr);
                ierr = MatShellSetOperation(matrix, MATOP_MULT,           (void(*)())Petsc_MatVecProd);  CHKERRQ(ierr);
                ierr = MatShellSetOperation(matrix, MATOP_MULT_TRANSPOSE, (void(*)())Petsc_MatVecProd);  CHKERRQ(ierr);
                return ierr;
            }


            PetscErrorCode SetDimensions(PetscInt num_evals, PetscInt max_subspace_size) {
                // set defaults, the last arg, mpd is a max projected dimension and is needed for 
                // solving many eigenpairs. 
                ierr = EPSSetDimensions(eps, num_evals, max_subspace_size, PETSC_DEFAULT);
                Nevals = num_evals; 
                return ierr;
            }
            
            PetscErrorCode SetBlockSize(PetscInt blocksize) {
                ierr = EPSJDSetBlockSize(eps, blocksize);   CHKERRQ(ierr);
                BlockSize = blocksize;
                return ierr;
            }
            
            PetscErrorCode SetTol(double tolerance, int max_it=200){
                tol = tolerance; 
                ierr = EPSSetTolerances(eps, tol, max_it); CHKERRQ(ierr);
                return ierr;
            }

            PetscErrorCode SetFromOptions() {
                ierr = EPSSetFromOptions(eps);  CHKERRQ(ierr);
                return ierr;
            }

            PetscErrorCode Solve() {
                ierr = EPSSetFromOptions(eps); CHKERRQ(ierr);
                ierr = EPSSolve(eps); CHKERRQ(ierr);
                
                // Retrieve Solutions
                ierr = EPSGetConverged(eps, &nconv); CHKERRQ(ierr);
                rVals.resize(nconv, 0.0);
                iVals.resize(nconv, 0.0);
                rVecs.resize(nconv);
                iVecs.resize(nconv);

                std::vector<PetscInt> indices(N); 
                for (int i = 0; i < N; ++i) {
                    indices[i] = i;
                }

                for (int i = 0; i < nconv; ++i) {
                    // Initialize inputs to GetEigenPair
                    PetscScalar rVal, iVal;
                    Vec PetscrVec, PetsciVec;
                    MatCreateVecs(matrix, &PetscrVec, NULL);
                    VecSet(PetscrVec, 0.0); 
                    MatCreateVecs(matrix, &PetsciVec, NULL);
                    VecSet(PetsciVec, 0.0); 
                    ierr = EPSGetEigenpair(eps, i, &rVal, &iVal, PetscrVec, PetsciVec); CHKERRQ(ierr);

                    // Store everything in a std::vector
                    iVals[i] = iVal;
                    rVals[i] = rVal;
                    iVecs[i].resize(N, 0.0);
                    rVecs[i].resize(N, 0.0);
                    ierr = VecGetValues(PetscrVec, N, &indices[0], &rVecs[i][0]);
                    ierr = VecGetValues(PetsciVec, N, &indices[0], &iVecs[i][0]);
                }
                return ierr;
            }


            void PrintEvals(const char* format="%10.5f\n") {
                for (int i = 0; i < nconv; ++i) {
                    printf(format, rVals[i]);
                }
            }

            void PrintEvecs(const char* format="%10.5f") {
                for (int i = 0; i < N; ++i) {
                    for (int j = 0; j < nconv; ++j) {
                        printf(format, rVecs[j][i]);
                    }
                    printf("\n");
                }
            }


            PetscErrorCode print(){
                ierr = PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD, PETSC_VIEWER_ASCII_INFO_DETAIL);    CHKERRQ(ierr);
                ierr = EPSReasonView(eps, PETSC_VIEWER_STDOUT_WORLD);                                       CHKERRQ(ierr);
                ierr = EPSErrorView(eps, EPS_ERROR_RELATIVE,PETSC_VIEWER_STDOUT_WORLD);                     CHKERRQ(ierr);
                ierr = PetscViewerPopFormat(PETSC_VIEWER_STDOUT_WORLD);                                     CHKERRQ(ierr);
                return ierr;
            }
    }; // Class EpS

    #undef __FUNCT__
    #define __FUNCT__ "Petsc_MatVecProd"
    PetscErrorCode Petsc_MatVecProd(Mat matrix, Vec x, Vec y) {
        void*             ctx;
        int               nx;
        const PetscReal*  px;
        PetscReal*        py;
        PetscErrorCode    ierr;

        PetscFunctionBeginUser;
        ierr = MatShellGetContext(matrix, &ctx);    CHKERRQ(ierr);

        nx = *(int*)ctx;
        ierr = VecGetArrayRead(x, &px);             CHKERRQ(ierr);
        ierr = VecGetArray(y, &py);                 CHKERRQ(ierr);

        double* v = (double*) &px[0];        // cast to non-const pointer for armadillo initialization
        double* Mv = (double*) &py[0];      
        arma::vec myvec(v, nx, false, true); // copy_aux_mem (first bool) must be false for this to work
        arma::vec myMv(Mv, nx, false, true); // since the memory pointed to by y is what PETSc sees. 
        SLEPc::matvec_product(myvec, myMv);  // modifies myMv inplace

        ierr = VecRestoreArrayRead(x, &px);         CHKERRQ(ierr);
        ierr = VecRestoreArray(y, &py);             CHKERRQ(ierr);
        PetscFunctionReturn(0);
    }

}; // Namespace SLEPc

