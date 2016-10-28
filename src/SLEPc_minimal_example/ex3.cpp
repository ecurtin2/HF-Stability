
static char help[] = "Solves the same eigenproblem as in example ex2, but using a shell matrix. "
                     "The problem is a standard symmetric eigenproblem corresponding to the 2-D Laplacian operator.\n\n"
                     "The command line options are:\n"
                     "  -n <n>, where <n> = number of grid subdivisions in both x and y dimensions.\n\n";

#include <slepceps.h>
#include <petscblaslapack.h>
#include <armadillo>
#include <iostream>
#include <iomanip>
#include <vector>


PetscErrorCode Petsc_MatVecProd(Mat matrix, Vec x, Vec y);
arma::uword nmat;
arma::mat mymat;


PetscErrorCode EPSContext (EPS eps, Mat matrix) {
    PetscErrorCode ierr;
    ierr = EPSSetOperators(eps, matrix, NULL);              CHKERRQ(ierr);  // Set Operators, null = non-general eigevalue problem
    ierr = EPSSetProblemType(eps, EPS_HEP);                 CHKERRQ(ierr);  // Hermitian eigenvalue?
    ierr = EPSSetWhichEigenpairs(eps, EPS_SMALLEST_REAL);   CHKERRQ(ierr);  // Set default searching
    ierr = EPSSetType(eps, EPSJD);                          CHKERRQ(ierr);  // Set default solver to Jacobi-Davidson
    return ierr;
}

PetscErrorCode SetInitialSpace(EPS eps, Mat matrix, std::vector<std::vector<double>> vecs) {
    
    int Nvecs = vecs.size();
    int N     = vecs[0].size();

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

    PetscErrorCode ierr;
    ierr = EPSSetInitialSpace(eps, Nvecs, *petsc_vecpointers.begin());
    return ierr;
}

PetscErrorCode PETSCMatCreate(Mat& matrix) {
    // only matvec prod supported
    PetscErrorCode ierr;
    ierr = MatSetFromOptions(matrix);                                                        CHKERRQ(ierr);
    ierr = MatShellSetOperation(matrix, MATOP_MULT,           (void(*)())Petsc_MatVecProd);  CHKERRQ(ierr);
    ierr = MatShellSetOperation(matrix, MATOP_MULT_TRANSPOSE, (void(*)())Petsc_MatVecProd);  CHKERRQ(ierr);
    return ierr;
}

void myarma_Matvec_Prodec(arma::vec& v, arma::vec& Mv) {
    Mv = mymat * v;
}

void My_Matvec_Prod(int nx, const PetscReal* x, PetscReal* y){
    double* v = (double*) x; 
    double* Mv = (double*) y; 
    arma::vec myvec(v, nx, false, true); // copy_aux_mem (first bool) must be false for this to work
    arma::vec myMv(Mv, nx, false, true); // since the memory pointed to by y is what PETSc sees. 
    myarma_Matvec_Prodec(myvec, myMv);   // modifies myMv inplace
}

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

    My_Matvec_Prod(nx, &px[0], &py[0]);

    ierr = VecRestoreArrayRead(x, &px);         CHKERRQ(ierr);
    ierr = VecRestoreArray(y, &py);             CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

void test_eigs(int nmat) {
    for (int i = 0; i < nmat; ++i) {
        for (int j = 0; j < nmat; ++j) {
            if (i == j) {
                mymat(i, j) = i+1;
            } else {
                mymat(i,j) = 0.0000001;
            }
        }
    }
    arma::mat eigvecs;
    arma::vec eigvals;

    arma::eig_sym(eigvals, eigvecs, mymat);
    arma::vec sortvals = arma::sort(eigvals);

    sortvals.head(5).print("Actual evals");
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv){
    PetscMPIInt    size;
    PetscErrorCode ierr;

    std::cout << "Enter nmat:" << std::endl; std::cin >> nmat;
    mymat.set_size(nmat, nmat);
    PetscInt       N, nev;
    N = nmat;

    test_eigs(N);

    SlepcInitialize(&argc, &argv, (char*)0, help);
    ierr = MPI_Comm_size(PETSC_COMM_WORLD, &size);      CHKERRQ(ierr);
    if (size != 1) SETERRQ(PETSC_COMM_WORLD, 1, "This is a uniprocessor example only");

    //Compute the operator matrix that defines the eigensystem, Ax=kx
    Mat matrix;
    ierr = MatCreateShell(PETSC_COMM_WORLD, N, N, N, N, &N, &matrix);   CHKERRQ(ierr);
    ierr = PETSCMatCreate(matrix);  CHKERRQ(ierr);

    //Create the eigensolver and set various options
    EPS eps; 
    EPSType type;
    ierr = EPSCreate(PETSC_COMM_WORLD, &eps);               CHKERRQ(ierr);  // Create Eigenvalue Solver Problem context
    EPSContext(eps, matrix);
    
    int nguess = 10;
    std::vector< std::vector<double> > vecs(nguess, std::vector<double>(N, 0.0));
    for (int i = 0; i < nguess; ++i) {
        vecs[i][i] = 1.0;
    }

    ierr = SetInitialSpace(eps, matrix, vecs);

    int num_evals = 5;
    int max_subspace_size = 1000;
    ierr = EPSSetDimensions(eps, num_evals, max_subspace_size, PETSC_DEFAULT);      CHKERRQ(ierr); // set defaults, the last arg, mpd is a max
                                                                                                   // projected dimension and is needed for 
                                                                                                   // solving many eigenpairs. 
    int blocksize = 5;
    ierr = EPSJDSetBlockSize(eps, blocksize);   CHKERRQ(ierr);
    ierr = EPSSetFromOptions(eps);              CHKERRQ(ierr); // Setting solver params at runtime (overrides what came before)
    ierr = EPSSolve(eps);                       CHKERRQ(ierr); 


    ierr = PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD, PETSC_VIEWER_ASCII_INFO_DETAIL); CHKERRQ(ierr);
    ierr = EPSReasonView(eps, PETSC_VIEWER_STDOUT_WORLD);                                    CHKERRQ(ierr);
    ierr = EPSErrorView(eps, EPS_ERROR_RELATIVE,PETSC_VIEWER_STDOUT_WORLD);                  CHKERRQ(ierr);
    ierr = PetscViewerPopFormat(PETSC_VIEWER_STDOUT_WORLD);                                  CHKERRQ(ierr);

    // Clean up
    ierr = EPSDestroy(&eps);    CHKERRQ(ierr);
    ierr = MatDestroy(&matrix); CHKERRQ(ierr);
    ierr = SlepcFinalize();
    return ierr;
}
