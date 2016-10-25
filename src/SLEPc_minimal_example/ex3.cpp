
static char help[] = "Solves the same eigenproblem as in example ex2, but using a shell matrix. "
                     "The problem is a standard symmetric eigenproblem corresponding to the 2-D Laplacian operator.\n\n"
                     "The command line options are:\n"
                     "  -n <n>, where <n> = number of grid subdivisions in both x and y dimensions.\n\n";

#include <slepceps.h>
#include <petscblaslapack.h>
#include <armadillo>
#include <iostream>
#include <iomanip>


PetscErrorCode Petsc_MatVecProd(Mat A, Vec x, Vec y);
arma::uword nmat = 100;
arma::mat mymat(nmat, nmat);

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
    Mat            A;               /* operator matrix */
    EPS            eps;             /* eigenproblem solver context */
    EPSType        type;
    PetscMPIInt    size;
    PetscInt       N, n=nmat, nev;
    PetscBool      terse;
    PetscErrorCode ierr;

    SlepcInitialize(&argc,&argv,(char*)0,help);
    ierr = MPI_Comm_size(PETSC_COMM_WORLD, &size);
    CHKERRQ(ierr);
    if (size != 1) SETERRQ(PETSC_COMM_WORLD,1,"This is a uniprocessor example only");

    ierr = PetscOptionsGetInt(NULL, NULL, "-n", &n, NULL);
    CHKERRQ(ierr);
    N = n;
    ierr = PetscPrintf(PETSC_COMM_WORLD,"\nTesting SLEPC Interfaced with armadillo matrix-vector product, N=%D \n\n", N);
    CHKERRQ(ierr);

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
       Compute the operator matrix that defines the eigensystem, Ax=kx
       - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    ierr = MatCreateShell(PETSC_COMM_WORLD, N, N, N, N, &n, &A);
    CHKERRQ(ierr);
    ierr = MatSetFromOptions(A);
    CHKERRQ(ierr);
    ierr = MatShellSetOperation(A,MATOP_MULT,(void(*)())Petsc_MatVecProd);
    CHKERRQ(ierr);
    ierr = MatShellSetOperation(A,MATOP_MULT_TRANSPOSE,(void(*)())Petsc_MatVecProd);
    CHKERRQ(ierr);

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                  Create the eigensolver and set various options
       - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    for (arma::uword i = 0; i < nmat; ++i) {
        for (arma::uword j = 0; j < nmat; ++j) {
            if (i == j) {
                mymat(i, j) = i+1;
            } else {
                mymat(i,j) = 0.0001;
            }
        }
    }
    arma::mat eigvecs;
    arma::vec eigvals;

    arma::eig_sym(eigvals, eigvecs, mymat);
    std::cout << "exact min = " << std::setprecision(10) << eigvals.min() << std::endl;

    ierr = EPSCreate(PETSC_COMM_WORLD,&eps); // create eigensolver context
    CHKERRQ(ierr);

    ierr = EPSSetOperators(eps,A,NULL); // Set Operators, null = non-general eigevalue problem
    CHKERRQ(ierr);
    ierr = EPSSetProblemType(eps,EPS_HEP);  // Hermitian eigenvalue?
    CHKERRQ(ierr);

    ierr = EPSSetFromOptions(eps); // Setting solver params at runtime
    CHKERRQ(ierr);

    ierr = EPSSolve(eps); // Solve the eigensystem
    CHKERRQ(ierr);

    // Optional: Get some information from the solver and display it
    ierr = EPSGetType(eps,&type);
    CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD," Solution method: %s\n\n",type);
    CHKERRQ(ierr);
    ierr = EPSGetDimensions(eps,&nev,NULL,NULL);
    CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD," Number of requested eigenvalues: %D\n",nev);
    CHKERRQ(ierr);

    // Display solution and clean up

    //show detailed info unless -terse option is given by user
    ierr = PetscOptionsHasName(NULL,NULL,"-terse",&terse);
    CHKERRQ(ierr);
    if (terse)
    {
        ierr = EPSErrorView(eps,EPS_ERROR_RELATIVE,NULL);
        CHKERRQ(ierr);
    }
    else
    {
        ierr = PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_ASCII_INFO_DETAIL);
        CHKERRQ(ierr);
        ierr = EPSReasonView(eps,PETSC_VIEWER_STDOUT_WORLD);
        CHKERRQ(ierr);
        ierr = EPSErrorView(eps,EPS_ERROR_RELATIVE,PETSC_VIEWER_STDOUT_WORLD);
        CHKERRQ(ierr);
        ierr = PetscViewerPopFormat(PETSC_VIEWER_STDOUT_WORLD);
        CHKERRQ(ierr);
    }
    ierr = EPSDestroy(&eps);
    CHKERRQ(ierr);
    ierr = MatDestroy(&A);
    CHKERRQ(ierr);
    ierr = SlepcFinalize();
    return ierr;
}


void myarma_Matvec_Prodec(arma::vec& v, arma::vec& Mv) {
    arma::uword N = v.n_elem;

    for (arma::uword j = 0; j < N; ++j) {
        double sum = 0.0;
        for (arma::uword i = 0; i < N; ++i) {
            sum += mymat(j, i) * v(i);
        }
        Mv(j) = sum;
    }
    arma::vec Mv2 = mymat * v;
}

static void My_Matvec_Prod(int nx, const PetscReal* x, PetscReal* y)
{
    double* v = (double*) x; 
    double* Mv = (double*) y; 
    arma::vec myvec(v, nx, false, true); // copy_aux_mem (first bool) must be false for this to work
    arma::vec myMv(Mv, nx, false, true); // since the memory pointed to by y is what PETSc sees. 
    myarma_Matvec_Prodec(myvec, myMv); // modifies myMv inplace
}


#undef __FUNCT__
#define __FUNCT__ "Petsc_MatVecProd"
PetscErrorCode Petsc_MatVecProd(Mat A, Vec x, Vec y)
{
    void*             ctx;
    int               nx;
    const PetscReal*  px;
    PetscReal*        py;
    PetscErrorCode    ierr;

    PetscFunctionBeginUser;
    ierr = MatShellGetContext(A, &ctx);
    CHKERRQ(ierr);

    nx = *(int*)ctx;
    ierr = VecGetArrayRead(x, &px);
    CHKERRQ(ierr);

    ierr = VecGetArray(y, &py);
    CHKERRQ(ierr);

    My_Matvec_Prod(nx, &px[0], &py[0]);

    ierr = VecRestoreArrayRead(x, &px);
    CHKERRQ(ierr);

    ierr = VecRestoreArray(y, &py);
    CHKERRQ(ierr);

    PetscFunctionReturn(0);
}
