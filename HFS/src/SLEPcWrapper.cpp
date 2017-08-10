#include "SLEPcWrapper.hpp"

void (*SLEPc::matvec_product)(arma::vec&, arma::vec&);

SLEPc::EpS::EpS(PetscInt Ninput, void (*matvec_product)(arma::vec&, arma::vec&), int argc, char* argv[]) {
    static char help[] = "Solves the same eigenproblem as in example ex2, but using a shell matrix. "
                         "The problem is a standard symmetric eigenproblem corresponding to the 2-D Laplacian operator.\n\n"
                         "The command line options are:\n"
                         "  -n <n>, where <n> = number of grid subdivisions in both x and y dimensions.\n\n";


    N = Ninput;
    SLEPc::matvec_product = matvec_product;
    SlepcInitialize(&argc, &argv, (char*)0, help);
    MPI_Comm_size(PETSC_COMM_WORLD, &nprocs);
    MatCreateShell(PETSC_COMM_WORLD, N / nprocs + 1, N / nprocs + 1, PETSC_DETERMINE, PETSC_DETERMINE, &N, &matrix);

    PETSCMatShellCreate(matrix);
    EPSCreate(PETSC_COMM_WORLD, &eps);
    EPSContext();
}

SLEPc::EpS::~EpS() {
    ierr = EPSDestroy(&eps);    //CHKERRQ(ierr);
    ierr = MatDestroy(&matrix); //CHKERRQ(ierr);
    ierr = SlepcFinalize();
}

PetscErrorCode SLEPc::EpS::SetInitialSpace(std::vector<std::vector<PetscScalar>> vecs) {

    int Nvecs = vecs.size();

    std::vector<Vec> petsc_vecs(Nvecs);
    std::vector<Vec*> petsc_vecpointers(Nvecs);
    std::vector<PetscInt> indices(N);

    for (int i = 0; i < N; ++i) {
        indices[i] = i;
    }

    for (int i = 0; i < Nvecs; ++i) {
        ierr = MatCreateVecs(matrix, &petsc_vecs[i], NULL);                                 CHKERRQ(ierr);
        ierr = VecSet(petsc_vecs[i], 0.0);                                                  CHKERRQ(ierr);
        ierr = VecSetValues(petsc_vecs[i], Nvecs, &indices[0], &vecs[i][0], INSERT_VALUES); CHKERRQ(ierr);
        petsc_vecpointers[i] = &petsc_vecs[i];                                              CHKERRQ(ierr);
        ierr = VecAssemblyBegin(petsc_vecs[i]);                                             CHKERRQ(ierr);
        ierr = VecAssemblyEnd(petsc_vecs[i]);                                               CHKERRQ(ierr);
    }
    nguess = Nvecs;
    ierr = EPSSetInitialSpace(eps, Nvecs, *petsc_vecpointers.begin());
    return ierr;
}

PetscErrorCode SLEPc::EpS::EPSContext () {
    ierr = EPSSetOperators(eps, matrix, NULL);              CHKERRQ(ierr);  // Set Operators, null = non-general eigevalue problem
    ierr = EPSSetProblemType(eps, EPS_HEP);                 CHKERRQ(ierr);  // Hermitian eigenvalue?
    ierr = EPSSetWhichEigenpairs(eps, EPS_SMALLEST_REAL);   CHKERRQ(ierr);  // Set default searching
    ierr = EPSSetType(eps, EPSKRYLOVSCHUR);                          CHKERRQ(ierr);  // Set default solver to Jacobi-Davidson
    return ierr;
}


PetscErrorCode SLEPc::EpS::PETSCMatShellCreate(Mat &matrix) {
    // only matvec prod supported
    ierr = MatSetFromOptions(matrix);                                                        CHKERRQ(ierr);
    //ierr = MatShellSetOperation(matrix, MATOP_MULT,           (void(*)())Petsc_MatVecProd);  CHKERRQ(ierr);
    //ierr = MatShellSetOperation(matrix, MATOP_MULT_TRANSPOSE, (void(*)())Petsc_MatVecProd);  CHKERRQ(ierr);
    ierr = MatShellSetOperation(matrix, MATOP_MULT,           (void(*)())Petsc_Mv_TripletH);  CHKERRQ(ierr);
    ierr = MatShellSetOperation(matrix, MATOP_MULT_TRANSPOSE, (void(*)())Petsc_Mv_TripletH);  CHKERRQ(ierr);
    //ierr = MatShellSetOperation(matrix, MATOP_GET_DIAGONAL  , (void(*)())Petsc_MatDiags);    CHKERRQ(ierr);
    return ierr;
}

PetscErrorCode SLEPc::EpS::SetFromOptions() {
    ierr = EPSSetFromOptions(eps);  CHKERRQ(ierr);
    return ierr;
}


PetscErrorCode SLEPc::EpS::SetDimensions(PetscInt num_evals, PetscInt max_subspace_size) {
    // set defaults, the last arg, mpd is a max projected dimension and is needed for
    // solving many eigenpairs.
    ierr = EPSSetDimensions(eps, num_evals, max_subspace_size, PETSC_DEFAULT);
    Nevals = num_evals;
    return ierr;
}

PetscErrorCode SLEPc::EpS::SetBlockSize(PetscInt blocksize) {
    ierr = EPSJDSetBlockSize(eps, blocksize);   CHKERRQ(ierr);
    BlockSize = blocksize;
    return ierr;
}

PetscErrorCode SLEPc::EpS::SetTol(PetscScalar tolerance, int max_it){
    tol = tolerance;
    maxits = max_it;
    ierr = EPSSetTolerances(eps, tol, max_it); CHKERRQ(ierr);
    return ierr;
}

PetscErrorCode SLEPc::EpS::Solve() {
    ierr = EPSSetFromOptions(eps); CHKERRQ(ierr);
    ierr = EPSSolve(eps); CHKERRQ(ierr);

    // Retrieve Solutions
    ierr = EPSGetConverged(eps, &nconv); CHKERRQ(ierr);
    ierr = EPSGetIterationNumber(eps, &niter);
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
        //ierr = VecGetValues(PetscrVec, N, &indices[0], &rVecs[i][0]);
        //ierr = VecGetValues(PetsciVec, N, &indices[0], &iVecs[i][0]);
    }
    return ierr;
}


void SLEPc::EpS::PrintEvals(const char* format) {
    for (int i = 0; i < nconv; ++i) {
        printf(format, rVals[i]);
    }
}

void SLEPc::EpS::PrintEvecs(const char* format) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < nconv; ++j) {
            printf(format, rVecs[j][i]);
        }
        printf("\n");
    }
}

PetscErrorCode SLEPc::EpS::print() {
    ierr = PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD, PETSC_VIEWER_ASCII_INFO_DETAIL);    CHKERRQ(ierr);
    ierr = EPSReasonView(eps, PETSC_VIEWER_STDOUT_WORLD);                                       CHKERRQ(ierr);
    ierr = EPSErrorView(eps, EPS_ERROR_RELATIVE,PETSC_VIEWER_STDOUT_WORLD);                     CHKERRQ(ierr);
    ierr = PetscViewerPopFormat(PETSC_VIEWER_STDOUT_WORLD);                                     CHKERRQ(ierr);
    return ierr;
}



#undef __FUNCT__
#define __FUNCT__ "Petsc_MatVecProd"
PetscErrorCode SLEPc::Petsc_MatVecProd(Mat matrix, Vec x, Vec y) {
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


    PetscScalar* v = (PetscScalar*) &px[0];        // cast to non-const pointer for armadillo initialization
    PetscScalar* Mv = (PetscScalar*) &py[0];
    arma::vec myvec(v, nx, false, true); // copy_aux_mem (first bool) must be false for this to work
    arma::vec myMv(Mv, nx, false, true); // since the memory pointed to by y is what PETSc sees.
    SLEPc::matvec_product(myvec, myMv);  // modifies myMv inplace
    ierr = VecRestoreArrayRead(x, &px);         CHKERRQ(ierr);
    ierr = VecRestoreArray(y, &py);             CHKERRQ(ierr);
    HFS::N_MV_PROD += 1;
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "Petsc_MatDiags"
PetscErrorCode SLEPc::Petsc_MatDiags(Mat M, Vec diag) {

    PetscErrorCode    ierr;
    PetscInt N;

    PetscInt indices[HFS::Nmat];
    const PetscScalar* values;
    N = static_cast<PetscInt> (HFS::Nmat);

    PetscFunctionBeginUser;
    for (unsigned i = 0; i < HFS::Nmat; ++i) {
        indices[i] = i;
    }

    values = HFS::exc_energies.memptr();
    ierr = VecSetValues(diag, N, indices, values, INSERT_VALUES);CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "Petsc_Mv_TripletH"
PetscErrorCode SLEPc::Petsc_Mv_TripletH(Mat M, Vec v, Vec Mv) {
    const PetscScalar* v_ptr, *v_local_copy_ptr;
    PetscScalar* Mv_ptr;
    PetscErrorCode ierr;
    PetscInt Mvstart, Mvend, vstart, vend, i, local_idx;

    Vec v_local_copy;
    VecScatter ctx;


    PetscFunctionBeginUser;
    ierr = VecGetArrayRead(v, &v_ptr);                 CHKERRQ(ierr);
    ierr = VecGetArray(Mv, &Mv_ptr);                   CHKERRQ(ierr);

    ierr = VecGetOwnershipRange(v, &vstart, &vend);    CHKERRQ(ierr);
    ierr = VecGetOwnershipRange(Mv, &Mvstart, &Mvend); CHKERRQ(ierr);

    VecScatterCreateToAll(v, &ctx, &v_local_copy);
    VecScatterBegin(ctx, v, v_local_copy, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(ctx, v, v_local_copy, INSERT_VALUES, SCATTER_FORWARD);

    ierr = VecGetArrayRead(v_local_copy, &v_local_copy_ptr);

    assert(Mvstart == vstart && Mvend == vend);

    std::cout << "vstart = " << vstart << " vend = " << vend;// << std::endl;
    printf(" 0x%016x\n", v_ptr);


    for (i = vstart, local_idx = 0; i < vend; ++i, ++local_idx) {
        if (i < 2*HFS::Nexc) {
            Mv_ptr[local_idx] = 0.0;
            // [ A B ] Portion
            if (i < HFS::Nexc) {
                // A Portion
                for (PetscInt j = 0; j < HFS::Nexc; ++j) {
                    Mv_ptr[local_idx] += HFS::Matrix::Gen::A_E_delta_ij_delta_ab_minus_aj_bi(i, j) * v_local_copy_ptr[j];
                }
                // B Portion
                for (PetscInt j = HFS::Nexc; j < 2*HFS::Nexc; ++j) {
                    Mv_ptr[local_idx] += HFS::Matrix::Gen::B_minus_ab_ji(i, j - HFS::Nexc) * v_local_copy_ptr[j];
                }

            // [ B A ] Portion
            } else {

                // B Portion
                for (PetscInt j = 0; j < HFS::Nexc; ++j) {
                    Mv_ptr[local_idx] += HFS::Matrix::Gen::B_minus_ab_ji(i - HFS::Nexc, j) * v_local_copy_ptr[j];
                }
                // A Portion
                for (PetscInt j = HFS::Nexc; j < 2*HFS::Nexc; ++j) {

                    Mv_ptr[local_idx] += HFS::Matrix::Gen::A_E_delta_ij_delta_ab_minus_aj_bi(i - HFS::Nexc, j - HFS::Nexc) * v_local_copy_ptr[j];
                }
            }
        }

    }

    ierr = VecScatterDestroy(&ctx);CHKERRQ(ierr);
    ierr = VecDestroy(&v_local_copy);CHKERRQ(ierr);

    ierr = VecRestoreArrayRead(v, &v_ptr);         CHKERRQ(ierr);
    ierr = VecRestoreArray(Mv, &Mv_ptr);             CHKERRQ(ierr);
    PetscFunctionReturn(0);
}
