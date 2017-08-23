#include "SLEPcWrapper.hpp"
#include "matrix_vectorproducts.hpp"

SLEPc::EpS::EpS(PetscInt Ninput, PetscErrorCode (*matvec_product)(Mat, Vec, Vec), int argc, char* argv[]) {
    static char help[] = "Solves the same eigenproblem as in example ex2, but using a shell matrix. "
                         "The problem is a standard symmetric eigenproblem corresponding to the 2-D Laplacian operator.\n\n"
                         "The command line options are:\n"
                         "  -n <n>, where <n> = number of grid subdivisions in both x and y dimensions.\n\n";


    N = Ninput;
    SlepcInitialize(&argc, &argv, (char*)0, help);
    MPI_Comm_size(PETSC_COMM_WORLD, &nprocs);
    MatCreateShell(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, N, N, &N, &matrix);
    PETSCMatShellCreate(matrix, matvec_product);
    EPSCreate(PETSC_COMM_WORLD, &eps);
    EPSContext();
}

SLEPc::EpS::~EpS() {
    ierr = EPSDestroy(&eps);    //CHKERRQ(ierr);
    ierr = MatDestroy(&matrix); //CHKERRQ(ierr);
    //ierr = SlepcFinalize();
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
    ierr = EPSSetType(eps, EPSJD);                          CHKERRQ(ierr);  // Set default solver to Jacobi-Davidson
    return ierr;
}


PetscErrorCode SLEPc::EpS::PETSCMatShellCreate(Mat &matrix, PetscErrorCode (*matvec_product)(Mat, Vec, Vec)) {
    // only matvec prod supported
    ierr = MatSetFromOptions(matrix);                                                        CHKERRQ(ierr);
    ierr = MatShellSetOperation(matrix, MATOP_MULT,           (void(*)())matvec_product);  CHKERRQ(ierr);
    ierr = MatShellSetOperation(matrix, MATOP_MULT_TRANSPOSE, (void(*)())matvec_product);  CHKERRQ(ierr);
    //ierr = MatShellSetOperation(matrix, MATOP_MULT,           (void(*)())Petsc_Mv_TripletH);  CHKERRQ(ierr);
    //ierr = MatShellSetOperation(matrix, MATOP_MULT_TRANSPOSE, (void(*)())Petsc_Mv_TripletH);  CHKERRQ(ierr);
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
    ierr = EPSSetUp(eps);
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


class PetscLocalVec {
/* Class wrapping the local access to a Petsc Vector. It is indexed
using the global index, and the conversion to the local index is
handled behind the scenes. Basically, it looks basically the same
as a normal vector but should parallelize automatically.
*/
    public:
        PetscLocalVec (Vec* v, bool Copy_to_Local) {
            parent = v;
            is_local_copy = Copy_to_Local;

            if (is_local_copy) {
                ierr = VecScatterCreateToAll(*parent, &VecScatterCtx, &LocalVec);//CHKERRQ(ierr);
                ierr = VecScatterBegin(VecScatterCtx, *parent, LocalVec, INSERT_VALUES, SCATTER_FORWARD);//CHKERRQ(ierr);
                ierr = VecScatterEnd(VecScatterCtx, *parent, LocalVec, INSERT_VALUES, SCATTER_FORWARD);//CHKERRQ(ierr);
            } else {
                LocalVec = *parent;
            }
            ierr = VecGetArray(LocalVec, &_data);//CHKERRQ(ierr);
            ierr = VecGetOwnershipRange(LocalVec, &global_begin_idx, &global_end_idx);//CHKERRQ(ierr);
            _size = global_end_idx - global_begin_idx;
        }

        PetscInt size() { return _size; }
        PetscScalar * data() { return _data; }
        PetscInt begin() { return global_begin_idx; }
        PetscInt end() { return global_end_idx; }
        PetscScalar& operator[] (PetscInt i) {
            PetscInt local_i = i - global_begin_idx;
            assert( (local_i >= 0) && (local_i < _size) && "PetscLocalVec index out of bounds!");
            return _data[i - global_begin_idx];
        }

        void cleanup () {
            if (is_local_copy) {
                ierr = VecScatterDestroy(&VecScatterCtx);//CHKERRQ(ierr);
                ierr = VecDestroy(&LocalVec);//CHKERRQ(ierr);
            } else {
                ierr = VecRestoreArray(*parent, &_data);//CHKERRQ(ierr);
            }

        }


    private:
        PetscScalar* _data;
        PetscErrorCode ierr;
        VecScatter VecScatterCtx;
        Vec *parent, LocalVec;
        PetscInt global_begin_idx, global_end_idx, _size;
        bool is_local_copy;
};


class PetscLocalVecReadOnly {
/* Class wrapping the local access to a Petsc Vector. It is indexed
using the global index, and the conversion to the local index is
handled behind the scenes. Basically, it looks basically the same
as a normal vector but should parallelize automatically.
*/
    public:
        PetscLocalVecReadOnly (Vec* v, bool Copy_to_Local) {
            parent = v;
            is_local_copy = Copy_to_Local;

            if (is_local_copy) {
                ierr = VecScatterCreateToAll(*parent, &VecScatterCtx, &LocalVec);//CHKERRQ(ierr);
                ierr = VecScatterBegin(VecScatterCtx, *parent, LocalVec, INSERT_VALUES, SCATTER_FORWARD);//CHKERRQ(ierr);
                ierr = VecScatterEnd(VecScatterCtx, *parent, LocalVec, INSERT_VALUES, SCATTER_FORWARD);//CHKERRQ(ierr);
            } else {
                LocalVec = *parent;
            }
            ierr = VecGetArrayRead(LocalVec, &_data);//CHKERRQ(ierr);
            ierr = VecGetOwnershipRange(LocalVec, &global_begin_idx, &global_end_idx);//CHKERRQ(ierr);
            _size = global_end_idx - global_begin_idx;
        }

        PetscInt size() { return _size; }
        PetscInt begin() { return global_begin_idx; }
        PetscInt end() { return global_end_idx; }
        const PetscScalar* data() {return _data; }
        PetscScalar operator[] (PetscInt i) {
            PetscInt local_i = i - global_begin_idx;
            assert( (local_i >= 0) && (local_i < _size) && "PetscLocalVec index out of bounds!");
            return _data[i - global_begin_idx];
        }

        void cleanup () {
            if (is_local_copy) {
                ierr = VecScatterDestroy(&VecScatterCtx);//CHKERRQ(ierr);
                ierr = VecDestroy(&LocalVec);//CHKERRQ(ierr);
            } else {
                ierr = VecRestoreArrayRead(*parent, &_data);//CHKERRQ(ierr);
            }

        }


    private:
        const PetscScalar* _data;
        PetscErrorCode ierr;
        VecScatter VecScatterCtx;
        Vec* parent;
        Vec LocalVec;
        PetscInt global_begin_idx, global_end_idx, _size;
        bool is_local_copy, is_readonly;
};


#undef __FUNCT__
#define __FUNCT__ "Petsc_Mv_Triplet_A_Minus_B"
PetscErrorCode SLEPc::Petsc_Mv_Triplet_A_Minus_B(Mat M, Vec v, Vec Mv) {
    PetscFunctionBeginUser;
    PetscLocalVec LocalMv(&Mv, false);
    PetscLocalVecReadOnly LocalV(&v, true);


    for (PetscInt s = LocalMv.begin(); s < LocalMv.end(); ++s) {
        LocalMv[s] = 0.0;

        if (s < HFS::Nexc) { // due to padding local may have extra elements
                             // Happens when matrix size not divisible by n procs.

            uint i = HFS::excitations(0, s), a = HFS::excitations(1, s);

            arma::vec ki(NDIM), ka(NDIM);
            HFS::occIndexToK(i, ki);
            ka = HFS::virIndexToK(a);
            for (uint j = 0; j < HFS::Nocc; ++j) {
                arma::vec kj(NDIM), kb(NDIM);
                HFS::occIndexToK(j, kj);



                // The contribution due to A
                kb = ka + kj - ki; // Momentum conservation for <aj|ib> or <aj|bi>
                HFS::toFirstBrillouinZone(kb);
                if (arma::norm(kb) > (HFS::kf + SMALLNUMBER)) {
                    // only if momentum conserving state is virtual
                    uint t = HFS::Matrix::calcTfromKbAndJ(kb, j);
                    if (s == t) {
                        LocalMv[s] += HFS::exc_energies(s) * LocalV[t];
                    } else {
                        LocalMv[s] += -HFS::twoElectron(ka, kb) * LocalV[t];
                    }
                } // if

                // The contribution due to B
                kb = ki +  kj - ka; // Momentum conservation for <aj|ib> or <aj|bi>
                HFS::toFirstBrillouinZone(kb);
                if (arma::norm(kb) > (HFS::kf + SMALLNUMBER)) {
                    // only if momentum conserving state is virtual
                    uint t = HFS::Matrix::calcTfromKbAndJ(kb, j);
                    LocalMv[s] += HFS::twoElectron(ka, kj) * LocalV[t];
                } // if

            } // j
        }
    } // s

    LocalMv.cleanup();
    LocalV.cleanup();
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "Petsc_Mv_Triplet_A_Plus_B"
PetscErrorCode SLEPc::Petsc_Mv_Triplet_A_Plus_B(Mat M, Vec v, Vec Mv) {
    PetscFunctionBeginUser;
    PetscLocalVec LocalMv(&Mv, false);
    PetscLocalVecReadOnly LocalV(&v, true);

    for (PetscInt s = LocalMv.begin(); s < LocalMv.end(); ++s) {
        LocalMv[s] = 0.0;
        if (s < HFS::Nexc) { // due to padding local may have extra elements
                             // Happens when matrix size not divisible by n procs.

            uint i = HFS::excitations(0, s), a = HFS::excitations(1, s);

            arma::vec ki(NDIM), ka(NDIM);
            HFS::occIndexToK(i, ki);
            ka = HFS::virIndexToK(a);
            for (uint j = 0; j < HFS::Nocc; ++j) {
                arma::vec kj(NDIM), kb(NDIM);
                HFS::occIndexToK(j, kj);



                // The contribution due to A
                kb = ka + kj - ki; // Momentum conservation for <aj|ib> or <aj|bi>
                HFS::toFirstBrillouinZone(kb);
                if (arma::norm(kb) > (HFS::kf + SMALLNUMBER)) {
                    // only if momentum conserving state is virtual
                    uint t = HFS::Matrix::calcTfromKbAndJ(kb, j);
                    if (s == t) {
                        LocalMv[s] += HFS::exc_energies(s) * LocalV[t];
                    }
                    LocalMv[s] += -HFS::twoElectron(ka, kb) * LocalV[t];

                } // if

                // The contribution due to B
                kb = ki +  kj - ka; // Momentum conservation for <aj|ib> or <aj|bi>
                HFS::toFirstBrillouinZone(kb);
                if (arma::norm(kb) > (HFS::kf + SMALLNUMBER)) {
                    // only if momentum conserving state is virtual
                    uint t = HFS::Matrix::calcTfromKbAndJ(kb, j);
                    LocalMv[s] += -HFS::twoElectron(ka, kj) * LocalV[t];
                } // if

            } // j
        }
    } // s

    LocalMv.cleanup();
    LocalV.cleanup();
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "Petsc_Mv_Singlet_A_Minus_B"
PetscErrorCode SLEPc::Petsc_Mv_Singlet_A_Minus_B(Mat M, Vec v, Vec Mv) {
    PetscFunctionBeginUser;
    PetscLocalVec LocalMv(&Mv, false);
    PetscLocalVecReadOnly LocalV(&v, true);

    for (PetscInt s = LocalMv.begin(); s < LocalMv.end(); ++s) {
        LocalMv[s] = 0.0;
        if (s < HFS::Nexc) { // due to padding local may have extra elements
                             // Happens when matrix size not divisible by n procs.

            uint i = HFS::excitations(0, s), a = HFS::excitations(1, s);

            arma::vec ki(NDIM), ka(NDIM);
            HFS::occIndexToK(i, ki);
            ka = HFS::virIndexToK(a);
            for (uint j = 0; j < HFS::Nocc; ++j) {
                arma::vec kj(NDIM), kb(NDIM);
                HFS::occIndexToK(j, kj);



                // The contribution due to A
                kb = ka + kj - ki; // Momentum conservation for <aj|ib> or <aj|bi>
                HFS::toFirstBrillouinZone(kb);
                if (arma::norm(kb) > (HFS::kf + SMALLNUMBER)) {
                    // only if momentum conserving state is virtual
                    uint t = HFS::Matrix::calcTfromKbAndJ(kb, j);
                    if (s == t) {
                        LocalMv[s] += HFS::exc_energies(s) * LocalV[t];
                    }
                    LocalMv[s] += (2.0 * HFS::twoElectron(ka, ki) - HFS::twoElectron(ka, kb)) * LocalV[t];

                } // if

                // The contribution due to B
                kb = ki +  kj - ka; // Momentum conservation for <aj|ib> or <aj|bi>
                HFS::toFirstBrillouinZone(kb);
                if (arma::norm(kb) > (HFS::kf + SMALLNUMBER)) {
                    // only if momentum conserving state is virtual
                    uint t = HFS::Matrix::calcTfromKbAndJ(kb, j);
                    LocalMv[s] +=  - (2.0 * HFS::twoElectron(ka, ki) - HFS::twoElectron(ka, kj)) * LocalV[t];
                } // if

            } // j
        }
    } // s

    LocalMv.cleanup();
    LocalV.cleanup();
    PetscFunctionReturn(0);
}



#undef __FUNCT__
#define __FUNCT__ "Petsc_Mv_Singlet_A_Plus_B"
PetscErrorCode SLEPc::Petsc_Mv_Singlet_A_Plus_B(Mat M, Vec v, Vec Mv) {
    PetscFunctionBeginUser;
    PetscLocalVec LocalMv(&Mv, false);
    PetscLocalVecReadOnly LocalV(&v, true);

    for (PetscInt s = LocalMv.begin(); s < LocalMv.end(); ++s) {
        LocalMv[s] = 0.0;
        if (s < HFS::Nexc) { // due to padding local may have extra elements
                             // Happens when matrix size not divisible by n procs.
            uint i = HFS::excitations(0, s), a = HFS::excitations(1, s);

            arma::vec ki(NDIM), ka(NDIM);
            HFS::occIndexToK(i, ki);
            ka = HFS::virIndexToK(a);
            for (uint j = 0; j < HFS::Nocc; ++j) {
                arma::vec kj(NDIM), kb(NDIM);
                HFS::occIndexToK(j, kj);



                // The contribution due to A
                kb = ka + kj - ki; // Momentum conservation for <aj|ib> or <aj|bi>
                HFS::toFirstBrillouinZone(kb);
                if (arma::norm(kb) > (HFS::kf + SMALLNUMBER)) {
                    // only if momentum conserving state is virtual
                    uint t = HFS::Matrix::calcTfromKbAndJ(kb, j);
                    if (s == t) {
                        LocalMv[s] += HFS::exc_energies(s) * LocalV[t];
                    }
                    LocalMv[s] += (2.0 * HFS::twoElectron(ka, ki) - HFS::twoElectron(ka, kb)) * LocalV[t];
                } // if

                // The contribution due to B
                kb = ki +  kj - ka; // Momentum conservation for <aj|ib> or <aj|bi>
                HFS::toFirstBrillouinZone(kb);
                if (arma::norm(kb) > (HFS::kf + SMALLNUMBER)) {
                    // only if momentum conserving state is virtual
                    uint t = HFS::Matrix::calcTfromKbAndJ(kb, j);
                    LocalMv[s] += (2.0 * HFS::twoElectron(ka, ki) - HFS::twoElectron(ka, kj)) * LocalV[t];
                } // if

            } // j
        }
    } // s

    LocalMv.cleanup();
    LocalV.cleanup();
    PetscFunctionReturn(0);
}

