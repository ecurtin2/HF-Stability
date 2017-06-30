import sys
import time
import pprint


import numpy as np
import slepc4py
from petsc4py import PETSc
from slepc4py import SLEPc


def matrix_func(i, j):
    if i != j:
        return float(1e-4 * (j + i))
    else:
        return float(3 * i)


class EigenValueProblemWrapper(object):

    def __init__(self, n_rows, n_cols, mpi_comm=PETSc.COMM_WORLD, n_evals=1):

        self.mpi_comm = mpi_comm
        self.scalar = PETSc.ScalarType

        self.eps = SLEPc.EPS().create(self.mpi_comm)
        self.eps.setProblemType(SLEPc.EPS.ProblemType.HEP)  # Hermitian Eigenvalue Problem

        self.mat = PETSc.Mat()
        self.mat.create(self.mpi_comm)
        self.n_rows, self.n_cols = n_rows, n_cols
        self.mat.setSizes([self.n_rows, self.n_cols])
        self.mat.setType('mpiaij')
        self.mat.setUp()
        self.mat.setPreallocationNNZ(self.n_cols)
        self.mat.setFromOptions()
        self.mat_from_func(matrix_func)

        self.eps.setType(SLEPc.EPS.Type.JD)  # Jacobi - Davidson
        self.eps.setWhichEigenpairs(SLEPc.EPS.Which.SMALLEST_REAL)
        self.eps.max_it = 100

        self.eps.setOperators(self.mat, None)
        self.eps.setDimensions(nev=n_evals, ncv=PETSc.DECIDE, mpd=PETSc.DECIDE)

        self.eps.setFromOptions()

        self.solution_time = None
        self.eigs = None
        self.n_iters = None
        self.n_converged = None

    def mat_from_func(self, f):
        Istart, Iend = self.mat.getOwnershipRange()
        row = np.zeros(self.n_cols)
        print(
        'Rank is {rank:.0f}, Istart is {ist:.0f} and Iend is {ie:.0f}'.format(rank=self.mpi_comm.getRank(), ist=Istart,
                                                                              ie=Iend))
        for i in range(Istart, Iend):
            for j in range(self.n_cols):
                row[j] = f(i, j)

        self.mat.assemblyBegin()
        self.mat.assemblyEnd()

    def set_initial_space(self, n_initial=1, which='identity'):
        """Initial guess vectors, create from mat to ensure compatible parallel layout."""

        vecs = [self.mat.createVecLeft() for _ in range(n_initial)]

        if which == 'identity':
            for i, v in enumerate(vecs):
                v[i] = 1.0
                v.assemblyBegin()
                v.assemblyEnd()

        self.eps.setInitialSpace(vecs)

    def solve(self):
        t = time.time()
        self.eps.solve()
        self.solution_time = time.time() - t
        self.n_iters = self.eps.getIterationNumber()
        self.n_converged = self.eps.getConverged()
        if self.n_converged > 0:
            for i in range(self.n_converged):
                self.eigs.append(self.eps.getEigenpair(i))

    def __str__(self):
        return "Eigenvalue Problem Solver:\n" + pprint.pformat(self.__dict__)


def main():

    N = 80
    EVPW = EigenValueProblemWrapper(N, N)
    EVPW.mat_from_func(matrix_func)
    # EVPW.solve()
    print(EVPW)

    if EVPW.mpi_comm.getRank() == 0:
        A = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                A[i, j] = matrix_func(i, j)
        t1 = time.time()
        evals, evecs = np.linalg.eigh(A)

        print('np min eval = {mineval:10.8f}'.format(mineval=np.amin(evals)))
        print("Np took this long: ", time.time() - t1)


if __name__ == '__main__':
    main()
